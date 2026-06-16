"""
hand.py — Dual hand tracking with full finger state and proper 3D orientation.

Each HandState exposes:
  index_tip         (x,y) normalized
  all_tips          list of (x,y) for all 5 fingertips [thumb,index,middle,ring,pinky]
  is_pinching       bool
  pinch_strength    0..1
  is_pointing       bool  — index extended, others curled
  orient_x/y/z      float radians — pitch / yaw / roll of the hand
  delta_ox/oy/oz    frame-to-frame orientation deltas (drive cube rotation)
  wrist_pos         (x,y,z) normalized
  visible           bool

HandTracker holds .left and .right HandState, updated each frame.
"""

import math


def _dist2d(a, b):
    ax, ay = float(a[0]), float(a[1])
    bx, by = float(b[0]), float(b[1])
    return math.sqrt((ax - bx) ** 2 + (ay - by) ** 2)


def _dist3d(a, b):
    return math.sqrt(sum((float(a[i]) - float(b[i])) ** 2 for i in range(3)))


def _lerp(a, b, t):
    return a + (b - a) * t


# MediaPipe landmark indices
WRIST      = 0
THUMB_CMC  = 1;  THUMB_MCP  = 2;  THUMB_IP   = 3;  THUMB_TIP  = 4
INDEX_MCP  = 5;  INDEX_PIP  = 6;  INDEX_DIP  = 7;  INDEX_TIP  = 8
MIDDLE_MCP = 9;  MIDDLE_PIP = 10; MIDDLE_DIP = 11; MIDDLE_TIP = 12
RING_MCP   = 13; RING_PIP   = 14; RING_DIP   = 15; RING_TIP   = 16
PINKY_MCP  = 17; PINKY_PIP  = 18; PINKY_DIP  = 19; PINKY_TIP  = 20

# Fingertip and base indices (for extension detection)
FINGERTIPS = [THUMB_TIP, INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP]
FINGER_MCPS = [THUMB_MCP, INDEX_MCP, MIDDLE_MCP, RING_MCP, PINKY_MCP]
FINGER_PIPS = [THUMB_IP,  INDEX_PIP, MIDDLE_PIP, RING_PIP, PINKY_PIP]


class HandState:
    """Tracks one hand — left or right."""

    PINCH_CLOSE    = 0.11
    PINCH_OPEN     = 0.17
    CONFIRM_FRAMES = 2
    RELEASE_FRAMES = 3
    TIP_SMOOTH     = 0.45
    PINCH_SMOOTH   = 0.35
    ORIENT_SMOOTH  = 0.30

    def __init__(self, label="right"):
        self.label = label        # "left" | "right"
        self.visible = False

        # Cursor (index tip, normalized)
        self.index_tip = (0.5, 0.5)

        # All 5 fingertips [(x,y) normalized]
        self.all_tips = [(0.5, 0.5)] * 5

        # Which fingers are extended (bool per finger: thumb→pinky)
        self.fingers_extended = [False] * 5

        # Pinch state
        self.is_pinching    = False
        self.pinch_strength = 0.0

        # Gesture flags
        self.is_pointing    = False   # index only extended
        self.is_open        = False   # all fingers extended
        self.is_fist        = False   # all fingers curled
        self.num_extended   = 0

        # 3D orientation (radians)
        self.orient_x = 0.0   # pitch  (hand tilts toward/away from camera)
        self.orient_y = 0.0   # yaw    (hand swings left/right)
        self.orient_z = 0.0   # roll   (hand rolls CW/CCW)

        # Wrist world position (normalized, for plane anchoring)
        self.wrist_pos = (0.5, 0.5, 0.0)

        # Raw landmarks (21 points)
        self.landmarks = None

        # ── Private state ──────────────────────────────────────────────
        self._smooth_index  = None
        self._pinch_raw     = 1.0
        self._pinching      = False
        self._confirm_count = 0
        self._release_count = 0

        self._prev_ox = 0.0
        self._prev_oy = 0.0
        self._prev_oz = 0.0
        self._sx = 0.0
        self._sy = 0.0
        self._sz = 0.0

    # ── Public deltas ──────────────────────────────────────────────────

    @property
    def delta_ox(self):
        return self.orient_x - self._prev_ox

    @property
    def delta_oy(self):
        return self.orient_y - self._prev_oy

    @property
    def delta_oz(self):
        return self.orient_z - self._prev_oz

    # ── Update ─────────────────────────────────────────────────────────

    def update(self, landmarks):
        if landmarks is None:
            self.visible = False
            return

        self.visible   = True
        self.landmarks = landmarks
        lm = landmarks

        # ── Wrist ──────────────────────────────────────────────────────
        w  = lm[WRIST]
        mm = lm[MIDDLE_MCP]
        self.wrist_pos = (float(w[0]), float(w[1]), float(w[2]))

        # ── Hand scale (wrist → middle MCP distance) ───────────────────
        scale = max(0.04, _dist2d(w, mm))

        # ── All fingertips ─────────────────────────────────────────────
        self.all_tips = [(float(lm[t][0]), float(lm[t][1])) for t in FINGERTIPS]

        # ── Index tip (smoothed) ───────────────────────────────────────
        raw = (float(lm[INDEX_TIP][0]), float(lm[INDEX_TIP][1]))
        if self._smooth_index is None:
            self._smooth_index = raw
        else:
            t = self.TIP_SMOOTH
            self._smooth_index = (
                _lerp(self._smooth_index[0], raw[0], t),
                _lerp(self._smooth_index[1], raw[1], t),
            )
        self.index_tip = self._smooth_index

        # ── Finger extension detection ─────────────────────────────────
        # Thumb: compare tip x vs MCP x (depends on handedness, use distance)
        thumb_ext = _dist2d(lm[THUMB_TIP], lm[THUMB_CMC]) > _dist2d(lm[THUMB_IP], lm[THUMB_CMC])
        exts = [thumb_ext]
        for tip_idx, pip_idx, mcp_idx in [
            (INDEX_TIP,  INDEX_PIP,  INDEX_MCP),
            (MIDDLE_TIP, MIDDLE_PIP, MIDDLE_MCP),
            (RING_TIP,   RING_PIP,   RING_MCP),
            (PINKY_TIP,  PINKY_PIP,  PINKY_MCP),
        ]:
            # Extended if tip is farther from wrist than PIP
            ext = _dist2d(lm[tip_idx], w) > _dist2d(lm[pip_idx], w) * 1.05
            exts.append(ext)

        self.fingers_extended = exts
        self.num_extended     = sum(exts)
        self.is_pointing      = exts[1] and not exts[2] and not exts[3] and not exts[4]
        self.is_open          = all(exts[1:])   # 4 fingers (ignore thumb)
        self.is_fist          = not any(exts[1:])

        # ── Pinch (index+thumb OR middle+thumb) ────────────────────────
        d_index  = _dist2d(lm[THUMB_TIP], lm[INDEX_TIP])  / scale
        d_middle = _dist2d(lm[THUMB_TIP], lm[MIDDLE_TIP]) / scale
        raw_dist = min(d_index, d_middle)
        self._pinch_raw = _lerp(self._pinch_raw, raw_dist, self.PINCH_SMOOTH)

        below_close = self._pinch_raw < self.PINCH_CLOSE
        above_open  = self._pinch_raw > self.PINCH_OPEN

        if not self._pinching:
            if below_close:
                self._confirm_count += 1
                self._release_count  = 0
                if self._confirm_count >= self.CONFIRM_FRAMES:
                    self._pinching      = True
                    self._confirm_count = 0
            else:
                self._confirm_count = max(0, self._confirm_count - 1)
        else:
            if above_open:
                self._release_count += 1
                self._confirm_count  = 0
                if self._release_count >= self.RELEASE_FRAMES:
                    self._pinching      = False
                    self._release_count = 0
            else:
                self._release_count = max(0, self._release_count - 1)

        self.is_pinching    = self._pinching
        self.pinch_strength = max(0.0, min(1.0,
            1.0 - (self._pinch_raw / self.PINCH_OPEN)
        ))

        # ── Hand orientation ──────────────────────────────────────────
        # Spine vector: wrist → middle MCP (gives yaw + pitch base)
        wx, wy, wz = float(w[0]), float(w[1]), float(w[2])
        mx, my     = float(mm[0]), float(mm[1])
        mmz        = float(mm[2])

        spine_x = mx - wx
        spine_y = my - wy

        # Yaw: spine angle in screen plane
        raw_oy = math.atan2(spine_x, -spine_y)

        # Pitch: Z depth difference between wrist and middle MCP
        raw_ox = (wz - mmz) * 8.0

        # Roll: angle of index MCP → pinky MCP axis
        ix5,  iy5  = float(lm[INDEX_MCP][0]),  float(lm[INDEX_MCP][1])
        px17, py17 = float(lm[PINKY_MCP][0]),  float(lm[PINKY_MCP][1])
        raw_oz = math.atan2(py17 - iy5, px17 - ix5)

        # Smooth orientations
        t = self.ORIENT_SMOOTH
        self._sx = _lerp(self._sx, raw_ox, t)
        self._sy = _lerp(self._sy, raw_oy, t)
        self._sz = _lerp(self._sz, raw_oz, t)

        self._prev_ox = self.orient_x
        self._prev_oy = self.orient_y
        self._prev_oz = self.orient_z

        self.orient_x = self._sx
        self.orient_y = self._sy
        self.orient_z = self._sz

    def debug_str(self):
        fingers = "".join(
            c if e else "·"
            for c, e in zip("TIMRP", self.fingers_extended)
        )
        gesture = "FIST" if self.is_fist else ("POINT" if self.is_pointing else
                  ("OPEN" if self.is_open else f"{self.num_extended}up"))
        return (
            f"[{self.label.upper()}] {gesture} [{fingers}]  "
            f"pinch={self._pinch_raw:.2f}  "
            f"ox={self.orient_x:.2f} oy={self.orient_y:.2f} oz={self.orient_z:.2f}"
        )


class HandTracker:
    """Wraps MediaPipe result → two HandState objects."""

    def __init__(self):
        self.left  = HandState("left")
        self.right = HandState("right")

    def update(self, result):
        """
        result: mediapipe HandLandmarker result.
        Reads result.hand_landmarks and result.handedness to route to left/right.
        """
        left_lm  = None
        right_lm = None

        if result is not None and result.hand_landmarks:
            for i, hand_lms in enumerate(result.hand_landmarks):
                lm_list = [(lm.x, lm.y, lm.z) for lm in hand_lms]

                # handedness[i].classification[0].label is "Left" or "Right"
                # Note: MediaPipe labels are from the camera's perspective (mirrored).
                # After cv2.flip(frame,1) the Left label = user's right hand.
                label = "unknown"
                if result.handedness and i < len(result.handedness):
                    h = result.handedness[i]
                    # MediaPipe Tasks API: handedness[i] is a list of Category objects
                    # each Category has .category_name (or .label on older builds)
                    if isinstance(h, list) and len(h) > 0:
                        cat = h[0]
                        label = (getattr(cat, "category_name", None)
                                 or getattr(cat, "label", "unknown")).lower()
                    elif hasattr(h, "classification"):
                        label = h.classification[0].label.lower()

                # After horizontal flip: "left" from MP = user's right and vice versa
                if label == "left":
                    right_lm = lm_list   # flipped → user's right
                elif label == "right":
                    left_lm  = lm_list   # flipped → user's left
                else:
                    # Fallback: assign by x position of wrist
                    if right_lm is None:
                        right_lm = lm_list
                    else:
                        left_lm  = lm_list

        self.left.update(left_lm)
        self.right.update(right_lm)

    @property
    def any_visible(self):
        return self.left.visible or self.right.visible