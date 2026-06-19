"""
hand.py — Phase 8: Gesture detection for spatial construction.

Each HandState exposes the gestures the interaction system is built on:

  is_open        all 5 fingers extended           -> spawn CUBE
  is_ok          thumb+index touching, M/R/P out   -> spawn SPHERE
  is_peace       index+middle extended only        -> spawn CYLINDER
  is_three       index+middle+ring extended        -> spawn PRISM
  is_four        index..pinky extended, no thumb   -> spawn PLANE

  is_grab        thumb+index+middle all touching   -> PRIMARY GRAB
  is_pointing    index only, thumb tucked          -> hover / connect
  is_finger_gun  index+thumb extended, rest closed -> DELETE
  is_fist        all four fingers curled           -> LEFT: rotate modifier
  is_pinching    thumb+index/middle pinch (2-pt)    -> LEFT: scale modifier

  orient_x/y/z, delta_ox/oy/oz   hand pitch/yaw/roll + frame deltas
  index_tip, wrist_pos           cursor + 3D anchor

Every gesture uses hysteresis (separate confirm/release frame counts) so
nothing flickers on a single noisy frame.
"""

import math


def _dist2d(a, b):
    ax, ay = float(a[0]), float(a[1])
    bx, by = float(b[0]), float(b[1])
    return math.sqrt((ax - bx) ** 2 + (ay - by) ** 2)


def _lerp(a, b, t):
    return a + (b - a) * t


WRIST      = 0
THUMB_CMC  = 1;  THUMB_MCP  = 2;  THUMB_IP   = 3;  THUMB_TIP  = 4
INDEX_MCP  = 5;  INDEX_PIP  = 6;  INDEX_DIP  = 7;  INDEX_TIP  = 8
MIDDLE_MCP = 9;  MIDDLE_PIP = 10; MIDDLE_DIP = 11; MIDDLE_TIP = 12
RING_MCP   = 13; RING_PIP   = 14; RING_DIP   = 15; RING_TIP   = 16
PINKY_MCP  = 17; PINKY_PIP  = 18; PINKY_DIP  = 19; PINKY_TIP  = 20

FINGERTIPS = [THUMB_TIP, INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP]


SHAPE_CUBE     = "cube"
SHAPE_SPHERE   = "sphere"
SHAPE_CYLINDER = "cylinder"
SHAPE_PRISM    = "prism"
SHAPE_PLANE    = "plane"


class HandState:
    """Tracks one hand — left or right — and all gesture flags."""


    PINCH_CLOSE = 0.11
    PINCH_OPEN  = 0.17


    GRAB_CLOSE  = 0.17
    GRAB_OPEN   = 0.25


    OK_CLOSE    = 0.13
    OK_OPEN     = 0.19

    CONFIRM_FRAMES = 2
    RELEASE_FRAMES = 3
    TIP_SMOOTH     = 0.45
    DIST_SMOOTH    = 0.35
    ORIENT_SMOOTH  = 0.30

    def __init__(self, label="right"):
        self.label   = label
        self.visible = False

        self.index_tip = (0.5, 0.5)
        self.all_tips  = [(0.5, 0.5)] * 5
        self.fingers_extended = [False] * 5
        self.num_extended     = 0


        self.is_open        = False
        self.is_fist         = False
        self.is_pointing    = False
        self.is_finger_gun  = False
        self.is_three       = False
        self.is_four        = False
        self.is_ok          = False
        self.is_pinching    = False
        self.is_grab        = False

        self.pinch_strength = 0.0
        self.grab_strength  = 0.0


        self.orient_x = 0.0
        self.orient_y = 0.0
        self.orient_z = 0.0

        self.wrist_pos = (0.5, 0.5, 0.0)
        self.landmarks = None


        self._smooth_index = None
        self._pinch_raw  = 1.0
        self._pinching   = False
        self._pinch_cc   = 0
        self._pinch_rc   = 0

        self._grab_raw   = 1.0
        self._grabbing   = False
        self._grab_cc    = 0
        self._grab_rc    = 0

        self._ok_raw     = 1.0
        self._ok_touch   = False
        self._ok_cc      = 0
        self._ok_rc      = 0

        self._prev_ox = 0.0
        self._prev_oy = 0.0
        self._prev_oz = 0.0
        self._sx = 0.0
        self._sy = 0.0
        self._sz = 0.0


    @property
    def delta_ox(self): return self.orient_x - self._prev_ox

    @property
    def delta_oy(self): return self.orient_y - self._prev_oy

    @property
    def delta_oz(self): return self.orient_z - self._prev_oz


    def classify_shape_gesture(self):
        """Returns one of the SHAPE_* constants, or None."""
        if not self.visible:
            return None
        if self.is_open:
            return SHAPE_CUBE
        if self.is_ok:
            return SHAPE_SPHERE
        if self.is_finger_gun:
            return None
        if self.is_four:
            return SHAPE_PLANE
        if self.is_three:
            return SHAPE_PRISM
        if self.fingers_extended[1] and self.fingers_extended[2] and \
           not self.fingers_extended[0] and not self.fingers_extended[3] \
           and not self.fingers_extended[4]:
            return SHAPE_CYLINDER
        return None


    def update(self, landmarks):
        if landmarks is None:
            self.visible = False
            return

        self.visible   = True
        self.landmarks = landmarks
        lm = landmarks

        w  = lm[WRIST]
        mm = lm[MIDDLE_MCP]
        self.wrist_pos = (float(w[0]), float(w[1]), float(w[2]))
        scale = max(0.04, _dist2d(w, mm))

        self.all_tips = [(float(lm[t][0]), float(lm[t][1])) for t in FINGERTIPS]

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


        thumb_ext = _dist2d(lm[THUMB_TIP], lm[THUMB_CMC]) > _dist2d(lm[THUMB_IP], lm[THUMB_CMC])
        exts = [thumb_ext]
        for tip_idx, pip_idx in [
            (INDEX_TIP,  INDEX_PIP),
            (MIDDLE_TIP, MIDDLE_PIP),
            (RING_TIP,   RING_PIP),
            (PINKY_TIP,  PINKY_PIP),
        ]:
            ext = _dist2d(lm[tip_idx], w) > _dist2d(lm[pip_idx], w) * 1.05
            exts.append(ext)

        self.fingers_extended = exts
        self.num_extended     = sum(exts)
        t_, i_, m_, r_, p_ = exts

        self.is_open       = t_ and i_ and m_ and r_ and p_
        self.is_fist        = not (i_ or m_ or r_ or p_)
        self.is_pointing    = i_ and not m_ and not r_ and not p_ and not t_
        self.is_finger_gun  = i_ and not m_ and not r_ and not p_ and t_
        self.is_three       = i_ and m_ and r_ and not p_ and not t_
        self.is_four        = i_ and m_ and r_ and p_ and not t_


        d_index  = _dist2d(lm[THUMB_TIP], lm[INDEX_TIP])  / scale
        d_middle = _dist2d(lm[THUMB_TIP], lm[MIDDLE_TIP]) / scale
        pinch_raw = min(d_index, d_middle)
        self._pinch_raw = _lerp(self._pinch_raw, pinch_raw, self.DIST_SMOOTH)
        self._pinching, self._pinch_cc, self._pinch_rc = self._hysteresis(
            self._pinch_raw, self._pinching, self._pinch_cc, self._pinch_rc,
            self.PINCH_CLOSE, self.PINCH_OPEN)
        self.is_pinching = self._pinching
        self.pinch_strength = max(0.0, min(1.0, 1.0 - self._pinch_raw / self.PINCH_OPEN))


        ok_raw = d_index
        self._ok_raw = _lerp(self._ok_raw, ok_raw, self.DIST_SMOOTH)
        self._ok_touch, self._ok_cc, self._ok_rc = self._hysteresis(
            self._ok_raw, self._ok_touch, self._ok_cc, self._ok_rc,
            self.OK_CLOSE, self.OK_OPEN)
        self.is_ok = self._ok_touch and m_ and r_ and p_ and not i_


        d_ti = d_index
        d_tm = d_middle
        d_im = _dist2d(lm[INDEX_TIP], lm[MIDDLE_TIP]) / scale
        grab_raw = (d_ti + d_tm + d_im) / 3.0
        self._grab_raw = _lerp(self._grab_raw, grab_raw, self.DIST_SMOOTH)
        self._grabbing, self._grab_cc, self._grab_rc = self._hysteresis(
            self._grab_raw, self._grabbing, self._grab_cc, self._grab_rc,
            self.GRAB_CLOSE, self.GRAB_OPEN)
        self.is_grab = self._grabbing
        self.grab_strength = max(0.0, min(1.0, 1.0 - self._grab_raw / self.GRAB_OPEN))


        if not (i_ and m_):
            self.is_grab = False


        wx, wy, wz = float(w[0]), float(w[1]), float(w[2])
        mx, my, mmz = float(mm[0]), float(mm[1]), float(mm[2])

        spine_x = mx - wx
        spine_y = my - wy
        raw_oy  = math.atan2(spine_x, -spine_y)
        raw_ox  = (wz - mmz) * 8.0

        ix5,  iy5  = float(lm[INDEX_MCP][0]), float(lm[INDEX_MCP][1])
        px17, py17 = float(lm[PINKY_MCP][0]), float(lm[PINKY_MCP][1])
        raw_oz = math.atan2(py17 - iy5, px17 - ix5)

        t = self.ORIENT_SMOOTH
        self._sx = _lerp(self._sx, raw_ox, t)
        self._sy = _lerp(self._sy, raw_oy, t)
        self._sz = _lerp(self._sz, raw_oz, t)

        self._prev_ox, self._prev_oy, self._prev_oz = self.orient_x, self.orient_y, self.orient_z
        self.orient_x, self.orient_y, self.orient_z = self._sx, self._sy, self._sz

    @staticmethod
    def _hysteresis(raw_dist, active, confirm_count, release_count,
                     close_thresh, open_thresh):
        """
        Generic confirm/release hysteresis for a 'closeness' metric.
        Requires CONFIRM_FRAMES consecutive close readings to activate,
        and RELEASE_FRAMES consecutive open readings to deactivate —
        this kills single-frame jitter from landmark noise.
        """
        below_close = raw_dist < close_thresh
        above_open  = raw_dist > open_thresh

        if not active:
            if below_close:
                confirm_count += 1
                release_count  = 0
                if confirm_count >= HandState.CONFIRM_FRAMES:
                    active, confirm_count = True, 0
            else:
                confirm_count = max(0, confirm_count - 1)
        else:
            if above_open:
                release_count += 1
                confirm_count  = 0
                if release_count >= HandState.RELEASE_FRAMES:
                    active, release_count = False, 0
            else:
                release_count = max(0, release_count - 1)

        return active, confirm_count, release_count

    def debug_str(self):
        flags = []
        if self.is_grab: flags.append("GRAB")
        if self.is_pointing: flags.append("POINT")
        if self.is_finger_gun: flags.append("GUN")
        if self.is_open: flags.append("OPEN")
        if self.is_fist: flags.append("FIST")
        if self.is_ok: flags.append("OK")
        if self.is_three: flags.append("THREE")
        if self.is_four: flags.append("FOUR")
        if self.is_pinching: flags.append("PINCH")
        return f"[{self.label.upper()}] {'|'.join(flags) or 'none'}"


class HandTracker:
    """Wraps MediaPipe result -> two HandState objects."""

    def __init__(self):
        self.left  = HandState("left")
        self.right = HandState("right")

    def update(self, result):
        left_lm  = None
        right_lm = None

        if result is not None and result.hand_landmarks:
            for i, hand_lms in enumerate(result.hand_landmarks):
                lm_list = [(lm.x, lm.y, lm.z) for lm in hand_lms]

                label = "unknown"
                if result.handedness and i < len(result.handedness):
                    h = result.handedness[i]
                    if isinstance(h, list) and len(h) > 0:
                        cat = h[0]
                        label = (getattr(cat, "category_name", None)
                                 or getattr(cat, "label", "unknown")).lower()
                    elif hasattr(h, "classification"):
                        label = h.classification[0].label.lower()


                if label == "left":
                    right_lm = lm_list
                elif label == "right":
                    left_lm = lm_list
                else:
                    if right_lm is None:
                        right_lm = lm_list
                    else:
                        left_lm = lm_list

        self.left.update(left_lm)
        self.right.update(right_lm)

    @property
    def any_visible(self):
        return self.left.visible or self.right.visible