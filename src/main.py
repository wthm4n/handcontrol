"""
Hand Gesture Recognizer — PERCEPTION ONLY, zero actions
========================================================
Detects gestures with maximum accuracy using:
  - 3D world landmarks (not screen-space) for angle math
  - Per-joint curl angles (MCP + PIP) for each finger
  - Multi-frame temporal confirmation before any state change
  - Hysteresis: must pass through NONE before re-triggering same gesture
  - Confidence score displayed per frame

Gestures recognized:
  IDLE          open/neutral hand
  POINTING      index extended, others curled
  PINCH_INDEX   thumb + index tip close  (→ left click intent)
  PINCH_MIDDLE  thumb + middle tip close (→ right click intent)
  PINCH_DRAG    thumb + index + middle close (→ drag intent)
  FIST          all fingers curled
  OPEN_HAND     all five fingers extended
  PEACE         index + middle extended, others curled

Camera: /dev/video10 (DroidCam - DO NOT CHANGE)
Model:  downloaded once to ~/.cache/mediapipe/hand_landmarker.task
"""

import sys, os, math, time, pathlib, urllib.request, collections
import cv2
import mediapipe as mp
import numpy as np


MODEL_URLS = [
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task",
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
]
MODEL_PATH = pathlib.Path.home() / ".cache" / "mediapipe" / "hand_landmarker.task"
MIN_MODEL_BYTES = 100_000

def ensure_model():
    if MODEL_PATH.exists() and MODEL_PATH.stat().st_size >= MIN_MODEL_BYTES:
        print(f"[INFO] Model: {MODEL_PATH}  ({MODEL_PATH.stat().st_size:,} bytes)")
        return
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    print("[INFO] Downloading hand landmarker model …")
    for url in MODEL_URLS:
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = resp.read()
            if len(data) < MIN_MODEL_BYTES:
                print(f"[WARN] Response too small ({len(data)} B), skipping {url}")
                continue
            MODEL_PATH.write_bytes(data)
            print(f"[INFO] Downloaded {len(data):,} bytes → {MODEL_PATH}")
            return
        except Exception as e:
            print(f"[WARN] {url}\n       {e}")
    print("[ERROR] All download attempts failed.")
    print("        Download manually:")
    print(f"          {MODEL_URLS[0]}")
    print(f"        Save to: {MODEL_PATH}")
    sys.exit(1)

ensure_model()


CAMERA_DEVICE = "/dev/video10"
cap = cv2.VideoCapture(CAMERA_DEVICE)
if not cap.isOpened():
    print(f"[ERROR] Cannot open {CAMERA_DEVICE}. Exiting.")
    sys.exit(1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
CAM_W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
CAM_H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"[INFO] Camera: {CAM_W}×{CAM_H}")


_v            = mp.tasks.vision
RunningMode   = _v.RunningMode
HandLandmarker       = _v.HandLandmarker
HandLandmarkerOptions= _v.HandLandmarkerOptions
BaseOptions   = mp.tasks.BaseOptions
draw_utils    = _v.drawing_utils
draw_styles   = _v.drawing_styles
HandConns     = _v.HandLandmarksConnections

_options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=str(MODEL_PATH)),
    running_mode=RunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.75,
    min_hand_presence_confidence=0.70,
    min_tracking_confidence=0.65,
)


WRIST       = 0
THUMB_CMC, THUMB_MCP, THUMB_IP,  THUMB_TIP  = 1,  2,  3,  4
INDEX_MCP,  INDEX_PIP,  INDEX_DIP,  INDEX_TIP  = 5,  6,  7,  8
MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP = 9,  10, 11, 12
RING_MCP,   RING_PIP,   RING_DIP,   RING_TIP   = 13, 14, 15, 16
PINKY_MCP,  PINKY_PIP,  PINKY_DIP,  PINKY_TIP  = 17, 18, 19, 20


def _v3(lm, idx):
    """Return landmark as numpy (x,y,z) using 3-D world coords."""
    p = lm[idx]
    return np.array([p.x, p.y, p.z], dtype=np.float32)

def angle_at_joint(lm, a, b, c):
    """
    Angle (degrees) at joint B, given three landmark indices A-B-C.
    Uses 3-D world landmarks for perspective-invariant measurement.
    0° = fully straight, 180° = fully bent back (impossible).
    Typical extended finger: <30°, curled finger: >70°.
    """
    va = _v3(lm, a) - _v3(lm, b)
    vc = _v3(lm, c) - _v3(lm, b)
    n_a, n_c = np.linalg.norm(va), np.linalg.norm(vc)
    if n_a < 1e-6 or n_c < 1e-6:
        return 0.0
    cos = np.clip(np.dot(va, vc) / (n_a * n_c), -1.0, 1.0)
    return math.degrees(math.acos(cos))

def dist3d(lm, i, j):
    """Euclidean 3-D distance between two world landmarks."""
    return float(np.linalg.norm(_v3(lm, i) - _v3(lm, j)))

def hand_scale(lm):
    """Approximate hand size = wrist→middle-MCP distance (normalization factor)."""
    return max(dist3d(lm, WRIST, MIDDLE_MCP), 1e-6)


CURL_STRAIGHT_DEG = 30.0
CURL_BENT_DEG     = 65.0

def _curl_score(angle_deg: float) -> float:
    """Map angle to [0,1]: 0=straight, 1=curled."""
    return float(np.clip(
        (angle_deg - CURL_STRAIGHT_DEG) / (CURL_BENT_DEG - CURL_STRAIGHT_DEG),
        0.0, 1.0
    ))

def finger_curls(world_lm):
    """
    Returns dict of curl scores (0=extended, 1=curled) for each finger.
    Thumb uses a different joint chain.
    """

    th_a1 = angle_at_joint(world_lm, THUMB_CMC,  THUMB_MCP, THUMB_IP)
    th_a2 = angle_at_joint(world_lm, THUMB_MCP,  THUMB_IP,  THUMB_TIP)

    ix_a1 = angle_at_joint(world_lm, WRIST,      INDEX_MCP,  INDEX_PIP)
    ix_a2 = angle_at_joint(world_lm, INDEX_MCP,  INDEX_PIP,  INDEX_TIP)

    mi_a1 = angle_at_joint(world_lm, WRIST,      MIDDLE_MCP, MIDDLE_PIP)
    mi_a2 = angle_at_joint(world_lm, MIDDLE_MCP, MIDDLE_PIP, MIDDLE_TIP)

    ri_a1 = angle_at_joint(world_lm, WRIST,      RING_MCP,   RING_PIP)
    ri_a2 = angle_at_joint(world_lm, RING_MCP,   RING_PIP,   RING_TIP)

    pi_a1 = angle_at_joint(world_lm, WRIST,      PINKY_MCP,  PINKY_PIP)
    pi_a2 = angle_at_joint(world_lm, PINKY_MCP,  PINKY_PIP,  PINKY_TIP)

    def avg_curl(a1, a2):
        return (_curl_score(a1) + _curl_score(a2)) * 0.5

    return {
        "thumb":  avg_curl(th_a1, th_a2),
        "index":  avg_curl(ix_a1, ix_a2),
        "middle": avg_curl(mi_a1, mi_a2),
        "ring":   avg_curl(ri_a1, ri_a2),
        "pinky":  avg_curl(pi_a1, pi_a2),
    }


PINCH_THRESH  = 0.18
PINCH_RELEASE = 0.25

def pinch_distances(world_lm):
    scale = hand_scale(world_lm)
    return {
        "ti":  dist3d(world_lm, THUMB_TIP, INDEX_TIP)  / scale,
        "tm":  dist3d(world_lm, THUMB_TIP, MIDDLE_TIP) / scale,
        "im":  dist3d(world_lm, INDEX_TIP, MIDDLE_TIP) / scale,
    }


GESTURES = [
    "IDLE",
    "POINTING",
    "PINCH_INDEX",
    "PINCH_MIDDLE",
    "PINCH_DRAG",
    "FIST",
    "OPEN_HAND",
    "PEACE",
]


EXTENDED_MAX = 0.35
CURLED_MIN   = 0.60

def is_ext(c): return c < EXTENDED_MAX
def is_cur(c): return c > CURLED_MIN

def classify_frame(world_lm) -> tuple[str, float]:
    """
    Returns (gesture_name, confidence ∈ [0,1]).
    confidence = fraction of expected binary conditions that hold.
    """
    curl = finger_curls(world_lm)
    pd   = pinch_distances(world_lm)

    th, ix, mi, ri, pi = (curl["thumb"], curl["index"], curl["middle"],
                           curl["ring"],  curl["pinky"])


    if pd["ti"] < PINCH_THRESH and pd["tm"] < PINCH_THRESH:

        tightness = 1.0 - (pd["ti"] + pd["tm"]) / (2 * PINCH_THRESH)
        conds = [pd["ti"] < PINCH_THRESH, pd["tm"] < PINCH_THRESH,
                 pd["im"] < PINCH_THRESH * 1.8]
        conf = sum(conds) / len(conds) * 0.7 + tightness * 0.3
        return "PINCH_DRAG", float(np.clip(conf, 0, 1))


    if pd["ti"] < PINCH_THRESH:
        tightness = 1.0 - pd["ti"] / PINCH_THRESH
        conds = [pd["ti"] < PINCH_THRESH,
                 pd["tm"] > PINCH_RELEASE * 0.8,
                 is_cur(ri), is_cur(pi)]
        conf = (sum(conds) / len(conds)) * 0.6 + tightness * 0.4
        return "PINCH_INDEX", float(np.clip(conf, 0, 1))


    if pd["tm"] < PINCH_THRESH:
        tightness = 1.0 - pd["tm"] / PINCH_THRESH
        conds = [pd["tm"] < PINCH_THRESH,
                 pd["ti"] > PINCH_RELEASE * 0.8,
                 is_cur(ri), is_cur(pi)]
        conf = (sum(conds) / len(conds)) * 0.6 + tightness * 0.4
        return "PINCH_MIDDLE", float(np.clip(conf, 0, 1))


    if is_ext(th) and is_ext(ix) and is_ext(mi) and is_ext(ri) and is_ext(pi):
        margin = min(EXTENDED_MAX - th, EXTENDED_MAX - ix,
                     EXTENDED_MAX - mi, EXTENDED_MAX - ri, EXTENDED_MAX - pi)
        conf = 0.7 + 0.3 * np.clip(margin / EXTENDED_MAX, 0, 1)
        return "OPEN_HAND", float(conf)


    if is_cur(ix) and is_cur(mi) and is_cur(ri) and is_cur(pi):
        surplus = min(ix, mi, ri, pi) - CURLED_MIN
        conf = 0.7 + 0.3 * np.clip(surplus / (1 - CURLED_MIN), 0, 1)
        return "FIST", float(conf)


    if is_ext(ix) and is_ext(mi) and is_cur(ri) and is_cur(pi) and not is_ext(th):
        conds = [is_ext(ix), is_ext(mi), is_cur(ri), is_cur(pi)]
        conf = sum(conds) / 4
        return "PEACE", float(conf)


    if is_ext(ix) and is_cur(mi) and is_cur(ri) and is_cur(pi):
        conds = [is_ext(ix), is_cur(mi), is_cur(ri), is_cur(pi), not is_ext(th)]
        conf = sum(conds) / len(conds)
        return "POINTING", float(conf)


    return "IDLE", 0.5


CONFIRM_FRAMES = 4
HISTORY_LEN    = 8

class TemporalSmoother:
    def __init__(self, confirm: int = CONFIRM_FRAMES):
        self._confirm  = confirm
        self._pending  = None
        self._pending_n= 0
        self.state     = "IDLE"
        self.conf      = 0.0

    def update(self, gesture: str, conf: float) -> tuple[str, float]:
        if gesture == self._pending:
            self._pending_n += 1
        else:
            self._pending   = gesture
            self._pending_n = 1

        if self._pending_n >= self._confirm:
            self.state = gesture
            self.conf  = conf

        return self.state, self.conf


GESTURE_COLOUR = {
    "IDLE":         (180, 180, 180),
    "POINTING":     (0,   220, 255),
    "PINCH_INDEX":  (0,   255, 80),
    "PINCH_MIDDLE": (80,  80,  255),
    "PINCH_DRAG":   (255, 160, 0),
    "FIST":         (0,   80,  255),
    "OPEN_HAND":    (180, 255, 130),
    "PEACE":        (255, 200, 0),
    "NO HAND":      (60,  60,  60),
}

GESTURE_LABEL = {
    "IDLE":         "IDLE",
    "POINTING":     "POINTING  ☝",
    "PINCH_INDEX":  "PINCH  (thumb+index)",
    "PINCH_MIDDLE": "PINCH  (thumb+middle)",
    "PINCH_DRAG":   "PINCH  (drag – 3 finger)",
    "FIST":         "FIST  ✊",
    "OPEN_HAND":    "OPEN HAND  🖐",
    "PEACE":        "PEACE  ✌",
    "NO HAND":      "NO HAND",
}


PANEL_H  = 120
BAR_W    = 280
FONT     = cv2.FONT_HERSHEY_SIMPLEX

def draw_hud(frame, gesture, conf, fps, curl, pd, history):
    h, w = frame.shape[:2]
    col  = GESTURE_COLOUR.get(gesture, (200, 200, 200))


    cv2.rectangle(frame, (0, 0), (w, 52), (12, 12, 12), -1)

    cv2.putText(frame, f"{fps:.0f} fps", (12, 36),
                FONT, 0.85, (0, 200, 100), 2, cv2.LINE_AA)

    label = GESTURE_LABEL.get(gesture, gesture)
    (tw, _), _ = cv2.getTextSize(label, FONT, 0.9, 2)
    cv2.putText(frame, label, (w//2 - tw//2, 36),
                FONT, 0.9, col, 2, cv2.LINE_AA)

    bar_x = w - BAR_W - 12
    cv2.rectangle(frame, (bar_x, 14), (bar_x + BAR_W, 38), (50, 50, 50), -1)
    filled = int(BAR_W * conf)
    cv2.rectangle(frame, (bar_x, 14), (bar_x + filled, 38), col, -1)
    cv2.putText(frame, f"{conf*100:.0f}%", (bar_x + BAR_W + 6, 36),
                FONT, 0.7, col, 2, cv2.LINE_AA)

    if curl is None:
        return


    fingers = ["thumb", "index", "middle", "ring", "pinky"]
    colours = [(0,200,255),(0,255,80),(255,160,0),(180,100,255),(255,80,180)]
    px, py = 10, 65
    for i, (fn, fc) in enumerate(zip(fingers, colours)):
        cv = curl.get(fn, 0.0)

        cv2.putText(frame, fn[:3].upper(), (px, py + i*22),
                    FONT, 0.50, fc, 1, cv2.LINE_AA)

        cv2.rectangle(frame, (px+42, py+i*22-12), (px+42+100, py+i*22+2),
                      (50,50,50), -1)

        cv2.rectangle(frame, (px+42, py+i*22-12),
                      (px+42+int(100*cv), py+i*22+2), fc, -1)

        cv2.putText(frame, f"{cv:.2f}", (px+148, py+i*22),
                    FONT, 0.45, fc, 1, cv2.LINE_AA)


    rx = w - 190
    ry = 65
    pinch_items = [
        ("TI", pd.get("ti", 1.0), (0, 255, 80)),
        ("TM", pd.get("tm", 1.0), (80, 80, 255)),
        ("IM", pd.get("im", 1.0), (255, 160, 0)),
    ]
    for i, (name, val, pc) in enumerate(pinch_items):
        active = val < PINCH_THRESH
        cv2.putText(frame, f"{name}: {val:.3f}", (rx, ry + i*22),
                    FONT, 0.52,
                    (0, 255, 0) if active else pc, 1, cv2.LINE_AA)
        if active:
            cv2.putText(frame, "PINCH", (rx+110, ry+i*22),
                        FONT, 0.45, (0, 255, 0), 1, cv2.LINE_AA)


    strip_y = h - 28
    cv2.rectangle(frame, (0, strip_y - 4), (w, h), (18, 18, 18), -1)
    slot_w = w // HISTORY_LEN
    for i, (hg, hc) in enumerate(history):
        hcol = GESTURE_COLOUR.get(hg, (120, 120, 120))
        sx   = i * slot_w + 4
        cv2.putText(frame, hg.replace("_", " "), (sx, strip_y + 16),
                    FONT, 0.35, hcol, 1, cv2.LINE_AA)


print("[INFO] Starting – press Q in window to quit.")
print()
print("  Gesture map:")
for k, v in GESTURE_LABEL.items():
    print(f"    {v}")
print()

smoother   = TemporalSmoother(confirm=CONFIRM_FRAMES)
history    = collections.deque(maxlen=HISTORY_LEN)
fps        = 30.0
t0         = time.perf_counter()
frame_ts   = 0

with HandLandmarker.create_from_options(_options) as detector:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Camera read failed. Exiting.")
            break


        frame = cv2.flip(frame, 1)


        now = time.perf_counter()
        fps = 0.92 * fps + 0.08 / max(now - t0, 1e-9)
        t0  = now
        frame_ts += 33


        mp_img = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        )
        result = detector.detect_for_video(mp_img, frame_ts)

        if result.hand_landmarks and result.hand_world_landmarks:
            screen_lm = result.hand_landmarks[0]
            world_lm  = result.hand_world_landmarks[0]


            raw_gesture, raw_conf = classify_frame(world_lm)
            gesture, conf         = smoother.update(raw_gesture, raw_conf)

            curl = finger_curls(world_lm)
            pd   = pinch_distances(world_lm)


            if not history or history[-1][0] != gesture:
                history.append((gesture, conf))
                ts = time.strftime("%H:%M:%S")
                print(f"[{ts}]  {gesture:<16}  conf={conf:.2f}")


            draw_utils.draw_landmarks(
                frame, screen_lm,
                HandConns.HAND_CONNECTIONS,
                draw_styles.get_default_hand_landmarks_style(),
                draw_styles.get_default_hand_connections_style(),
            )

            tip_cols = [
                (THUMB_TIP,  (0,  200, 255)),
                (INDEX_TIP,  (0,  255,  80)),
                (MIDDLE_TIP, (255,160,   0)),
                (RING_TIP,   (180,100, 255)),
                (PINKY_TIP,  (255, 80, 180)),
            ]
            for tid, tc in tip_cols:
                px_ = int(screen_lm[tid].x * CAM_W)
                py_ = int(screen_lm[tid].y * CAM_H)
                cv2.circle(frame, (px_, py_), 9, tc, -1)
                cv2.circle(frame, (px_, py_), 11, (255,255,255), 1)

        else:
            gesture, conf = smoother.update("IDLE", 0.0)
            if gesture == "IDLE":
                gesture = "NO HAND"
            curl = None
            pd   = {}
            if not history or history[-1][0] != "NO HAND":
                history.append(("NO HAND", 0.0))


        draw_hud(frame, gesture, conf, fps, curl, pd, list(history))

        cv2.imshow("Hand Gesture Recognizer  [Q=quit]", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
print("[INFO] Stopped.")

