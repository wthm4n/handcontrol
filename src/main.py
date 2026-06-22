"""
Hand Gesture PC Controller — mediapipe 0.10+ (tasks API)
---------------------------------------------------------
Gestures:
  Move index finger               → cursor
  Thumb + Index pinch             → left click
  Thumb + Middle pinch            → right click
  Thumb + Index + Middle pinch    → drag

Run once to auto-download the model (~1 MB), then it's cached.
"""

import cv2
import mediapipe as mp
import time, sys, math, re, subprocess, os, urllib.request, pathlib


MODEL_URL  = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
MODEL_PATH = pathlib.Path.home() / ".cache" / "mediapipe" / "hand_landmarker.task"

if not MODEL_PATH.exists():
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Downloading hand landmarker model → {MODEL_PATH}")
    try:
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("[INFO] Model downloaded OK")
    except Exception as e:
        print(f"[ERROR] Model download failed: {e}")
        print(f"        Download manually from:\n        {MODEL_URL}")
        print(f"        Save to: {MODEL_PATH}")
        sys.exit(1)
else:
    print(f"[INFO] Model found: {MODEL_PATH}")


try:
    from pynput.mouse import Button, Controller as MouseController
    _mouse = MouseController()
    move_mouse  = lambda x, y: setattr(_mouse, 'position', (int(x), int(y)))
    left_click  = lambda: _mouse.click(Button.left)
    right_click = lambda: _mouse.click(Button.right)
    mouse_down  = lambda: _mouse.press(Button.left)
    mouse_up    = lambda: _mouse.release(Button.left)
    print("[INFO] Using pynput")
except ImportError:
    import pyautogui
    pyautogui.FAILSAFE = False; pyautogui.PAUSE = 0
    move_mouse  = lambda x, y: pyautogui.moveTo(int(x), int(y))
    left_click  = lambda: pyautogui.click()
    right_click = lambda: pyautogui.rightClick()
    mouse_down  = lambda: pyautogui.mouseDown()
    mouse_up    = lambda: pyautogui.mouseUp()
    print("[INFO] Using pyautogui")


SCREEN_W, SCREEN_H = 1920, 1080
try:
    r = subprocess.run(['xrandr', '--current'], capture_output=True, text=True)
    for line in r.stdout.split('\n'):
        if ' connected' in line:
            m = re.search(r'(\d+)x(\d+)', line)
            if m:
                SCREEN_W, SCREEN_H = int(m.group(1)), int(m.group(2))
                break
except Exception:
    pass
print(f"[INFO] Screen: {SCREEN_W}x{SCREEN_H}")


CAMERA_DEVICE = "/dev/video10"
cap = cv2.VideoCapture(CAMERA_DEVICE)
if not cap.isOpened():
    print(f"[ERROR] Cannot open {CAMERA_DEVICE}. Exiting.")
    sys.exit(1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
CAM_W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
CAM_H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"[INFO] Camera: {CAM_W}x{CAM_H}")


VisionRunningMode = mp.tasks.vision.RunningMode
HandLandmarker    = mp.tasks.vision.HandLandmarker
HandLandmarkerOpt = mp.tasks.vision.HandLandmarkerOptions
BaseOptions       = mp.tasks.BaseOptions
draw_utils        = mp.tasks.vision.drawing_utils
draw_styles       = mp.tasks.vision.drawing_styles
HandConnections   = mp.tasks.vision.HandLandmarksConnections

options = HandLandmarkerOpt(
    base_options=BaseOptions(model_asset_path=str(MODEL_PATH)),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.70,
    min_hand_presence_confidence=0.65,
    min_tracking_confidence=0.60,
)


THUMB_TIP  = 4
INDEX_TIP  = 8
MIDDLE_TIP = 12


PINCH_ON         = 0.052
PINCH_OFF        = 0.075
SMOOTH_ALPHA     = 0.30
MARGIN           = 0.10
CLICK_COOLDOWN   = 0.35
DRAG_HOLD_FRAMES = 3


cursor_x = cursor_y = 0.0
last_lclick = last_rclick = 0.0
ti_active = tm_active = False
dragging  = False
drag_cnt  = 0
gesture_label = "NO HAND"

def dist(a, b):
    return math.hypot(a.x - b.x, a.y - b.y)

def screen_pos(lm):
    m  = MARGIN
    rx = max(m, min(1 - m, lm[INDEX_TIP].x))
    ry = max(m, min(1 - m, lm[INDEX_TIP].y))
    return (rx - m) / (1 - 2 * m) * SCREEN_W, \
           (ry - m) / (1 - 2 * m) * SCREEN_H

def draw_hud(frame, label, fps):
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (w, 42), (15, 15, 15), -1)
    cv2.putText(frame, f"{fps:.0f} fps", (8, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 200, 90), 2)
    col = {"MOVE":(200,200,200),"LEFT CLICK":(0,255,80),
           "RIGHT CLICK":(60,60,255),"DRAGGING":(0,210,255),
           "NO HAND":(90,90,90)}.get(label,(200,200,200))
    cv2.putText(frame, label, (w//2-85, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, col, 2)
    cx = int(cursor_x / SCREEN_W * w)
    cy = int(cursor_y / SCREEN_H * h)
    cv2.circle(frame, (cx, cy), 7, (0, 255, 255), -1)
    cv2.circle(frame, (cx, cy), 10, (0, 180, 180), 1)

def draw_hand(frame, hand_landmarks_list):
    """Draw skeleton using new tasks drawing API."""
    for hand_lms in hand_landmarks_list:
        draw_utils.draw_landmarks(
            frame,
            hand_lms,
            HandConnections.HAND_CONNECTIONS,
            draw_styles.get_default_hand_landmarks_style(),
            draw_styles.get_default_hand_connections_style(),
        )

    for hand_lms in hand_landmarks_list:
        lm = hand_lms
        for tip_idx, col in [(THUMB_TIP,(0,200,255)),
                              (INDEX_TIP,(0,255,80)),
                              (MIDDLE_TIP,(255,100,0))]:
            px = int(lm[tip_idx].x * CAM_W)
            py = int(lm[tip_idx].y * CAM_H)
            cv2.circle(frame, (px, py), 11, col, -1)


print("[INFO] Running – press Q to quit")
print("  Move index    → cursor")
print("  Thumb+Index   → left click")
print("  Thumb+Middle  → right click")
print("  All three     → drag")

t0  = time.perf_counter()
fps = 30.0
frame_ts = 0

with HandLandmarker.create_from_options(options) as detector:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Camera read failed. Exiting.")
            break

        frame = cv2.flip(frame, 0)

        now = time.perf_counter()
        fps = 0.9 * fps + 0.1 / max(now - t0, 1e-9)
        t0  = now


        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        )

        frame_ts += int(1000 / 30)
        result = detector.detect_for_video(mp_image, frame_ts)

        if result.hand_landmarks:
            lm = result.hand_landmarks[0]

            d_ti = dist(lm[THUMB_TIP], lm[INDEX_TIP])
            d_tm = dist(lm[THUMB_TIP], lm[MIDDLE_TIP])
            d_im = dist(lm[INDEX_TIP],  lm[MIDDLE_TIP])

            triple = (d_ti < PINCH_ON and
                      d_tm < PINCH_ON and
                      d_im < PINCH_ON * 1.9)
            ti_now = (d_ti < PINCH_ON) and not triple
            tm_now = (d_tm < PINCH_ON) and not triple


            tx, ty   = screen_pos(lm)
            cursor_x = (1 - SMOOTH_ALPHA) * cursor_x + SMOOTH_ALPHA * tx
            cursor_y = (1 - SMOOTH_ALPHA) * cursor_y + SMOOTH_ALPHA * ty
            move_mouse(cursor_x, cursor_y)

            now_t = time.perf_counter()

            if triple:
                drag_cnt += 1
                if drag_cnt >= DRAG_HOLD_FRAMES and not dragging:
                    mouse_down(); dragging = True
                gesture_label = "DRAGGING"
                ti_active = tm_active = False
            else:
                if dragging:
                    mouse_up(); dragging = False
                drag_cnt = 0

                if ti_now and not ti_active:
                    ti_active = True
                    if now_t - last_lclick > CLICK_COOLDOWN:
                        left_click(); last_lclick = now_t
                        gesture_label = "LEFT CLICK"
                elif not ti_now and d_ti > PINCH_OFF:
                    ti_active = False

                if tm_now and not tm_active:
                    tm_active = True
                    if now_t - last_rclick > CLICK_COOLDOWN:
                        right_click(); last_rclick = now_t
                        gesture_label = "RIGHT CLICK"
                elif not tm_now and d_tm > PINCH_OFF:
                    tm_active = False

                if not ti_now and not tm_now:
                    gesture_label = "MOVE"

            draw_hand(frame, result.hand_landmarks)

        else:
            gesture_label = "NO HAND"
            if dragging:
                mouse_up(); dragging = False
            drag_cnt = ti_active = tm_active = False

        draw_hud(frame, gesture_label, fps)
        cv2.imshow("Hand Controller  [Q=quit]", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

if dragging:
    mouse_up()
cap.release()
cv2.destroyAllWindows()
print("[INFO] Stopped.")