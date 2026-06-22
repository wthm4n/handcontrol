"""
Hand Gesture PC Controller
--------------------------
Gestures:
  - Index finger moves            → Move cursor
  - Thumb + Index pinch           → Left click
  - Thumb + Middle pinch          → Right click
  - Thumb + Index + Middle pinch  → Drag (hold left button)

Camera: /dev/video10 (DroidCam via ADB)
"""

import cv2
import mediapipe as mp
import numpy as np
import time
import sys
import math
import subprocess
import re


try:
    from pynput.mouse import Button, Controller as MouseController
    _mouse = MouseController()

    def move_mouse(x, y):
        _mouse.position = (int(x), int(y))

    def left_click():
        _mouse.click(Button.left)

    def right_click():
        _mouse.click(Button.right)

    def mouse_down():
        _mouse.press(Button.left)

    def mouse_up():
        _mouse.release(Button.left)

    print("[INFO] Using pynput for mouse control")

except ImportError:
    try:
        import pyautogui
        pyautogui.FAILSAFE = False
        pyautogui.PAUSE = 0

        def move_mouse(x, y):
            pyautogui.moveTo(int(x), int(y))

        def left_click():
            pyautogui.click()

        def right_click():
            pyautogui.rightClick()

        def mouse_down():
            pyautogui.mouseDown()

        def mouse_up():
            pyautogui.mouseUp()

        print("[INFO] Using pyautogui for mouse control")

    except ImportError:
        print("[ERROR] Install pynput:  pip install pynput")
        sys.exit(1)


SCREEN_W, SCREEN_H = 1920, 1080
try:
    r = subprocess.run(['xrandr', '--current'], capture_output=True, text=True)
    for line in r.stdout.split('\n'):
        if ' connected' in line:
            m = re.search(r'(\d+)x(\d+)', line)
            if m:
                SCREEN_W, SCREEN_H = int(m.group(1)), int(m.group(2))
                break
    print(f"[INFO] Screen resolution: {SCREEN_W}x{SCREEN_H}")
except Exception:
    print(f"[INFO] Defaulting to {SCREEN_W}x{SCREEN_H}")


CAMERA_DEVICE = "/dev/video10"

cap = cv2.VideoCapture(CAMERA_DEVICE)
if not cap.isOpened():
    print(f"[ERROR] Cannot open camera {CAMERA_DEVICE}. Exiting.")
    sys.exit(1)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

CAM_W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
CAM_H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"[INFO] Camera opened: {CAM_W}x{CAM_H}")


mp_hands  = mp.solutions.hands
mp_draw   = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(
    model_complexity=0,
    max_num_hands=1,
    min_detection_confidence=0.70,
    min_tracking_confidence=0.60,
)


WRIST      = 0
THUMB_TIP  = 4
INDEX_TIP  = 8
INDEX_PIP  = 6
MIDDLE_TIP = 12
MIDDLE_PIP = 10


PINCH_ON          = 0.052
PINCH_OFF         = 0.075
SMOOTH_ALPHA      = 0.30
MARGIN            = 0.10
CLICK_COOLDOWN    = 0.35
DRAG_HOLD_FRAMES  = 3


cursor_x = cursor_y = 0.0

last_lclick  = 0.0
last_rclick  = 0.0
ti_active    = False
tm_active    = False
dragging     = False
drag_cnt     = 0

gesture_label = "NO HAND"

def dist(a, b):
    return math.hypot(a.x - b.x, a.y - b.y)

def screen_pos(lm):
    """Map index-tip position from active zone → screen coordinates."""
    m = MARGIN
    rx = max(m, min(1 - m, lm[INDEX_TIP].x))
    ry = max(m, min(1 - m, lm[INDEX_TIP].y))
    sx = (rx - m) / (1 - 2 * m) * SCREEN_W
    sy = (ry - m) / (1 - 2 * m) * SCREEN_H
    return sx, sy

def draw_hud(frame, label, fps):
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (w, 42), (15, 15, 15), -1)
    cv2.putText(frame, f"{fps:.0f} fps", (8, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 200, 90), 2)
    colours = {
        "MOVE":        (200, 200, 200),
        "LEFT CLICK":  (0,   255, 80),
        "RIGHT CLICK": (60,  60,  255),
        "DRAGGING":    (0,   210, 255),
        "NO HAND":     (90,  90,  90),
    }
    col = colours.get(label, (200, 200, 200))
    cv2.putText(frame, label, (w // 2 - 85, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, col, 2)

    cx = int(cursor_x / SCREEN_W * w)
    cy = int(cursor_y / SCREEN_H * h)
    cv2.circle(frame, (cx, cy), 7, (0, 255, 255), -1)
    cv2.circle(frame, (cx, cy), 10, (0, 180, 180), 1)


print("[INFO] Running – press Q in window to quit.")
print("  Move index finger  → cursor")
print("  Thumb+Index        → left click")
print("  Thumb+Middle       → right click")
print("  Thumb+Idx+Mid      → drag")

t0  = time.perf_counter()
fps = 30.0

while True:
    ret, frame = cap.read()
    if not ret:
        print("[ERROR] Camera read failed. Exiting.")
        break

    frame = cv2.flip(frame, 1)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb.flags.writeable = False
    res = hands.process(rgb)
    rgb.flags.writeable = True

    now = time.perf_counter()
    fps = 0.9 * fps + 0.1 / max(now - t0, 1e-9)
    t0  = now

    if res.multi_hand_landmarks:
        lm = res.multi_hand_landmarks[0].landmark


        d_ti = dist(lm[THUMB_TIP], lm[INDEX_TIP])
        d_tm = dist(lm[THUMB_TIP], lm[MIDDLE_TIP])
        d_im = dist(lm[INDEX_TIP], lm[MIDDLE_TIP])

        triple = (d_ti < PINCH_ON and
                  d_tm < PINCH_ON and
                  d_im < PINCH_ON * 1.9)

        ti_now = (d_ti < PINCH_ON) and not triple
        tm_now = (d_tm < PINCH_ON) and not triple


        tx, ty   = screen_pos(lm)
        cursor_x = (1 - SMOOTH_ALPHA) * cursor_x + SMOOTH_ALPHA * tx
        cursor_y = (1 - SMOOTH_ALPHA) * cursor_y + SMOOTH_ALPHA * ty
        move_mouse(cursor_x, cursor_y)


        if triple:
            drag_cnt += 1
            if drag_cnt >= DRAG_HOLD_FRAMES and not dragging:
                mouse_down()
                dragging = True
            gesture_label = "DRAGGING"
            ti_active = tm_active = False

        else:
            if dragging:
                mouse_up()
                dragging  = False
            drag_cnt = 0


            if ti_now and not ti_active:
                ti_active = True
                if now - last_lclick > CLICK_COOLDOWN:
                    left_click()
                    last_lclick   = now
                    gesture_label = "LEFT CLICK"
            elif not ti_now and d_ti > PINCH_OFF:
                ti_active = False


            if tm_now and not tm_active:
                tm_active = True
                if now - last_rclick > CLICK_COOLDOWN:
                    right_click()
                    last_rclick   = now
                    gesture_label = "RIGHT CLICK"
            elif not tm_now and d_tm > PINCH_OFF:
                tm_active = False

            if not ti_now and not tm_now:
                gesture_label = "MOVE"


        mp_draw.draw_landmarks(
            frame,
            res.multi_hand_landmarks[0],
            mp_hands.HAND_CONNECTIONS,
            mp_styles.get_default_hand_landmarks_style(),
            mp_styles.get_default_hand_connections_style(),
        )

        for tip, col in [(THUMB_TIP,  (0, 200, 255)),
                         (INDEX_TIP,  (0, 255,  80)),
                         (MIDDLE_TIP, (255, 100,  0))]:
            px = int(lm[tip].x * CAM_W)
            py = int(lm[tip].y * CAM_H)
            cv2.circle(frame, (px, py), 11, col, -1)

    else:
        gesture_label = "NO HAND"
        if dragging:
            mouse_up()
            dragging = False
        drag_cnt  = 0
        ti_active = tm_active = False

    draw_hud(frame, gesture_label, fps)
    cv2.imshow("Hand Controller  [Q=quit]", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


if dragging:
    mouse_up()
cap.release()
cv2.destroyAllWindows()
hands.close()
print("[INFO] Stopped.")