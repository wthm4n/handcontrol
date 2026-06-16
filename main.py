"""
Aether — Phase 3: Throw physics.

RIGHT hand  = cursor / fist-grab / wrist-twist rotate / THROW on open
LEFT  hand  = anchor the virtual plane (tilt workspace)

New in Phase 3:
  - Hand velocity is tracked while grabbing (rolling window).
  - Opening fist throws the cube with that velocity.
  - Cubes fly, tumble, bounce off floor and walls, then settle.
  - 'G' toggles gravity on/off.
  - 'R' resets all cubes to starting positions.
  - Physics debug line shown per cube when airborne.

Gestures (unchanged from Phase 2):
  Right fist near cube     → grab
  Right wrist rotation     → spin while held
  Left hand tilt           → tilt workspace plane
  Both fists spread/close  → scale grabbed cube
"""

import cv2
import math
import time
import os
import sys
import urllib.request

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

from hand import HandTracker
from cube import Cube
from physics import HandVelocityTracker


# ── Config ────────────────────────────────────────────────────────────

CAMERA_DEVICE = "/dev/video10"
WINDOW_W      = 1280
WINDOW_H      = 720

MODEL_PATH = os.path.join(os.path.dirname(__file__), "hand_landmarker.task")
MODEL_URL  = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
)

HOVER_RADIUS = 110
HOVER_EXIT   = 145

# How far above the frame bottom the floor sits (so cubes don't go off-screen)
FLOOR_MARGIN = 40

HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17),
]

COLOR_RIGHT = (0, 220, 255)
COLOR_LEFT  = (255, 160,  40)
COLOR_FLOOR = (40,  80,  60)


# ── Model ─────────────────────────────────────────────────────────────

def ensure_model():
    if os.path.exists(MODEL_PATH) and os.path.getsize(MODEL_PATH) > 50_000:
        return
    print("Downloading hand_landmarker.task (~5 MB)...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("Done.")


# ── Hand skeleton ──────────────────────────────────────────────────────

def draw_hand_skeleton(img, hand_state, W, H, color, label):
    if not hand_state.visible or hand_state.landmarks is None:
        return
    lm  = hand_state.landmarks
    pts = [(int(p[0]*W), int(p[1]*H)) for p in lm]
    for a, b in HAND_CONNECTIONS:
        cv2.line(img, pts[a], pts[b], color, 2, cv2.LINE_AA)
    for i, (px, py) in enumerate(pts):
        r = 5 if i in (4,8,12,16,20) else 3
        cv2.circle(img, (px, py), r, color, -1, cv2.LINE_AA)
    tip_labels = {4:"T", 8:"I", 12:"M", 16:"R", 20:"P"}
    for idx, lbl in tip_labels.items():
        px, py = pts[idx]
        cv2.putText(img, lbl, (px+4, py-4), cv2.FONT_HERSHEY_SIMPLEX,
                    0.3, color, 1, cv2.LINE_AA)
    wx, wy = pts[0]
    for fi, ext in enumerate(hand_state.fingers_extended):
        dot_col = color if ext else (50, 50, 50)
        cv2.circle(img, (wx-20+fi*10, wy-20), 4, dot_col, -1, cv2.LINE_AA)
    cv2.putText(img, label, (wx-10, wy+20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)


# ── Plane grid ────────────────────────────────────────────────────────

def draw_plane(img, left_hand, W, H):
    if not left_hand.visible:
        return
    cx, cy  = W//2, H//2
    tilt_x  = left_hand.orient_x * 0.4
    tilt_z  = left_hand.orient_z * 0.3
    step    = 80
    cols    = W // step + 2
    rows    = H // step + 2

    def transform(gx, gy):
        skew_y = gy * math.sin(tilt_x) * 0.3
        roll_x = gx * math.cos(tilt_z) - gy * math.sin(tilt_z) * 0.15
        roll_y = gx * math.sin(tilt_z) * 0.15 + gy * math.cos(tilt_z)
        return int(cx + roll_x + skew_y), int(cy + roll_y)

    ov = img.copy()
    for r in range(-rows//2, rows//2+1):
        gy = r * step
        cv2.line(ov, transform(-cols//2*step, gy),
                     transform( cols//2*step, gy), (30,60,50), 1, cv2.LINE_AA)
    for c in range(-cols//2, cols//2+1):
        gx = c * step
        cv2.line(ov, transform(gx, -rows//2*step),
                     transform(gx,  rows//2*step), (30,60,50), 1, cv2.LINE_AA)
    cv2.addWeighted(ov, 0.35, img, 0.65, 0, img)


# ── Floor line ────────────────────────────────────────────────────────

def draw_floor(img, floor_y, W):
    """Subtle glowing floor line."""
    ov = img.copy()
    cv2.line(ov, (0, floor_y), (W, floor_y), COLOR_FLOOR, 6, cv2.LINE_AA)
    cv2.addWeighted(ov, 0.5, img, 0.5, 0, img)
    cv2.line(img, (0, floor_y), (W, floor_y), (60, 140, 100), 1, cv2.LINE_AA)


# ── Velocity arrow ────────────────────────────────────────────────────

def draw_velocity_arrow(img, cube):
    """Draw a small velocity vector arrow on flying cubes."""
    if cube.body.sleeping or cube.grabbed:
        return
    speed = math.sqrt(cube.body.vx**2 + cube.body.vy**2)
    if speed < 20:
        return
    sx, sy  = int(cube.sx), int(cube.sy)
    scale   = min(60.0, speed * 0.06)
    ex = int(sx + cube.body.vx / speed * scale)
    ey = int(sy + cube.body.vy / speed * scale)
    cv2.arrowedLine(img, (sx, sy), (ex, ey), (180, 120, 255), 2,
                    cv2.LINE_AA, tipLength=0.4)


# ── Cursor ────────────────────────────────────────────────────────────

def draw_cursor(img, cx, cy, hand_state):
    cx, cy  = int(cx), int(cy)
    is_fist = hand_state.is_fist
    color   = (50, 255, 120) if is_fist else (20, 230, 200)
    ring_r  = 12 if is_fist else 22

    ov = img.copy()
    cv2.circle(ov, (cx, cy), ring_r+18, color, -1, cv2.LINE_AA)
    cv2.addWeighted(ov, 0.07, img, 0.93, 0, img)

    ov = img.copy()
    cv2.circle(ov, (cx, cy), ring_r+7, color, -1, cv2.LINE_AA)
    cv2.addWeighted(ov, 0.18, img, 0.82, 0, img)

    cv2.circle(img, (cx, cy), ring_r, color, 2, cv2.LINE_AA)
    cv2.circle(img, (cx, cy), 4,      color, -1, cv2.LINE_AA)

    if not is_fist:
        arm = 10
        cv2.line(img, (cx-arm, cy), (cx+arm, cy), color, 1, cv2.LINE_AA)
        cv2.line(img, (cx, cy-arm), (cx, cy+arm), color, 1, cv2.LINE_AA)


# ── Two-hand scale ────────────────────────────────────────────────────

_prev_two_hand_dist = None

def two_hand_scale_delta(left, right, W, H):
    global _prev_two_hand_dist
    if not (left.visible and right.visible and left.is_fist and right.is_fist):
        _prev_two_hand_dist = None
        return None, None
    lx = left.index_tip[0] * W;  ly = left.index_tip[1] * H
    rx = right.index_tip[0] * W; ry = right.index_tip[1] * H
    dist = math.sqrt((rx-lx)**2 + (ry-ly)**2)
    mid  = ((lx+rx)/2, (ly+ry)/2)
    if _prev_two_hand_dist is None or _prev_two_hand_dist < 1.0:
        _prev_two_hand_dist = dist
        return 1.0, mid
    scale = dist / _prev_two_hand_dist
    _prev_two_hand_dist = dist
    return scale, mid


# ── HUD ───────────────────────────────────────────────────────────────

def draw_hud(img, tracker, grabbed_cube, gravity_on, W, H):
    r = tracker.right
    l = tracker.left
    h_img = img.shape[0]

    if r.visible:
        if grabbed_cube:
            state = "GRAB"
        elif r.is_fist:
            state = "FIST"
        elif r.is_pointing:
            state = "POINT"
        else:
            state = "TRACK"
        col = (50, 255, 120) if grabbed_cube else (20, 200, 200)
    else:
        state, col = "NO R HAND", (60, 60, 60)

    cv2.putText(img, f"R: {state}", (16, h_img-36),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1, cv2.LINE_AA)
    if r.visible:
        cv2.putText(img, r.debug_str(), (16, h_img-16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, (80,80,80), 1, cv2.LINE_AA)

    if l.visible:
        lstate = "SCALE" if (l.is_fist and r.is_fist) else "PLANE"
        cv2.putText(img, f"L: {lstate}", (16, h_img-56),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_LEFT, 1, cv2.LINE_AA)
        cv2.putText(img, l.debug_str(), (16, h_img-72),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, (80,80,80), 1, cv2.LINE_AA)

    # Physics debug for grabbed cube
    if grabbed_cube:
        cv2.putText(img, grabbed_cube.body.debug_str(), (16, h_img-90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, (100, 200, 255), 1, cv2.LINE_AA)

    # Gravity indicator
    grav_col = (50, 255, 120) if gravity_on else (60, 60, 60)
    cv2.putText(img, f"G: {'ON' if gravity_on else 'OFF'}", (W-80, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, grav_col, 1, cv2.LINE_AA)

    # Legend
    legend = [
        "FIST = grab & throw",
        "Twist wrist = spin",
        "L hand tilt = plane",
        "Both fists  = scale",
        "G = gravity  R = reset",
    ]
    for i, line in enumerate(legend):
        cv2.putText(img, line, (16, 20+i*16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, (60,60,60), 1, cv2.LINE_AA)


# ── Cube factory ──────────────────────────────────────────────────────

def make_cubes(W, H):
    return [
        Cube(int(W*0.25), int(H*0.40), size=int(min(W,H)*0.10)),
        Cube(int(W*0.50), int(H*0.42), size=int(min(W,H)*0.10)),
        Cube(int(W*0.75), int(H*0.40), size=int(min(W,H)*0.10)),
    ]


# ── Main ──────────────────────────────────────────────────────────────

def main():
    ensure_model()

    cap = cv2.VideoCapture(CAMERA_DEVICE)
    if not cap.isOpened():
        print(f"ERROR: Cannot open camera {CAMERA_DEVICE}")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  WINDOW_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, WINDOW_H)
    cap.set(cv2.CAP_PROP_FPS, 30)

    ret, _ = cap.read()
    if not ret:
        print("ERROR: Camera opened but can't read frames.")
        sys.exit(1)

    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Camera: {CAMERA_DEVICE}  {W}x{H}")

    floor_y = H - FLOOR_MARGIN

    cv2.namedWindow("Aether", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Aether", WINDOW_W, WINDOW_H)

    base_opts  = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
    opts       = mp_vision.HandLandmarkerOptions(
        base_options=base_opts,
        running_mode=mp_vision.RunningMode.IMAGE,
        num_hands=2,
        min_hand_detection_confidence=0.55,
        min_hand_presence_confidence=0.50,
        min_tracking_confidence=0.50,
    )
    landmarker = mp_vision.HandLandmarker.create_from_options(opts)

    tracker      = HandTracker()
    vel_tracker  = HandVelocityTracker()   # tracks hand speed while grabbing
    cubes        = make_cubes(W, H)
    grabbed_cube = None
    hovered_cube = None
    prev_was_fist = False
    gravity_on    = True

    cx, cy    = float(W/2), float(H/2)
    prev_time = time.time()

    print("Controls:")
    print("  FIST over cube = grab; open hand = THROW")
    print("  Twist wrist    = spin cube")
    print("  Left hand tilt = tilt plane")
    print("  Both fists     = scale")
    print("  G = toggle gravity   R = reset   Q = quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Camera read failed.")
            break

        frame = cv2.flip(frame, 1)

        now = time.time()
        dt  = min(now - prev_time, 0.05)
        prev_time = now

        # ── Hand tracking ──────────────────────────────────────────────
        rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = landmarker.detect(mp_img)
        tracker.update(result)

        right = tracker.right
        left  = tracker.left

        # ── Cursor ────────────────────────────────────────────────────
        if right.visible:
            cx = right.index_tip[0] * W
            cy = right.index_tip[1] * H

        # ── Hand velocity tracking (while grabbing) ────────────────────
        if grabbed_cube is not None and right.visible:
            vel_tracker.record(cx, cy, now)

        # ── Two-hand scale ─────────────────────────────────────────────
        scale_delta, scale_mid = two_hand_scale_delta(left, right, W, H)
        if scale_delta is not None and scale_delta != 1.0 and grabbed_cube is not None:
            grabbed_cube.size = int(max(20, min(300, grabbed_cube.size * scale_delta)))

        # ── Hover ──────────────────────────────────────────────────────
        if grabbed_cube is None:
            best_dist, best_cube = 9999.0, None
            for cube in cubes:
                d = cube.screen_dist(cx, cy)
                if d < best_dist:
                    best_dist, best_cube = d, cube

            if hovered_cube is not None:
                if hovered_cube.screen_dist(cx, cy) > HOVER_EXIT:
                    hovered_cube.hovered = False
                    hovered_cube = None

            if hovered_cube is None and best_cube is not None and best_dist < HOVER_RADIUS:
                hovered_cube         = best_cube
                hovered_cube.hovered = True

        # ── Grab / throw ───────────────────────────────────────────────
        just_fisted = right.is_fist and not prev_was_fist
        just_opened = not right.is_fist and prev_was_fist

        if just_fisted and hovered_cube is not None and grabbed_cube is None:
            grabbed_cube         = hovered_cube
            hovered_cube.hovered = False
            hovered_cube         = None
            grabbed_cube.grab(cx, cy)
            vel_tracker.reset()

        if just_opened and grabbed_cube is not None:
            # Transfer hand velocity → throw
            vx, vy = vel_tracker.release_velocity()
            grabbed_cube.release(vx, vy)
            grabbed_cube = None

        prev_was_fist = right.is_fist

        # ── Update cubes ───────────────────────────────────────────────
        for cube in cubes:
            cube.update(dt, cx, cy, right, left, W, H, floor_y, gravity_on)

        # ── Render ────────────────────────────────────────────────────
        out = frame.copy()

        draw_floor(out, floor_y, W)
        draw_plane(out, left, W, H)
        draw_hand_skeleton(out, right, W, H, COLOR_RIGHT, "R")
        draw_hand_skeleton(out, left,  W, H, COLOR_LEFT,  "L")

        for cube in cubes:
            if cube is not grabbed_cube:
                cube.draw(out)
                draw_velocity_arrow(out, cube)
        if grabbed_cube:
            grabbed_cube.draw(out)

        if right.visible:
            draw_cursor(out, cx, cy, right)

        if scale_delta is not None and left.is_fist and right.is_fist:
            mx, my = int(scale_mid[0]), int(scale_mid[1])
            cv2.circle(out, (mx, my), 8, (200, 200, 50), 2, cv2.LINE_AA)
            cv2.putText(out, f"SCALE {scale_delta:.2f}x", (mx+10, my),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200,200,50), 1, cv2.LINE_AA)

        draw_hud(out, tracker, grabbed_cube, gravity_on, W, H)

        fps = 1.0/dt if dt > 0 else 0.0
        cv2.putText(out, f"{fps:.0f} fps", (W-70, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (50,50,50), 1, cv2.LINE_AA)

        if W < WINDOW_W or H < WINDOW_H:
            out = cv2.resize(out, (WINDOW_W, WINDOW_H), interpolation=cv2.INTER_LINEAR)

        cv2.imshow("Aether", out)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('g'):
            gravity_on = not gravity_on
            print(f"Gravity: {'ON' if gravity_on else 'OFF'}")
        elif key == ord('r'):
            # Reset — drop grabbed cube first
            if grabbed_cube:
                grabbed_cube.release(0, 0)
                grabbed_cube = None
            hovered_cube = None
            cubes = make_cubes(W, H)
            vel_tracker.reset()
            print("Reset.")

    landmarker.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()