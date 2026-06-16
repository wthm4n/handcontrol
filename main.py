"""
Aether — Phase 5: Spatial Computing Environment

RIGHT hand gestures:
  FIST over cube          → grab & hold
  Open hand (release)     → throw
  Twist wrist             → spin cube
  OPEN PALM held 0.5s     → spawn new cube at cursor depth
  FIST held over cube 1s  → delete (countdown ring)
  PINCH near cube         → select / deselect

LEFT hand:
  Tilt/roll               → workspace plane

BOTH FISTS:
  Spread / close          → scale grabbed cube

Keys:
  G = toggle gravity
  R = reset scene
  S = toggle snap on grabbed cube
  Q = quit
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
from physics import HandVelocityTracker, HandDepthCalibrator
from effects import (
    AnimatedCursor, DepthPresenceHUD,
    SpawnSystem, DeleteSystem, SelectionSystem,
    SnapSystem, CalibrationOverlay,
    draw_floor_enhanced, draw_minimal_hud,
)


# ── Config ────────────────────────────────────────────────────────────

CAMERA_DEVICE = "/dev/video10"
WINDOW_W      = 1280
WINDOW_H      = 720

MODEL_PATH = os.path.join(os.path.dirname(__file__), "hand_landmarker.task")
MODEL_URL  = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
)

HOVER_RADIUS  = 110
HOVER_EXIT    = 145
FLOOR_MARGIN  = 40

# Hand skeleton colours
COLOR_RIGHT = (  255, 255, 255)
COLOR_LEFT  = (255, 160,  40)

HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17),
]


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
        cv2.line(img, pts[a], pts[b], color, 1, cv2.LINE_AA)
    for i, (px, py) in enumerate(pts):
        r = 4 if i in (4, 8, 12, 16, 20) else 2
        cv2.circle(img, (px, py), r, color, -1, cv2.LINE_AA)


# ── Plane grid ────────────────────────────────────────────────────────

def draw_plane(img, left_hand, W, H):
    if not left_hand.visible:
        return
    cx, cy = W//2, H//2
    tilt_x = left_hand.orient_x * 0.4
    tilt_z = left_hand.orient_z * 0.3
    step   = 80
    cols   = W // step + 2
    rows   = H // step + 2

    def T(gx, gy):
        skew_y = gy * math.sin(tilt_x) * 0.3
        roll_x = gx * math.cos(tilt_z) - gy * math.sin(tilt_z) * 0.15
        roll_y = gx * math.sin(tilt_z) * 0.15 + gy * math.cos(tilt_z)
        return int(cx+roll_x+skew_y), int(cy+roll_y)

    ov = img.copy()
    for r in range(-rows//2, rows//2+1):
        cv2.line(ov, T(-cols//2*step, r*step),
                     T( cols//2*step, r*step), (20, 50, 38), 1, cv2.LINE_AA)
    for c in range(-cols//2, cols//2+1):
        cv2.line(ov, T(c*step, -rows//2*step),
                     T(c*step,  rows//2*step), (20, 50, 38), 1, cv2.LINE_AA)
    cv2.addWeighted(ov, 0.30, img, 0.70, 0, img)


# ── Two-hand scale ────────────────────────────────────────────────────

_prev_two_hand_dist = None

def two_hand_scale_delta(left, right, W, H):
    global _prev_two_hand_dist
    if not (left.visible and right.visible and left.is_fist and right.is_fist):
        _prev_two_hand_dist = None
        return None, None
    lx = left.index_tip[0]*W;  ly = left.index_tip[1]*H
    rx = right.index_tip[0]*W; ry = right.index_tip[1]*H
    dist = math.sqrt((rx-lx)**2+(ry-ly)**2)
    mid  = ((lx+rx)/2, (ly+ry)/2)
    if _prev_two_hand_dist is None or _prev_two_hand_dist < 1.0:
        _prev_two_hand_dist = dist
        return 1.0, mid
    scale = dist / _prev_two_hand_dist
    _prev_two_hand_dist = dist
    return scale, mid


# ── Cube factory ──────────────────────────────────────────────────────

def make_cubes(W, H):
    return [
        Cube(int(W*0.25), int(H*0.40), z3d=0.0, size=int(min(W,H)*0.10)),
        Cube(int(W*0.50), int(H*0.42), z3d=0.0, size=int(min(W,H)*0.10)),
        Cube(int(W*0.75), int(H*0.40), z3d=0.0, size=int(min(W,H)*0.10)),
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
    floor_y = H - FLOOR_MARGIN
    print(f"Camera: {CAMERA_DEVICE}  {W}x{H}  floor_y={floor_y}")

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

    # ── Systems ───────────────────────────────────────────────────────
    tracker      = HandTracker()
    vel_tracker  = HandVelocityTracker()
    depth_calib  = HandDepthCalibrator(z_scale=600.0, smooth=0.18)

    cursor       = AnimatedCursor()
    depth_hud    = DepthPresenceHUD()
    spawn_sys    = SpawnSystem()
    delete_sys   = DeleteSystem()
    select_sys   = SelectionSystem()
    snap_sys     = SnapSystem()
    calib_overlay = CalibrationOverlay()

    cubes        = make_cubes(W, H)
    grabbed_cube = None
    hovered_cube = None
    prev_was_fist = False
    gravity_on    = True
    snap_on       = False

    hand_z_world  = 0.0
    cx, cy        = float(W/2), float(H/2)
    prev_time     = time.time()

    print("AETHER Phase 5")
    print("  FIST  = grab & throw")
    print("  OPEN PALM held = spawn cube")
    print("  FIST held over cube = delete")
    print("  PINCH = select / deselect")
    print("  Twist = spin | L-hand tilt = plane")
    print("  Both fists = scale | G=gravity S=snap R=reset Q=quit")

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

        # ── Right-hand XY cursor ───────────────────────────────────────
        if right.visible:
            cx = right.index_tip[0] * W
            cy = right.index_tip[1] * H

        # ── Right-hand Z via calibrator ────────────────────────────────
        if right.visible:
            raw_mp_z    = right.wrist_pos[2]
            hand_z_world = depth_calib.update(raw_mp_z, now)

        # ── Record velocity (3D) ───────────────────────────────────────
        if grabbed_cube is not None and right.visible:
            vel_tracker.record(cx, cy, hand_z_world, now)

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
            grabbed_cube.grab(cx, cy, hand_z_world)
            grabbed_cube.snap_enabled = snap_on
            vel_tracker.reset()

        if just_opened and grabbed_cube is not None:
            vx, vy, vz = vel_tracker.release_velocity()
            # Augment Z velocity with calibrator's depth velocity
            vz += depth_calib.vel_z * 0.4
            grabbed_cube.release(vx, vy, vz)
            grabbed_cube = None

        prev_was_fist = right.is_fist

        # ── Spawn system ───────────────────────────────────────────────
        if grabbed_cube is None:
            spawn_sys.update(right, cx, cy, hand_z_world, now, cubes, W, H)

        # ── Delete system ──────────────────────────────────────────────
        to_delete = delete_sys.update(right, cx, cy, cubes, grabbed_cube)
        for dead in to_delete:
            if dead in cubes:
                cubes.remove(dead)

        # ── Selection system ───────────────────────────────────────────
        select_sys.update(right, cx, cy, cubes, grabbed_cube)

        # ── Update cubes ───────────────────────────────────────────────
        for cube in cubes:
            cube.update(
                dt, cx, cy, hand_z_world,
                right, left,
                W, H, floor_y,
                gravity_on=gravity_on,
            )

        # ── Render ────────────────────────────────────────────────────
        out = frame.copy()

        draw_floor_enhanced(out, floor_y, W)
        draw_plane(out, left, W, H)
        draw_hand_skeleton(out, right, W, H, COLOR_RIGHT, "R")
        draw_hand_skeleton(out, left,  W, H, COLOR_LEFT,  "L")

        # Sort by z3d: far objects first
        for cube in sorted(cubes, key=lambda c: c.z3d, reverse=True):
            if cube is not grabbed_cube:
                cube.draw(out)
        if grabbed_cube:
            grabbed_cube.draw(out)

        # Snap guides
        if grabbed_cube and snap_on:
            snap_sys.draw_guides(out, grabbed_cube, W, H)

        # Cursor
        if right.visible:
            is_hovering = hovered_cube is not None
            cursor.draw(out, cx, cy, right, is_hovering,
                        depth_z=hand_z_world, dt=dt)

        # Depth presence HUD
        if right.visible and depth_calib.is_calibrated:
            depth_hud.draw(out, cx, cy, hand_z_world,
                           active=(grabbed_cube is not None or abs(hand_z_world) > 20))

        # Spawn charge indicator
        if right.visible and right.is_open and grabbed_cube is None:
            spawn_sys.draw_progress(out, cx, cy)

        # Selection feedback
        select_sys.draw_selection_feedback(out, cubes, W, H)

        # Scale feedback
        if scale_delta is not None and left.is_fist and right.is_fist:
            mx, my = int(scale_mid[0]), int(scale_mid[1])
            cv2.putText(out, f"{scale_delta:.2f}×", (mx + 10, my),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (30, 190, 255), 1, cv2.LINE_AA)

        # Minimal HUD
        num_sel = sum(1 for c in cubes if c.selected)
        draw_minimal_hud(out, tracker, grabbed_cube, gravity_on, W, H,
                         len(cubes), num_sel)

        # Snap indicator
        if snap_on:
            cv2.putText(out, "SNAP", (W // 2 - 18, H - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.32,
                        (20, 60, 40), 1, cv2.LINE_AA)

        # FPS (very subtle)
        fps = 1.0 / dt if dt > 0 else 0.0
        cv2.putText(out, f"{fps:.0f}", (W - 28, H - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.28, (30, 30, 30), 1, cv2.LINE_AA)

        # Calibration overlay (fades away after ~45 frames)
        calib_overlay.draw(out, depth_calib.calibration_progress, W, H)

        if W < WINDOW_W or H < WINDOW_H:
            out = cv2.resize(out, (WINDOW_W, WINDOW_H),
                             interpolation=cv2.INTER_LINEAR)

        cv2.imshow("Aether", out)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('g'):
            gravity_on = not gravity_on
            print(f"Gravity: {'ON' if gravity_on else 'OFF'}")
        elif key == ord('s'):
            snap_on = not snap_on
            if grabbed_cube:
                grabbed_cube.snap_enabled = snap_on
            print(f"Snap: {'ON' if snap_on else 'OFF'}")
        elif key == ord('r'):
            if grabbed_cube:
                grabbed_cube.release(0, 0, 0)
                grabbed_cube = None
            hovered_cube = None
            cubes = make_cubes(W, H)
            vel_tracker.reset()
            depth_calib.reset()
            hand_z_world = 0.0
            print("Reset.")

    landmarker.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()