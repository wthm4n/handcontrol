"""
Aether — Phase 7: Spatial Knowledge Workspace

Every concept gets the most physically intuitive 3D form:
  arrays         → rows of ArrayBlock cubes
  trees/graphs   → TreeSphere node networks
  workflows      → PipeSegment pipelines
  stacks/queues  → StackDisc columns
  loops          → LoopMarker orbiting rings
  bitmasks       → BitField cube rows
  variables      → single ArrayBlock
  fallback       → ArrayBlock with label

RIGHT hand gestures:
  FIST over object        → grab & hold
  Open hand (release)     → throw
  Twist wrist             → spin object
  OPEN PALM held 2s       → spawn blank ArrayBlock at cursor depth
  FIST held over object 1s→ delete (countdown ring)
  PINCH near object       → select / deselect

LEFT hand:
  Tilt/roll               → workspace plane

BOTH FISTS:
  Spread / close          → scale grabbed object

Keys:
  G = toggle gravity
  R = reset scene
  S = toggle snap on grabbed object
  T = AI workspace generation (type topic in terminal) / stop teaching
  N = next teaching step (auto-starts teaching mode)
  P = previous teaching step
  C = cluster objects by category
  L = auto-layout objects
  Q = quit
"""

import cv2
import math
import time
import os
import sys
import urllib.request
import threading

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

from hand import HandTracker
from physics import HandVelocityTracker, HandDepthCalibrator
from objects import (
    SceneGraph, ArrayBlock, TreeSphere, PipeSegment, StackDisc,
    LoopMarker, BitField, ConnectionRenderer
)
from workspace_generator import WorkspaceGenerator
from teaching import TeachingScene
from effects import (
    AnimatedCursor, DepthPresenceHUD,
    SpawnSystem, DeleteSystem, SelectionSystem,
    SnapSystem, CalibrationOverlay,
    draw_floor_enhanced,
)


CAMERA_DEVICE = "/dev/video10"
WINDOW_W      = 1280
WINDOW_H      = 720

MODEL_PATH = os.path.join(os.path.dirname(__file__), "hand_landmarker.task")
MODEL_URL  = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
)

HOVER_RADIUS = 130
HOVER_EXIT   = 160
FLOOR_MARGIN = 40

COLOR_RIGHT = (  0, 220, 255)
COLOR_LEFT  = (255, 160,  40)
C_DIM       = ( 40,  40,  40)
C_TEAL      = (220, 240,  20)
C_GREEN     = ( 50, 255, 120)
C_AMBER     = ( 30, 190, 255)
C_PURPLE    = (230,  90, 215)

HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17),
]


class PromptInputThread:
    def __init__(self):
        self._result = None
        self._done   = False

    def start(self, banner=""):
        if banner: print(f"\n{banner}")
        print("Topic (blank to cancel): ", end="", flush=True)
        self._result = None
        self._done   = False
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self):
        try:
            self._result = input().strip()
        except EOFError:
            self._result = ""
        self._done = True

    @property
    def done(self): return self._done

    @property
    def result(self): return self._result


def ensure_model():
    if os.path.exists(MODEL_PATH) and os.path.getsize(MODEL_PATH) > 50_000:
        return
    print("Downloading hand_landmarker.task (~5 MB)…")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("Done.")


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


def draw_plane(img, left_hand, W, H):
    if not left_hand.visible: return
    cx, cy = W // 2, H // 2
    tilt_x = left_hand.orient_x * 0.4
    tilt_z = left_hand.orient_z * 0.3
    step = 80
    cols = W // step + 2
    rows = H // step + 2

    def T(gx, gy):
        skew_y = gy * math.sin(tilt_x) * 0.3
        roll_x = gx * math.cos(tilt_z) - gy * math.sin(tilt_z) * 0.15
        roll_y = gx * math.sin(tilt_z) * 0.15 + gy * math.cos(tilt_z)
        return int(cx + roll_x + skew_y), int(cy + roll_y)

    ov = img.copy()
    for r in range(-rows // 2, rows // 2 + 1):
        cv2.line(ov, T(-cols // 2 * step, r * step),
                     T( cols // 2 * step, r * step), (20, 50, 38), 1, cv2.LINE_AA)
    for c in range(-cols // 2, cols // 2 + 1):
        cv2.line(ov, T(c * step, -rows // 2 * step),
                     T(c * step,  rows // 2 * step), (20, 50, 38), 1, cv2.LINE_AA)
    cv2.addWeighted(ov, 0.30, img, 0.70, 0, img)


_prev_two_hand_dist = None

def two_hand_scale_delta(left, right, W, H):
    global _prev_two_hand_dist
    if not (left.visible and right.visible and left.is_fist and right.is_fist):
        _prev_two_hand_dist = None
        return None, None
    lx = left.index_tip[0] * W;  ly = left.index_tip[1] * H
    rx = right.index_tip[0] * W; ry = right.index_tip[1] * H
    dist = math.sqrt((rx - lx)**2 + (ry - ly)**2)
    mid  = ((lx + rx) / 2, (ly + ry) / 2)
    if _prev_two_hand_dist is None or _prev_two_hand_dist < 1.0:
        _prev_two_hand_dist = dist
        return 1.0, mid
    scale = dist / _prev_two_hand_dist
    _prev_two_hand_dist = dist
    return scale, mid


def make_default_scene(W, H):
    """
    Default scene: a small 5-element array + a binary tree root + a pipeline
    to demonstrate every major physical object type at startup.
    """
    graph = SceneGraph()


    values = [3, 1, 4, 1, 5]
    blocks = []
    for i, v in enumerate(values):
        b = ArrayBlock(value=v, index=i,
                       x3d=W*0.18 + i*110, y3d=H*0.35)
        b.label = f"arr[{i}]"
        graph.add_node(b)
        blocks.append(b)
    for i in range(len(blocks)-1):
        graph.connect(blocks[i], blocks[i+1], "→", "follows")


    root = TreeSphere(radius=32, label="root",
                      x3d=W*0.65, y3d=H*0.22, category="structure")
    root.body = "Binary tree root"
    left_child  = TreeSphere(radius=26, label="left",
                             x3d=W*0.55, y3d=H*0.42, category="structure")
    right_child = TreeSphere(radius=26, label="right",
                             x3d=W*0.75, y3d=H*0.42, category="structure")
    for n in (root, left_child, right_child):
        graph.add_node(n)
    graph.connect(root, left_child,  "left",  "contains")
    graph.connect(root, right_child, "right", "contains")


    steps = ["Input", "Process", "Output"]
    pipes = []
    for i, s in enumerate(steps):
        p = PipeSegment(pipe_w=72, pipe_h=40, label=s,
                        x3d=W*0.25 + i*210, y3d=H*0.70,
                        category="process")
        p.body = f"Step {i+1}"
        graph.add_node(p)
        pipes.append(p)
    for i in range(len(pipes)-1):
        graph.connect(pipes[i], pipes[i+1], "→", "follows")

    return graph


def draw_hud(img, tracker, grabbed_obj, gravity_on, W, H,
             num_objs, num_selected, generating, teach_active):
    r = tracker.right
    if r.visible:
        if grabbed_obj:
            state, col = "HOLDING", C_GREEN
        elif r.is_fist:
            state, col = "FIST",    C_GREEN
        elif r.is_open:
            state, col = "SPAWN",   C_TEAL
        else:
            state, col = "TRACK",   C_DIM
    else:
        state, col = "WAITING", C_DIM

    cv2.putText(img, state, (16, H - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.40, col, 1, cv2.LINE_AA)
    grav_col = C_GREEN if gravity_on else C_DIM
    cv2.putText(img, "G" if gravity_on else "g", (W - 28, H - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.40, grav_col, 1, cv2.LINE_AA)

    if num_objs > 0:
        count_str = f"{num_objs} obj"
        if num_selected > 0:
            count_str += f"  {num_selected} sel"
        cv2.putText(img, count_str, (W - 90, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, C_DIM, 1, cv2.LINE_AA)

    if generating:
        dots = "." * (int(time.time() * 3) % 4)
        cv2.putText(img, f"AI{dots}", (W // 2 - 18, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.36, C_PURPLE, 1, cv2.LINE_AA)

    if teach_active:
        cv2.putText(img, "TEACH", (W // 2 - 22, H - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.36, C_AMBER, 1, cv2.LINE_AA)

    legend = ["G=gravity", "R=reset", "S=snap", "T=AI/teach",
              "N=next", "P=prev", "C=cluster", "L=layout", "Q=quit"]
    for i, line in enumerate(legend):
        cv2.putText(img, line, (16, 18 + i * 13),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.24, (32, 32, 32), 1, cv2.LINE_AA)


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
        base_options   = base_opts,
        running_mode   = mp_vision.RunningMode.IMAGE,
        num_hands      = 2,
        min_hand_detection_confidence = 0.55,
        min_hand_presence_confidence  = 0.50,
        min_tracking_confidence       = 0.50,
    )
    landmarker = mp_vision.HandLandmarker.create_from_options(opts)


    tracker       = HandTracker()
    vel_tracker   = HandVelocityTracker()
    depth_calib   = HandDepthCalibrator(z_scale=600.0, smooth=0.18)
    cursor        = AnimatedCursor()
    depth_hud     = DepthPresenceHUD()
    delete_sys    = DeleteSystem()
    select_sys    = SelectionSystem()
    snap_sys      = SnapSystem()
    calib_overlay = CalibrationOverlay()
    conn_renderer = ConnectionRenderer()


    graph = make_default_scene(W, H)


    _spawn_types = [ArrayBlock, TreeSphere, PipeSegment, StackDisc]
    _sidx = [0]

    def node_factory(cx, cy, depth_z, size):
        ShapeClass = _spawn_types[_sidx[0] % len(_spawn_types)]
        _sidx[0] += 1
        if ShapeClass is ArrayBlock:
            obj = ArrayBlock(value="?", x3d=cx, y3d=cy, z3d=depth_z)
            obj.half = max(20, size // 3)
        elif ShapeClass is TreeSphere:
            obj = TreeSphere(radius=max(20, size // 3),
                             x3d=cx, y3d=cy, z3d=depth_z)
        elif ShapeClass is PipeSegment:
            obj = PipeSegment(pipe_w=max(40, size // 2), pipe_h=max(24, size // 3),
                              x3d=cx, y3d=cy, z3d=depth_z)
        else:
            obj = StackDisc(disc_rx=max(30, size // 2), disc_ry=max(10, size // 6),
                            x3d=cx, y3d=cy, z3d=depth_z)
        obj.label = "new"
        graph.add_node(obj)
        return obj

    spawn_sys = SpawnSystem(factory=node_factory)


    ollama_host  = os.environ.get("OLLAMA_HOST",  "http://localhost:11434")
    ollama_model = os.environ.get("OLLAMA_MODEL", "qwen2.5-coder:7b-instruct-q4_K_M")
    generator    = WorkspaceGenerator(host=ollama_host, model=ollama_model)
    generating   = False
    gen_status   = ""
    gen_status_t = 0.0

    prompt_thread  = PromptInputThread()
    waiting_prompt = False

    def on_generated(new_nodes, error):
        nonlocal generating, gen_status, gen_status_t
        generating   = False
        gen_status_t = time.time()
        if error:
            gen_status = f"Error: {error}"
            print(f"\n[WorkspaceGenerator] {error}")
        else:
            gen_status = f"Added {len(new_nodes)} objects"
            print(f"\n[WorkspaceGenerator] Added {len(new_nodes)} objects")


    teach = TeachingScene(graph)


    grabbed_obj   = None
    hovered_obj   = None
    prev_was_fist = False
    gravity_on    = True
    snap_on       = False
    hand_z_world  = 0.0
    cx, cy        = float(W / 2), float(H / 2)
    prev_time     = time.time()

    print("\nAETHER — Spatial Physical Workspace")
    print("  Arrays → cube rows  |  Trees → sphere networks")
    print("  Workflows → pipelines  |  Stacks → disc columns")
    print("  FIST = grab/throw  |  OPEN PALM 2s = spawn")
    print("  T = AI scene (type topic)  |  N/P = teach steps")
    print(f"  Ollama: {ollama_model} @ {ollama_host}")
    print()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Camera read failed.")
            break

        frame = cv2.flip(frame, 1)
        now   = time.time()
        dt    = min(now - prev_time, 0.05)
        prev_time = now


        rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = landmarker.detect(mp_img)
        tracker.update(result)

        right = tracker.right
        left  = tracker.left

        if right.visible:
            cx = right.index_tip[0] * W
            cy = right.index_tip[1] * H
            hand_z_world = depth_calib.update(right.wrist_pos[2], now)

        if grabbed_obj is not None and right.visible:
            vel_tracker.record(cx, cy, hand_z_world, now)


        scale_delta, scale_mid = two_hand_scale_delta(left, right, W, H)
        if scale_delta is not None and scale_delta != 1.0 and grabbed_obj is not None:
            grabbed_obj.size = max(0.3, min(4.0, grabbed_obj.size * scale_delta))

            if isinstance(grabbed_obj, ArrayBlock):
                grabbed_obj.half = max(10, grabbed_obj.half * scale_delta)
            elif isinstance(grabbed_obj, TreeSphere):
                grabbed_obj.radius = max(8, grabbed_obj.radius * scale_delta)
            elif isinstance(grabbed_obj, PipeSegment):
                grabbed_obj.pipe_w = max(20, grabbed_obj.pipe_w * scale_delta)
                grabbed_obj.pipe_h = max(12, grabbed_obj.pipe_h * scale_delta)
            elif isinstance(grabbed_obj, StackDisc):
                grabbed_obj.disc_rx = max(15, grabbed_obj.disc_rx * scale_delta)
                grabbed_obj.disc_ry = max(5,  grabbed_obj.disc_ry * scale_delta)


        objects = graph.nodes


        if grabbed_obj is None:
            best_dist, best_obj = 9999.0, None
            for n in objects:
                d = n.screen_dist(cx, cy)
                if d < best_dist:
                    best_dist, best_obj = d, n

            if hovered_obj is not None and hovered_obj.screen_dist(cx, cy) > HOVER_EXIT:
                hovered_obj.hovered = False
                hovered_obj = None

            if hovered_obj is None and best_obj is not None and best_dist < HOVER_RADIUS:
                hovered_obj = best_obj
                hovered_obj.hovered = True


        just_fisted = right.is_fist and not prev_was_fist
        just_opened = not right.is_fist and prev_was_fist

        if just_fisted and hovered_obj is not None and grabbed_obj is None:
            grabbed_obj = hovered_obj
            hovered_obj.hovered = False
            hovered_obj = None
            grabbed_obj.grab(cx, cy, hand_z_world)
            grabbed_obj.snap_enabled = snap_on
            vel_tracker.reset()

        if just_opened and grabbed_obj is not None:
            vx, vy, vz = vel_tracker.release_velocity()
            vz += depth_calib.vel_z * 0.4
            grabbed_obj.release(vx, vy, vz)
            grabbed_obj = None

        prev_was_fist = right.is_fist


        if grabbed_obj is None:
            spawn_sys.update(right, cx, cy, hand_z_world, now, objects, W, H)


        to_delete = delete_sys.update(right, cx, cy, objects, grabbed_obj)
        for dead in to_delete:
            if dead is grabbed_obj: grabbed_obj = None
            if dead is hovered_obj: hovered_obj = None
            graph.remove_node(dead)


        objects = graph.nodes
        select_sys.update(right, cx, cy, objects, grabbed_obj)


        for n in objects:
            n.update(dt, cx, cy, hand_z_world, right, left, W, H, floor_y,
                     gravity_on=gravity_on)


        if waiting_prompt and prompt_thread.done:
            waiting_prompt = False
            topic = prompt_thread.result
            if topic:
                generating = True
                gen_status = ""
                print(f"[WorkspaceGenerator] Generating scene for: '{topic}'")
                generator.generate_async(
                    prompt   = topic,
                    graph    = graph,
                    callback = on_generated,
                    screen_w = W,
                    screen_h = H,
                )
            else:
                print("[WorkspaceGenerator] Cancelled.")

        if gen_status and (now - gen_status_t) > 3.0:
            gen_status = ""


        teach.update(dt, W, H)


        out = frame.copy()

        draw_floor_enhanced(out, floor_y, W)
        draw_plane(out, left, W, H)
        draw_hand_skeleton(out, right, W, H, COLOR_RIGHT, "R")
        draw_hand_skeleton(out, left,  W, H, COLOR_LEFT,  "L")


        objects = graph.nodes
        objects_by_id = {n.id: n for n in objects}
        conn_renderer.render(out, objects_by_id, graph.edges, dt)


        sorted_objs = sorted(objects, key=lambda n: n.z3d, reverse=True)
        render_list = [n for n in sorted_objs if n is not grabbed_obj]
        if grabbed_obj:
            render_list.append(grabbed_obj)

        glow_layer = out.copy()
        glow_layer[:] = 0
        for n in render_list:
            n.draw_glow(glow_layer)
        cv2.addWeighted(glow_layer, 0.55, out, 1.0, 0, out)

        for n in render_list:
            n.draw_crisp(out)

        if grabbed_obj and snap_on:
            snap_sys.draw_guides(out, grabbed_obj, W, H)

        if right.visible:
            cursor.draw(out, cx, cy, right, hovered_obj is not None,
                        depth_z=hand_z_world, dt=dt)

        if right.visible and depth_calib.is_calibrated:
            depth_hud.draw(out, cx, cy, hand_z_world,
                           active=(grabbed_obj is not None or abs(hand_z_world) > 20))

        if right.visible and right.is_open and grabbed_obj is None:
            spawn_sys.draw_progress(out, cx, cy)

        select_sys.draw_selection_feedback(out, objects, W, H)

        if scale_delta is not None and left.is_fist and right.is_fist:
            mx, my = int(scale_mid[0]), int(scale_mid[1])
            cv2.putText(out, f"{scale_delta:.2f}×", (mx + 10, my),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, C_AMBER, 1, cv2.LINE_AA)

        teach.draw_hud(out, W, H)

        if gen_status:
            cv2.putText(out, gen_status, (W // 2 - 80, H - 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, C_PURPLE, 1, cv2.LINE_AA)

        num_sel = sum(1 for n in objects if n.selected)
        draw_hud(out, tracker, grabbed_obj, gravity_on, W, H,
                 len(objects), num_sel, generating, teach.active)

        fps = 1.0 / dt if dt > 0 else 0.0
        cv2.putText(out, f"{fps:.0f}", (W - 28, H - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.28, (30, 30, 30), 1, cv2.LINE_AA)

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
            if grabbed_obj:
                grabbed_obj.snap_enabled = snap_on
            print(f"Snap: {'ON' if snap_on else 'OFF'}")

        elif key == ord('r'):
            if grabbed_obj:
                grabbed_obj.release(0, 0, 0)
                grabbed_obj = None
            hovered_obj = None
            teach.stop()
            graph      = make_default_scene(W, H)
            teach      = TeachingScene(graph)
            spawn_sys  = SpawnSystem(factory=node_factory)
            vel_tracker.reset()
            depth_calib.reset()
            hand_z_world = 0.0
            generating   = False
            gen_status   = ""
            print("Reset.")

        elif key == ord('t'):
            if teach.active:
                teach.stop()
                print("Teaching mode stopped.")
            elif waiting_prompt or generating:
                print("Already generating…")
            else:
                waiting_prompt = True
                prompt_thread.start("─" * 40 + "\nAI Workspace Generator\n" + "─" * 40)

        elif key == ord('n'):
            if not teach.active:
                teach.sync_graph(graph)
                teach.start()
                print(f"Teaching mode: step {teach.step_label}")
            else:
                teach.next()
                print(f"Teaching step: {teach.step_label}")

        elif key == ord('p'):
            if teach.active:
                teach.prev()
                print(f"Teaching step: {teach.step_label}")

        elif key == ord('c'):
            if teach.active:
                teach.stop()
            graph.cluster_by_category(W, H)
            print("Clustered by category.")

        elif key == ord('l'):
            graph.auto_layout(W, H)
            print("Auto-layout applied.")

    landmarker.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
