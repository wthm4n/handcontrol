"""
Aether — Phase 6: Spatial Knowledge Workspace

RIGHT hand gestures:
  FIST over node          → grab & hold
  Open hand (release)     → throw
  Twist wrist             → spin node
  OPEN PALM held 2s       → spawn blank node at cursor depth
  FIST held over node 1s  → delete (countdown ring)
  PINCH near node         → select / deselect

LEFT hand:
  Tilt/roll               → workspace plane

BOTH FISTS:
  Spread / close          → scale grabbed node

Keys:
  G = toggle gravity
  R = reset scene (blank workspace)
  S = toggle snap on grabbed node
  T = AI scene generation (type topic in terminal) / stop teaching
  N = next teaching step (auto-starts teaching mode)
  P = previous teaching step
  C = cluster nodes by category
  L = auto-layout nodes
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
from node import SpatialNode
from graph import KnowledgeGraph, ConnectionRenderer
from scene_generator import SceneGenerator
from effects import (
    AnimatedCursor, DepthPresenceHUD,
    SpawnSystem, DeleteSystem, SelectionSystem,
    SnapSystem, CalibrationOverlay,
    draw_floor_enhanced,
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

HOVER_RADIUS  = 130
HOVER_EXIT    = 160
FLOOR_MARGIN  = 40

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


# ── Teaching scene ────────────────────────────────────────────────────

class TeachingScene:
    """
    Walks through a KnowledgeGraph node-by-node in BFS order,
    highlighting the focused node and dimming all others.
    """

    def __init__(self, graph):
        self._graph  = graph
        self._step   = 0
        self._order  = []
        self._active = False

    def sync_graph(self, graph):
        self._graph = graph

    def build_order(self):
        nodes = self._graph.nodes
        if not nodes:
            self._order = []
            return
        adj  = {n.id: list(self._graph._adj.get(n.id,  set())) for n in nodes}
        radj = {n.id: list(self._graph._radj.get(n.id, set())) for n in nodes}
        roots = [n.id for n in nodes if not radj[n.id]]
        if not roots:
            roots = [nodes[0].id]
        visited, queue, order = set(roots), list(roots), list(roots)
        while queue:
            nxt = []
            for nid in queue:
                for child in adj.get(nid, []):
                    if child not in visited:
                        visited.add(child)
                        nxt.append(child)
                        order.append(child)
            queue = nxt
        for n in nodes:
            if n.id not in visited:
                order.append(n.id)
        self._order = order
        self._step  = 0

    def start(self):
        self.build_order()
        self._active = True
        self._apply()

    def stop(self):
        self._active = False
        for n in self._graph.nodes:
            n.focused = False
            n.dimmed  = False

    @property
    def active(self):
        return self._active

    @property
    def current_node(self):
        if not self._order or self._step >= len(self._order):
            return None
        return self._graph.get_node(self._order[self._step])

    @property
    def step_label(self):
        if not self._order:
            return ""
        return f"{self._step + 1} / {len(self._order)}"

    def next(self):
        if not self._active or not self._order:
            return
        self._step = min(self._step + 1, len(self._order) - 1)
        self._apply()

    def prev(self):
        if not self._active or not self._order:
            return
        self._step = max(self._step - 1, 0)
        self._apply()

    def _apply(self):
        focused_id = self._order[self._step] if self._order else None
        for n in self._graph.nodes:
            n.focused = (n.id == focused_id)
            n.dimmed  = (n.id != focused_id)

    def draw_hud(self, img, W, H):
        if not self._active:
            return
        node = self.current_node
        if node is None:
            return
        bx, by = W // 2 - 200, H - 80
        ov = img.copy()
        cv2.rectangle(ov, (bx, by), (bx + 400, by + 60), (10, 20, 15), -1)
        cv2.addWeighted(ov, 0.75, img, 0.25, 0, img)
        cv2.rectangle(img, (bx, by), (bx + 400, by + 60), C_TEAL, 1, cv2.LINE_AA)
        cv2.putText(img, node.title, (bx + 12, by + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, C_TEAL, 1, cv2.LINE_AA)
        cv2.putText(img, node.body[:60], (bx + 12, by + 44),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, (160, 200, 140), 1, cv2.LINE_AA)
        cv2.putText(img, self.step_label, (bx + 350, by + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, C_AMBER, 1, cv2.LINE_AA)
        cv2.putText(img, "N=next  P=prev  T=stop teach",
                    (bx + 60, by + 56),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.26, C_DIM, 1, cv2.LINE_AA)


# ── Prompt input thread ───────────────────────────────────────────────

class PromptInputThread:
    def __init__(self):
        self._result = None
        self._done   = False

    def start(self, banner=""):
        if banner:
            print(f"\n{banner}")
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
    def done(self):
        return self._done

    @property
    def result(self):
        return self._result


# ── Helpers ───────────────────────────────────────────────────────────

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
    if not left_hand.visible:
        return
    cx, cy  = W // 2, H // 2
    tilt_x  = left_hand.orient_x * 0.4
    tilt_z  = left_hand.orient_z * 0.3
    step    = 80
    cols    = W // step + 2
    rows    = H // step + 2

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
    dist = math.sqrt((rx - lx) ** 2 + (ry - ly) ** 2)
    mid  = ((lx + rx) / 2, (ly + ry) / 2)
    if _prev_two_hand_dist is None or _prev_two_hand_dist < 1.0:
        _prev_two_hand_dist = dist
        return 1.0, mid
    scale = dist / _prev_two_hand_dist
    _prev_two_hand_dist = dist
    return scale, mid


def make_default_graph(W, H):
    graph = KnowledgeGraph()
    intro   = SpatialNode(title="Welcome to Aether",
                          body="AI-powered spatial knowledge workspace.",
                          category="concept", icon="bulb")
    gesture = SpatialNode(title="Hand Gestures",
                          body="Grab, throw, pinch-select, open-palm spawn.",
                          category="system", icon="flow")
    ai_node = SpatialNode(title="AI Scenes",
                          body="Press T to generate a topic spatially.",
                          category="process", icon="cpu")
    teach   = SpatialNode(title="Teaching Mode",
                          body="N/P keys walk through any scene step-by-step.",
                          category="concept", icon="scroll")
    for n in (intro, gesture, ai_node, teach):
        graph.add_node(n)
    graph.connect(intro, gesture,  "uses")
    graph.connect(intro, ai_node,  "generates")
    graph.connect(intro, teach,    "enables")
    graph.auto_layout(W, H)
    return graph


def draw_minimal_hud(img, tracker, grabbed_node, gravity_on, W, H,
                     num_nodes, num_selected, generating, teach_active):
    r = tracker.right
    if r.visible:
        if grabbed_node:
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
    if num_nodes > 0:
        count_str = f"{num_nodes} nodes"
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
        base_options   = base_opts,
        running_mode   = mp_vision.RunningMode.IMAGE,
        num_hands      = 2,
        min_hand_detection_confidence = 0.55,
        min_hand_presence_confidence  = 0.50,
        min_tracking_confidence       = 0.50,
    )
    landmarker = mp_vision.HandLandmarker.create_from_options(opts)

    # ── Systems ───────────────────────────────────────────────────────
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

    # ── Knowledge graph ───────────────────────────────────────────────
    graph = make_default_graph(W, H)

    # Spawn factory — creates SpatialNode and registers it in the graph
    _spawn_cats = ["concept", "structure", "code", "math", "process", "system"]
    _sidx = [0]

    def node_factory(cx, cy, depth_z, size):
        cat = _spawn_cats[_sidx[0] % len(_spawn_cats)]
        _sidx[0] += 1
        n = SpatialNode(title="New Node", body="Tap and explore this idea.",
                        category=cat, size=size)
        n.x3d = cx;  n.y3d = cy;  n.z3d = depth_z
        graph.add_node(n)
        return n

    spawn_sys = SpawnSystem(factory=node_factory)

    # ── AI scene generator ────────────────────────────────────────────
    api_key    = os.environ.get("ANTHROPIC_API_KEY", "")
    generator  = SceneGenerator(api_key=api_key)
    generating = False
    gen_status = ""
    gen_status_t = 0.0

    prompt_thread  = PromptInputThread()
    waiting_prompt = False

    def on_generated(new_nodes, error):
        nonlocal generating, gen_status, gen_status_t
        generating   = False
        gen_status_t = time.time()
        if error:
            gen_status = f"Error: {error}"
            print(f"\n[SceneGenerator] {error}")
        else:
            gen_status = f"Added {len(new_nodes)} nodes"
            print(f"\n[SceneGenerator] Added {len(new_nodes)} nodes")

    # ── Teaching mode ─────────────────────────────────────────────────
    teach = TeachingScene(graph)

    # ── Interaction state ─────────────────────────────────────────────
    grabbed_node  = None
    hovered_node  = None
    prev_was_fist = False
    gravity_on    = True
    snap_on       = False
    hand_z_world  = 0.0
    cx, cy        = float(W / 2), float(H / 2)
    prev_time     = time.time()

    print("\nAETHER Phase 6 — Spatial Knowledge Workspace")
    print("  FIST         = grab & throw node")
    print("  OPEN PALM    = spawn blank node (hold 2s)")
    print("  PINCH        = select / deselect")
    print("  T            = AI scene generation (type topic in terminal)")
    print("                 or stop teaching mode if active")
    print("  N / P        = teaching mode next / prev step")
    print("  C            = cluster nodes by category")
    print("  L            = auto-layout")
    print("  G / S / R / Q = gravity / snap / reset / quit")
    if not api_key:
        print("\n  ⚠  ANTHROPIC_API_KEY not set — AI generation disabled.")
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

        # ── Hand tracking ──────────────────────────────────────────────
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

        if grabbed_node is not None and right.visible:
            vel_tracker.record(cx, cy, hand_z_world, now)

        # ── Two-hand scale ─────────────────────────────────────────────
        scale_delta, scale_mid = two_hand_scale_delta(left, right, W, H)
        if scale_delta is not None and scale_delta != 1.0 and grabbed_node is not None:
            grabbed_node.size = int(max(20, min(300, grabbed_node.size * scale_delta)))

        # ── Live node list ─────────────────────────────────────────────
        nodes = graph.nodes

        # ── Hover ──────────────────────────────────────────────────────
        if grabbed_node is None:
            best_dist, best_node = 9999.0, None
            for n in nodes:
                d = n.screen_dist(cx, cy)
                if d < best_dist:
                    best_dist, best_node = d, n

            if hovered_node is not None and hovered_node.screen_dist(cx, cy) > HOVER_EXIT:
                hovered_node.hovered = False
                hovered_node = None

            if hovered_node is None and best_node is not None and best_dist < HOVER_RADIUS:
                hovered_node = best_node
                hovered_node.hovered = True

        # ── Grab / throw ───────────────────────────────────────────────
        just_fisted = right.is_fist and not prev_was_fist
        just_opened = not right.is_fist and prev_was_fist

        if just_fisted and hovered_node is not None and grabbed_node is None:
            grabbed_node = hovered_node
            hovered_node.hovered = False
            hovered_node = None
            grabbed_node.grab(cx, cy, hand_z_world)
            grabbed_node.snap_enabled = snap_on
            vel_tracker.reset()

        if just_opened and grabbed_node is not None:
            vx, vy, vz = vel_tracker.release_velocity()
            vz += depth_calib.vel_z * 0.4
            grabbed_node.release(vx, vy, vz)
            grabbed_node = None

        prev_was_fist = right.is_fist

        # ── Spawn ──────────────────────────────────────────────────────
        if grabbed_node is None:
            spawn_sys.update(right, cx, cy, hand_z_world, now, nodes, W, H)

        # ── Delete ──────────────────────────────────────────────────────
        to_delete = delete_sys.update(right, cx, cy, nodes, grabbed_node)
        for dead in to_delete:
            if dead is grabbed_node:
                grabbed_node = None
            if dead is hovered_node:
                hovered_node = None
            graph.remove_node(dead)

        # ── Selection ─────────────────────────────────────────────────
        nodes = graph.nodes   # refresh after possible deletion
        select_sys.update(right, cx, cy, nodes, grabbed_node)

        # ── Update nodes ───────────────────────────────────────────────
        for n in nodes:
            n.update(dt, cx, cy, hand_z_world, right, left, W, H, floor_y,
                     gravity_on=gravity_on)

        # ── Check prompt thread ────────────────────────────────────────
        if waiting_prompt and prompt_thread.done:
            waiting_prompt = False
            topic = prompt_thread.result
            if topic:
                generating = True
                gen_status = ""
                print(f"[SceneGenerator] Generating scene for: '{topic}'")
                generator.generate_async(
                    prompt   = topic,
                    graph    = graph,
                    callback = on_generated,
                    screen_w = W,
                    screen_h = H,
                )
            else:
                print("[SceneGenerator] Cancelled.")

        # ── Clear stale status message ─────────────────────────────────
        if gen_status and (now - gen_status_t) > 3.0:
            gen_status = ""

        # ── Render ────────────────────────────────────────────────────
        out = frame.copy()

        draw_floor_enhanced(out, floor_y, W)
        draw_plane(out, left, W, H)
        draw_hand_skeleton(out, right, W, H, COLOR_RIGHT, "R")
        draw_hand_skeleton(out, left,  W, H, COLOR_LEFT,  "L")

        # Connections (behind nodes)
        nodes = graph.nodes
        conn_renderer.render(out, graph, dt)

        # Depth-sorted node render
        sorted_nodes = sorted(nodes, key=lambda n: n.z3d, reverse=True)
        render_list  = [n for n in sorted_nodes if n is not grabbed_node]
        if grabbed_node:
            render_list.append(grabbed_node)

        glow_layer = out.copy()
        glow_layer[:] = 0
        for n in render_list:
            n.draw_glow(glow_layer)
        cv2.addWeighted(glow_layer, 0.55, out, 1.0, 0, out)

        for n in render_list:
            n.draw_crisp(out)

        if grabbed_node and snap_on:
            snap_sys.draw_guides(out, grabbed_node, W, H)

        if right.visible:
            cursor.draw(out, cx, cy, right, hovered_node is not None,
                        depth_z=hand_z_world, dt=dt)

        if right.visible and depth_calib.is_calibrated:
            depth_hud.draw(out, cx, cy, hand_z_world,
                           active=(grabbed_node is not None or abs(hand_z_world) > 20))

        if right.visible and right.is_open and grabbed_node is None:
            spawn_sys.draw_progress(out, cx, cy)

        select_sys.draw_selection_feedback(out, nodes, W, H)

        if scale_delta is not None and left.is_fist and right.is_fist:
            mx, my = int(scale_mid[0]), int(scale_mid[1])
            cv2.putText(out, f"{scale_delta:.2f}×", (mx + 10, my),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, C_AMBER, 1, cv2.LINE_AA)

        teach.draw_hud(out, W, H)

        if gen_status:
            cv2.putText(out, gen_status, (W // 2 - 80, H - 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, C_PURPLE, 1, cv2.LINE_AA)

        num_sel = sum(1 for n in nodes if n.selected)
        draw_minimal_hud(out, tracker, grabbed_node, gravity_on, W, H,
                         len(nodes), num_sel, generating, teach.active)

        fps = 1.0 / dt if dt > 0 else 0.0
        cv2.putText(out, f"{fps:.0f}", (W - 28, H - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.28, (30, 30, 30), 1, cv2.LINE_AA)

        calib_overlay.draw(out, depth_calib.calibration_progress, W, H)

        if W < WINDOW_W or H < WINDOW_H:
            out = cv2.resize(out, (WINDOW_W, WINDOW_H),
                             interpolation=cv2.INTER_LINEAR)

        cv2.imshow("Aether", out)

        # ── Keys ───────────────────────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break

        elif key == ord('g'):
            gravity_on = not gravity_on
            print(f"Gravity: {'ON' if gravity_on else 'OFF'}")

        elif key == ord('s'):
            snap_on = not snap_on
            if grabbed_node:
                grabbed_node.snap_enabled = snap_on
            print(f"Snap: {'ON' if snap_on else 'OFF'}")

        elif key == ord('r'):
            if grabbed_node:
                grabbed_node.release(0, 0, 0)
                grabbed_node = None
            hovered_node = None
            teach.stop()
            graph    = make_default_graph(W, H)
            teach    = TeachingScene(graph)
            spawn_sys = SpawnSystem(factory=node_factory)
            vel_tracker.reset()
            depth_calib.reset()
            hand_z_world = 0.0
            generating   = False
            gen_status   = ""
            print("Reset.")

        elif key == ord('t'):
            # If teaching active, T stops it; otherwise prompt for AI scene
            if teach.active:
                teach.stop()
                print("Teaching mode stopped.")
            elif waiting_prompt or generating:
                print("Already generating…")
            else:
                if not api_key:
                    print("No ANTHROPIC_API_KEY set — cannot generate scenes.")
                else:
                    waiting_prompt = True
                    prompt_thread.start("─" * 40 + "\nAI Scene Generator\n" + "─" * 40)

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