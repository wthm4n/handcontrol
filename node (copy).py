"""
node.py — Phase 6: SpatialNode, the knowledge-card object that replaces
the cube as Aether's primary spatial object.

A SpatialNode represents an *idea*, not a piece of geometry: a title, a
short body of explanatory text, an icon, a category, and free-form
metadata. Many nodes are linked into a KnowledgeGraph (see graph.py) to
represent a whole topic spatially — "Recursion", "Binary Tree", "CPU
Cache", "French Revolution", "Neural Network" all live here as the same
underlying object, only their content differs.

Render / interaction API intentionally mirrors cube.Cube so a SpatialNode
drops into the existing gesture pipeline (SpawnSystem, DeleteSystem,
SelectionSystem, SnapSystem, the main.py render loop) with no changes
required on the caller's side:

    grab(cx, cy, hand_z_world)
    release(vx, vy, vz)
    update(dt, cx, cy, hand_z_world, right_hand, left_hand,
           screen_w, screen_h, floor_y, gravity_on=True)
    draw_glow(glow_layer)
    draw_crisp(img)
    screen_dist(cx, cy)
    begin_delete_countdown() / cancel_delete_countdown()
    .sx .sy .x3d .y3d .z3d .size .hovered .grabbed .selected
    .marked_for_delete .delete_progress

Cards are billboards: they always face the camera so their text stays
readable, but they still live at a real (x3d, y3d, z3d) in the same
world the cubes used, so depth scaling, fog, shadows and glow all behave
consistently with the rest of Aether.
"""

import cv2
import math
import time
import itertools

from physics import PhysicsBody

# ── Constants ─────────────────────────────────────────────────────────

FOV = 900.0  # matches cube.py's perspective strength

CARD_W = 230
CARD_H = 138
CORNER_R = 16

Z_SMOOTH_GRAB = 0.12
Z_SMOOTH_FREE = 0.25

SPAWN_DURATION = 0.35
DELETE_HOLD_TIME = 1.0

FOG_NEAR = 0.0
FOG_FAR = 700.0
FOG_MAX = 0.55

TRAIL_MAX_PTS = 8
TRAIL_MIN_SPEED = 250

SNAP_GRID = 80
SNAP_THRESHOLD = 30

# ── Phase 6 colour palette (BGR) — visual states from the spec ────────
COLOR_IDLE     = (180, 200,  50)   # cyan
COLOR_HOVER    = (220, 240,  20)   # teal
COLOR_GRAB     = ( 50, 255, 120)   # green
COLOR_SELECTED = ( 30, 190, 255)   # amber
COLOR_AI       = (230,  90, 215)   # purple — AI-generated
COLOR_DELETE   = ( 40,  40, 240)   # red

# Category → default icon glyph (drawn as small vector art, not a font
# glyph, so it never depends on what's installed on the host machine).
CATEGORY_ICONS = {
    "concept":   "bulb",
    "structure": "tree",
    "code":      "brackets",
    "math":      "function",
    "history":   "scroll",
    "system":    "cpu",
    "process":   "flow",
    "default":   "dot",
}

_id_counter = itertools.count(1)


def _next_id():
    return f"node_{next(_id_counter)}"


def _lerp(a, b, t):
    return a + (b - a) * t


def _wrap_text(text, max_chars):
    """Greedy word-wrap to a fixed character budget per line."""
    if not text:
        return []
    words = text.split()
    lines, cur = [], ""
    for w in words:
        candidate = (cur + " " + w).strip()
        if len(candidate) > max_chars and cur:
            lines.append(cur)
            cur = w
        else:
            cur = candidate
    if cur:
        lines.append(cur)
    return lines


def _rounded_rect(img, x1, y1, x2, y2, r, color, thickness=-1):
    """Draw a filled or stroked rounded rectangle with plain cv2 calls."""
    r = max(0, min(r, int(min(x2 - x1, y2 - y1) / 2)))
    if thickness < 0:
        cv2.rectangle(img, (x1 + r, y1), (x2 - r, y2), color, -1, cv2.LINE_AA)
        cv2.rectangle(img, (x1, y1 + r), (x2, y2 - r), color, -1, cv2.LINE_AA)
        for cx, cy in [(x1 + r, y1 + r), (x2 - r, y1 + r),
                        (x1 + r, y2 - r), (x2 - r, y2 - r)]:
            cv2.circle(img, (cx, cy), r, color, -1, cv2.LINE_AA)
    else:
        cv2.line(img, (x1 + r, y1), (x2 - r, y1), color, thickness, cv2.LINE_AA)
        cv2.line(img, (x1 + r, y2), (x2 - r, y2), color, thickness, cv2.LINE_AA)
        cv2.line(img, (x1, y1 + r), (x1, y2 - r), color, thickness, cv2.LINE_AA)
        cv2.line(img, (x2, y1 + r), (x2, y2 - r), color, thickness, cv2.LINE_AA)
        cv2.ellipse(img, (x1 + r, y1 + r), (r, r), 180, 0, 90, color, thickness, cv2.LINE_AA)
        cv2.ellipse(img, (x2 - r, y1 + r), (r, r), 270, 0, 90, color, thickness, cv2.LINE_AA)
        cv2.ellipse(img, (x1 + r, y2 - r), (r, r),  90, 0, 90, color, thickness, cv2.LINE_AA)
        cv2.ellipse(img, (x2 - r, y2 - r), (r, r),   0, 0, 90, color, thickness, cv2.LINE_AA)


def _draw_icon(img, kind, cx, cy, r, color, thickness=2):
    """Small vector icon, drawn entirely with cv2 primitives (no fonts,
    no image assets — keeps the render pipeline dependency-free)."""
    cx, cy, r = int(cx), int(cy), max(4, int(r))

    if kind == "bulb":
        cv2.circle(img, (cx, cy - r // 4), int(r * 0.8), color, thickness, cv2.LINE_AA)
        cv2.line(img, (cx - r // 3, cy + r // 2), (cx + r // 3, cy + r // 2), color, thickness, cv2.LINE_AA)
        cv2.line(img, (cx - r // 4, cy + int(r * 0.8)), (cx + r // 4, cy + int(r * 0.8)), color, thickness, cv2.LINE_AA)

    elif kind == "tree":
        cv2.circle(img, (cx, cy - r), max(2, r // 3), color, thickness, cv2.LINE_AA)
        cv2.line(img, (cx, cy - int(r * 0.7)), (cx, cy + r // 4), color, thickness, cv2.LINE_AA)
        cv2.line(img, (cx, cy + r // 4), (cx - r, cy + r), color, thickness, cv2.LINE_AA)
        cv2.line(img, (cx, cy + r // 4), (cx + r, cy + r), color, thickness, cv2.LINE_AA)
        cv2.circle(img, (cx - r, cy + r), max(2, r // 4), color, thickness, cv2.LINE_AA)
        cv2.circle(img, (cx + r, cy + r), max(2, r // 4), color, thickness, cv2.LINE_AA)

    elif kind == "brackets":
        cv2.ellipse(img, (cx - r // 2, cy), (r // 3, r), 0, 110, 250, color, thickness, cv2.LINE_AA)
        cv2.ellipse(img, (cx + r // 2, cy), (r // 3, r), 0, -70, 70, color, thickness, cv2.LINE_AA)

    elif kind == "function":
        cv2.ellipse(img, (cx, cy - r // 2), (r // 2, r // 2), 0, 200, 360, color, thickness, cv2.LINE_AA)
        cv2.line(img, (cx - r // 2, cy), (cx + r // 3, cy), color, thickness, cv2.LINE_AA)
        cv2.line(img, (cx - r // 4, cy), (cx - r // 2, cy + r), color, thickness, cv2.LINE_AA)

    elif kind == "scroll":
        cv2.rectangle(img, (cx - r, cy - r // 2), (cx + r, cy + r // 2), color, thickness, cv2.LINE_AA)
        cv2.line(img, (cx - r, cy - r // 6), (cx + r, cy - r // 6), color, 1, cv2.LINE_AA)
        cv2.line(img, (cx - r, cy + r // 6), (cx + r, cy + r // 6), color, 1, cv2.LINE_AA)

    elif kind == "cpu":
        cv2.rectangle(img, (cx - r // 2, cy - r // 2), (cx + r // 2, cy + r // 2), color, thickness, cv2.LINE_AA)
        for i in (-1, 0, 1):
            off = int(i * r * 0.45)
            cv2.line(img, (cx + off, cy - r // 2), (cx + off, cy - r), color, 1, cv2.LINE_AA)
            cv2.line(img, (cx + off, cy + r // 2), (cx + off, cy + r), color, 1, cv2.LINE_AA)
            cv2.line(img, (cx - r // 2, cy + off), (cx - r, cy + off), color, 1, cv2.LINE_AA)
            cv2.line(img, (cx + r // 2, cy + off), (cx + r, cy + off), color, 1, cv2.LINE_AA)

    elif kind == "flow":
        cv2.circle(img, (cx, cy - r), max(2, r // 4), color, thickness, cv2.LINE_AA)
        cv2.line(img, (cx, cy - int(r * 0.7)), (cx, cy + int(r * 0.5)), color, thickness, cv2.LINE_AA)
        cv2.circle(img, (cx, cy + r), max(2, r // 4), color, thickness, cv2.LINE_AA)
        cv2.line(img, (cx - r // 4, cy + r // 6), (cx, cy + r // 2), color, thickness, cv2.LINE_AA)
        cv2.line(img, (cx + r // 4, cy + r // 6), (cx, cy + r // 2), color, thickness, cv2.LINE_AA)

    elif kind == "nodes":
        pts = [(cx, cy - r), (cx - r, cy + r // 2), (cx + r, cy + r // 2)]
        for a, b in [(0, 1), (1, 2), (2, 0)]:
            cv2.line(img, pts[a], pts[b], color, 1, cv2.LINE_AA)
        for p in pts:
            cv2.circle(img, p, max(2, r // 4), color, -1, cv2.LINE_AA)

    else:  # "dot" fallback
        cv2.circle(img, (cx, cy), r // 2, color, thickness, cv2.LINE_AA)


def _fog_alpha(z3d):
    t = (z3d - FOG_NEAR) / max(FOG_FAR - FOG_NEAR, 1.0)
    return max(0.0, min(1.0, t)) * FOG_MAX


def _depth_glow_scale(z3d):
    t = (z3d - FOG_NEAR) / max(FOG_FAR - FOG_NEAR, 1.0)
    t = max(0.0, min(1.0, t))
    return 1.0 - t * 0.7


class SpatialNode:
    """A single knowledge card living in Aether's 3D workspace."""

    def __init__(self, title, body="", icon=None, category="concept",
                 x3d=0.0, y3d=0.0, z3d=0.0,
                 ai_generated=False, metadata=None, node_id=None):
        self.id = node_id or _next_id()

        self.title = title
        self.body = body
        self.category = category
        self.icon = icon or CATEGORY_ICONS.get(category, CATEGORY_ICONS["default"])
        self.metadata = metadata if metadata is not None else {}
        self.ai_generated = ai_generated

        # World position (same coordinate convention as cube.py / physics.py)
        self.x3d = float(x3d)
        self.y3d = float(y3d)
        self.z3d = float(z3d)
        self.size = 1.0          # uniform scale multiplier on CARD_W/H

        self.sx = float(x3d)     # last projected screen position
        self.sy = float(y3d)

        self.rz = 0.0            # roll — cards can spin like a playing card
        self._yaw_wobble = 0.0   # cosmetic foreshortening, not real rotation

        # Interaction state
        self.hovered = False
        self.grabbed = False
        self.selected = False

        self.glow = 0.0
        self.scale_v = 1.0

        self.grab_off_x = 0.0
        self.grab_off_y = 0.0
        self.grab_off_z = 0.0

        # Physics is opt-in for nodes — knowledge cards float gently by
        # default rather than falling, but can be "thrown" like a cube
        # if a teaching scene or the user wants that behaviour.
        self.physics_enabled = False
        self.physics = PhysicsBody()

        # Spawn animation
        self._spawn_t = time.time()
        self._spawning = True
        self._spawn_glow = 2.0

        # Delete countdown
        self._delete_hold_start = None
        self.delete_progress = 0.0
        self.marked_for_delete = False

        # Motion trail (only meaningful if physics_enabled)
        self._trail = []

        self.snap_enabled = False

        # Relationship cache — the KnowledgeGraph is the source of truth,
        # this just lets a node answer "what am I connected to" cheaply.
        self.connections = []

        # Teaching-mode focus state (set by TeachingScene)
        self.focused = False
        self.dimmed = False

    # ── Projection ────────────────────────────────────────────────────

    def _project_point(self, x3d, y3d, z3d, screen_cx, screen_cy):
        scale = FOV / max(FOV + z3d, 1.0)
        sx = screen_cx + (x3d - screen_cx) * scale
        sy = screen_cy + (y3d - screen_cy) * scale
        return sx, sy, scale

    # ── Interaction ───────────────────────────────────────────────────

    def grab(self, cx, cy, hand_z_world):
        self.grabbed = True
        self.grab_off_x = self.x3d - cx
        self.grab_off_y = self.y3d - cy
        self.grab_off_z = self.z3d - hand_z_world
        self.scale_v = 1.08
        self.physics.stop()
        self._delete_hold_start = None
        self.delete_progress = 0.0

    def release(self, vx=0.0, vy=0.0, vz=0.0):
        self.grabbed = False
        self.scale_v = 0.94
        if self.physics_enabled:
            self.physics.launch(vx, vy, vz)
        self._delete_hold_start = None
        self.delete_progress = 0.0

    def begin_delete_countdown(self):
        if self._delete_hold_start is None:
            self._delete_hold_start = time.time()

    def cancel_delete_countdown(self):
        self._delete_hold_start = None
        self.delete_progress = 0.0

    def _apply_snap(self):
        for attr in ('x3d', 'y3d'):
            val = getattr(self, attr)
            nearest = round(val / SNAP_GRID) * SNAP_GRID
            if abs(val - nearest) < SNAP_THRESHOLD:
                setattr(self, attr, _lerp(val, nearest, 0.25))

    # ── Update ────────────────────────────────────────────────────────

    def update(self, dt, cx, cy, hand_z_world,
               right_hand, left_hand,
               screen_w, screen_h, floor_y,
               gravity_on=True):
        screen_cx = screen_w / 2.0
        screen_cy = screen_h / 2.0
        now = time.time()

        # Spawn animation — ease-out scale-in, identical feel to Cube
        if self._spawning:
            elapsed = now - self._spawn_t
            frac = min(1.0, elapsed / SPAWN_DURATION)
            t = 1.0 - (1.0 - frac) ** 3
            self.scale_v = t
            self._spawn_glow = max(0.0, 2.0 * (1.0 - frac))
            if frac >= 1.0:
                self._spawning = False
                self.scale_v = 1.0

        # Delete countdown
        if self._delete_hold_start is not None:
            elapsed = now - self._delete_hold_start
            self.delete_progress = min(1.0, elapsed / DELETE_HOLD_TIME)
            if self.delete_progress >= 1.0:
                self.marked_for_delete = True
        else:
            self.delete_progress = max(0.0, self.delete_progress - dt * 3)

        if self.grabbed:
            self.x3d = cx + self.grab_off_x
            self.y3d = cy + self.grab_off_y
            target_z = hand_z_world + self.grab_off_z
            self.z3d = _lerp(self.z3d, target_z, Z_SMOOTH_GRAB)

            # Twist rolls the card; pitch/yaw only add a cosmetic
            # foreshortening wobble so the title text never goes upside
            # down or unreadable.
            self.rz += right_hand.delta_oz * 3.0
            self._yaw_wobble += right_hand.delta_oy * 1.2
            self._yaw_wobble = max(-0.9, min(0.9, self._yaw_wobble))

            if self.snap_enabled:
                self._apply_snap()

        elif self.physics_enabled:
            self.x3d, self.y3d, self.z3d, drx, dry, drz = self.physics.step(
                dt, self.x3d, self.y3d, self.z3d,
                floor_y, screen_w, screen_h,
                gravity_on=gravity_on,
            )
            self.rz += drz
        else:
            # Free-floating card: gently relax wobble/roll back to neutral
            self.rz = _lerp(self.rz, 0.0, dt * 1.5)
            self._yaw_wobble = _lerp(self._yaw_wobble, 0.0, dt * 1.5)

        self.sx, self.sy, _ = self._project_point(
            self.x3d, self.y3d, self.z3d, screen_cx, screen_cy
        )

        # Motion trail (cosmetic, only shows up if a node is thrown)
        speed = math.sqrt(self.physics.vx ** 2 + self.physics.vy ** 2)
        if self.physics_enabled and not self.grabbed and speed > TRAIL_MIN_SPEED:
            self._trail.append((self.sx, self.sy, 1.0))
            if len(self._trail) > TRAIL_MAX_PTS:
                self._trail.pop(0)
        else:
            self._trail = [(x, y, max(0.0, a - dt * 4)) for x, y, a in self._trail if a > 0.05]

        # Glow / scale animation toward target state
        if not self._spawning:
            glow_target = (1.0 if self.grabbed else
                           0.85 if self.focused else
                           0.85 if self.selected else
                           0.65 if self.hovered else
                           0.18 if self.ai_generated else
                           0.08)
            if self.dimmed and not (self.grabbed or self.hovered or self.selected or self.focused):
                glow_target *= 0.25
            self.glow = _lerp(self.glow, glow_target, 0.15)
            self.scale_v = _lerp(self.scale_v, 1.0, 0.18)

    # ── Colour ────────────────────────────────────────────────────────

    def _pick_color(self):
        if self.delete_progress > 0.05:
            base = COLOR_AI if self.ai_generated else COLOR_IDLE
            t = self.delete_progress
            return tuple(int(COLOR_DELETE[i] * t + base[i] * (1 - t)) for i in range(3))
        if self.grabbed:
            return COLOR_GRAB
        if self.focused or self.selected:
            return COLOR_SELECTED
        if self.hovered:
            return COLOR_HOVER
        if self.ai_generated:
            return COLOR_AI
        return COLOR_IDLE

    # ── Geometry: rotated rounded-rect corners in screen space ────────

    def _card_quad(self, screen_cx, screen_cy):
        scale = FOV / max(FOV + self.z3d, 1.0)
        yaw_fold = max(0.4, abs(math.cos(self._yaw_wobble)))
        w = CARD_W * self.size * self.scale_v * scale * yaw_fold * 0.5
        h = CARD_H * self.size * self.scale_v * scale * 0.5

        c, s = math.cos(self.rz), math.sin(self.rz)
        local = [(-w, -h), (w, -h), (w, h), (-w, h)]
        pts = []
        for lx, ly in local:
            rx = lx * c - ly * s
            ry = lx * s + ly * c
            pts.append((int(self.sx + rx), int(self.sy + ry)))
        return pts, scale

    # ── Rendering (two-pass, matches cube.py's pipeline) ──────────────

    def draw_glow(self, glow_layer):
        H, W = glow_layer.shape[:2]
        screen_cx, screen_cy = W / 2.0, H / 2.0

        color = self._pick_color()
        quad, scale = self._card_quad(screen_cx, screen_cy)
        eff_glow = (self.glow + self._spawn_glow) * _depth_glow_scale(self.z3d)
        if eff_glow < 0.02:
            return

        w1 = max(2, int(16 * eff_glow))
        w2 = max(1, int(9 * eff_glow))
        w3 = max(1, int(4 * eff_glow))

        pts = [quad, quad]  # closed polygon needs first point repeated at end
        poly = quad + [quad[0]]
        for i in range(len(poly) - 1):
            for width in (w1, w2, w3):
                cv2.line(glow_layer, poly[i], poly[i + 1], color, width, cv2.LINE_AA)

        if self.hovered or self.grabbed or self.selected or self.focused:
            margin = max(6, int(10 * scale))
            hx1 = min(p[0] for p in quad) - margin
            hy1 = min(p[1] for p in quad) - margin
            hx2 = max(p[0] for p in quad) + margin
            hy2 = max(p[1] for p in quad) + margin
            ring_w = max(1, int(5 * eff_glow))
            _rounded_rect(glow_layer, hx1, hy1, hx2, hy2,
                          int((CORNER_R + margin) * scale), color, ring_w)

    def draw_crisp(self, img):
        H, W = img.shape[:2]
        screen_cx, screen_cy = W / 2.0, H / 2.0

        color = self._pick_color()
        quad, scale = self._card_quad(screen_cx, screen_cy)
        fog = _fog_alpha(self.z3d)

        self._draw_shadow_direct(img, screen_cx, screen_cy, H, scale)
        self._draw_trail_direct(img, color)

        x1 = min(p[0] for p in quad); x2 = max(p[0] for p in quad)
        y1 = min(p[1] for p in quad); y2 = max(p[1] for p in quad)

        axis_aligned = abs(self.rz) < 0.08
        if axis_aligned:
            # Fast, crisp path — also the only path where text stays legible
            card_color = tuple(int(c * 0.16) for c in color)
            _rounded_rect(img, x1, y1, x2, y2, int(CORNER_R * scale), card_color, -1)
            _rounded_rect(img, x1, y1, x2, y2, int(CORNER_R * scale), color, 2)
            self._draw_content(img, x1, y1, x2, y2, color, scale)
        else:
            # Spinning card — draw the rotated quad outline only; text
            # is suppressed mid-spin so nothing renders illegibly.
            poly = quad + [quad[0]]
            overlay_color = tuple(int(c * 0.16) for c in color)
            pts_arr = _np_array(quad)
            cv2.fillConvexPoly(img, pts_arr, overlay_color, cv2.LINE_AA)
            for i in range(len(poly) - 1):
                cv2.line(img, poly[i], poly[i + 1], color, 2, cv2.LINE_AA)
            ix, iy = int(self.sx), int(self.sy)
            _draw_icon(img, self.icon, ix, iy, int(22 * scale), color, 2)

        if self.delete_progress > 0.02:
            self._draw_delete_ring_direct(img, color, scale)

        if fog > 0.02:
            x1c, y1c = max(0, x1), max(0, y1)
            x2c, y2c = min(W, x2), min(H, y2)
            roi = img[y1c:y2c, x1c:x2c]
            if roi.size > 0:
                import numpy as np
                np.multiply(roi, (1.0 - fog), out=roi, casting='unsafe')

        # Teaching-mode dimming: visibly darken cards that aren't the
        # current focus, rather than only reducing their glow.
        if self.dimmed and not (self.hovered or self.grabbed or self.selected or self.focused):
            x1c, y1c = max(0, x1), max(0, y1)
            x2c, y2c = min(W, x2), min(H, y2)
            roi = img[y1c:y2c, x1c:x2c]
            if roi.size > 0:
                import numpy as np
                np.multiply(roi, 0.45, out=roi, casting='unsafe')

    # ── Card content (title / icon / body / footer) ───────────────────

    def _draw_content(self, img, x1, y1, x2, y2, color, scale):
        pad = max(6, int(12 * scale))
        icon_r = max(8, int(16 * scale))
        icon_cx = x1 + pad + icon_r
        icon_cy = y1 + pad + icon_r
        _draw_icon(img, self.icon, icon_cx, icon_cy, icon_r, color, max(1, int(2 * scale)))

        title_x = icon_cx + icon_r + pad // 2
        title_y = y1 + pad + icon_r // 2 + 4
        font_scale = max(0.32, 0.50 * scale)
        cv2.putText(img, self.title, (title_x, title_y),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 1, cv2.LINE_AA)

        # AI-generated badge — small purple dot in the corner, independent
        # of whatever interaction colour the card is currently showing.
        if self.ai_generated:
            cv2.circle(img, (x2 - pad, y1 + pad), max(2, int(4 * scale)), COLOR_AI, -1, cv2.LINE_AA)

        body_y0 = y1 + pad + icon_r * 2 + 6
        max_chars = max(14, int((x2 - x1 - pad * 2) / max(4.0, 7.5 * scale)))
        line_h = max(10, int(15 * scale))
        text_color = tuple(int(c * 0.78 + 60) for c in color)
        for i, line in enumerate(_wrap_text(self.body, max_chars)):
            ly = body_y0 + i * line_h
            if ly > y2 - pad:
                break
            cv2.putText(img, line, (x1 + pad, ly),
                        cv2.FONT_HERSHEY_SIMPLEX, max(0.28, 0.36 * scale),
                        text_color, 1, cv2.LINE_AA)

        # Category footer tag
        footer_y = y2 - max(4, int(8 * scale))
        cv2.putText(img, self.category.upper(), (x1 + pad, footer_y),
                    cv2.FONT_HERSHEY_SIMPLEX, max(0.22, 0.26 * scale),
                    tuple(int(c * 0.5) for c in color), 1, cv2.LINE_AA)

    # ── Shadow / trail / delete ring ────────────────────────────────────

    def _draw_shadow_direct(self, img, screen_cx, screen_cy, frame_h, scale):
        floor_y = int(frame_h * 0.94)
        dist_to_floor = floor_y - self.y3d
        if dist_to_floor < 0 or dist_to_floor > frame_h * 0.9:
            return
        shadow_sx = int(screen_cx + (self.x3d - screen_cx) * scale)
        shadow_sy = floor_y - 2
        height_frac = max(0.0, min(1.0, dist_to_floor / (frame_h * 0.5)))
        rx = max(10, int(CARD_W * 0.5 * scale * (1.0 - height_frac * 0.6)))
        ry = max(3, int(rx * 0.22))
        darkness = int(28 * (1.0 - height_frac))
        if darkness < 2:
            return
        cv2.ellipse(img, (shadow_sx, shadow_sy), (rx, ry), 0, 0, 360,
                    (darkness, darkness, darkness), -1, cv2.LINE_AA)

    def _draw_trail_direct(self, img, color):
        for tx, ty, alpha in self._trail:
            c = tuple(int(v * alpha * 0.5) for v in color)
            r = max(1, int(3 * alpha))
            cv2.circle(img, (int(tx), int(ty)), r, c, -1, cv2.LINE_AA)

    def _draw_delete_ring_direct(self, img, color, scale):
        ring_r = max(18, int(max(CARD_W, CARD_H) * 0.7 * self.scale_v * scale))
        cx_i, cy_i = int(self.sx), int(self.sy)
        angle = int(360 * self.delete_progress)
        cv2.ellipse(img, (cx_i, cy_i), (ring_r, ring_r), -90, 0, 360, (60, 40, 40), 2, cv2.LINE_AA)
        cv2.ellipse(img, (cx_i, cy_i), (ring_r, ring_r), -90, 0, angle, COLOR_DELETE, 3, cv2.LINE_AA)

    # ── Misc ─────────────────────────────────────────────────────────

    def screen_dist(self, cx, cy):
        return math.sqrt((cx - self.sx) ** 2 + (cy - self.sy) ** 2)

    def debug_str(self):
        return (f"[{self.id}] {self.title!r} cat={self.category} "
                f"ai={self.ai_generated} conn={len(self.connections)}")


def _np_array(pts):
    import numpy as np
    return np.array(pts, dtype=int)