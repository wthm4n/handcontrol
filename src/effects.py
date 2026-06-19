"""
effects.py — Phase 8: Visual feedback for the gesture construction system.

  AnimatedCursor      reticle that reflects point / grab state
  DepthPresenceHUD     floating depth rings
  ShapeSpawnSystem     hold a shape gesture -> ghost preview + countdown -> spawn
  DeleteSystem         finger-gun hold over an object -> countdown -> delete
  ModeHUD              shows which modifier is currently active (rotate/scale/axis)
  GroupFeedback        draws the bounding link between grouped / selected objects
  SnapSystem           grid snap guide drawing
  CalibrationOverlay   startup calibration progress display
  draw_floor_enhanced  floor with depth cues
  draw_minimal_hud     bottom/top corner status text
"""

import cv2
import math
import time

from hand import (
    SHAPE_CUBE, SHAPE_SPHERE, SHAPE_CYLINDER, SHAPE_PRISM, SHAPE_PLANE,
)


C_CYAN   = (180, 200,  50)
C_TEAL   = (220, 240,  20)
C_GREEN  = ( 50, 255, 120)
C_PURPLE = (255,  80, 180)
C_AMBER  = ( 30, 190, 255)
C_RED    = ( 40,  40, 240)
C_FLOOR  = ( 30,  80,  55)
C_GRID   = ( 20,  60,  40)
C_DIM    = ( 40,  40,  40)

SNAP_GRID      = 80
SNAP_THRESHOLD = 30

SHAPE_HOLD_TIME  = 0.5
SHAPE_COOLDOWN   = 0.6
DELETE_HOVER_RADIUS = 110


def _lerp(a, b, t):
    return a + (b - a) * t


class AnimatedCursor:
    def __init__(self):
        self._pulse_t = 0.0
        self._hover_t = 0.0

    def draw(self, img, cx, cy, hand_state, is_hovering, depth_z=0.0, dt=0.016):
        cx_i, cy_i = int(cx), int(cy)
        is_grab    = hand_state.is_grab
        is_pointing = hand_state.is_pointing

        self._pulse_t += dt * 3.5
        if is_hovering:
            self._hover_t = min(1.0, self._hover_t + dt * 5)
        else:
            self._hover_t = max(0.0, self._hover_t - dt * 4)

        depth_scale = max(0.5, min(1.5, 1.0 - depth_z / 800.0))
        base_r = int(18 * depth_scale)
        pulse_amp   = math.sin(self._pulse_t) * 0.5 + 0.5
        hover_extra = int(self._hover_t * 10 * pulse_amp)

        if is_grab:
            color  = C_GREEN
            ring_r = base_r - 6
            ov = img.copy()
            cv2.circle(ov, (cx_i, cy_i), ring_r + 12, color, -1, cv2.LINE_AA)
            cv2.addWeighted(ov, 0.12, img, 0.88, 0, img)
            cv2.circle(img, (cx_i, cy_i), ring_r, color, 2, cv2.LINE_AA)
            cv2.circle(img, (cx_i, cy_i), 5, color, -1, cv2.LINE_AA)

        elif is_pointing:
            color  = C_AMBER
            ring_r = base_r + hover_extra
            cv2.circle(img, (cx_i, cy_i), ring_r, color, 2, cv2.LINE_AA)
            cv2.line(img, (cx_i - ring_r - 6, cy_i), (cx_i - ring_r + 4, cy_i), color, 1, cv2.LINE_AA)
            cv2.line(img, (cx_i + ring_r - 4, cy_i), (cx_i + ring_r + 6, cy_i), color, 1, cv2.LINE_AA)
            cv2.circle(img, (cx_i, cy_i), 3, color, -1, cv2.LINE_AA)

        else:
            color  = C_TEAL
            ring_r = base_r + hover_extra
            if self._hover_t > 0.05:
                outer_r = ring_r + int(6 * pulse_amp)
                ov = img.copy()
                cv2.circle(ov, (cx_i, cy_i), outer_r, color, 1, cv2.LINE_AA)
                cv2.addWeighted(ov, self._hover_t * 0.4, img, 1 - self._hover_t * 0.4, 0, img)
            cv2.circle(img, (cx_i, cy_i), ring_r, color, 2, cv2.LINE_AA)
            cv2.circle(img, (cx_i, cy_i), 3, color, -1, cv2.LINE_AA)
            cv2.line(img, (cx_i - 8, cy_i), (cx_i + 8, cy_i), color, 1, cv2.LINE_AA)
            cv2.line(img, (cx_i, cy_i - 8), (cx_i, cy_i + 8), color, 1, cv2.LINE_AA)


class DepthPresenceHUD:
    def draw(self, img, cx, cy, depth_z, active=True):
        if not active: return
        cx_i, cy_i = int(cx), int(cy)
        from physics import Z_NEAR_LIMIT, Z_FAR_LIMIT
        t = (depth_z - Z_NEAR_LIMIT) / max(Z_FAR_LIMIT - Z_NEAR_LIMIT, 1.0)
        t = max(0.0, min(1.0, t))

        for r_scale, base_alpha in [(1.8, 0.35), (2.4, 0.20), (3.2, 0.10)]:
            ring_r = max(20, int(25 * r_scale * (1.0 - t * 0.4)))
            alpha  = base_alpha * (1.0 - t * 0.6)
            if alpha < 0.02: continue
            ov = img.copy()
            cv2.circle(ov, (cx_i, cy_i), ring_r, C_TEAL, 1, cv2.LINE_AA)
            cv2.addWeighted(ov, alpha, img, 1 - alpha, 0, img)

        if abs(depth_z) > 15:
            label = f"{'↗' if depth_z > 0 else '↙'} {abs(depth_z):.0f}"
            cv2.putText(img, label, (cx_i + 28, cy_i - 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.32, C_TEAL, 1, cv2.LINE_AA)


_GHOST_LABEL = {
    SHAPE_CUBE: "CUBE", SHAPE_SPHERE: "SPHERE", SHAPE_CYLINDER: "CYLINDER",
    SHAPE_PRISM: "PRISM", SHAPE_PLANE: "PLANE",
}


class ShapeSpawnSystem:
    """
    Holds one of the 5 shape gestures for SHAPE_HOLD_TIME -> spawns that
    shape at the cursor. Shows a ghost wireframe + countdown ring while
    holding, exactly the visual feedback the spec calls for.
    """

    def __init__(self, factory):
        """factory(shape_name, cx, cy, depth_z) -> new SpatialObject, added to scene."""
        self._factory     = factory
        self._held_shape  = None
        self._held_since  = None
        self._progress    = 0.0
        self._last_spawn  = 0.0

    @property
    def progress(self):
        return self._progress

    @property
    def pending_shape(self):
        return self._held_shape

    def update(self, hand_state, cx, cy, depth_z, t):
        since_spawn = t - self._last_spawn
        shape = hand_state.classify_shape_gesture() if hand_state.visible else None

        if shape is None or since_spawn < SHAPE_COOLDOWN:
            self._held_shape = None
            self._held_since = None
            self._progress   = 0.0
            return None

        if shape != self._held_shape:
            self._held_shape = shape
            self._held_since = t
            self._progress   = 0.0
            return None

        held = t - self._held_since
        self._progress = min(1.0, held / SHAPE_HOLD_TIME)

        if held >= SHAPE_HOLD_TIME:
            new_obj = self._factory(shape, cx, cy, depth_z)
            self._held_shape = None
            self._held_since = None
            self._progress   = 0.0
            self._last_spawn = t
            return new_obj

        return None

    def draw_preview(self, img, cx, cy):
        if self._held_shape is None or self._progress < 0.04:
            return
        cx_i, cy_i = int(cx), int(cy)
        ghost_r = int(_lerp(14, 34, self._progress))
        ov = img.copy()
        cv2.circle(ov, (cx_i, cy_i), ghost_r, C_TEAL, 1, cv2.LINE_AA)
        cv2.rectangle(ov, (cx_i - ghost_r, cy_i - ghost_r),
                      (cx_i + ghost_r, cy_i + ghost_r), C_TEAL, 1, cv2.LINE_AA)
        cv2.addWeighted(ov, 0.35, img, 0.65, 0, img)

        ring_r = 38
        angle  = int(360 * self._progress)
        ov2 = img.copy()
        cv2.ellipse(ov2, (cx_i, cy_i), (ring_r, ring_r), -90, 0, angle, C_TEAL, 3, cv2.LINE_AA)
        cv2.addWeighted(ov2, 0.85, img, 0.15, 0, img)

        label = _GHOST_LABEL.get(self._held_shape, "?")
        cv2.putText(img, label, (cx_i - 28, cy_i + ring_r + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, C_TEAL, 1, cv2.LINE_AA)


class DeleteSystem:
    def update(self, hand_state, cx, cy, objects, grabbed_obj):
        to_delete = []
        is_gun = hand_state.visible and hand_state.is_finger_gun

        for obj in objects:
            if obj is grabbed_obj:
                obj.cancel_delete_countdown()
                continue
            dist = obj.screen_dist(cx, cy)
            if is_gun and dist < DELETE_HOVER_RADIUS:
                obj.begin_delete_countdown()
            else:
                obj.cancel_delete_countdown()
            if obj.marked_for_delete:
                to_delete.append(obj)

        return to_delete

    def draw_crosshair(self, img, cx, cy, active):
        if not active: return
        cx_i, cy_i = int(cx), int(cy)
        cv2.line(img, (cx_i - 16, cy_i), (cx_i + 16, cy_i), C_RED, 1, cv2.LINE_AA)
        cv2.line(img, (cx_i, cy_i - 16), (cx_i, cy_i + 16), C_RED, 1, cv2.LINE_AA)
        cv2.circle(img, (cx_i, cy_i), 22, C_RED, 1, cv2.LINE_AA)


class ModeHUD:
    LABELS = {
        "rotate":  ("ROTATE",  C_AMBER),
        "scale":   ("SCALE",   C_PURPLE),
        "axis_x":  ("RESIZE X", (50, 50, 230)),
        "axis_y":  ("RESIZE Y", (60, 220, 60)),
        "axis_z":  ("RESIZE Z", (230, 130, 40)),
        "connect": ("CONNECTING", C_TEAL),
        "delete":  ("DELETE", C_RED),
        "group":   ("GROUPING", C_GROUP if (C_GROUP := (230, 90, 215)) else None),
        "explode": ("EXPLODING", C_RED),
    }

    def draw(self, img, mode, cx, cy):
        if mode is None or mode not in self.LABELS:
            return
        label, color = self.LABELS[mode]
        cv2.putText(img, label, (int(cx) - 40, int(cy) - 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)


class GroupFeedback:
    def draw(self, img, objects, groups, pull_progress=None):

        selected = [o for o in objects if o.selected]
        for i, a in enumerate(selected):
            for b in selected[i+1:]:
                cv2.line(img, (int(a.sx), int(a.sy)), (int(b.sx), int(b.sy)),
                          C_AMBER, 1, cv2.LINE_AA)
        if len(selected) >= 2:
            cx = sum(o.sx for o in selected) / len(selected)
            cy = sum(o.sy for o in selected) / len(selected)
            cv2.putText(img, f"{len(selected)} selected", (int(cx) - 40, int(cy) - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.32, C_AMBER, 1, cv2.LINE_AA)


        for group in groups:
            members = group.members
            for i, a in enumerate(members):
                for b in members[i+1:]:
                    cv2.line(img, (int(a.sx), int(a.sy)), (int(b.sx), int(b.sy)),
                              (230, 90, 215), 1, cv2.LINE_AA)

        if pull_progress is not None:
            H, W = img.shape[:2]
            label = "PULL APART TO EXPLODE" if pull_progress > 0 else "PUSH TOGETHER TO GROUP"
            cv2.putText(img, label, (W // 2 - 130, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, C_PURPLE, 1, cv2.LINE_AA)


class SnapSystem:
    def draw_guides(self, img, obj, W, H):
        if not obj.grabbed or not obj.snap_enabled: return
        nx = round(obj.x3d / SNAP_GRID) * SNAP_GRID
        ny = round(obj.y3d / SNAP_GRID) * SNAP_GRID
        dist_x = abs(obj.x3d - nx)
        dist_y = abs(obj.y3d - ny)
        alpha = 0.25

        if dist_x < SNAP_THRESHOLD * 2:
            snap_sx = int(obj.sx - (obj.x3d - nx))
            ov = img.copy()
            cv2.line(ov, (snap_sx, 0), (snap_sx, H), C_TEAL, 1, cv2.LINE_AA)
            a = alpha * max(0.0, 1.0 - dist_x / (SNAP_THRESHOLD * 2))
            cv2.addWeighted(ov, a, img, 1 - a, 0, img)

        if dist_y < SNAP_THRESHOLD * 2:
            snap_sy = int(obj.sy - (obj.y3d - ny))
            ov = img.copy()
            cv2.line(ov, (0, snap_sy), (W, snap_sy), C_TEAL, 1, cv2.LINE_AA)
            a = alpha * max(0.0, 1.0 - dist_y / (SNAP_THRESHOLD * 2))
            cv2.addWeighted(ov, a, img, 1 - a, 0, img)


class CalibrationOverlay:
    def draw(self, img, progress, W, H):
        if progress >= 1.0: return
        cx, cy = W // 2, H // 2
        ov = img.copy()
        cv2.rectangle(ov, (0, 0), (W, H), (0, 0, 0), -1)
        cv2.addWeighted(ov, 0.35, img, 0.65, 0, img)
        ring_r = 55
        angle  = int(360 * progress)
        cv2.ellipse(img, (cx, cy), (ring_r, ring_r), -90, 0, 360, C_DIM, 2, cv2.LINE_AA)
        cv2.ellipse(img, (cx, cy), (ring_r, ring_r), -90, 0, angle, C_TEAL, 3, cv2.LINE_AA)
        cv2.putText(img, "Hold hand steady", (cx - 68, cy + ring_r + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, C_TEAL, 1, cv2.LINE_AA)
        cv2.putText(img, "Calibrating depth...", (cx - 78, cy + ring_r + 42),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, C_DIM, 1, cv2.LINE_AA)


def draw_floor_enhanced(img, floor_y, W):
    strip_h = 12
    for i in range(strip_h):
        alpha = (strip_h - i) / strip_h * 0.5
        ov = img.copy()
        cv2.line(ov, (0, floor_y - i), (W, floor_y - i), C_GRID, 1)
        cv2.addWeighted(ov, alpha, img, 1 - alpha, 0, img)
    cv2.line(img, (0, floor_y), (W, floor_y), C_FLOOR, 2, cv2.LINE_AA)
    cx = W // 2
    for step in range(1, 6):
        x_offset = step * (W // 6)
        for sign in (-1, 1):
            px = cx + sign * x_offset
            if 0 <= px <= W:
                ov = img.copy()
                cv2.line(ov, (px, floor_y - 4), (px, floor_y + 2), C_GRID, 1)
                cv2.addWeighted(ov, 0.4, img, 0.6, 0, img)


def draw_minimal_hud(img, tracker, grabbed_obj, gravity_on, W, H, num_objs, num_groups):
    r = tracker.right
    if r.visible:
        if grabbed_obj:        state, col = "HOLDING", C_GREEN
        elif r.is_grab:         state, col = "GRAB",    C_GREEN
        elif r.is_pointing:    state, col = "POINT",   C_AMBER
        elif r.classify_shape_gesture(): state, col = "SPAWN", C_TEAL
        else:                   state, col = "TRACK",   C_DIM
    else:
        state, col = "WAITING", C_DIM

    cv2.putText(img, state, (16, H - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.40, col, 1, cv2.LINE_AA)
    grav_col = C_GREEN if gravity_on else C_DIM
    cv2.putText(img, "G" if gravity_on else "g", (W - 28, H - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.40, grav_col, 1, cv2.LINE_AA)

    if num_objs > 0:
        count_str = f"{num_objs} obj"
        if num_groups > 0:
            count_str += f"  {num_groups} grp"
        cv2.putText(img, count_str, (W - 100, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, C_DIM, 1, cv2.LINE_AA)

    legend = [
        "Right: open=cube ok=sphere peace=cylinder 3fing=prism 4fing=plane",
        "Right: pinch3=grab  point=target/connect  fingergun=delete",
        "Left while grabbing: fist=rotate  pinch=scale  point=axis resize",
        "Both grab + pull/push: explode / group selected",
        "G=gravity  R=reset  Q=quit",
    ]
    for i, line in enumerate(legend):
        cv2.putText(img, line, (16, 18 + i * 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.26, (35, 35, 35), 1, cv2.LINE_AA)