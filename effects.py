"""
effects.py — Phase 5: Reusable visual effects and UI systems for Aether.

Provides:
  • AnimatedCursor       — reticle with depth-aware size, hover pulse, grab state
  • DepthPresenceHUD     — floating depth rings + subtle presence indicators
  • SpawnSystem          — gesture detection (open palm hold) + cube factory
  • DeleteSystem         — fist-hold-over-cube countdown management
  • SelectionSystem      — pinch-to-select + group move/scale feedback
  • SnapSystem           — grid snap guide drawing
  • CalibrationOverlay   — startup calibration progress display
  • draw_floor_enhanced  — improved floor with depth cues
"""

import cv2
import math
import time


# ── Palette (BGR) ─────────────────────────────────────────────────────
C_CYAN    = (180, 200,  50)
C_TEAL    = (220, 240,  20)
C_GREEN   = ( 50, 255, 120)
C_PURPLE  = (255,  80, 180)
C_AMBER   = ( 30, 190, 255)
C_RED     = ( 40,  40, 240)
C_FLOOR   = ( 30,  80,  55)
C_GRID    = ( 20,  60,  40)
C_DIM     = ( 40,  40,  40)


def _lerp(a, b, t):
    return a + (b - a) * t


# ── Animated Cursor ───────────────────────────────────────────────────

class AnimatedCursor:
    """
    Animated reticle cursor.
    • Depth-aware size (bigger when hand is near, smaller when far).
    • Pulsing ring when hovering over an object.
    • Fills in (circle → dot) when grabbing.
    """

    def __init__(self):
        self._pulse_t  = 0.0
        self._hover_t  = 0.0

    def draw(self, img, cx, cy, hand_state, is_hovering, depth_z=0.0, dt=0.016):
        cx_i, cy_i = int(cx), int(cy)
        is_fist = hand_state.is_fist
        is_open = hand_state.is_open

        # Animate
        self._pulse_t += dt * 3.5
        if is_hovering:
            self._hover_t = min(1.0, self._hover_t + dt * 5)
        else:
            self._hover_t = max(0.0, self._hover_t - dt * 4)

        # Depth-aware size: near (z < 0) = larger, far (z > 400) = smaller
        depth_scale = max(0.5, min(1.5, 1.0 - depth_z / 800.0))
        base_r = int(18 * depth_scale)

        pulse_amp   = math.sin(self._pulse_t) * 0.5 + 0.5
        hover_extra = int(self._hover_t * 10 * pulse_amp)

        if is_fist:
            color  = C_GREEN
            ring_r = base_r - 6
            # Filled grab circle
            ov = img.copy()
            cv2.circle(ov, (cx_i, cy_i), ring_r + 12, color, -1, cv2.LINE_AA)
            cv2.addWeighted(ov, 0.12, img, 0.88, 0, img)
            cv2.circle(img, (cx_i, cy_i), ring_r, color, 2, cv2.LINE_AA)
            cv2.circle(img, (cx_i, cy_i), 5, color, -1, cv2.LINE_AA)

        elif is_open:
            # Spawn mode — pulsing open ring
            color  = C_TEAL
            ring_r = base_r + hover_extra
            ov = img.copy()
            cv2.circle(ov, (cx_i, cy_i), ring_r + 8, color, -1, cv2.LINE_AA)
            cv2.addWeighted(ov, 0.08 + 0.05 * pulse_amp, img, 1 - 0.13, 0, img)
            for gap in range(0, 360, 90):
                cv2.ellipse(img, (cx_i, cy_i), (ring_r, ring_r),
                            0, gap + 10, gap + 80, color, 2, cv2.LINE_AA)
            cv2.circle(img, (cx_i, cy_i), 3, color, -1, cv2.LINE_AA)

        else:
            # Normal pointing cursor
            color  = C_TEAL
            ring_r = base_r + hover_extra

            if self._hover_t > 0.05:
                # Outer pulse ring when hovering
                outer_r = ring_r + int(6 * pulse_amp)
                ov = img.copy()
                cv2.circle(ov, (cx_i, cy_i), outer_r, color, 1, cv2.LINE_AA)
                cv2.addWeighted(ov, self._hover_t * 0.4, img,
                                1 - self._hover_t * 0.4, 0, img)

            # Inner glow
            ov = img.copy()
            cv2.circle(ov, (cx_i, cy_i), ring_r + 6, color, -1, cv2.LINE_AA)
            cv2.addWeighted(ov, 0.10, img, 0.90, 0, img)

            cv2.circle(img, (cx_i, cy_i), ring_r, color, 2, cv2.LINE_AA)
            cv2.circle(img, (cx_i, cy_i), 3, color, -1, cv2.LINE_AA)
            # Crosshair lines
            cv2.line(img, (cx_i - 8, cy_i), (cx_i + 8, cy_i), color, 1, cv2.LINE_AA)
            cv2.line(img, (cx_i, cy_i - 8), (cx_i, cy_i + 8), color, 1, cv2.LINE_AA)


# ── Depth Presence HUD ────────────────────────────────────────────────

class DepthPresenceHUD:
    """
    Floating depth indicator — replaces the developer bar.
    Shows depth rings around the cursor and a subtle floating label.
    Far objects appear dimmer; near objects have crisp concentric rings.
    """

    def draw(self, img, cx, cy, depth_z, active=True):
        if not active:
            return
        cx_i, cy_i = int(cx), int(cy)

        # Normalise depth to 0..1
        from physics import Z_NEAR_LIMIT, Z_FAR_LIMIT
        t = (depth_z - Z_NEAR_LIMIT) / max(Z_FAR_LIMIT - Z_NEAR_LIMIT, 1.0)
        t = max(0.0, min(1.0, t))

        # Depth rings — 3 concentric circles fading with depth
        for i, (r_scale, base_alpha) in enumerate([(1.8, 0.35), (2.4, 0.20), (3.2, 0.10)]):
            ring_r = max(20, int(25 * r_scale * (1.0 - t * 0.4)))
            alpha  = base_alpha * (1.0 - t * 0.6)
            if alpha < 0.02:
                continue
            ov = img.copy()
            cv2.circle(ov, (cx_i, cy_i), ring_r, C_TEAL, 1, cv2.LINE_AA)
            cv2.addWeighted(ov, alpha, img, 1 - alpha, 0, img)

        # Floating depth label — minimal, no developer aesthetics
        if abs(depth_z) > 15:
            label = f"{'↗' if depth_z > 0 else '↙'} {abs(depth_z):.0f}"
            text_x = cx_i + 28
            text_y = cy_i - 16
            cv2.putText(img, label, (text_x, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.32, C_TEAL, 1, cv2.LINE_AA)


# ── Spawn System ──────────────────────────────────────────────────────

SPAWN_HOLD_TIME  = 2    # seconds open palm held
SPAWN_COOLDOWN   = 0.8     # seconds between spawns

class SpawnSystem:
    """
    Detects open-palm-hold gesture and spawns new objects.
    Spawn location = cursor at current hand depth.

    `factory(cx, cy, depth_z, size)` builds whatever object type the
    caller wants (Cube, SpatialNode, ...) — defaults to Cube so any
    Phase ≤5 caller that doesn't pass a factory keeps working unchanged.
    """

    def __init__(self, factory=None):
        self._open_since  = None
        self._last_spawn  = 0.0
        self._progress    = 0.0   # 0..1
        self._factory      = factory or self._default_factory

    @staticmethod
    def _default_factory(cx, cy, depth_z, size):
        from cube import Cube
        return Cube(cx, cy, z3d=depth_z, size=size)

    @property
    def progress(self):
        return self._progress

    def update(self, hand_state, cx, cy, depth_z, t, objects, W, H):
        """
        Returns a newly spawned object if spawning triggered, else None.
        Modifies `objects` list in place.
        """
        is_open = hand_state.visible and hand_state.is_open
        since_spawn = t - self._last_spawn

        if not is_open or since_spawn < SPAWN_COOLDOWN:
            self._open_since = None
            self._progress   = 0.0
            return None

        if self._open_since is None:
            self._open_since = t
            self._progress   = 0.0
            return None

        held = t - self._open_since
        self._progress = min(1.0, held / SPAWN_HOLD_TIME)

        if held >= SPAWN_HOLD_TIME:
            self._open_since = None
            self._last_spawn = t
            self._progress   = 0.0
            size = int(min(W, H) * 0.10)
            new_obj = self._factory(cx, cy, depth_z, size)
            objects.append(new_obj)
            return new_obj

        return None

    def draw_progress(self, img, cx, cy):
        """Draw spawn charge-up indicator around cursor."""
        if self._progress < 0.05:
            return
        cx_i, cy_i = int(cx), int(cy)
        ring_r = 35
        angle  = int(360 * self._progress)
        ov = img.copy()
        cv2.ellipse(ov, (cx_i, cy_i), (ring_r, ring_r), -90,
                    0, angle, C_TEAL, 3, cv2.LINE_AA)
        cv2.addWeighted(ov, 0.85, img, 0.15, 0, img)
        if self._progress > 0.8:
            label = "SPAWN"
            cv2.putText(img, label, (cx_i - 20, cy_i + ring_r + 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, C_TEAL, 1, cv2.LINE_AA)


# ── Delete System ─────────────────────────────────────────────────────

DELETE_HOVER_RADIUS = 100   # px — how close fist must be to start countdown

class DeleteSystem:
    """
    Manages delete-by-fist-hold for all cubes.
    Call update() each frame; it returns a list of cubes to remove.
    """

    def update(self, hand_state, cx, cy, cubes, grabbed_cube):
        """Returns list of cubes to delete."""
        to_delete = []
        is_fist   = hand_state.visible and hand_state.is_fist

        for cube in cubes:
            if cube is grabbed_cube:
                cube.cancel_delete_countdown()
                continue

            dist = cube.screen_dist(cx, cy)
            if is_fist and dist < DELETE_HOVER_RADIUS:
                cube.begin_delete_countdown()
            else:
                cube.cancel_delete_countdown()

            if cube.marked_for_delete:
                to_delete.append(cube)

        return to_delete


# ── Selection System ──────────────────────────────────────────────────

class SelectionSystem:
    """
    Pinch-to-select cubes. Selected cubes move together.
    Right hand: pinch = select/deselect nearest cube
    """

    def __init__(self):
        self._prev_pinching = False

    def update(self, hand_state, cx, cy, cubes, grabbed_cube):
        """Toggle selection on pinch-start over a cube."""
        is_pinching = hand_state.visible and hand_state.is_pinching
        just_pinched = is_pinching and not self._prev_pinching
        self._prev_pinching = is_pinching

        if just_pinched and grabbed_cube is None:
            # Find nearest cube within range
            best_d, best_c = 9999.0, None
            for cube in cubes:
                d = cube.screen_dist(cx, cy)
                if d < best_d:
                    best_d, best_c = d, cube
            if best_c is not None and best_d < 120:
                best_c.selected = not best_c.selected

    def draw_selection_feedback(self, img, cubes, W, H):
        """Draw a subtle bounding hint if multiple cubes are selected."""
        selected = [c for c in cubes if c.selected]
        if len(selected) < 2:
            return
        xs = [int(c.sx) for c in selected]
        ys = [int(c.sy) for c in selected]
        x1, y1 = max(0, min(xs) - 20), max(0, min(ys) - 20)
        x2, y2 = min(W, max(xs) + 20), min(H, max(ys) + 20)
        ov = img.copy()
        cv2.rectangle(ov, (x1, y1), (x2, y2), C_AMBER, 1, cv2.LINE_AA)
        cv2.addWeighted(ov, 0.35, img, 0.65, 0, img)
        label = f"{len(selected)} selected"
        cv2.putText(img, label, (x1, y1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, C_AMBER, 1, cv2.LINE_AA)


# ── Snap System ───────────────────────────────────────────────────────

from cube import SNAP_GRID

class SnapSystem:
    """Draws subtle grid snap guides when a cube is near a snap point."""

    def draw_guides(self, img, cube, W, H):
        if not cube.grabbed or not cube.snap_enabled:
            return

        # Find nearest grid lines
        nx = round(cube.x3d / SNAP_GRID) * SNAP_GRID
        ny = round(cube.y3d / SNAP_GRID) * SNAP_GRID
        dist_x = abs(cube.x3d - nx)
        dist_y = abs(cube.y3d - ny)

        from cube import SNAP_THRESHOLD
        alpha = 0.25

        if dist_x < SNAP_THRESHOLD * 2:
            snap_sx = int(cube.sx - (cube.x3d - nx))
            ov = img.copy()
            cv2.line(ov, (snap_sx, 0), (snap_sx, H), C_TEAL, 1, cv2.LINE_AA)
            a = alpha * max(0.0, 1.0 - dist_x / (SNAP_THRESHOLD * 2))
            cv2.addWeighted(ov, a, img, 1 - a, 0, img)

        if dist_y < SNAP_THRESHOLD * 2:
            snap_sy = int(cube.sy - (cube.y3d - ny))
            ov = img.copy()
            cv2.line(ov, (0, snap_sy), (W, snap_sy), C_TEAL, 1, cv2.LINE_AA)
            a = alpha * max(0.0, 1.0 - dist_y / (SNAP_THRESHOLD * 2))
            cv2.addWeighted(ov, a, img, 1 - a, 0, img)


# ── Calibration Overlay ───────────────────────────────────────────────

class CalibrationOverlay:
    """
    Shown during the first ~45 frames while depth baseline is being established.
    Clean, non-developer aesthetic.
    """

    def draw(self, img, progress, W, H):
        if progress >= 1.0:
            return

        cx, cy = W // 2, H // 2

        # Background vignette
        ov = img.copy()
        cv2.rectangle(ov, (0, 0), (W, H), (0, 0, 0), -1)
        cv2.addWeighted(ov, 0.35, img, 0.65, 0, img)

        # Progress arc
        ring_r = 55
        angle  = int(360 * progress)
        cv2.ellipse(img, (cx, cy), (ring_r, ring_r), -90,
                    0, 360, C_DIM, 2, cv2.LINE_AA)
        cv2.ellipse(img, (cx, cy), (ring_r, ring_r), -90,
                    0, angle, C_TEAL, 3, cv2.LINE_AA)

        cv2.putText(img, "Hold hand steady", (cx - 68, cy + ring_r + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, C_TEAL, 1, cv2.LINE_AA)
        cv2.putText(img, "Calibrating depth...", (cx - 78, cy + ring_r + 42),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, C_DIM, 1, cv2.LINE_AA)


# ── Enhanced floor ────────────────────────────────────────────────────

def draw_floor_enhanced(img, floor_y, W):
    """
    Improved floor: gradient fade, subtle grid marks, no dev-HUD look.
    """
    # Soft gradient strip
    strip_h = 12
    for i in range(strip_h):
        alpha = (strip_h - i) / strip_h * 0.5
        ov = img.copy()
        cv2.line(ov, (0, floor_y - i), (W, floor_y - i), C_GRID, 1)
        cv2.addWeighted(ov, alpha, img, 1 - alpha, 0, img)

    # Solid floor line
    cv2.line(img, (0, floor_y), (W, floor_y), C_FLOOR, 2, cv2.LINE_AA)

    # Perspective grid marks along floor
    cx = W // 2
    for step in range(1, 6):
        x_offset = step * (W // 6)
        for sign in (-1, 1):
            px = cx + sign * x_offset
            if 0 <= px <= W:
                ov = img.copy()
                cv2.line(ov, (px, floor_y - 4), (px, floor_y + 2), C_GRID, 1)
                cv2.addWeighted(ov, 0.4, img, 0.6, 0, img)


# ── Minimal HUD ───────────────────────────────────────────────────────

def draw_minimal_hud(img, tracker, grabbed_cube, gravity_on, W, H,
                     num_cubes, num_selected):
    """
    Clean, non-developer HUD. Replaces the old debug-heavy overlay.
    Shows only what a spatial OS user needs.
    """
    r = tracker.right

    # State indicator (top-left, subtle)
    if r.visible:
        if grabbed_cube:
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

    # Gravity indicator
    grav_col = C_GREEN if gravity_on else C_DIM
    cv2.putText(img, "G" if gravity_on else "g", (W - 28, H - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.40, grav_col, 1, cv2.LINE_AA)

    # Object count
    if num_cubes > 0:
        count_str = f"{num_cubes} obj"
        if num_selected > 0:
            count_str += f"  {num_selected} sel"
        cv2.putText(img, count_str, (W - 80, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, C_DIM, 1, cv2.LINE_AA)

    # Quick-key legend (top-left, very subtle)
    legend = ["G=gravity", "R=reset", "S=snap", "Q=quit"]
    for i, line in enumerate(legend):
        cv2.putText(img, line, (16, 18 + i * 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.28, (35, 35, 35), 1, cv2.LINE_AA)