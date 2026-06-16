"""
cube.py — Phase 5: Spatial computing visual redesign + spawn/delete/select.

Phase 5 changes over Phase 4:
  • New colour palette: cyan/teal/green/purple/amber/red for high contrast.
  • Bloom-like multi-layer glow with depth-aware intensity.
  • Better perspective: enhanced FOV scaling + depth fog.
  • Ground shadow (soft ellipse projected onto floor plane).
  • Motion trails for fast throws.
  • Spawn animation: scale-in + glow pulse.
  • Delete countdown: shrink + fade with visual ring timer.
  • Selection state: amber highlight, can be moved in groups.
  • Snap grid support: x/y grid snap with guide lines.

Coordinate system unchanged from Phase 4.
"""

import cv2
import math
import time
from physics import PhysicsBody


# ── Constants ─────────────────────────────────────────────────────────

FOV = 900.0          # increased from 800 → stronger perspective

Z_SMOOTH_GRAB   = 0.12
Z_SMOOTH_FREE   = 0.25
Z_DEFAULT       = 0.0

# ── Phase 5 colour palette ────────────────────────────────────────────
#   (BGR tuples)
COLOR_IDLE     = (180, 200,  50)   # soft cyan
COLOR_HOVER    = (220, 240,  20)   # bright teal
COLOR_GRAB     = ( 50, 255, 120)   # vivid green
COLOR_FLYING   = (255,  80, 180)   # electric purple
COLOR_SELECTED = ( 30, 190, 255)   # amber
COLOR_DELETE   = ( 40,  40, 240)   # red

# Cube geometry
EDGES = [
    (0,1),(1,2),(2,3),(3,0),
    (4,5),(5,6),(6,7),(7,4),
    (0,4),(1,5),(2,6),(3,7),
]

# Spawn / delete timing
SPAWN_DURATION   = 0.35   # seconds for spawn scale-in
DELETE_HOLD_TIME = 1.0    # seconds fist must hover to delete

# Depth fog
FOG_NEAR = 0.0    # world-Z where fog = 0
FOG_FAR  = 700.0  # world-Z where fog = max
FOG_MAX  = 0.55   # maximum fog darkening factor

# Motion trail
TRAIL_MAX_PTS  = 8
TRAIL_MIN_SPEED = 250   # px/s before trail shows

# Grid snap
SNAP_GRID = 80          # world-px grid cell size
SNAP_THRESHOLD = 30     # snap if within this distance of grid line


# ── Math helpers ──────────────────────────────────────────────────────

def _lerp(a, b, t):
    return a + (b - a) * t

def _rot_x(v, a):
    c, s = math.cos(a), math.sin(a)
    x, y, z = v
    return (x, c*y - s*z, s*y + c*z)

def _rot_y(v, a):
    c, s = math.cos(a), math.sin(a)
    x, y, z = v
    return (c*x + s*z, y, -s*x + c*z)

def _rot_z(v, a):
    c, s = math.cos(a), math.sin(a)
    x, y, z = v
    return (c*x - s*y, s*x + c*y, z)

def _fog_alpha(z3d):
    """0 = no fog (near), 1 = full fog (far)."""
    t = (z3d - FOG_NEAR) / max(FOG_FAR - FOG_NEAR, 1.0)
    return max(0.0, min(1.0, t)) * FOG_MAX

def _depth_glow_scale(z3d):
    """Near objects glow more strongly (1.0), far objects less (0.3)."""
    t = (z3d - FOG_NEAR) / max(FOG_FAR - FOG_NEAR, 1.0)
    t = max(0.0, min(1.0, t))
    return 1.0 - t * 0.7


class Cube:
    def __init__(self, x3d, y3d, z3d=Z_DEFAULT, size=90):
        self.x3d  = float(x3d)
        self.y3d  = float(y3d)
        self.z3d  = float(z3d)
        self.size = size

        self.sx = float(x3d)
        self.sy = float(y3d)

        self.rx = 0.35
        self.ry = 0.45
        self.rz = 0.0

        # Interaction
        self.hovered  = False
        self.grabbed  = False
        self.selected = False      # Phase 5: multi-select

        # Visuals
        self.glow    = 0.0
        self.scale_v = 1.0

        # Grab offsets
        self.grab_off_x = 0.0
        self.grab_off_y = 0.0
        self.grab_off_z = 0.0

        # Physics
        self.body = PhysicsBody()

        # ── Phase 5: Spawn animation ──────────────────────────────────
        self._spawn_t      = time.time()
        self._spawning     = True
        self._spawn_glow   = 2.0   # extra glow pulse on creation

        # ── Phase 5: Delete countdown ─────────────────────────────────
        self._delete_hold_start = None   # time fist entered delete zone
        self.delete_progress    = 0.0    # 0..1
        self.marked_for_delete  = False

        # ── Phase 5: Motion trail ─────────────────────────────────────
        self._trail = []   # list of (sx, sy, alpha)

        # ── Phase 5: Snap ─────────────────────────────────────────────
        self.snap_enabled = False

    # ── Projection ────────────────────────────────────────────────────

    def _project_point(self, x3d, y3d, z3d, screen_cx, screen_cy):
        scale = FOV / max(FOV + z3d, 1.0)
        sx = screen_cx + (x3d - screen_cx) * scale
        sy = screen_cy + (y3d - screen_cy) * scale
        return sx, sy, scale

    # ── Interaction ───────────────────────────────────────────────────

    def grab(self, cx, cy, hand_z_world):
        self.grabbed    = True
        self.grab_off_x = self.x3d - cx
        self.grab_off_y = self.y3d - cy
        self.grab_off_z = self.z3d - hand_z_world
        self.scale_v    = 1.10
        self.body.stop()
        self._delete_hold_start = None
        self.delete_progress    = 0.0

    def release(self, vx=0.0, vy=0.0, vz=0.0):
        self.grabbed = False
        self.scale_v = 0.92
        self.body.launch(vx, vy, vz)
        self._delete_hold_start = None
        self.delete_progress    = 0.0

    def begin_delete_countdown(self):
        if self._delete_hold_start is None:
            self._delete_hold_start = time.time()

    def cancel_delete_countdown(self):
        self._delete_hold_start = None
        self.delete_progress    = 0.0

    def _apply_snap(self):
        """Snap x3d/y3d to nearest grid lines if within threshold."""
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

        # ── Spawn animation ───────────────────────────────────────────
        if self._spawning:
            elapsed = now - self._spawn_t
            frac    = min(1.0, elapsed / SPAWN_DURATION)
            # Ease-out cubic
            t = 1.0 - (1.0 - frac) ** 3
            self.scale_v  = t
            self._spawn_glow = max(0.0, 2.0 * (1.0 - frac))
            if frac >= 1.0:
                self._spawning = False
                self.scale_v   = 1.0

        # ── Delete countdown ──────────────────────────────────────────
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

            self.rx += right_hand.delta_ox * 3.5
            self.ry += right_hand.delta_oy * 3.5
            self.rz += right_hand.delta_oz * 3.0

            if self.snap_enabled:
                self._apply_snap()

        else:
            prev_x, prev_y = self.x3d, self.y3d
            self.x3d, self.y3d, self.z3d, drx, dry, drz = self.body.step(
                dt, self.x3d, self.y3d, self.z3d,
                floor_y, screen_w, screen_h,
                gravity_on=gravity_on,
            )
            self.rx += drx
            self.ry += dry
            self.rz += drz

        # ── Perspective-correct screen position ───────────────────────
        self.sx, self.sy, _ = self._project_point(
            self.x3d, self.y3d, self.z3d, screen_cx, screen_cy
        )

        # ── Motion trail ──────────────────────────────────────────────
        speed = math.sqrt(self.body.vx**2 + self.body.vy**2)
        if not self.grabbed and speed > TRAIL_MIN_SPEED:
            self._trail.append((self.sx, self.sy, 1.0))
            if len(self._trail) > TRAIL_MAX_PTS:
                self._trail.pop(0)
        else:
            # Fade existing trail
            self._trail = [(x, y, max(0.0, a - dt * 4)) for x, y, a in self._trail if a > 0.05]

        # ── Glow / scale animation ────────────────────────────────────
        if not self._spawning:
            is_airborne = (not self.body.sleeping and
                           not self.grabbed and
                           not self.body.on_floor)
            glow_target = (1.0 if self.grabbed else
                           0.80 if is_airborne else
                           0.85 if self.selected else
                           0.65 if self.hovered else
                           0.05)
            self.glow    = _lerp(self.glow, glow_target, 0.15)
            self.scale_v = _lerp(self.scale_v, 1.0, 0.18)

    # ── Rendering ─────────────────────────────────────────────────────

    def _vertices(self, screen_cx, screen_cy):
        h = self.size * self.scale_v
        local = [
            (-h,-h,-h),(h,-h,-h),(h,h,-h),(-h,h,-h),
            (-h,-h, h),(h,-h, h),(h,h, h),(-h,h, h),
        ]
        pts = []
        for v in local:
            v = _rot_x(v, self.rx)
            v = _rot_y(v, self.ry)
            v = _rot_z(v, self.rz)
            lx, ly, lz = v
            wx = self.x3d + lx
            wy = self.y3d + ly
            wz = self.z3d + lz
            sx, sy, _ = self._project_point(wx, wy, wz, screen_cx, screen_cy)
            pts.append((int(sx), int(sy), lz, wz))
        return pts

    def _pick_color(self):
        is_airborne = (not self.body.sleeping and
                       not self.grabbed and
                       not self.body.on_floor)
        if self.delete_progress > 0.05:
            t = self.delete_progress
            r = int(COLOR_DELETE[0] * t + COLOR_IDLE[0] * (1 - t))
            g = int(COLOR_DELETE[1] * t + COLOR_IDLE[1] * (1 - t))
            b = int(COLOR_DELETE[2] * t + COLOR_IDLE[2] * (1 - t))
            return (r, g, b)
        if self.grabbed:   return COLOR_GRAB
        if is_airborne:    return COLOR_FLYING
        if self.selected:  return COLOR_SELECTED
        if self.hovered:   return COLOR_HOVER
        return COLOR_IDLE

    def draw(self, img):
        H, W = img.shape[:2]
        screen_cx = W / 2.0
        screen_cy = H / 2.0

        color = self._pick_color()
        pts   = self._vertices(screen_cx, screen_cy)
        glow  = self.glow + self._spawn_glow
        depth_glow = _depth_glow_scale(self.z3d)
        fog   = _fog_alpha(self.z3d)

        # ── Ground shadow ─────────────────────────────────────────────
        self._draw_shadow(img, screen_cx, screen_cy, H)

        # ── Motion trail ─────────────────────────────────────────────
        self._draw_trail(img, color)

        # ── Edges ─────────────────────────────────────────────────────
        def edge_z(e):
            return (pts[e[0]][2] + pts[e[1]][2]) * 0.5

        for ia, ib in sorted(EDGES, key=edge_z):
            pa, pb = pts[ia], pts[ib]
            avg_wz  = (pa[3] + pb[3]) * 0.5
            depth_t = max(0.0, min(1.0, (avg_wz + 100) / 600.0))
            lw      = max(1, int(1 + depth_t * 2.5))

            eff_glow = glow * depth_glow
            if eff_glow > 0.02:
                # Multi-layer bloom
                for radius, alpha_mult in [(14, 0.06), (8, 0.12), (4, 0.20)]:
                    ov = img.copy()
                    cv2.line(ov, (pa[0],pa[1]), (pb[0],pb[1]), color, lw + radius, cv2.LINE_AA)
                    cv2.addWeighted(ov, eff_glow * alpha_mult,
                                    img, 1 - eff_glow * alpha_mult, 0, img)
            cv2.line(img, (pa[0],pa[1]), (pb[0],pb[1]), color, lw, cv2.LINE_AA)

        # ── Vertices ──────────────────────────────────────────────────
        for sx, sy, lz, wz in pts:
            depth_t = max(0.0, min(1.0, (wz + 100) / 600.0))
            r = max(2, int(2 + depth_t * 3.0))
            eff_glow = glow * depth_glow
            if eff_glow > 0.02:
                ov = img.copy()
                cv2.circle(ov, (sx, sy), r+6, color, -1, cv2.LINE_AA)
                cv2.addWeighted(ov, eff_glow * 0.14, img, 1 - eff_glow * 0.14, 0, img)
            cv2.circle(img, (sx, sy), r, color, -1, cv2.LINE_AA)

        # ── Hover / grab / select ring ────────────────────────────────
        if self.hovered or self.grabbed or self.selected:
            centre_scale = FOV / max(FOV + self.z3d, 1.0)
            ring_r = max(10, int(self.size * 1.5 * self.scale_v * centre_scale))
            ov = img.copy()
            cv2.circle(ov, (int(self.sx), int(self.sy)), ring_r, color, 1, cv2.LINE_AA)
            cv2.addWeighted(ov, 0.20 + glow * 0.35,
                            img, 1 - (0.20 + glow * 0.35), 0, img)

        # ── Delete countdown ring ─────────────────────────────────────
        if self.delete_progress > 0.02:
            self._draw_delete_ring(img, color)

        # ── Depth fog ─────────────────────────────────────────────────
        if fog > 0.02:
            # Darken far objects by blending toward black at cube bounding box
            centre_scale = FOV / max(FOV + self.z3d, 1.0)
            fr = max(20, int(self.size * 2.0 * centre_scale))
            sx_i, sy_i = int(self.sx), int(self.sy)
            x1 = max(0, sx_i - fr);  y1 = max(0, sy_i - fr)
            x2 = min(W, sx_i + fr);  y2 = min(H, sy_i + fr)
            roi = img[y1:y2, x1:x2]
            if roi.size > 0:
                dark = (roi * (1.0 - fog)).astype(roi.dtype)
                img[y1:y2, x1:x2] = dark

    def _draw_shadow(self, img, screen_cx, screen_cy, frame_h):
        """Soft ellipse shadow projected on floor plane."""
        floor_y = int(frame_h * 0.94)
        # Shadow only when not too far in Z or too high above floor
        dist_to_floor = floor_y - self.y3d
        if dist_to_floor < 0 or dist_to_floor > frame_h * 0.9:
            return

        centre_scale = FOV / max(FOV + self.z3d, 1.0)
        # Shadow x tracks cube sx; y is always at floor
        shadow_sx = int(screen_cx + (self.x3d - screen_cx) * centre_scale)
        shadow_sy = floor_y - 2

        # Shadow size depends on height above floor and depth
        height_frac = max(0.0, min(1.0, dist_to_floor / (frame_h * 0.5)))
        rx = max(8, int(self.size * centre_scale * (1.0 - height_frac * 0.6)))
        ry = max(3, int(rx * 0.25))

        # Opacity fades as cube rises
        opacity = (1.0 - height_frac) * 0.45

        ov = img.copy()
        cv2.ellipse(ov, (shadow_sx, shadow_sy), (rx, ry),
                    0, 0, 360, (0, 0, 0), -1, cv2.LINE_AA)
        cv2.addWeighted(ov, opacity, img, 1.0 - opacity, 0, img)

    def _draw_trail(self, img, color):
        for i, (tx, ty, alpha) in enumerate(self._trail):
            r = max(1, int(3 * alpha))
            ov = img.copy()
            cv2.circle(ov, (int(tx), int(ty)), r, color, -1, cv2.LINE_AA)
            cv2.addWeighted(ov, alpha * 0.4, img, 1.0 - alpha * 0.4, 0, img)

    def _draw_delete_ring(self, img, color):
        """Animated countdown ring around the cube."""
        centre_scale = FOV / max(FOV + self.z3d, 1.0)
        ring_r = max(15, int(self.size * 1.8 * self.scale_v * centre_scale))
        cx_i, cy_i = int(self.sx), int(self.sy)
        angle = int(360 * self.delete_progress)

        ov = img.copy()
        # Background ring
        cv2.ellipse(ov, (cx_i, cy_i), (ring_r, ring_r), -90,
                    0, 360, (60, 40, 40), 2, cv2.LINE_AA)
        # Progress arc
        cv2.ellipse(ov, (cx_i, cy_i), (ring_r, ring_r), -90,
                    0, angle, COLOR_DELETE, 3, cv2.LINE_AA)
        cv2.addWeighted(ov, 0.8, img, 0.2, 0, img)

    def screen_dist(self, cx, cy):
        return math.sqrt((cx - self.sx)**2 + (cy - self.sy)**2)