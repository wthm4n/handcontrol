
"""
cube.py — Phase 4: Full 3D cube with depth interaction and perspective scaling.

Changes from Phase 3:
  - Proper perspective projection: cube.sx/sy are computed from x3d/y3d/z3d
    and a fixed FOV, so cubes further away appear smaller and shifted toward
    the vanishing point (screen centre).
  - grab() records z3d offset so it doesn't snap to hand depth.
  - update() drives z3d from right-hand wrist Z while grabbed.
  - release() takes (vx, vy, vz) for full 3D throw.
  - PhysicsBody.step() now returns (new_x, new_y, new_z, drx, dry, drz).
  - screen_dist() uses projected sx/sy (already perspective-corrected).
  - Draw: edge thickness and vertex size now also modulated by z3d depth.

Coordinate system:
  x3d, y3d  — world pixels (origin = top-left of frame)
  z3d        — depth in world pixels (0 = at-camera plane, +ve = further away)
  sx, sy     — projected screen pixels (perspective-correct, used for drawing)

Projection (centred at screen_cx, screen_cy):
  scale = FOV / (FOV + z3d)
  sx = screen_cx + (x3d - screen_cx) * scale
  sy = screen_cy + (y3d - screen_cy) * scale
"""

import cv2
import math
from physics import PhysicsBody


# ── Constants ─────────────────────────────────────────────────────────

FOV = 800.0          # perspective focal length (px); larger = less distortion

# Z-depth tracking while grabbed
Z_SCALE         = 600.0   # maps MediaPipe wrist Z (≈±0.3) to world-px range
Z_SMOOTH_GRAB   = 0.10    # lerp weight while grabbing (lower = less jitter)
Z_SMOOTH_FREE   = 0.25    # lerp weight when not grabbed (faster to settle)
Z_DEFAULT       = 0.0     # resting depth for new cubes

# Colours
COLOR_IDLE   = ( 40, 180, 150)
COLOR_HOVER  = ( 20, 240, 200)
COLOR_GRAB   = ( 50, 255, 120)
COLOR_FLYING = (180, 120, 255)

# Cube geometry
EDGES = [
    (0,1),(1,2),(2,3),(3,0),
    (4,5),(5,6),(6,7),(7,4),
    (0,4),(1,5),(2,6),(3,7),
]


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


class Cube:
    def __init__(self, x3d, y3d, z3d=Z_DEFAULT, size=90):
        self.x3d  = float(x3d)
        self.y3d  = float(y3d)
        self.z3d  = float(z3d)
        self.size = size          # base half-size in world px at z=0

        # Projected screen position — recomputed every frame
        self.sx = float(x3d)
        self.sy = float(y3d)

        # Euler rotation (radians)
        self.rx = 0.35
        self.ry = 0.45
        self.rz = 0.0

        # Interaction state
        self.hovered = False
        self.grabbed = False

        # Visuals
        self.glow    = 0.0
        self.scale_v = 1.0

        # Grab offsets (world space)
        self.grab_off_x = 0.0
        self.grab_off_y = 0.0
        self.grab_off_z = 0.0   # z offset at grab time → preserved until release

        # Physics
        self.body = PhysicsBody()

    # ── Projection helpers ────────────────────────────────────────────

    def _project_point(self, x3d, y3d, z3d, screen_cx, screen_cy):
        """Project a world point to screen space."""
        scale = FOV / max(FOV + z3d, 1.0)   # guard against z3d = -FOV
        sx = screen_cx + (x3d - screen_cx) * scale
        sy = screen_cy + (y3d - screen_cy) * scale
        return sx, sy, scale

    # ── Interaction ───────────────────────────────────────────────────

    def grab(self, cx, cy, hand_z_world):
        """
        cx, cy       — screen cursor position
        hand_z_world — current world-Z of the hand at grab time
        """
        self.grabbed    = True
        self.grab_off_x = self.x3d - cx
        self.grab_off_y = self.y3d - cy
        self.grab_off_z = self.z3d - hand_z_world   # depth offset preserved
        self.scale_v    = 1.10
        self.body.stop()

    def release(self, vx=0.0, vy=0.0, vz=0.0):
        self.grabbed = False
        self.scale_v = 0.92
        self.body.launch(vx, vy, vz)

    # ── Update ────────────────────────────────────────────────────────

    def update(self, dt, cx, cy, hand_z_world,
               right_hand, left_hand,
               screen_w, screen_h, floor_y,
               gravity_on=True):
        """
        cx, cy        — right-hand cursor in screen pixels
        hand_z_world  — right-hand wrist Z in world pixels (smoothed externally)
        """
        screen_cx = screen_w / 2.0
        screen_cy = screen_h / 2.0

        if self.grabbed:
            # ── XY follows cursor ─────────────────────────────────────
            self.x3d = cx + self.grab_off_x
            self.y3d = cy + self.grab_off_y

            # ── Z follows hand depth with offset ─────────────────────
            target_z = hand_z_world + self.grab_off_z
            self.z3d = _lerp(self.z3d, target_z, Z_SMOOTH_GRAB)

            # ── Wrist rotation drives spin ────────────────────────────
            self.rx += right_hand.delta_ox * 3.5
            self.ry += right_hand.delta_oy * 3.5
            self.rz += right_hand.delta_oz * 3.0

        else:
            # ── Physics-driven ────────────────────────────────────────
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

        # ── Glow / scale animation ────────────────────────────────────
        is_airborne  = (not self.body.sleeping and
                        not self.grabbed and
                        not self.body.on_floor)
        glow_target  = (1.0 if self.grabbed else
                        0.80 if is_airborne else
                        0.65 if self.hovered else
                        0.05)
        self.glow    = _lerp(self.glow,    glow_target, 0.15)
        self.scale_v = _lerp(self.scale_v, 1.0,         0.18)

    # ── Rendering ─────────────────────────────────────────────────────

    def _vertices(self, screen_cx, screen_cy):
        """
        Compute projected (sx, sy, local_z) for all 8 corners.
        Size is modulated by perspective scale so cube looks correct in 3D.
        """
        # Perspective scale at cube centre
        centre_scale = FOV / max(FOV + self.z3d, 1.0)
        h = self.size * self.scale_v   # half-size in local space

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

            # World position of this vertex
            wx = self.x3d + lx
            wy = self.y3d + ly
            wz = self.z3d + lz

            # Project to screen
            sx, sy, _ = self._project_point(wx, wy, wz, screen_cx, screen_cy)
            pts.append((int(sx), int(sy), lz, wz))   # lz for sorting, wz for size

        return pts

    def draw(self, img):
        H, W = img.shape[:2]
        screen_cx = W / 2.0
        screen_cy = H / 2.0

        pts  = self._vertices(screen_cx, screen_cy)
        glow = self.glow

        is_airborne = (not self.body.sleeping and
                       not self.grabbed and
                       not self.body.on_floor)
        if self.grabbed:
            color = COLOR_GRAB
        elif is_airborne:
            color = COLOR_FLYING
        elif self.hovered:
            color = COLOR_HOVER
        else:
            color = COLOR_IDLE

        def edge_z(e):
            return (pts[e[0]][2] + pts[e[1]][2]) * 0.5

        # Edges — back to front
        for ia, ib in sorted(EDGES, key=edge_z):
            pa, pb = pts[ia], pts[ib]

            # Depth-modulated line weight
            avg_wz = (pa[3] + pb[3]) * 0.5
            depth_t = max(0.0, min(1.0, (avg_wz + 100) / 600.0))
            lw = max(1, int(1 + depth_t * 2.5))

            if glow > 0.02:
                ov = img.copy()
                cv2.line(ov, (pa[0],pa[1]), (pb[0],pb[1]), color, lw+8, cv2.LINE_AA)
                cv2.addWeighted(ov, glow * 0.18, img, 1 - glow * 0.18, 0, img)
            cv2.line(img, (pa[0],pa[1]), (pb[0],pb[1]), color, lw, cv2.LINE_AA)

        # Vertices
        for sx, sy, lz, wz in pts:
            depth_t = max(0.0, min(1.0, (wz + 100) / 600.0))
            r = max(2, int(2 + depth_t * 3.0))
            if glow > 0.02:
                ov = img.copy()
                cv2.circle(ov, (sx, sy), r+4, color, -1, cv2.LINE_AA)
                cv2.addWeighted(ov, glow * 0.15, img, 1 - glow * 0.15, 0, img)
            cv2.circle(img, (sx, sy), r, color, -1, cv2.LINE_AA)

        # Hover / grab ring (perspective-scaled radius)
        if self.hovered or self.grabbed:
            centre_scale = FOV / max(FOV + self.z3d, 1.0)
            ring_r = max(10, int(self.size * 1.5 * self.scale_v * centre_scale))
            ov = img.copy()
            cv2.circle(ov, (int(self.sx), int(self.sy)), ring_r, color, 1, cv2.LINE_AA)
            cv2.addWeighted(ov, 0.20 + glow * 0.35,
                            img, 1 - (0.20 + glow * 0.35), 0, img)

    def screen_dist(self, cx, cy):
        """Distance in screen space (perspective-projected)."""
        return math.sqrt((cx - self.sx)**2 + (cy - self.sy)**2)
