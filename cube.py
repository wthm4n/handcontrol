"""
cube.py — Phase 3: 3D cube with throw physics.

Changes from Phase 2:
  - Each Cube owns a PhysicsBody (from physics.py).
  - update() calls body.step() when not grabbed (free flight / resting).
  - grab() stops physics; release() is now release(vx, vy) to seed velocity.
  - Floor Y is passed in from main (= bottom of frame minus cube margin).
"""

import cv2
import math
from physics import PhysicsBody


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


# ── Cube geometry ─────────────────────────────────────────────────────

EDGES = [
    (0,1),(1,2),(2,3),(3,0),
    (4,5),(5,6),(6,7),(7,4),
    (0,4),(1,5),(2,6),(3,7),
]

COLOR_IDLE    = ( 40, 180, 150)
COLOR_HOVER   = ( 20, 240, 200)
COLOR_GRAB    = ( 50, 255, 120)
COLOR_FLYING  = (180, 120, 255)   # purple tint while airborne


class Cube:
    def __init__(self, x3d, y3d, z3d=0.0, size=90):
        self.x3d  = float(x3d)
        self.y3d  = float(y3d)
        self.z3d  = float(z3d)
        self.size = size

        # Screen position (updated from 3D each frame)
        self.sx = float(x3d)
        self.sy = float(y3d)

        # Euler rotation
        self.rx = 0.35
        self.ry = 0.45
        self.rz = 0.0

        # State
        self.hovered = False
        self.grabbed = False

        # Visuals
        self.glow    = 0.0
        self.scale_v = 1.0

        # Grab offset
        self.grab_off_x = 0.0
        self.grab_off_y = 0.0

        # Physics body — always present, stepped when not grabbed
        self.body = PhysicsBody()

    # ── Interaction ───────────────────────────────────────────────────

    def grab(self, cx, cy):
        self.grabbed    = True
        self.grab_off_x = self.x3d - cx
        self.grab_off_y = self.y3d - cy
        self.scale_v    = 1.10
        self.body.stop()   # suppress physics while held

    def release(self, vx=0.0, vy=0.0):
        """Release and throw. vx/vy in px/s."""
        self.grabbed = False
        self.scale_v = 0.92
        self.body.launch(vx, vy)

    # ── Update ────────────────────────────────────────────────────────

    def update(self, dt, cx, cy, right_hand, left_hand,
               screen_w, screen_h, floor_y, gravity_on=True):
        if self.grabbed:
            # Hand-driven movement
            self.x3d = cx + self.grab_off_x
            self.y3d = cy + self.grab_off_y

            # Wrist rotation drives spin
            self.rx += right_hand.delta_ox * 3.5
            self.ry += right_hand.delta_oy * 3.5
            self.rz += right_hand.delta_oz * 3.0

            # Left hand Z depth
            if left_hand and left_hand.visible:
                target_z = left_hand.wrist_pos[2] * 400.0
                self.z3d = _lerp(self.z3d, target_z, 0.08)

        else:
            # Physics-driven movement
            self.x3d, self.y3d, drx, dry, drz = self.body.step(
                dt, self.x3d, self.y3d,
                floor_y, screen_w, screen_h,
                gravity_on=gravity_on,
            )
            self.rx += drx
            self.ry += dry
            self.rz += drz

        # Visual state
        is_airborne = not self.body.sleeping and not self.grabbed and not self.body.on_floor
        glow_target  = 1.0 if self.grabbed else (0.75 if is_airborne else
                        (0.65 if self.hovered else 0.05))
        self.glow    = _lerp(self.glow,    glow_target, 0.15)
        self.scale_v = _lerp(self.scale_v, 1.0,         0.18)

        # Screen position (simple: sx == x3d while z3d is small)
        self.sx = self.x3d
        self.sy = self.y3d

    # ── Rendering ─────────────────────────────────────────────────────

    def _vertices(self):
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
            x, y, z = v
            effective_z = z + self.z3d
            fov   = self.size * 6.0
            persp = 1.0 + effective_z / fov
            sx = int(self.sx + x * persp)
            sy = int(self.sy - y * persp)
            pts.append((sx, sy, z))
        return pts

    def draw(self, img):
        pts = self._vertices()
        glow = self.glow

        is_airborne = not self.body.sleeping and not self.grabbed and not self.body.on_floor
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

        for ia, ib in sorted(EDGES, key=edge_z):
            pa, pb = pts[ia], pts[ib]
            t  = ((pa[2] + pb[2]) * 0.5 + self.size) / (self.size * 2.0)
            lw = max(1, int(1 + t * 2.5))

            if glow > 0.02:
                ov = img.copy()
                cv2.line(ov, (pa[0],pa[1]), (pb[0],pb[1]), color, lw+8, cv2.LINE_AA)
                cv2.addWeighted(ov, glow * 0.18, img, 1 - glow * 0.18, 0, img)
            cv2.line(img, (pa[0],pa[1]), (pb[0],pb[1]), color, lw, cv2.LINE_AA)

        for px, py, pz in pts:
            r = max(2, int(3 + (pz / self.size) * 1.5))
            if glow > 0.02:
                ov = img.copy()
                cv2.circle(ov, (px,py), r+4, color, -1, cv2.LINE_AA)
                cv2.addWeighted(ov, glow * 0.15, img, 1 - glow * 0.15, 0, img)
            cv2.circle(img, (px,py), r, color, -1, cv2.LINE_AA)

        if self.hovered or self.grabbed:
            ring_r = int(self.size * 1.5 * self.scale_v)
            ov = img.copy()
            cv2.circle(ov, (int(self.sx), int(self.sy)), ring_r, color, 1, cv2.LINE_AA)
            cv2.addWeighted(ov, 0.20 + glow * 0.35, img, 1 - (0.20 + glow * 0.35), 0, img)

    def screen_dist(self, cx, cy):
        return math.sqrt((cx - self.sx)**2 + (cy - self.sy)**2)