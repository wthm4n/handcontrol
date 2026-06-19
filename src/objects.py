"""
objects.py — Phase 8: SpatialObject and the five primitive shapes.

Every object is just a SpatialObject with a half-extent triple `dim`
= [half_x, half_y, half_z], in world pixels. That single triple is what
makes independent axis resizing (X/Y/Z) work identically for every
shape — resizing just edits dim[0], dim[1], or dim[2].

Shapes:
  CubeObject       — equal half-extents by default
  SphereObject      — ellipsoid; dim doubles as the 3 radii
  CylinderObject    — vertical cylinder; dim = (radius_x, half_height, radius_z)
  PrismObject        — rectangular prism; unequal half-extents by default
  PlaneObject        — a very flat box (tiny dim[1])

Also:
  Group             — binds selected objects so grabbing one moves all
"""

import cv2
import math
import time
import itertools
import numpy as np

from physics import PhysicsBody, Z_FAR_LIMIT

FOV = 900.0


def _proj(x3d, y3d, z3d, scx, scy):
    s = FOV / max(FOV + z3d, 1.0)
    return scx + (x3d - scx) * s, scy + (y3d - scy) * s, s


def _lerp(a, b, t):
    return a + (b - a) * t


_id_counter = itertools.count(1)
def _next_id(): return f"obj_{next(_id_counter)}"


C_IDLE    = (180, 200,  50)
C_HOVER   = (220, 240,  20)
C_GRAB    = ( 50, 255, 120)
C_SELECT  = ( 30, 190, 255)
C_DELETE  = ( 40,  40, 240)
C_GROUP   = (230,  90, 215)

C_AXIS_X  = ( 50,  50, 230)
C_AXIS_Y  = ( 60, 220,  60)
C_AXIS_Z  = (230, 130,  40)

DELETE_HOLD = 0.75
SNAP_GRID   = 80
SNAP_THRESH = 30
Z_SMOOTH_G  = 0.12

DIM_MIN = 10.0
DIM_MAX = 280.0


def _box_corners(cx, cy, cz, hx, hy, hz, rx, ry, rz):
    lc = [
        (-1,-1,-1),( 1,-1,-1),( 1, 1,-1),(-1, 1,-1),
        (-1,-1, 1),( 1,-1, 1),( 1, 1, 1),(-1, 1, 1),
    ]
    crx, srx = math.cos(rx), math.sin(rx)
    cry, sry = math.cos(ry), math.sin(ry)
    crz, srz = math.cos(rz), math.sin(rz)
    out = []
    for lx, ly, lz in lc:
        x, y, z = lx*hx, ly*hy, lz*hz
        x, y = x*crz - y*srz, x*srz + y*crz
        y, z = y*crx - z*srx, y*srx + z*crx
        x, z = x*cry + z*sry, -x*sry + z*cry
        out.append((cx+x, cy+y, cz+z))
    return out

BOX_EDGES = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]
BOX_FACES = [(0,1,2,3),(4,7,6,5),(0,4,5,1),(2,6,7,3),(0,3,7,4),(1,5,6,2)]


def _draw_box_3d(img, corners, scx, scy, color, face_color=None, thickness=2):
    pts = []
    for cx, cy, cz in corners:
        sx, sy, _ = _proj(cx, cy, cz, scx, scy)
        pts.append((int(sx), int(sy)))
    if face_color:
        for face in BOX_FACES:
            poly = np.array([pts[i] for i in face], dtype=np.int32)
            ov = img.copy()
            cv2.fillConvexPoly(ov, poly, face_color)
            cv2.addWeighted(ov, 0.16, img, 0.84, 0, img)
    for a, b in BOX_EDGES:
        cv2.line(img, pts[a], pts[b], color, thickness, cv2.LINE_AA)
    return pts


class SpatialObject:
    """Base for every physical object in Aether."""

    shape_kind = "object"

    def __init__(self, x3d=640.0, y3d=360.0, z3d=0.0, label="", dim=(45.0, 45.0, 45.0)):
        self.id    = _next_id()
        self.label = label
        self.dim   = [float(dim[0]), float(dim[1]), float(dim[2])]

        self.x3d = float(x3d); self.y3d = float(y3d); self.z3d = float(z3d)
        self.sx  = float(x3d); self.sy  = float(y3d)

        self.rx = 0.0; self.ry = 0.0; self.rz = 0.0

        self.hovered  = False
        self.grabbed  = False
        self.selected = False

        self.glow = 0.0
        self.scale_v = 1.0

        self._grab_ox = 0.0; self._grab_oy = 0.0; self._grab_oz = 0.0

        self.physics_enabled = True
        self.physics = PhysicsBody()

        self._del_start = None
        self.delete_progress = 0.0
        self.marked_for_delete = False

        self._spawn_t = time.time()
        self._spawning = True
        self._spawn_glow = 2.0

        self.snap_enabled = False
        self.connections = []
        self.group_id = None


    def grab(self, cx, cy, hand_z_world):
        self.grabbed = True
        self._grab_ox = self.x3d - cx
        self._grab_oy = self.y3d - cy
        self._grab_oz = self.z3d - hand_z_world
        self.scale_v  = 1.08
        self.physics.stop()
        self._del_start = None
        self.delete_progress = 0.0

    def release(self, vx=0.0, vy=0.0, vz=0.0):
        self.grabbed = False
        self.scale_v = 0.95
        if self.physics_enabled:
            self.physics.launch(vx, vy, vz)
        self._del_start = None
        self.delete_progress = 0.0

    def begin_delete_countdown(self):
        if self._del_start is None:
            self._del_start = time.time()

    def cancel_delete_countdown(self):
        self._del_start = None
        self.delete_progress = 0.0

    def screen_dist(self, cx, cy):
        return math.sqrt((cx - self.sx) ** 2 + (cy - self.sy) ** 2)

    def resize_axis(self, axis_index, delta):
        """axis_index: 0=X, 1=Y, 2=Z. Clamped to [DIM_MIN, DIM_MAX]."""
        self.dim[axis_index] = max(DIM_MIN, min(DIM_MAX, self.dim[axis_index] + delta))

    def resize_uniform(self, factor):
        for i in range(3):
            self.dim[i] = max(DIM_MIN, min(DIM_MAX, self.dim[i] * factor))


    def update(self, dt, cx, cy, hand_z_world, screen_w, screen_h, floor_y, gravity_on=True):
        now = time.time()
        scx, scy = screen_w / 2.0, screen_h / 2.0

        if self._spawning:
            elapsed = now - self._spawn_t
            frac = min(1.0, elapsed / 0.35)
            t = 1.0 - (1.0 - frac) ** 3
            self.scale_v = t
            self._spawn_glow = max(0.0, 2.0 * (1.0 - frac))
            if frac >= 1.0:
                self._spawning = False
                self.scale_v = 1.0

        if self._del_start is not None:
            elapsed = now - self._del_start
            self.delete_progress = min(1.0, elapsed / DELETE_HOLD)
            if self.delete_progress >= 1.0:
                self.marked_for_delete = True
        else:
            self.delete_progress = max(0.0, self.delete_progress - dt * 3)

        if self.grabbed:
            self.x3d = cx + self._grab_ox
            self.y3d = cy + self._grab_oy
            self.z3d = _lerp(self.z3d, hand_z_world + self._grab_oz, Z_SMOOTH_G)
            if self.snap_enabled:
                for attr in ('x3d', 'y3d'):
                    v = getattr(self, attr)
                    n = round(v / SNAP_GRID) * SNAP_GRID
                    if abs(v - n) < SNAP_THRESH:
                        setattr(self, attr, _lerp(v, n, 0.25))
        elif self.physics_enabled and self.group_id is None:
            self.x3d, self.y3d, self.z3d, drx, dry, drz = self.physics.step(
                dt, self.x3d, self.y3d, self.z3d,
                floor_y, screen_w, screen_h, gravity_on=gravity_on)
            self.rx += drx; self.ry += dry; self.rz += drz

        self.sx, self.sy, _ = _proj(self.x3d, self.y3d, self.z3d, scx, scy)

        if not self._spawning:
            gt = (1.0 if self.grabbed else
                  0.85 if self.selected else
                  0.65 if self.hovered else
                  0.10 if self.group_id else
                  0.06)
            self.glow    = _lerp(self.glow,    gt,  0.15)
            self.scale_v = _lerp(self.scale_v, 1.0, 0.18)


    def _color(self):
        if self.delete_progress > 0.05:
            t = self.delete_progress
            return tuple(int(C_DELETE[i]*t + C_IDLE[i]*(1-t)) for i in range(3))
        if self.grabbed:  return C_GRAB
        if self.selected: return C_SELECT
        if self.hovered:  return C_HOVER
        if self.group_id: return C_GROUP
        return C_IDLE

    def _fog_mul(self):
        t = max(0.0, min(1.0, self.z3d / max(Z_FAR_LIMIT, 1.0)))
        return 1.0 - t * 0.55

    def _depth_glow(self):
        t = max(0.0, min(1.0, self.z3d / max(Z_FAR_LIMIT, 1.0)))
        return 1.0 - t * 0.7

    def _draw_delete_ring(self, img, size_px):
        cx_i, cy_i = int(self.sx), int(self.sy)
        r = max(18, size_px + 8)
        angle = int(360 * self.delete_progress)
        cv2.ellipse(img, (cx_i, cy_i), (r, r), -90, 0, 360, (60,40,40), 2, cv2.LINE_AA)
        cv2.ellipse(img, (cx_i, cy_i), (r, r), -90, 0, angle, C_DELETE, 3, cv2.LINE_AA)

    def draw_axis_gizmo(self, img, screen_w, screen_h, active_axis=None):
        """Tony-Stark style colored XYZ gizmo + live dimension readout.
        active_axis: 0/1/2 or None — highlights the axis currently being resized."""
        scx, scy = screen_w / 2.0, screen_h / 2.0
        _, _, s = _proj(self.x3d, self.y3d, self.z3d, scx, scy)
        cx_i, cy_i = int(self.sx), int(self.sy)

        axes = [
            (0, "X", C_AXIS_X, (self.dim[0] * s, 0)),
            (1, "Y", C_AXIS_Y, (0, -self.dim[1] * s)),
            (2, "Z", C_AXIS_Z, (self.dim[2] * s * 0.6, self.dim[2] * s * 0.35)),
        ]
        for idx, name, color, (ex, ey) in axes:
            active = (active_axis == idx)
            lw = 3 if active else 1
            tip_color = color if active else tuple(int(c*0.55) for c in color)
            tip = (int(cx_i + ex), int(cy_i + ey))
            cv2.line(img, (cx_i, cy_i), tip, tip_color, lw, cv2.LINE_AA)
            cv2.circle(img, tip, 4 if active else 2, tip_color, -1, cv2.LINE_AA)
            if active:
                label = f"{name}: {int(self.dim[idx]*2)}px"
                cv2.putText(img, label, (tip[0] + 8, tip[1]),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.36, color, 1, cv2.LINE_AA)


    def draw_glow(self, glow_layer): pass
    def draw_crisp(self, img): pass


class _BoxShape(SpatialObject):
    def draw_glow(self, glow_layer):
        color = self._color()
        eff = (self.glow + self._spawn_glow) * self._depth_glow()
        if eff < 0.02: return
        scx, scy = glow_layer.shape[1]/2.0, glow_layer.shape[0]/2.0
        hx, hy, hz = (d * self.scale_v for d in self.dim)
        corners = _box_corners(self.x3d, self.y3d, self.z3d, hx, hy, hz,
                               self.rx, self.ry, self.rz)
        pts = []
        for cx, cy, cz in corners:
            sx, sy, _ = _proj(cx, cy, cz, scx, scy)
            pts.append((int(sx), int(sy)))
        for w in (max(1,int(12*eff)), max(1,int(6*eff)), max(1,int(3*eff))):
            for a, b in BOX_EDGES:
                cv2.line(glow_layer, pts[a], pts[b], color, w, cv2.LINE_AA)

    def draw_crisp(self, img):
        H, W = img.shape[:2]
        scx, scy = W/2.0, H/2.0
        color = self._color()
        fog   = self._fog_mul()
        hx, hy, hz = (d * self.scale_v for d in self.dim)
        corners = _box_corners(self.x3d, self.y3d, self.z3d, hx, hy, hz,
                               self.rx, self.ry, self.rz)
        face_c = tuple(int(c * 0.12) for c in color)
        pts = _draw_box_3d(img, corners, scx, scy,
                           tuple(int(c*fog) for c in color), face_c, thickness=2)

        if self.label:
            _, _, s = _proj(self.x3d, self.y3d, self.z3d, scx, scy)
            fs = max(0.24, 0.34 * s)
            (tw, th), _ = cv2.getTextSize(self.label, cv2.FONT_HERSHEY_SIMPLEX, fs, 1)
            cv2.putText(img, self.label, (int(self.sx)-tw//2, int(self.sy)+th//2),
                        cv2.FONT_HERSHEY_SIMPLEX, fs,
                        tuple(int(c*fog) for c in color), 1, cv2.LINE_AA)

        if self.delete_progress > 0.02:
            self._draw_delete_ring(img, int(max(hx, hy) * 1.0))


class CubeObject(_BoxShape):
    shape_kind = "cube"
    def __init__(self, **kwargs):
        kwargs.setdefault("dim", (45.0, 45.0, 45.0))
        kwargs.setdefault("label", "cube")
        super().__init__(**kwargs)


class PrismObject(_BoxShape):
    shape_kind = "prism"
    def __init__(self, **kwargs):
        kwargs.setdefault("dim", (60.0, 38.0, 30.0))
        kwargs.setdefault("label", "prism")
        super().__init__(**kwargs)


class PlaneObject(_BoxShape):
    shape_kind = "plane"
    def __init__(self, **kwargs):
        kwargs.setdefault("dim", (75.0, 4.0, 75.0))
        kwargs.setdefault("label", "plane")
        super().__init__(**kwargs)


class SphereObject(SpatialObject):
    shape_kind = "sphere"

    def __init__(self, **kwargs):
        kwargs.setdefault("dim", (40.0, 40.0, 40.0))
        kwargs.setdefault("label", "sphere")
        super().__init__(**kwargs)

    def _screen_r(self, scx, scy):
        _, _, s = _proj(self.x3d, self.y3d, self.z3d, scx, scy)
        rx = max(6, int(self.dim[0] * self.scale_v * s))
        ry = max(6, int(self.dim[1] * self.scale_v * s))
        return rx, ry, s

    def draw_glow(self, glow_layer):
        color = self._color()
        eff = (self.glow + self._spawn_glow) * self._depth_glow()
        if eff < 0.02: return
        H, W = glow_layer.shape[:2]
        rx, ry, _ = self._screen_r(W/2.0, H/2.0)
        cx_i, cy_i = int(self.sx), int(self.sy)
        for w in (max(1,int(14*eff)), max(1,int(8*eff)), max(1,int(4*eff))):
            cv2.ellipse(glow_layer, (cx_i, cy_i), (rx+w//2, ry+w//2), 0, 0, 360, color, w, cv2.LINE_AA)

    def draw_crisp(self, img):
        H, W = img.shape[:2]
        color = self._color()
        fog = self._fog_mul()
        rx, ry, s = self._screen_r(W/2.0, H/2.0)
        cx_i, cy_i = int(self.sx), int(self.sy)
        col_f = tuple(int(c*fog) for c in color)
        face_c = tuple(int(c*0.15) for c in color)

        cv2.ellipse(img, (cx_i, cy_i), (rx, ry), 0, 0, 360, face_c, -1, cv2.LINE_AA)
        cv2.ellipse(img, (cx_i, cy_i), (rx, ry), 0, 0, 360, col_f, 2, cv2.LINE_AA)
        if ry > 10:
            cv2.ellipse(img, (cx_i, cy_i), (rx, ry//3), 0, 0, 360,
                        tuple(int(c*0.5*fog) for c in color), 1, cv2.LINE_AA)
            cv2.ellipse(img, (cx_i, cy_i), (rx//3, ry), 0, 0, 360,
                        tuple(int(c*0.5*fog) for c in color), 1, cv2.LINE_AA)

        if self.label:
            fs = max(0.24, 0.34 * s)
            (tw, th), _ = cv2.getTextSize(self.label, cv2.FONT_HERSHEY_SIMPLEX, fs, 1)
            cv2.putText(img, self.label, (cx_i-tw//2, cy_i+th//2),
                        cv2.FONT_HERSHEY_SIMPLEX, fs, col_f, 1, cv2.LINE_AA)

        if self.delete_progress > 0.02:
            self._draw_delete_ring(img, rx)


class CylinderObject(SpatialObject):
    shape_kind = "cylinder"

    def __init__(self, **kwargs):
        kwargs.setdefault("dim", (32.0, 55.0, 32.0))
        kwargs.setdefault("label", "cylinder")
        super().__init__(**kwargs)

    def _metrics(self, scx, scy):
        _, _, s = _proj(self.x3d, self.y3d, self.z3d, scx, scy)
        rx = max(8, int(self.dim[0] * self.scale_v * s))
        hh = max(8, int(self.dim[1] * self.scale_v * s))
        cap = max(3, int(self.dim[2] * self.scale_v * s * 0.35))
        return rx, hh, cap, s

    def draw_glow(self, glow_layer):
        color = self._color()
        eff = (self.glow + self._spawn_glow) * self._depth_glow()
        if eff < 0.02: return
        H, W = glow_layer.shape[:2]
        rx, hh, cap, _ = self._metrics(W/2.0, H/2.0)
        cx_i, cy_i = int(self.sx), int(self.sy)
        for w in (max(1,int(12*eff)), max(1,int(6*eff)), max(1,int(3*eff))):
            cv2.ellipse(glow_layer, (cx_i, cy_i-hh), (rx, cap), 0, 0, 360, color, w, cv2.LINE_AA)
            cv2.ellipse(glow_layer, (cx_i, cy_i+hh), (rx, cap), 0, 0, 360, color, w, cv2.LINE_AA)
            cv2.line(glow_layer, (cx_i-rx, cy_i-hh), (cx_i-rx, cy_i+hh), color, w, cv2.LINE_AA)
            cv2.line(glow_layer, (cx_i+rx, cy_i-hh), (cx_i+rx, cy_i+hh), color, w, cv2.LINE_AA)

    def draw_crisp(self, img):
        H, W = img.shape[:2]
        color = self._color()
        fog = self._fog_mul()
        rx, hh, cap, s = self._metrics(W/2.0, H/2.0)
        cx_i, cy_i = int(self.sx), int(self.sy)
        col_f = tuple(int(c*fog) for c in color)
        face_c = tuple(int(c*0.12) for c in color)

        body = np.array([
            (cx_i-rx, cy_i-hh), (cx_i+rx, cy_i-hh),
            (cx_i+rx, cy_i+hh), (cx_i-rx, cy_i+hh),
        ], dtype=np.int32)
        ov = img.copy()
        cv2.fillConvexPoly(ov, body, face_c)
        cv2.addWeighted(ov, 0.16, img, 0.84, 0, img)

        cv2.line(img, (cx_i-rx, cy_i-hh), (cx_i-rx, cy_i+hh), col_f, 2, cv2.LINE_AA)
        cv2.line(img, (cx_i+rx, cy_i-hh), (cx_i+rx, cy_i+hh), col_f, 2, cv2.LINE_AA)
        cv2.ellipse(img, (cx_i, cy_i-hh), (rx, cap), 0, 0, 360, col_f, 2, cv2.LINE_AA)
        cv2.ellipse(img, (cx_i, cy_i+hh), (rx, cap), 0, 180, 360, col_f, 2, cv2.LINE_AA)

        if self.label:
            fs = max(0.24, 0.34 * s)
            (tw, th), _ = cv2.getTextSize(self.label, cv2.FONT_HERSHEY_SIMPLEX, fs, 1)
            cv2.putText(img, self.label, (cx_i-tw//2, cy_i+th//2),
                        cv2.FONT_HERSHEY_SIMPLEX, fs, col_f, 1, cv2.LINE_AA)

        if self.delete_progress > 0.02:
            self._draw_delete_ring(img, rx)


class Group:
    def __init__(self, members):
        self.id = _next_id()
        self.members = list(members)
        for m in self.members:
            m.group_id = self.id
            m.selected = False

    def centroid(self):
        n = max(1, len(self.members))
        xs = sum(m.x3d for m in self.members) / n
        ys = sum(m.y3d for m in self.members) / n
        zs = sum(m.z3d for m in self.members) / n
        return xs, ys, zs

    def ungroup(self):
        for m in self.members:
            m.group_id = None
        members = self.members
        self.members = []
        return members


SHAPE_FACTORY = {
    "cube":     CubeObject,
    "sphere":   SphereObject,
    "cylinder": CylinderObject,
    "prism":    PrismObject,
    "plane":    PlaneObject,
}