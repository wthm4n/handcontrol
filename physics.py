"""
physics.py — Phase 4: Full 3D throw physics for Aether.

Changes from Phase 3:
  - HandVelocityTracker now tracks (cx, cy, cz) — includes depth.
  - PhysicsBody upgraded from 2D (vx,vy) to 3D (vx,vy,vz).
  - Floor collision lives in world-Y space; Z is unconstrained by floor.
  - Z damping: depth velocity decays faster than XY (no Z "gravity").
  - Sleep check extended to include vz.
  - debug_str() shows all three velocity components.

Coordinate conventions (matches cube.py):
  X  — screen right   (pixels from left edge of frame)
  Y  — screen down    (pixels from top edge of frame)
  Z  — depth          (positive = further from camera, negative = closer)
  Gravity acts in +Y only.
"""

from collections import deque
import math


# ── Tuneable constants ────────────────────────────────────────────────

GRAVITY         = 900.0   # px/s²  Y-axis only
AIR_DAMPING_XY  = 0.985   # per-frame multiplier for X/Y velocity
AIR_DAMPING_Z   = 0.970   # Z damps faster — depth throws feel snappier
FLOOR_BOUNCE    = 0.52    # energy kept on floor hit
FLOOR_FRICTION  = 0.78    # X/Z multiplier on floor bounce
SPIN_DAMPING    = 0.97    # angular velocity decay
SPIN_TRANSFER   = 0.04    # linear speed → angular bleed on throw
VELOCITY_SMOOTH = 0.50    # EMA weight for velocity estimate
VELOCITY_WINDOW = 6       # rolling-window size (frames)
THROW_SCALE_XY  = 1.20   # amplify X/Y throw
THROW_SCALE_Z   = 1.50   # amplify Z throw a bit more (depth less sensitive)
MIN_BOUNCE_VY   = 40.0   # below this Y speed, stop bouncing
WALL_BOUNCE     = 0.42   # energy kept on side-wall hit
Z_NEAR_BOUNCE   = 0.35   # energy kept when hitting the near-Z wall
Z_NEAR_LIMIT    = -300.0 # world-Z: closest the cube can come (px)
Z_FAR_LIMIT     = 800.0  # world-Z: furthest the cube can go (px)


class HandVelocityTracker:
    """
    Rolling-window 3D velocity tracker.
    Records (cx, cy, cz, t) each frame while grabbing.
    On release, returns smoothed (vx, vy, vz) in world-px/s.
    """

    def __init__(self):
        self._buf             = deque(maxlen=VELOCITY_WINDOW)
        self._smoothed_vx     = 0.0
        self._smoothed_vy     = 0.0
        self._smoothed_vz     = 0.0

    def reset(self):
        self._buf.clear()
        self._smoothed_vx = 0.0
        self._smoothed_vy = 0.0
        self._smoothed_vz = 0.0

    def record(self, cx, cy, cz, t):
        """
        cx, cy  — screen pixels
        cz      — world-Z of the hand (same units as cube.z3d, i.e. pixels)
        t       — timestamp (seconds)
        """
        self._buf.append((cx, cy, cz, t))

        if len(self._buf) >= 2:
            x0, y0, z0, t0 = self._buf[0]
            x1, y1, z1, t1 = self._buf[-1]
            dt = t1 - t0
            if dt > 1e-4:
                w = VELOCITY_SMOOTH
                self._smoothed_vx = w*(x1-x0)/dt + (1-w)*self._smoothed_vx
                self._smoothed_vy = w*(y1-y0)/dt + (1-w)*self._smoothed_vy
                self._smoothed_vz = w*(z1-z0)/dt + (1-w)*self._smoothed_vz

    def release_velocity(self):
        """Return (vx, vy, vz) in world-px/s."""
        vx = self._smoothed_vx * THROW_SCALE_XY
        vy = self._smoothed_vy * THROW_SCALE_XY
        vz = self._smoothed_vz * THROW_SCALE_Z
        self.reset()
        return vx, vy, vz

    @property
    def current_velocity(self):
        return self._smoothed_vx, self._smoothed_vy, self._smoothed_vz


class PhysicsBody:
    """
    Full 3D physics body attached to each Cube.

    step() returns (new_x, new_y, new_z, drx, dry, drz).
    The cube applies position directly and accumulates rotation deltas.

    Floor is a world-Y plane. Z has soft near/far limits.
    No gravity in Z — depth throws just coast with drag.
    """

    def __init__(self):
        self.vx  = 0.0   # px/s — screen right
        self.vy  = 0.0   # px/s — screen down
        self.vz  = 0.0   # px/s — depth (positive = away)
        self.vrx = 0.0   # rad/s
        self.vry = 0.0   # rad/s
        self.vrz = 0.0   # rad/s
        self.on_floor = False
        self.sleeping = False

    # ── Launch / stop ─────────────────────────────────────────────────

    def launch(self, vx, vy, vz=0.0):
        self.vx = vx
        self.vy = vy
        self.vz = vz

        # Spin bleed from XY speed
        speed_xy = math.sqrt(vx*vx + vy*vy)
        sign_x   = 1.0 if vx >= 0 else -1.0
        self.vrz += sign_x * speed_xy * SPIN_TRANSFER
        self.vry += vx * SPIN_TRANSFER * 0.5
        self.vrx += vy * SPIN_TRANSFER * 0.3

        # Z throw also creates yaw spin
        self.vry += vz * SPIN_TRANSFER * 0.4

        self.on_floor = False
        self.sleeping = False

    def stop(self):
        self.vx = self.vy = self.vz = 0.0
        self.vrx = self.vry = self.vrz = 0.0
        self.on_floor = False
        self.sleeping = False

    # ── Step ──────────────────────────────────────────────────────────

    def step(self, dt, x3d, y3d, z3d,
             floor_y, screen_w, screen_h,
             gravity_on=True):
        """
        Advance physics one frame.
        Returns (new_x, new_y, new_z, drx, dry, drz).
        """
        if self.sleeping:
            return x3d, y3d, z3d, 0.0, 0.0, 0.0

        # ── Gravity (Y only) ──────────────────────────────────────────
        if gravity_on and not self.on_floor:
            self.vy += GRAVITY * dt

        # ── Air drag ──────────────────────────────────────────────────
        damp_xy = AIR_DAMPING_XY ** (dt * 60)
        damp_z  = AIR_DAMPING_Z  ** (dt * 60)
        self.vx *= damp_xy
        self.vy *= damp_xy
        self.vz *= damp_z

        # ── Integrate ─────────────────────────────────────────────────
        new_x = x3d + self.vx * dt
        new_y = y3d + self.vy * dt
        new_z = z3d + self.vz * dt

        # ── Floor (world-Y plane) ─────────────────────────────────────
        if new_y >= floor_y:
            new_y = floor_y
            if abs(self.vy) < MIN_BOUNCE_VY:
                self.vy = 0.0
                self.on_floor = True
            else:
                self.vy      = -abs(self.vy) * FLOOR_BOUNCE
                self.on_floor = False
            self.vx *= FLOOR_FRICTION
            self.vz *= FLOOR_FRICTION   # also slows depth on floor bounce

        # ── Side walls (world-X) ──────────────────────────────────────
        margin = 30
        if new_x < margin:
            new_x    = margin
            self.vx  = abs(self.vx) * WALL_BOUNCE
        elif new_x > screen_w - margin:
            new_x    = screen_w - margin
            self.vx  = -abs(self.vx) * WALL_BOUNCE

        # ── Depth limits (world-Z) ────────────────────────────────────
        if new_z < Z_NEAR_LIMIT:
            new_z    = Z_NEAR_LIMIT
            self.vz  = abs(self.vz) * Z_NEAR_BOUNCE   # bounce back from near wall
        elif new_z > Z_FAR_LIMIT:
            new_z    = Z_FAR_LIMIT
            self.vz  = -abs(self.vz) * Z_NEAR_BOUNCE

        # ── Spin decay ────────────────────────────────────────────────
        sdamp    = SPIN_DAMPING ** (dt * 60)
        self.vrx *= sdamp
        self.vry *= sdamp
        self.vrz *= sdamp

        if self.on_floor:
            self.vrx *= 0.90
            self.vry *= 0.90
            self.vrz *= 0.90

        drx = self.vrx * dt
        dry = self.vry * dt
        drz = self.vrz * dt

        # ── Sleep ─────────────────────────────────────────────────────
        lin  = math.sqrt(self.vx**2 + self.vy**2 + self.vz**2)
        ang  = math.sqrt(self.vrx**2 + self.vry**2 + self.vrz**2)
        if self.on_floor and lin < 2.0 and ang < 0.005:
            self.vx = self.vy = self.vz = 0.0
            self.vrx = self.vry = self.vrz = 0.0
            self.sleeping = True

        return new_x, new_y, new_z, drx, dry, drz

    def debug_str(self):
        spd = math.sqrt(self.vx**2 + self.vy**2 + self.vz**2)
        return (f"v=({self.vx:+.0f},{self.vy:+.0f},{self.vz:+.0f}) "
                f"spd={spd:.0f}  floor={self.on_floor}  sleep={self.sleeping}")
