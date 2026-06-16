"""
physics.py — Phase 3: Throw physics for Aether.

Responsibilities:
  - Track hand velocity while a cube is grabbed (using a short rolling window).
  - On release, inject that velocity into the cube's PhysicsBody.
  - Each frame, step the PhysicsBody: gravity, air damping, floor bounce.
  - Angular momentum: spin rate decays independently.

Design notes:
  - Completely decoupled from rendering; Cube owns a PhysicsBody.
  - All units are screen-pixels and seconds.
  - Gravity acts in +Y (down, because screen Y grows downward).
  - The "floor" is at a configurable Y pixel value (default = bottom of frame).
  - Wall bounds optionally keep cubes on screen.
"""

from collections import deque
import math


# ── Tuneable constants ────────────────────────────────────────────────
# Override these at runtime if you want a config panel later.

GRAVITY          = 980.0   # px/s²  (~real gravity scaled to screen)
AIR_DAMPING      = 0.985   # velocity multiplier per frame (1.0 = no drag)
FLOOR_BOUNCE     = 0.55    # energy retained on floor hit (0=dead stop, 1=elastic)
FLOOR_FRICTION   = 0.80    # horizontal velocity multiplier on floor bounce
SPIN_DAMPING     = 0.97    # angular velocity multiplier per frame
SPIN_TRANSFER    = 0.04    # how much linear throw speed bleeds into spin
VELOCITY_SMOOTH  = 0.5     # lerp weight for velocity estimate (lower=smoother)
VELOCITY_WINDOW  = 6       # frames kept in rolling velocity buffer
THROW_SCALE      = 1.2     # amplify throw velocity slightly for feel
MIN_BOUNCE_VY    = 40.0    # below this speed, stop bouncing (rest on floor)
WALL_BOUNCE      = 0.45    # energy retained on wall hit


class HandVelocityTracker:
    """
    Maintains a rolling window of (cx, cy, timestamp) samples while a cube
    is grabbed. Call .record() every frame, .release_velocity() on drop.
    """

    def __init__(self):
        self._buf = deque(maxlen=VELOCITY_WINDOW)
        self._smoothed_vx = 0.0
        self._smoothed_vy = 0.0

    def reset(self):
        self._buf.clear()
        self._smoothed_vx = 0.0
        self._smoothed_vy = 0.0

    def record(self, cx, cy, t):
        self._buf.append((cx, cy, t))

        if len(self._buf) >= 2:
            # Compute velocity over the whole window for stability
            x0, y0, t0 = self._buf[0]
            x1, y1, t1 = self._buf[-1]
            dt = t1 - t0
            if dt > 1e-4:
                raw_vx = (x1 - x0) / dt
                raw_vy = (y1 - y0) / dt
                # Exponential smoothing so brief jitter doesn't corrupt the throw
                w = VELOCITY_SMOOTH
                self._smoothed_vx = w * raw_vx + (1 - w) * self._smoothed_vx
                self._smoothed_vy = w * raw_vy + (1 - w) * self._smoothed_vy

    def release_velocity(self):
        """Return (vx, vy) in px/s to apply on throw."""
        vx = self._smoothed_vx * THROW_SCALE
        vy = self._smoothed_vy * THROW_SCALE
        self.reset()
        return vx, vy

    @property
    def current_velocity(self):
        return self._smoothed_vx, self._smoothed_vy


class PhysicsBody:
    """
    Attached to each Cube. Holds velocity, spin, and stepping logic.
    The cube calls body.step(dt, floor_y, screen_w) each frame when not grabbed.
    When grabbed, the cube bypasses physics entirely (position is hand-driven).
    """

    def __init__(self):
        self.vx  = 0.0   # px/s
        self.vy  = 0.0   # px/s
        self.vrx = 0.0   # rad/s  pitch
        self.vry = 0.0   # rad/s  yaw
        self.vrz = 0.0   # rad/s  roll
        self.on_floor  = False
        self.sleeping  = False   # True when nearly at rest; skip stepping

    def launch(self, vx, vy):
        """Called on release. Seeds linear and spin velocity from throw."""
        self.vx  = vx
        self.vy  = vy
        # Throw speed bleeds into spin (faster throw = more tumble)
        speed = math.sqrt(vx*vx + vy*vy)
        sign  = 1.0 if vx >= 0 else -1.0
        self.vrz += sign * speed * SPIN_TRANSFER
        self.vry += vx * SPIN_TRANSFER * 0.5
        self.vrx += vy * SPIN_TRANSFER * 0.3
        self.on_floor = False
        self.sleeping = False

    def stop(self):
        """Zero everything (e.g. grabbed)."""
        self.vx = self.vy = 0.0
        self.vrx = self.vry = self.vrz = 0.0
        self.on_floor = False
        self.sleeping = False

    def step(self, dt, x3d, y3d, floor_y, screen_w, screen_h, gravity_on=True):
        """
        Advance physics by dt seconds.
        Returns new (x3d, y3d, drx, dry, drz) — position deltas + rotation deltas.
        Caller applies these to the cube.
        """
        if self.sleeping:
            return x3d, y3d, 0.0, 0.0, 0.0

        # ── Gravity ───────────────────────────────────────────────────
        if gravity_on and not self.on_floor:
            self.vy += GRAVITY * dt

        # ── Air drag ──────────────────────────────────────────────────
        damp = AIR_DAMPING ** (dt * 60)   # frame-rate independent
        self.vx *= damp
        self.vy *= damp

        # ── Integrate position ────────────────────────────────────────
        new_x = x3d + self.vx * dt
        new_y = y3d + self.vy * dt

        # ── Floor collision ───────────────────────────────────────────
        if new_y >= floor_y:
            new_y = floor_y
            if abs(self.vy) < MIN_BOUNCE_VY:
                self.vy = 0.0
                self.on_floor = True
            else:
                self.vy = -abs(self.vy) * FLOOR_BOUNCE
                self.on_floor = False
            self.vx *= FLOOR_FRICTION

        # ── Wall collisions (left / right) ───────────────────────────
        margin = 30
        if new_x < margin:
            new_x = margin
            self.vx = abs(self.vx) * WALL_BOUNCE
        elif new_x > screen_w - margin:
            new_x = screen_w - margin
            self.vx = -abs(self.vx) * WALL_BOUNCE

        # ── Spin decay ────────────────────────────────────────────────
        sdamp = SPIN_DAMPING ** (dt * 60)
        self.vrx *= sdamp
        self.vry *= sdamp
        self.vrz *= sdamp

        # Extra spin friction when on floor
        if self.on_floor:
            self.vrx *= 0.92
            self.vry *= 0.92
            self.vrz *= 0.92

        drx = self.vrx * dt
        dry = self.vry * dt
        drz = self.vrz * dt

        # ── Sleep check ───────────────────────────────────────────────
        lin_speed = math.sqrt(self.vx**2 + self.vy**2)
        ang_speed = math.sqrt(self.vrx**2 + self.vry**2 + self.vrz**2)
        if self.on_floor and lin_speed < 2.0 and ang_speed < 0.005:
            self.vx = self.vy = 0.0
            self.vrx = self.vry = self.vrz = 0.0
            self.sleeping = True

        return new_x, new_y, drx, dry, drz

    def debug_str(self):
        speed = math.sqrt(self.vx**2 + self.vy**2)
        return (f"v=({self.vx:+.0f},{self.vy:+.0f}) spd={speed:.0f}  "
                f"floor={self.on_floor}  sleep={self.sleeping}")