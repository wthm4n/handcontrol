"""
physics.py — Phase 5: Full 3D throw physics for Aether.

Phase 5 additions:
  - HandDepthCalibrator: baseline calibration + dead zone + hysteresis for Z.
  - HandVelocityTracker extended with z-velocity/acceleration accessors.
  - PhysicsBody: no structural changes, same API.

Coordinate conventions (matches cube.py):
  X  — screen right   (pixels from left edge of frame)
  Y  — screen down    (pixels from top edge of frame)
  Z  — depth          (positive = further from camera, negative = closer)
  Gravity acts in +Y only.
"""

from collections import deque
import math


GRAVITY         = 900.0
AIR_DAMPING_XY  = 0.985
AIR_DAMPING_Z   = 0.970
FLOOR_BOUNCE    = 0.52
FLOOR_FRICTION  = 0.78
SPIN_DAMPING    = 0.97
SPIN_TRANSFER   = 0.04
VELOCITY_SMOOTH = 0.50
VELOCITY_WINDOW = 6
THROW_SCALE_XY  = 1.20
THROW_SCALE_Z   = 1.50
MIN_BOUNCE_VY   = 40.0
WALL_BOUNCE     = 0.42
Z_NEAR_BOUNCE   = 0.35
Z_NEAR_LIMIT    = -300.0
Z_FAR_LIMIT     = 800.0


DEPTH_CALIB_FRAMES  = 45
DEPTH_DEAD_ZONE     = 12.0
DEPTH_HYSTERESIS    = 6.0
DEPTH_DRIFT_DECAY   = 0.003


class HandDepthCalibrator:
    """
    Converts raw MediaPipe wrist-Z to stable world-Z depth.

    Phase 5 improvements over direct Z_SCALE multiplication:
      • Establishes a neutral baseline from the first N frames.
      • Computes relative depth from that baseline (eliminates person-to-
        person distance variation).
      • Dead zone + hysteresis kills micro-jitter.
      • Slow baseline drift correction keeps depth centred over time.
    """

    def __init__(self, z_scale=600.0, smooth=0.18):
        self.z_scale   = z_scale
        self.smooth    = smooth


        self._calib_buf   = deque(maxlen=DEPTH_CALIB_FRAMES)
        self._calibrated  = False
        self._baseline    = 0.0


        self._smooth_z    = 0.0
        self._in_deadzone = True
        self._dz_anchor   = 0.0


        self._prev_z      = 0.0
        self._prev_t      = None
        self._vel_z       = 0.0
        self._acc_z       = 0.0

    def update(self, raw_mp_z, t):
        """
        raw_mp_z : MediaPipe wrist Z (normalised, ~-0.15..+0.15)
        t        : current timestamp (seconds)
        Returns  : stable world-Z (pixels)
        """

        self._calib_buf.append(raw_mp_z)
        if not self._calibrated:
            if len(self._calib_buf) >= DEPTH_CALIB_FRAMES:
                self._baseline   = sum(self._calib_buf) / len(self._calib_buf)
                self._calibrated = True
            else:

                return 0.0


        relative_mp_z = raw_mp_z - self._baseline
        raw_world_z   = relative_mp_z * self.z_scale


        if len(self._calib_buf) >= DEPTH_CALIB_FRAMES:
            running_mean = sum(self._calib_buf) / len(self._calib_buf)
            self._baseline += (running_mean - self._baseline) * DEPTH_DRIFT_DECAY


        self._smooth_z = (self.smooth * raw_world_z +
                          (1.0 - self.smooth) * self._smooth_z)
        sz = self._smooth_z


        if self._in_deadzone:
            if abs(sz - self._dz_anchor) > DEPTH_DEAD_ZONE + DEPTH_HYSTERESIS:
                self._in_deadzone = False
        else:
            if abs(sz - self._dz_anchor) < DEPTH_DEAD_ZONE:
                self._in_deadzone = True
                self._dz_anchor   = sz

        if self._in_deadzone:
            output_z = self._dz_anchor
        else:

            direction = 1.0 if sz > self._dz_anchor else -1.0
            output_z  = self._dz_anchor + direction * (abs(sz - self._dz_anchor) - DEPTH_DEAD_ZONE)


        if self._prev_t is not None:
            dt = max(t - self._prev_t, 1e-4)
            inst_vel    = (output_z - self._prev_z) / dt
            prev_vel    = self._vel_z
            self._vel_z = 0.4 * inst_vel + 0.6 * self._vel_z
            self._acc_z = (self._vel_z - prev_vel) / dt
        self._prev_z = output_z
        self._prev_t = t

        return output_z

    @property
    def vel_z(self):
        return self._vel_z

    @property
    def acc_z(self):
        return self._acc_z

    @property
    def is_calibrated(self):
        return self._calibrated

    @property
    def calibration_progress(self):
        return min(1.0, len(self._calib_buf) / DEPTH_CALIB_FRAMES)

    def reset(self):
        self._calib_buf.clear()
        self._calibrated  = False
        self._baseline    = 0.0
        self._smooth_z    = 0.0
        self._in_deadzone = True
        self._dz_anchor   = 0.0
        self._prev_z      = 0.0
        self._prev_t      = None
        self._vel_z       = 0.0
        self._acc_z       = 0.0


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
    Unchanged from Phase 4 — same API, same behaviour.
    """

    def __init__(self):
        self.vx  = 0.0
        self.vy  = 0.0
        self.vz  = 0.0
        self.vrx = 0.0
        self.vry = 0.0
        self.vrz = 0.0
        self.on_floor = False
        self.sleeping = False

    def launch(self, vx, vy, vz=0.0):
        self.vx = vx
        self.vy = vy
        self.vz = vz
        speed_xy = math.sqrt(vx*vx + vy*vy)
        sign_x   = 1.0 if vx >= 0 else -1.0
        self.vrz += sign_x * speed_xy * SPIN_TRANSFER
        self.vry += vx * SPIN_TRANSFER * 0.5
        self.vrx += vy * SPIN_TRANSFER * 0.3
        self.vry += vz * SPIN_TRANSFER * 0.4
        self.on_floor = False
        self.sleeping = False

    def stop(self):
        self.vx = self.vy = self.vz = 0.0
        self.vrx = self.vry = self.vrz = 0.0
        self.on_floor = False
        self.sleeping = False

    def step(self, dt, x3d, y3d, z3d,
             floor_y, screen_w, screen_h,
             gravity_on=True):
        if self.sleeping:
            return x3d, y3d, z3d, 0.0, 0.0, 0.0

        if gravity_on and not self.on_floor:
            self.vy += GRAVITY * dt

        damp_xy = AIR_DAMPING_XY ** (dt * 60)
        damp_z  = AIR_DAMPING_Z  ** (dt * 60)
        self.vx *= damp_xy
        self.vy *= damp_xy
        self.vz *= damp_z

        new_x = x3d + self.vx * dt
        new_y = y3d + self.vy * dt
        new_z = z3d + self.vz * dt

        if new_y >= floor_y:
            new_y = floor_y
            if abs(self.vy) < MIN_BOUNCE_VY:
                self.vy = 0.0
                self.on_floor = True
            else:
                self.vy      = -abs(self.vy) * FLOOR_BOUNCE
                self.on_floor = False
            self.vx *= FLOOR_FRICTION
            self.vz *= FLOOR_FRICTION

        margin = 30
        if new_x < margin:
            new_x    = margin
            self.vx  = abs(self.vx) * WALL_BOUNCE
        elif new_x > screen_w - margin:
            new_x    = screen_w - margin
            self.vx  = -abs(self.vx) * WALL_BOUNCE

        if new_z < Z_NEAR_LIMIT:
            new_z    = Z_NEAR_LIMIT
            self.vz  = abs(self.vz) * Z_NEAR_BOUNCE
        elif new_z > Z_FAR_LIMIT:
            new_z    = Z_FAR_LIMIT
            self.vz  = -abs(self.vz) * Z_NEAR_BOUNCE

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