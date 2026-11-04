"""Kinematic model of the TurtleBot3 Burger (2-D unicycle).

Integration method
------------------
For constant ``(v, omega)`` over a duration ``t`` the pose is integrated
*exactly* along the circular arc (no Euler drift)::

    omega != 0:  x' = x + (v/omega) * (sin(theta + omega t) - sin(theta))
                 y' = y + (v/omega) * (cos(theta) - cos(theta + omega t))
    omega == 0:  x' = x + v cos(theta) t,   y' = y + v sin(theta) t
    theta'      = wrap(theta + omega t)

The brief's Euler formula is the ``t -> 0`` limit of this and agrees with it to
O(omega * t) per step.  Headings are wrapped to ``[-pi, pi)``.

This is a *kinematic* model: velocity changes instantly, there is no wheel
slip, inertia, motor lag or Gazebo contact physics.
"""

from __future__ import annotations

import math
from typing import Sequence, Tuple

import numpy as np

from .config import MotionConfig, RobotConfig


class InvalidActionError(ValueError):
    """Raised when an action id is not a valid index of the action table."""


def wrap_angle(angle: float) -> float:
    """Wrap an angle to ``[-pi, pi)``."""
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def arc_pose(x: float, y: float, theta: float, v: float, omega: float, t: float
             ) -> Tuple[float, float, float]:
    """Exact pose after driving constant ``(v, omega)`` for ``t`` seconds."""
    if abs(omega) < 1e-9:
        return x + v * math.cos(theta) * t, y + v * math.sin(theta) * t, wrap_angle(theta + omega * t)
    th1 = theta + omega * t
    r = v / omega
    return (x + r * (math.sin(th1) - math.sin(theta)),
            y + r * (math.cos(theta) - math.cos(th1)),
            wrap_angle(th1))


class ActionTable:
    """Maps discrete action ids to ``(v, omega)``; validates ids."""

    def __init__(self, actions: Sequence[Sequence[float]]):
        self._table = tuple((float(v), float(w)) for v, w in actions)

    def __len__(self) -> int:
        return len(self._table)

    @property
    def table(self) -> Tuple[Tuple[float, float], ...]:
        return self._table

    def __getitem__(self, action) -> Tuple[float, float]:
        return self._table[self.validate(action)]

    def validate(self, action) -> int:
        if isinstance(action, (bool, np.bool_)) or not isinstance(action, (int, np.integer)):
            raise InvalidActionError(
                f"Action must be an integer in [0, {len(self._table) - 1}], "
                f"got {action!r} ({type(action).__name__})")
        a = int(action)
        if not 0 <= a < len(self._table):
            raise InvalidActionError(
                f"Action {a} is out of range; valid actions are 0..{len(self._table) - 1}")
        return a


class Robot:
    """Pose + velocity state with exact-arc motion and a swept-path generator."""

    __slots__ = ("cfg", "motion", "x", "y", "theta", "v", "omega", "collided", "friction")

    def __init__(self, robot_cfg: RobotConfig, motion_cfg: MotionConfig):
        self.cfg = robot_cfg
        self.motion = motion_cfg
        self.x = self.y = self.theta = 0.0
        self.v = self.omega = 0.0
        self.collided = False
        self.friction = float(motion_cfg.friction)

    def reset(self, x: float, y: float, theta: float) -> None:
        self.x, self.y, self.theta = float(x), float(y), wrap_angle(float(theta))
        self.v = self.omega = 0.0
        self.collided = False

    @property
    def pose(self) -> Tuple[float, float, float]:
        return self.x, self.y, self.theta

    def wheel_speeds(self) -> Tuple[float, float]:
        """(left, right) wheel angular velocity in rad/s for the current twist."""
        half = 0.5 * self.cfg.wheel_separation
        return ((self.v - self.omega * half) / self.cfg.wheel_radius,
                (self.v + self.omega * half) / self.cfg.wheel_radius)

    # ---- motion ------------------------------------------------------
    def plan(self, v: float, omega: float, dt: float):
        """Return ``(end_pose, path_points, sagitta)`` for one sub-step.

        ``path_points`` is an ``(n+1, 2)`` polyline along the arc (including
        the start); each chord is at most ``max_chord_fraction * radius`` long
        so the world's swept test cannot tunnel.  ``sagitta`` is an upper bound
        on how far the true arc strays from the polyline.
        """
        end = arc_pose(self.x, self.y, self.theta, v, omega, dt)
        length = abs(v) * dt
        max_chord = self.motion.max_chord_fraction * self.cfg.collision_radius
        n = max(1, int(math.ceil(length / max_chord)))
        ts = np.linspace(0.0, dt, n + 1)
        if abs(omega) < 1e-9:
            c, s = math.cos(self.theta), math.sin(self.theta)
            pts = np.stack([self.x + v * c * ts, self.y + v * s * ts], axis=1)
            sag = 0.0
        else:
            th1 = self.theta + omega * ts
            r = v / omega
            pts = np.stack([self.x + r * (np.sin(th1) - math.sin(self.theta)),
                            self.y + r * (math.cos(self.theta) - np.cos(th1))], axis=1)
            chord = length / n
            rad = abs(r)
            sag = rad * (1.0 - math.cos(min(chord / (2.0 * rad), math.pi / 2))) if rad > 0 else 0.0
        return end, pts, sag

    def inertial_twist(self, action: str, dt: float) -> Tuple[float, float]:
        """Inertial model (RL_BTP env.Robot.step): apply the acceleration command,
        decay both velocities by friction, then clip.  Does not move the robot;
        the result is fed to plan()/commit() like a table action."""
        m = self.motion
        v, w = self.v, self.omega
        if action == "FORWARD":
            v += m.lin_accel * dt
        elif action == "BACKWARD":
            v -= m.lin_accel * dt
        elif action == "LEFT":
            w += m.ang_accel * dt
        elif action == "RIGHT":
            w -= m.ang_accel * dt
        elif action != "STOP":                      # STOP = coast under friction
            raise InvalidActionError(f"unknown accel action {action!r}")
        v *= 1.0 - self.friction
        w *= 1.0 - self.friction
        v = min(max(v, m.v_min), m.v_max)
        w = min(max(w, -m.w_max), m.w_max)
        return float(v), float(w)

    def commit(self, pose: Tuple[float, float, float], v: float, omega: float) -> None:
        self.x, self.y, self.theta = pose
        self.v, self.omega = v, omega

    def block(self) -> None:
        """Motion was blocked by a collision: keep the pose, zero the velocity."""
        self.v = self.omega = 0.0
        self.collided = True
