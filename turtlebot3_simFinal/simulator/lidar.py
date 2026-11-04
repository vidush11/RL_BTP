"""Geometric 2-D LiDAR (ray casting) -- no Gazebo plugin, no ROS message.

Beam convention (matches the existing ROS ``LaserScan`` usage in
``dqn_environment.py``: ``front_distance = scan.ranges[0]``,
``angle = angle_min + i * angle_increment``)
-----------------------------------------------------------------------------
* Beam ``i`` points at robot-frame angle ``angle_min + i * increment``,
  measured counter-clockwise from the robot's +x axis (straight ahead).
* Defaults: ``angle_min = 0``, 360 deg, ``increment = 2*pi / num_beams`` (15 deg
  for 24 beams, so beam 0 = ahead, beam 6 = left, beam 12 = behind, beam 18 =
  right).  For ``fov < 360`` the beams span ``[angle_min, angle_min + fov]``
  inclusive, i.e. ``increment = fov / (num_beams - 1)``.
* World direction of beam ``i`` = robot heading + its robot-frame angle.

For every beam the nearest intersection with the boundary (from the inside) and
with every obstacle rectangle (slab method) is found, the distance is clipped to
``max_range`` (rays that hit nothing within range read ``max_range``), and the
normalised reading is ``min(d, max_range) / max_range`` in ``[0, 1]``.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np

from .config import LidarConfig
from .world import World

_EPS = 1e-12


class Lidar:
    def __init__(self, cfg: LidarConfig):
        self.cfg = cfg
        n = int(cfg.num_beams)
        fov = math.radians(cfg.fov_deg)
        if cfg.fov_deg >= 360.0 or n == 1:
            inc = fov / n
        else:
            inc = fov / (n - 1)
        self.num_beams = n
        self.max_range = float(cfg.max_range)
        self.angles = math.radians(cfg.angle_min_deg) + np.arange(n) * inc   # robot frame
        self._cos = np.cos(self.angles)
        self._sin = np.sin(self.angles)

    @property
    def ros_angles(self) -> np.ndarray:
        """Beam angles wrapped to ``[0, 2*pi)`` like a ROS scan with angle_min=0."""
        return np.mod(self.angles, 2.0 * math.pi)

    def cast(self, world: World, x: float, y: float, theta: float,
             rng: Optional[np.random.Generator] = None) -> np.ndarray:
        """Distances in metres, shape ``(num_beams,)``, float64, in ``[0, max_range]``."""
        ct, st = math.cos(theta), math.sin(theta)
        dx = self._cos * ct - self._sin * st
        dy = self._sin * ct + self._cos * st
        dxs = np.where(np.abs(dx) < _EPS, _EPS, dx)
        dys = np.where(np.abs(dy) < _EPS, _EPS, dy)

        # Boundary, seen from the inside.
        tx = np.where(dxs > 0.0, (world.xmax - x) / dxs, (world.xmin - x) / dxs)
        ty = np.where(dys > 0.0, (world.ymax - y) / dys, (world.ymin - y) / dys)
        dist = np.minimum(np.minimum(tx, ty), self.max_range)
        np.maximum(dist, 0.0, out=dist)

        rects = world.obstacles
        if rects.shape[0]:
            t1x = (rects[None, :, 0] - x) / dxs[:, None]
            t2x = (rects[None, :, 2] - x) / dxs[:, None]
            t1y = (rects[None, :, 1] - y) / dys[:, None]
            t2y = (rects[None, :, 3] - y) / dys[:, None]
            tmin = np.maximum(np.minimum(t1x, t2x), np.minimum(t1y, t2y))
            tmax = np.minimum(np.maximum(t1x, t2x), np.maximum(t1y, t2y))
            entry = np.maximum(tmin, 0.0)
            hit = tmax >= entry
            hit_dist = np.where(hit, entry, np.inf).min(axis=1)
            dist = np.minimum(dist, hit_dist)

        if self.cfg.noise_std > 0.0:
            if rng is None:
                raise ValueError("A numpy Generator is required when lidar.noise_std > 0")
            dist = np.clip(dist + rng.normal(0.0, self.cfg.noise_std, self.num_beams),
                           0.0, self.max_range)
        return dist

    def normalize(self, ranges: np.ndarray) -> np.ndarray:
        """``min(d, max_range) / max_range`` as float32 in ``[0, 1]``."""
        return np.clip(ranges / self.max_range, 0.0, 1.0).astype(np.float32)
