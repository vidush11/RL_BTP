"""2-D world: rectangular boundary + static axis-aligned rectangular obstacles.

Collision checking
------------------
The robot is a disc of radius ``r``.  Its centre travels along a polyline
(see ``Robot.plan``).  For every chord we compute the *exact* minimum distance
between the chord and each obstacle rectangle and report a hit when it is below
``r``.  Because the test is a swept (continuous) test, no speed or obstacle
thickness can make the robot tunnel through an obstacle.  The boundary is a
convex region, so checking each polyline vertex against the inset boundary is
sufficient for the walls.
"""

from __future__ import annotations

from typing import Iterable, Optional, Sequence, Tuple

import numpy as np

from .config import WorldConfig


def points_to_rects_distance(px: np.ndarray, py: np.ndarray, rects: np.ndarray) -> np.ndarray:
    """Distance from points ``(S,)`` to rectangles ``(K,4)`` -> ``(S, K)`` (0 inside)."""
    dx = np.maximum(np.maximum(rects[None, :, 0] - px[:, None], 0.0), px[:, None] - rects[None, :, 2])
    dy = np.maximum(np.maximum(rects[None, :, 1] - py[:, None], 0.0), py[:, None] - rects[None, :, 3])
    return np.hypot(dx, dy)


def segments_to_rects_distance(p0: np.ndarray, p1: np.ndarray, rects: np.ndarray) -> np.ndarray:
    """Exact distance between segments ``p0->p1`` (each ``(S,2)``) and rectangles
    ``(K,4)`` as ``[xmin, ymin, xmax, ymax]``.  Returns ``(S, K)``; 0 on overlap.

    The minimum distance between a segment and a convex polygon is attained
    either at a segment end point or at a polygon vertex, unless they overlap.
    """
    S, K = p0.shape[0], rects.shape[0]
    d = p1 - p0                                           # (S, 2)

    # 1. segment/rectangle overlap (Liang-Barsky slab clipping), vectorised.
    t_lo = np.zeros((S, K))
    t_hi = np.ones((S, K))
    miss = np.zeros((S, K), dtype=bool)
    for axis in (0, 1):
        dd = d[:, axis][:, None]                          # (S,1)
        pp = p0[:, axis][:, None]
        lo = rects[None, :, axis]
        hi = rects[None, :, axis + 2]
        parallel = dd == 0.0
        safe = np.where(parallel, 1.0, dd)
        ta = (lo - pp) / safe
        tb = (hi - pp) / safe
        tmin = np.where(parallel, -np.inf, np.minimum(ta, tb))
        tmax = np.where(parallel, np.inf, np.maximum(ta, tb))
        miss |= parallel & ((pp < lo) | (pp > hi))
        t_lo = np.maximum(t_lo, tmin)
        t_hi = np.minimum(t_hi, tmax)
    overlap = (~miss) & (t_lo <= t_hi)

    # 2. end points -> rectangle.
    d0 = points_to_rects_distance(p0[:, 0], p0[:, 1], rects)
    d1 = points_to_rects_distance(p1[:, 0], p1[:, 1], rects)

    # 3. rectangle vertices -> segment.
    verts = np.stack([
        np.stack([rects[:, 0], rects[:, 1]], axis=1),
        np.stack([rects[:, 2], rects[:, 1]], axis=1),
        np.stack([rects[:, 2], rects[:, 3]], axis=1),
        np.stack([rects[:, 0], rects[:, 3]], axis=1)], axis=1)          # (K,4,2)
    rel = verts[None, :, :, :] - p0[:, None, None, :]                    # (S,K,4,2)
    l2 = np.einsum("sc,sc->s", d, d)                                      # (S,)
    num = np.einsum("skvc,sc->skv", rel, d)
    t = np.where(l2[:, None, None] > 0.0, num / np.where(l2 > 0.0, l2, 1.0)[:, None, None], 0.0)
    t = np.clip(t, 0.0, 1.0)
    closest = p0[:, None, None, :] + t[..., None] * d[:, None, None, :]
    dv = np.sqrt(np.einsum("skvc,skvc->skv", verts[None] - closest, verts[None] - closest)).min(axis=2)

    dist = np.minimum(np.minimum(d0, d1), dv)
    return np.where(overlap, 0.0, dist)


class World:
    """Static geometry.  Holds no randomness itself: callers pass an RNG."""

    def __init__(self, cfg: WorldConfig):
        self.cfg = cfg
        cx, cy = cfg.center
        self.xmin, self.xmax = cx - cfg.width / 2.0, cx + cfg.width / 2.0
        self.ymin, self.ymax = cy - cfg.height / 2.0, cy + cfg.height / 2.0
        self._static = self._as_rects(cfg.obstacles)
        self.obstacles = self._static.copy()

    # ----------------------------------------------------------- obstacles
    @staticmethod
    def _as_rects(obstacles) -> np.ndarray:
        arr = np.asarray(obstacles, dtype=np.float64).reshape(-1, 4)
        if arr.size and (np.any(arr[:, 2] <= arr[:, 0]) or np.any(arr[:, 3] <= arr[:, 1])):
            raise ValueError("Each obstacle must be [xmin, ymin, xmax, ymax] with max > min")
        return arr

    def set_obstacles(self, obstacles) -> None:
        self.obstacles = self._as_rects(obstacles)

    def reset_obstacles(self) -> None:
        self.obstacles = self._static.copy()

    @property
    def bounds(self) -> Tuple[float, float, float, float]:
        return self.xmin, self.ymin, self.xmax, self.ymax

    # ------------------------------------------------------------ distances
    def wall_distance(self, x: float, y: float) -> float:
        return float(min(x - self.xmin, self.xmax - x, y - self.ymin, self.ymax - y))

    def obstacle_distance(self, x: float, y: float) -> float:
        if self.obstacles.shape[0] == 0:
            return float("inf")
        return float(points_to_rects_distance(np.array([x]), np.array([y]), self.obstacles).min())

    def clearance(self, x: float, y: float) -> float:
        """Distance from ``(x, y)`` to the nearest wall or obstacle surface
        (negative outside the boundary, 0 inside an obstacle)."""
        return min(self.wall_distance(x, y), self.obstacle_distance(x, y))

    # ------------------------------------------------------------ collision
    def path_hits(self, points: np.ndarray, radius: float) -> bool:
        """True if a disc of ``radius`` whose centre follows the polyline
        ``points`` (``(n+1, 2)``) touches a wall or an obstacle."""
        xs, ys = points[:, 0], points[:, 1]
        if (xs.min() < self.xmin + radius or xs.max() > self.xmax - radius or
                ys.min() < self.ymin + radius or ys.max() > self.ymax - radius):
            return True
        rects = self.obstacles
        if rects.shape[0] == 0:
            return False
        # Broad phase: the swept disc lies inside the path's bounding box grown by
        # ``radius``; obstacles outside that box cannot be touched, so skip the
        # exact (more expensive) test for them.  Boundary contact keeps the obstacle.
        x0, x1, y0, y1 = xs.min() - radius, xs.max() + radius, ys.min() - radius, ys.max() + radius
        near = ((rects[:, 0] <= x1) & (rects[:, 2] >= x0) &
                (rects[:, 1] <= y1) & (rects[:, 3] >= y0))
        if not near.any():
            return False
        rects = rects[near]
        if points.shape[0] == 1:
            dist = points_to_rects_distance(xs, ys, rects)
        else:
            dist = segments_to_rects_distance(points[:-1], points[1:], rects)
        return bool((dist < radius).any())

    def point_hits(self, x: float, y: float, radius: float) -> bool:
        return self.path_hits(np.array([[x, y]]), radius)

    # ------------------------------------------------------------- sampling
    def sample_free(self, rng: np.random.Generator, clearance: float,
                    avoid: Optional[Iterable[Tuple[float, float, float]]] = None,
                    max_tries: int = 5000) -> Tuple[float, float]:
        """Uniform sample with at least ``clearance`` to every wall/obstacle and
        at least ``d`` from each ``(x, y, d)`` in ``avoid``."""
        avoid = list(avoid or [])
        lo_x, hi_x = self.xmin + clearance, self.xmax - clearance
        lo_y, hi_y = self.ymin + clearance, self.ymax - clearance
        if lo_x >= hi_x or lo_y >= hi_y:
            raise RuntimeError(f"clearance {clearance} m leaves no free space in the world")
        for _ in range(max_tries):
            x, y = float(rng.uniform(lo_x, hi_x)), float(rng.uniform(lo_y, hi_y))
            if self.clearance(x, y) < clearance:
                continue
            if any(np.hypot(x - ax, y - ay) < d for ax, ay, d in avoid):
                continue
            return x, y
        raise RuntimeError(
            f"Could not find a collision-free position (clearance={clearance} m) "
            f"after {max_tries} tries; the layout is too crowded")

    def sample_obstacles(self, rng: np.random.Generator, count: int,
                         size_range: Tuple[float, float],
                         keep_out: Optional[Sequence[Tuple[float, float, float]]] = None,
                         max_tries: int = 500) -> np.ndarray:
        """Random axis-aligned rectangles inside the world, none closer than
        ``d`` to any ``(x, y, d)`` in ``keep_out``."""
        keep_out = list(keep_out or [])
        lo, hi = size_range
        rects = []
        for _ in range(count):
            for _try in range(max_tries):
                w, h = rng.uniform(lo, hi), rng.uniform(lo, hi)
                cx = rng.uniform(self.xmin + w / 2, self.xmax - w / 2)
                cy = rng.uniform(self.ymin + h / 2, self.ymax - h / 2)
                rect = np.array([[cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]])
                if all(points_to_rects_distance(np.array([kx]), np.array([ky]), rect)[0, 0] >= kd
                       for kx, ky, kd in keep_out):
                    rects.append(rect[0])
                    break
            else:
                raise RuntimeError("Could not place a random obstacle; reduce count/size")
        return np.array(rects, dtype=np.float64).reshape(-1, 4)
