"""Discrete action spaces, mirror symmetry and the persistent target policy.

An action is a (v, omega) pair. Tables must form a full grid of
speed levels x turn rates (Part 1: 1 x 5, Part 2: 3 x 5). Mirroring the world
across the robot's x-axis maps omega -> -omega and keeps v, i.e.
turn_idx -> n_turn-1-turn_idx with the speed level unchanged.
"""
from __future__ import annotations

import numpy as np

TURNS5 = (-1.5, -0.75, 0.0, 0.75, 1.5)
SPEC5 = tuple((0.15, w) for w in TURNS5)                     # Part 1 (simulator default)
SPEEDS3 = (0.05, 0.12, 0.22)
SPEED15 = tuple((v, w) for v in SPEEDS3 for w in TURNS5)     # Part 2: index = speed_idx*5 + turn_idx


class ActionSpace:
    def __init__(self, table, persist_p: float = 0.1):
        t = np.asarray(table, dtype=np.float64)
        self.table = t
        self.n = len(t)
        self.v = t[:, 0].copy()
        self.w = t[:, 1].copy()
        self.speeds = np.unique(np.round(self.v, 9))
        self.turns = np.unique(np.round(self.w, 9))
        self.n_speed, self.n_turn = len(self.speeds), len(self.turns)
        self.speed_idx = np.searchsorted(self.speeds, np.round(self.v, 9))
        self.turn_idx = np.searchsorted(self.turns, np.round(self.w, 9))
        grid = -np.ones((self.n_speed, self.n_turn), dtype=int)
        grid[self.speed_idx, self.turn_idx] = np.arange(self.n)
        if (grid < 0).any() or self.n != self.n_speed * self.n_turn:
            raise ValueError("action table must be a full speed x turn grid")
        if not np.allclose(self.turns, -self.turns[::-1]):
            raise ValueError("turn rates must be symmetric about 0 for mirroring")
        self.grid = grid
        self.mirror = grid[self.speed_idx, self.n_turn - 1 - self.turn_idx]
        self.straight_turn = int(np.argmin(np.abs(self.turns)))
        self.v_max = float(self.speeds.max())
        self.w_max = float(np.abs(self.turns).max())
        self.persist_p = float(persist_p)
        self.persist = self.persistent_policy(persist_p)

    # ------------------------------------------------------------------
    def persistent_policy(self, p: float) -> np.ndarray:
        """pi(a'|a): keep a with prob 1-p, else move uniformly to a neighbouring
        turn rate OR speed level (grid neighbours)."""
        P = np.zeros((self.n, self.n))
        for a in range(self.n):
            s, t = self.speed_idx[a], self.turn_idx[a]
            nb = [self.grid[s, t2] for t2 in (t - 1, t + 1) if 0 <= t2 < self.n_turn]
            nb += [self.grid[s2, t] for s2 in (s - 1, s + 1) if 0 <= s2 < self.n_speed]
            P[a, a] = 1.0 - p if nb else 1.0
            for b in nb:
                P[a, b] += p / len(nb)
        return P

    def index(self, speed_idx, turn_idx):
        return self.grid[speed_idx, turn_idx]

    def default_action(self) -> int:
        """Episode-start 'previous action': straight at the middle speed level."""
        return int(self.grid[self.n_speed // 2, self.straight_turn])

    def remap_to(self, env_table) -> np.ndarray:
        """Map each of OUR actions to the env index with the same physical turn
        rate (closest omega, then closest v). Used to run the same trained agent
        on an env with a different action table (e.g. gazebo_compat)."""
        et = np.asarray(env_table, float)
        out = np.empty(self.n, dtype=int)
        for a in range(self.n):
            cost = 10.0 * np.abs(et[:, 1] - self.w[a]) + np.abs(et[:, 0] - self.v[a])
            out[a] = int(np.argmin(cost))
        return out


def mirror_beams(n_beams: int = 24) -> np.ndarray:
    """Beam i (angle 2*pi*i/n, CCW, 0 = ahead) -> beam (n-i) % n under a mirror
    across the robot's heading axis."""
    return (-np.arange(n_beams)) % n_beams
