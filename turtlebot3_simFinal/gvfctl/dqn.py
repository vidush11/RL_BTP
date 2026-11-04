"""DQN baselines.

A (standard): raw 24 LiDAR -> 128 -> 128 -> n, Huber loss, hard target network,
  epsilon-greedy 1 -> 0.05 over the first 40% of steps, replay, UTD 1, batch 64.
B (strong):   A + double-Q + left/right mirror augmentation + the 48 rich features.
Done-mask = collision only (truncation bootstraps). Rewards are computed by the
training loop (Part 1: +1/step; Part 2: v/v_max [- lam*|w|/w_max]) times reward_scale.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Tuple

import numpy as np

from .actions import ActionSpace
from .features import FEATURES, mirror_ranges
from .nn import MLP


@dataclass
class DQNConfig:
    features: str = "raw"
    double: bool = False
    mirror: bool = False
    hidden: Tuple[int, ...] = (128, 128)
    gamma: float = 0.99
    lr: float = 1e-3
    batch: int = 64
    target_update: int = 1000         # gradient steps between hard target copies
    replay: int = 100_000
    learning_starts: int = 1000
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_frac: float = 0.4
    huber: float = 1.0
    reward_scale: float = 1.0
    turn_penalty: float = 0.0         # Part 2 reward variant: - lam*|w|/w_max

    def to_dict(self):
        d = asdict(self); d["hidden"] = list(self.hidden); return d


def family(name: str, **kw) -> DQNConfig:
    if name == "A":
        return DQNConfig(features="raw", double=False, mirror=False, **kw)
    if name == "B":
        return DQNConfig(features="rich", double=True, mirror=True, **kw)
    raise ValueError(name)


class DQN:
    def __init__(self, cfg: DQNConfig, aspace: ActionSpace, seed: int = 0):
        self.cfg, self.A = cfg, aspace
        self.feat, in_dim = FEATURES[cfg.features]
        self.q = MLP([in_dim, *cfg.hidden, aspace.n], seed)
        self.t = MLP(self.q.sizes, seed); self.t.copy_from(self.q)
        self.updates = 0

    def q_values(self, ranges):
        return self.q.forward(self.feat(np.atleast_2d(ranges)))

    def reward(self, a: int) -> float:
        A, c = self.A, self.cfg
        if A.n_speed == 1:
            r = 1.0
        else:
            r = A.v[a] / A.v_max - c.turn_penalty * abs(A.w[a]) / A.w_max
        return c.reward_scale * r

    def update(self, b: dict) -> dict:
        c, A = self.cfg, self.A
        s, sp, a = b["s"], b["sp"], b["a"].astype(int)
        r, term = b["r"].astype(np.float32), b["term"].astype(np.float32)
        if c.mirror:
            s = np.concatenate([s, mirror_ranges(s)]); sp = np.concatenate([sp, mirror_ranges(sp)])
            a = np.concatenate([a, A.mirror[a]]); r = np.concatenate([r, r]); term = np.concatenate([term, term])
        N = len(a); ar = np.arange(N)
        Fn = self.feat(sp)
        qt = self.t.forward(Fn)
        if c.double:
            a_star = np.argmax(self.q.forward(Fn), 1)
            qn = qt[ar, a_star]
        else:
            qn = qt.max(1)
        y = r + c.gamma * (1 - term) * qn
        out = self.q.forward(self.feat(s), cache=True)
        err = out[ar, a] - y
        d = np.zeros_like(out)
        d[ar, a] = np.clip(err, -c.huber, c.huber) / N
        self.q.adam(self.q.backward(d), c.lr)
        self.updates += 1
        if self.updates % c.target_update == 0:
            self.t.copy_from(self.q)
        ae = np.abs(err)
        return {"loss": float(np.mean(np.where(ae <= c.huber, 0.5 * ae ** 2, c.huber * (ae - 0.5 * c.huber))))}

    def state(self):
        return {f"q_{k}": v for k, v in self.q.state().items()}

    def load(self, d):
        self.q.load({k[2:]: d[k] for k in d if k.startswith("q_")}); self.t.copy_from(self.q)
