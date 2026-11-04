"""DQN baseline (Part C): replay, hard-updated target network, Adam, Huber
loss, epsilon-greedy with linear decay over training EPISODES. Same observation
(3 stacked frames of [LiDAR, v, omega]), enabled actions and trunk size
(MLP 128-128) as the GVF predictors, and the same env-steps-per-gradient-step
ratio. Reward = the shared task reward (A8). Target:
  y = r + gamma * (1 - terminated) * max_a Q_target(s', a)
terminated = collision only; truncation bootstraps (H1)."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Sequence

import numpy as np

from gvfctl.nn import MLP


@dataclass
class DQNConfig:
    hidden: Sequence[int] = (128, 128)
    lr: float = 5e-4
    gamma: float = 0.99
    batch: int = 64
    replay: int = 100_000
    learning_starts: int = 2_000
    train_every: int = 4
    target_update: int = 1000          # gradient steps between hard copies
    huber: float = 1.0
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_decay_frac: float = 0.2        # fraction of the episode budget

    def to_dict(self):
        d = asdict(self); d["hidden"] = list(self.hidden); return d


class Replay:
    def __init__(self, cap, obs_dim, seed):
        self.cap = int(cap)
        self.s = np.zeros((self.cap, obs_dim), np.float32); self.sp = np.zeros((self.cap, obs_dim), np.float32)
        self.a = np.zeros(self.cap, np.int16); self.r = np.zeros(self.cap, np.float32)
        self.term = np.zeros(self.cap, np.float32)
        self.i, self.full = 0, False
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return self.cap if self.full else self.i

    def add(self, s, a, r, sp, terminated):
        i = self.i
        self.s[i], self.a[i], self.r[i], self.sp[i], self.term[i] = s, a, r, sp, float(terminated)
        self.i = (i + 1) % self.cap; self.full = self.full or self.i == 0

    def sample(self, m):
        idx = self.rng.integers(0, len(self), size=m)
        return self.s[idx], self.a[idx].astype(int), self.r[idx], self.sp[idx], self.term[idx]

    def state(self):
        n = len(self)
        return {"s": self.s[:n], "sp": self.sp[:n], "a": self.a[:n], "r": self.r[:n], "term": self.term[:n],
                "i": np.int64(self.i), "full": np.bool_(self.full)}

    def load(self, d):
        n = len(d["a"])
        for k in ("s", "sp", "a", "r", "term"):
            getattr(self, k)[:n] = d[k]
        self.i, self.full = int(d["i"]), bool(d["full"])


def dqn_targets(r, term, q_next_target, gamma):
    return r + gamma * (1.0 - term) * q_next_target.max(1)


class DQNAgent:
    def __init__(self, obs_dim, n_actions, cfg: DQNConfig, seed: int):
        self.cfg, self.n = cfg, n_actions
        self.q = MLP([obs_dim, *cfg.hidden, n_actions], seed)
        self.t = MLP(self.q.sizes, seed); self.t.copy_from(self.q)
        self.replay = Replay(cfg.replay, obs_dim, seed + 7)
        self.updates = 0

    def epsilon(self, episode, budget):
        c = self.cfg
        frac = min(1.0, episode / max(1.0, c.eps_decay_frac * budget))
        return c.eps_start + frac * (c.eps_end - c.eps_start)

    def greedy(self, obs) -> int:
        return int(np.argmax(self.q.forward(np.atleast_2d(obs))[0]))

    def maybe_train(self, step):
        c = self.cfg
        if len(self.replay) < c.learning_starts or step % c.train_every:
            return None
        s, a, r, sp, term = self.replay.sample(c.batch)
        y = dqn_targets(r, term, self.t.forward(sp), c.gamma)
        out = self.q.forward(s, cache=True)
        ar = np.arange(len(a))
        err = out[ar, a] - y
        d = np.zeros_like(out); d[ar, a] = np.clip(err, -c.huber, c.huber) / len(a)
        self.q.adam(self.q.backward(d), c.lr)
        self.updates += 1
        if self.updates % c.target_update == 0:
            self.t.copy_from(self.q)
        return float(np.mean(np.minimum(np.abs(err), c.huber) * (np.abs(err) - 0.5 * np.minimum(np.abs(err), c.huber))))

    def frozen_policy(self):
        q = MLP(self.q.sizes, 0); q.copy_from(self.q)
        return DQNPolicy(q)


class DQNPolicy:
    name = "dqn"

    def __init__(self, q):
        self.q = q

    def reset(self):
        pass

    def act(self, obs, info) -> int:
        return int(np.argmax(self.q.forward(np.atleast_2d(obs))[0]))
