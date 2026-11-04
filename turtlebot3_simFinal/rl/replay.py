"""Experience replay that stores RAW base observations, never augmented features.

Why raw: Agent C's GVF predictor changes during training, so an augmented
observation cached at insertion time would go stale. We therefore store the raw
base observation (normalised LiDAR [+ goal features]) for both s and s'. The
DQN learner rebuilds the GVF features from these raw vectors at minibatch time
using the CURRENT predictor (Agent C) or the frozen one (Agent B). The action,
reward, and terminal/truncated flags are stored exactly as experienced.
"""
from __future__ import annotations

from typing import NamedTuple

import numpy as np


class Batch(NamedTuple):
    obs: np.ndarray          # (b, base_dim) raw base observation
    action: np.ndarray       # (b,)
    reward: np.ndarray       # (b,)
    next_obs: np.ndarray     # (b, base_dim)
    terminated: np.ndarray   # (b,) real terminal (collision/goal) -> no bootstrap
    truncated: np.ndarray    # (b,) step-limit -> still bootstraps


class ReplayBuffer:
    def __init__(self, capacity: int, base_dim: int, seed: int = 0):
        self.capacity = int(capacity)
        self.base_dim = int(base_dim)
        self._obs = np.zeros((self.capacity, base_dim), dtype=np.float32)
        self._next = np.zeros((self.capacity, base_dim), dtype=np.float32)
        self._act = np.zeros(self.capacity, dtype=np.int64)
        self._rew = np.zeros(self.capacity, dtype=np.float32)
        self._term = np.zeros(self.capacity, dtype=np.float32)
        self._trunc = np.zeros(self.capacity, dtype=np.float32)
        self._i = 0
        self._full = False
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return self.capacity if self._full else self._i

    def add(self, obs, action, reward, next_obs, terminated, truncated) -> None:
        i = self._i
        self._obs[i] = obs
        self._next[i] = next_obs
        self._act[i] = int(action)
        self._rew[i] = float(reward)
        self._term[i] = float(bool(terminated))
        self._trunc[i] = float(bool(truncated))
        self._i = (i + 1) % self.capacity
        if self._i == 0:
            self._full = True

    def sample(self, batch_size: int) -> Batch:
        n = len(self)
        idx = self.rng.integers(0, n, size=batch_size)
        return Batch(self._obs[idx], self._act[idx], self._rew[idx],
                     self._next[idx], self._term[idx], self._trunc[idx])

    # ---- serialization (resume with the SAME buffer) -----------------------
    def get_state(self) -> dict:
        return {"obs": self._obs, "next": self._next, "act": self._act,
                "rew": self._rew, "term": self._term, "trunc": self._trunc,
                "i": self._i, "full": self._full,
                "rng": self.rng.bit_generator.state}

    def load_state(self, d: dict) -> None:
        self._obs = np.asarray(d["obs"]); self._next = np.asarray(d["next"])
        self._act = np.asarray(d["act"]); self._rew = np.asarray(d["rew"])
        self._term = np.asarray(d["term"]); self._trunc = np.asarray(d["trunc"])
        self._i = int(d["i"]); self._full = bool(d["full"])
        try:
            self.rng.bit_generator.state = d["rng"]
        except Exception:
            pass
