"""Minimal NumPy MLP (ReLU hidden layers, linear output) with Adam.

float32 by default for speed; tests gradient-check it in float64.
"""
from __future__ import annotations

from typing import Dict, Sequence

import numpy as np


class MLP:
    def __init__(self, sizes: Sequence[int], seed: int = 0, dtype=np.float32):
        rng = np.random.default_rng(seed)
        self.sizes = list(sizes)
        self.dtype = dtype
        self.W, self.b = [], []
        for i, (a, c) in enumerate(zip(sizes[:-1], sizes[1:])):
            scale = np.sqrt(2.0 / a) if i < len(sizes) - 2 else np.sqrt(1.0 / a)
            self.W.append((rng.standard_normal((a, c)) * scale).astype(dtype))
            self.b.append(np.zeros(c, dtype=dtype))
        self.L = len(self.W)
        self._m = [np.zeros_like(p) for p in self.params()]
        self._v = [np.zeros_like(p) for p in self.params()]
        self.t = 0
        self._cache = None

    def params(self):
        return [p for pair in zip(self.W, self.b) for p in pair]

    def forward(self, x: np.ndarray, cache: bool = False) -> np.ndarray:
        h = np.asarray(x, dtype=self.dtype)
        acts, pres = [h], []
        for i in range(self.L):
            z = h @ self.W[i] + self.b[i]
            if i < self.L - 1:
                pres.append(z)
                h = np.maximum(z, 0)
            else:
                h = z
            acts.append(h)
        if cache:
            self._cache = (acts, pres)
        return h

    def backward(self, d_out: np.ndarray):
        acts, pres = self._cache
        delta = np.asarray(d_out, dtype=self.dtype)
        gW, gb = [None] * self.L, [None] * self.L
        for i in reversed(range(self.L)):
            gW[i] = acts[i].T @ delta
            gb[i] = delta.sum(0)
            if i > 0:
                delta = (delta @ self.W[i].T) * (pres[i - 1] > 0)
        return [g for pair in zip(gW, gb) for g in pair]

    def adam(self, grads, lr: float, clip_norm: float = 10.0, b1=0.9, b2=0.999, eps=1e-8):
        if clip_norm:
            norm = float(np.sqrt(sum(float((g.astype(np.float64) ** 2).sum()) for g in grads)))
            if norm > clip_norm:
                grads = [g * (clip_norm / norm) for g in grads]
        self.t += 1
        c1, c2 = 1 - b1 ** self.t, 1 - b2 ** self.t
        for p, g, m, v in zip(self.params(), grads, self._m, self._v):
            m *= b1; m += (1 - b1) * g
            v *= b2; v += (1 - b2) * (g * g)
            p -= (lr * (m / c1) / (np.sqrt(v / c2) + eps)).astype(p.dtype)

    def copy_from(self, other: "MLP") -> None:
        for p, q in zip(self.params(), other.params()):
            p[...] = q

    def polyak_from(self, other: "MLP", tau: float) -> None:
        for p, q in zip(self.params(), other.params()):
            p *= (1 - tau); p += tau * q

    def state(self) -> Dict[str, np.ndarray]:
        return {f"p{i}": p.copy() for i, p in enumerate(self.params())}

    def load(self, d) -> None:
        for i, p in enumerate(self.params()):
            p[...] = d[f"p{i}"]


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))
