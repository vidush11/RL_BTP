"""A small, fully-NumPy MLP Q-network + Adam, and a DQN learner.

No torch/tf: the net is tiny (input 24-27 dims, two hidden layers, 5 outputs),
so a hand-written MLP keeps the whole project dependency-free and reproducible,
and runs fine on a CPU-only Omni node. Backprop is gradient-checked in
tests/test_dqn.py.

DQN target (standard), with the environment's terminal mask (NOT truncation):

    y = r + gamma * (1 - terminated) * max_a Q_target(s', a)

Only the Q-value of the action actually taken is regressed toward y (Huber loss).
Truncation (step-limit) is handled by the caller: a truncated transition still
bootstraps (it is not a real terminal), so ``terminated`` — not ``done`` — is
passed as the mask.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np


def _he_init(rng: np.random.Generator, n_in: int, n_out: int) -> np.ndarray:
    return rng.normal(0.0, np.sqrt(2.0 / n_in), size=(n_in, n_out))


class MLP:
    """Two-hidden-layer ReLU MLP with Adam. Deterministic given ``seed``."""

    def __init__(self, in_dim: int, hidden: Tuple[int, ...], out_dim: int, seed: int = 0):
        rng = np.random.default_rng(seed)
        sizes = [in_dim, *hidden, out_dim]
        self.W: List[np.ndarray] = []
        self.b: List[np.ndarray] = []
        for a, c in zip(sizes[:-1], sizes[1:]):
            self.W.append(_he_init(rng, a, c))
            self.b.append(np.zeros(c))
        self.n_layers = len(self.W)
        self.in_dim, self.out_dim = in_dim, out_dim
        # Adam state
        self._mW = [np.zeros_like(w) for w in self.W]
        self._vW = [np.zeros_like(w) for w in self.W]
        self._mb = [np.zeros_like(b) for b in self.b]
        self._vb = [np.zeros_like(b) for b in self.b]
        self._t = 0

    # ---- forward with cache ------------------------------------------------
    def forward(self, x: np.ndarray, cache: bool = False):
        x = np.atleast_2d(np.asarray(x, dtype=np.float64))
        acts = [x]
        pre = []
        h = x
        for i in range(self.n_layers):
            z = h @ self.W[i] + self.b[i]
            pre.append(z)
            h = np.maximum(z, 0.0) if i < self.n_layers - 1 else z  # linear output
            acts.append(h)
        if cache:
            self._cache = (acts, pre)
        return h

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x, cache=False)

    # ---- backward: dL/dq given, returns grads ------------------------------
    def _backward(self, d_out: np.ndarray):
        acts, pre = self._cache
        gW = [None] * self.n_layers
        gb = [None] * self.n_layers
        delta = d_out  # (batch, out)
        for i in reversed(range(self.n_layers)):
            gW[i] = acts[i].T @ delta
            gb[i] = delta.sum(axis=0)
            if i > 0:
                dh = delta @ self.W[i].T
                delta = dh * (pre[i - 1] > 0.0)  # ReLU grad
        return gW, gb

    def adam_step(self, gW, gb, lr: float, beta1=0.9, beta2=0.999, eps=1e-8,
                  clip: float = 10.0):
        self._t += 1
        t = self._t
        for i in range(self.n_layers):
            for g, m, v, p in ((gW[i], self._mW[i], self._vW[i], self.W[i]),
                               (gb[i], self._mb[i], self._vb[i], self.b[i])):
                if clip is not None:
                    np.clip(g, -clip, clip, out=g)
                m[...] = beta1 * m + (1 - beta1) * g
                v[...] = beta2 * v + (1 - beta2) * (g * g)
                mhat = m / (1 - beta1 ** t)
                vhat = v / (1 - beta2 ** t)
                p -= lr * mhat / (np.sqrt(vhat) + eps)

    # ---- serialization -----------------------------------------------------
    def get_state(self) -> dict:
        d = {}
        for i in range(self.n_layers):
            d[f"W{i}"] = self.W[i].copy()
            d[f"b{i}"] = self.b[i].copy()
        return d

    def load_state(self, d: dict) -> None:
        for i in range(self.n_layers):
            self.W[i] = np.asarray(d[f"W{i}"]).copy()
            self.b[i] = np.asarray(d[f"b{i}"]).copy()

    def copy_from(self, other: "MLP") -> None:
        for i in range(self.n_layers):
            self.W[i][...] = other.W[i]
            self.b[i][...] = other.b[i]


def huber_grad(pred: np.ndarray, target: np.ndarray, delta: float = 1.0) -> np.ndarray:
    """d/d(pred) of the Huber loss, clipped to [-delta, delta]."""
    err = pred - target
    return np.clip(err, -delta, delta)


@dataclass
class DQNConfig:
    hidden: Tuple[int, ...] = (64, 64)
    gamma: float = 0.99
    lr: float = 1e-3
    batch_size: int = 64
    target_update: int = 500        # gradient steps between target syncs
    huber_delta: float = 1.0
    grad_clip: float = 10.0


class DQN:
    """Online + target MLP with the standard DQN update. Agnostic to how the
    feature vectors are built (the agent/encoder handles raw-vs-GVF features)."""

    def __init__(self, in_dim: int, n_actions: int, cfg: DQNConfig, seed: int = 0):
        self.cfg = cfg
        self.n_actions = n_actions
        self.q = MLP(in_dim, cfg.hidden, n_actions, seed=seed)
        self.target = MLP(in_dim, cfg.hidden, n_actions, seed=seed)
        self.target.copy_from(self.q)
        self._updates = 0

    def q_values(self, feats: np.ndarray) -> np.ndarray:
        return self.q.predict(feats)

    def greedy_action(self, feats: np.ndarray) -> int:
        return int(np.argmax(self.q.predict(feats)[0]))

    def update(self, feats: np.ndarray, actions: np.ndarray, rewards: np.ndarray,
               next_feats: np.ndarray, terminated: np.ndarray) -> float:
        """One gradient step on a minibatch. ``feats``/``next_feats`` are the
        FINAL feature vectors (raw [+ GVF]); the caller builds them so that
        Agent C uses current-GVF features for both s and s'. Returns the mean
        Huber loss (diagnostic)."""
        cfg = self.cfg
        feats = np.asarray(feats, float)
        next_feats = np.asarray(next_feats, float)
        actions = np.asarray(actions, int)
        rewards = np.asarray(rewards, float)
        terminated = np.asarray(terminated, float)
        b = feats.shape[0]

        q_next = self.target.predict(next_feats)             # (b, A)
        y = rewards + cfg.gamma * (1.0 - terminated) * q_next.max(axis=1)

        q_pred = self.q.forward(feats, cache=True)           # (b, A)
        taken = q_pred[np.arange(b), actions]
        g = huber_grad(taken, y, cfg.huber_delta)            # (b,)
        d_out = np.zeros_like(q_pred)
        d_out[np.arange(b), actions] = g / b                 # mean over batch
        gW, gb = self.q._backward(d_out)
        self.q.adam_step(gW, gb, cfg.lr, clip=cfg.grad_clip)

        self._updates += 1
        if self._updates % cfg.target_update == 0:
            self.target.copy_from(self.q)
        loss = float(np.mean(np.where(np.abs(taken - y) <= cfg.huber_delta,
                                      0.5 * (taken - y) ** 2,
                                      cfg.huber_delta * (np.abs(taken - y) - 0.5 * cfg.huber_delta))))
        return loss

    # ---- serialization -----------------------------------------------------
    def get_state(self) -> dict:
        return {"q": self.q.get_state(), "target": self.target.get_state(),
                "updates": self._updates}

    def load_state(self, d: dict) -> None:
        self.q.load_state(d["q"])
        self.target.load_state(d["target"])
        self._updates = int(d.get("updates", 0))
