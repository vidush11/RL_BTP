"""Discrete analogues of the paper's policies (Sec. IV-A).

Paper (continuous throttle): target pi(a_t | a_{t-1}) = N(a_{t-1}, sigma^2) --
"what if I keep doing what I'm doing?"; behaviour mu = Wiener process (last
action + noise), occasionally interrupted by a uniform random action.

Here (documented in DECISIONS.md):
* accel mode (unordered FORWARD/LEFT/RIGHT/STOP):
    pi(a | a_prev) = 1 - eps_pi            if a == a_prev
                     eps_pi / (n - 1)      otherwise
    mu(a | a_prev) = p_interrupt / n + p_repeat * [a == a_prev]
                     + (1 - p_repeat - p_interrupt) / (n - 1) * [a != a_prev]
* velocity_table mode (ordered index): pi = discretised Gaussian over the action
  index centred on a_prev (sigma in index units); mu = the same random-walk kernel
  (wider) mixed with uniform interruptions.
mu != pi, so learning is off-policy, as in the paper.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class PolicyConfig:
    eps_pi: float = 0.1          # target: probability of not repeating
    sigma_pi: float = 0.5        # velocity_table target: Gaussian std in index units
    p_repeat: float = 0.70       # behaviour (accel): repeat the last action
    p_interrupt: float = 0.05    # behaviour: uniform "interruption"
    sigma_mu: float = 1.0        # velocity_table behaviour: random-walk std in index units


def _gauss_kernel(n: int, centre: int, sigma: float) -> np.ndarray:
    k = np.exp(-0.5 * ((np.arange(n) - centre) / sigma) ** 2)
    return k / k.sum()


class TargetPolicy:
    def __init__(self, n_actions: int, ordered: bool, cfg: PolicyConfig):
        self.n, self.ordered, self.cfg = n_actions, ordered, cfg
        if ordered:
            self.P = np.stack([_gauss_kernel(n_actions, a, cfg.sigma_pi) for a in range(n_actions)])
        else:
            off = cfg.eps_pi / (n_actions - 1) if n_actions > 1 else 0.0
            self.P = np.full((n_actions, n_actions), off)
            np.fill_diagonal(self.P, 1.0 - cfg.eps_pi if n_actions > 1 else 1.0)

    def probs(self, a_prev: int) -> np.ndarray:
        return self.P[int(a_prev)]

    def sample(self, a_prev: int, rng: np.random.Generator) -> int:
        return int(rng.choice(self.n, p=self.P[int(a_prev)]))


class BehaviourPolicy:
    def __init__(self, n_actions: int, ordered: bool, cfg: PolicyConfig):
        self.n, self.ordered, self.cfg = n_actions, ordered, cfg
        u = np.full(n_actions, cfg.p_interrupt / n_actions)
        if ordered:
            self.P = np.stack([(1 - cfg.p_interrupt) * _gauss_kernel(n_actions, a, cfg.sigma_mu) + u
                               for a in range(n_actions)])
        else:
            P = []
            for a in range(n_actions):
                p = u + np.full(n_actions, (1 - cfg.p_repeat - cfg.p_interrupt) / max(n_actions - 1, 1))
                p[a] = cfg.p_interrupt / n_actions + cfg.p_repeat
                P.append(p)
            self.P = np.stack(P)
        self.P /= self.P.sum(1, keepdims=True)

    def probs(self, a_prev: int) -> np.ndarray:
        return self.P[int(a_prev)]

    def sample(self, a_prev: int, rng: np.random.Generator) -> int:
        return int(rng.choice(self.n, p=self.P[int(a_prev)]))
