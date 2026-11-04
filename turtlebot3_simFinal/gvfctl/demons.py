"""Baseline D: the ORIGINAL linear Horde demons from gvf_horde_agent.py plugged
into the same controller.

* TTC demons: DirectionalTTCGVF (GQ(lambda)), target actions = straight and the
  two gentle turns (omega -0.75 / +0.75) at the middle speed level; features are
  the original phi(s,a) = clip(ranges/3.5) in the action's block. Hard turns and
  other speed levels reuse the demon of the same turning direction (the original
  has only 3 directional demons and no speed conditioning).
* Velocity demons (Part 2): FixedHorizonGVF (H=20) on commanded v and signed w,
  target = epsilon-greedy(0.05) toward middle-speed straight, behaviour uniform.
* Variants: "default" (alpha_theta 0.005, H-step IS product) and "fixed"
  (documented fixes: alpha_theta 0.1, velocity IS off).
* TTC -> controller safety:  Q_h = 1 - g_h^max(TTC, 0)  (inverse of ttc_from_safety;
  the original gamma cut-off "any ray <= 0.25 m" matches the safety cumulant).
Behaviour during training = uniform random over actions (as in the original agent).
"""
from __future__ import annotations

import importlib

import numpy as np

from .actions import ActionSpace

ALPHA_W, LAMBDA, THR = 1e-4, 0.4, 0.25


def _classes():
    from simulator.ros_compat import install
    from simulator import TurtleBot3LidarEnv
    import sys
    if "gvf_horde_agent" not in sys.modules:
        install(TurtleBot3LidarEnv(), seed=0)
    m = importlib.import_module("gvf_horde_agent")
    return m.DirectionalTTCGVF, m.FixedHorizonGVF


class OriginalDemons:
    def __init__(self, aspace: ActionSpace, variant: str = "default", horizons=(0.90, 0.97)):
        Dir, Fix = _classes()
        self.A, self.variant = aspace, variant
        self.g = np.asarray(horizons)
        n = aspace.n
        self.fd = 24 * n
        mid, st = aspace.n_speed // 2, aspace.straight_turn
        self.targets = {"right": int(aspace.grid[mid, st - 1]), "fwd": int(aspace.grid[mid, st]),
                        "left": int(aspace.grid[mid, st + 1])}
        at = 0.005 if variant == "default" else 0.1
        self.ttc = {k: Dir(self.fd, n, a, slice(0, 24), at, ALPHA_W, LAMBDA, THR, k)
                    for k, a in self.targets.items()}
        self.velocity = aspace.n_speed > 1
        if self.velocity:
            self.lin = Fix(self.fd, 20, 0.001, 50.0)
            self.ang = Fix(self.fd, 20, 0.001, 50.0)
        # demon used for each action (by turning direction)
        self.demon_of = np.array(["fwd" if aspace.turn_idx[a] == st else
                                  ("right" if aspace.turn_idx[a] < st else "left") for a in range(n)])

    def phi(self, ranges, a):
        f = np.zeros((self.A.n, 24))
        f[a] = np.clip(np.asarray(ranges, float) / 3.5, 0, 1)
        return f.ravel()

    def reset_episode(self):
        for d in self.ttc.values():
            d.reset_trace()
        if self.velocity:
            self.lin.reset_window(); self.ang.reset_window()

    def update(self, r_t, a, r_tp1, done):
        b = 1.0 / self.A.n
        phi_t = self.phi(r_t, a)
        for k, d in self.ttc.items():
            d.update(phi_t, self.phi(r_tp1, self.targets[k]), np.asarray(r_t), np.asarray(r_tp1),
                     int(a), b, bool(done))
        if self.velocity:
            fwd = self.targets["fwd"]
            pi = (0.95 + 0.05 / self.A.n) if a == fwd else 0.05 / self.A.n
            rho = 1.0 if self.variant == "fixed" else pi / b
            self.lin.push_and_maybe_update(phi_t, rho, self.A.v[a])
            self.ang.push_and_maybe_update(phi_t, rho, self.A.w[a])

    # ----------------------------------------------------------- predictions
    def ttc_pred(self, ranges):
        """(B, n) predicted steps for each action (via its direction demon).
        theta . phi(s, a_d) = clip(ranges/3.5) . theta.reshape(n, 24)[a_d]."""
        X = np.clip(np.atleast_2d(np.asarray(ranges, float)) / 3.5, 0, 1)
        per = {k: X @ self.ttc[k].theta.reshape(self.A.n, 24)[self.targets[k]] for k in self.ttc}
        out = np.stack([per[self.demon_of[a]] for a in range(self.A.n)], axis=1)
        return np.nan_to_num(out, nan=0.0, posinf=1e6, neginf=0.0)

    def decision_inputs(self, ranges):
        ttc = np.clip(self.ttc_pred(ranges), 0, 1e6)
        Q = 1.0 - self.g[None, :, None] ** ttc[:, None, :]          # (B, H, n)
        Qmin = Q.min(1)
        if not self.velocity:
            return Qmin, None, None
        X = np.clip(np.atleast_2d(np.asarray(ranges, float)) / 3.5, 0, 1)
        Vl = X @ self.lin.w.reshape(self.A.n, 24).T                  # w . phi(s, a) for every a
        Va = np.abs(X @ self.ang.w.reshape(self.A.n, 24).T)
        return Qmin, np.nan_to_num(Vl), np.nan_to_num(Va)

    def state(self):
        d = {f"theta_{k}": v.theta.copy() for k, v in self.ttc.items()}
        d.update({f"w_{k}": v.w.copy() for k, v in self.ttc.items()})
        if self.velocity:
            d["w_lin"], d["w_ang"] = self.lin.w.copy(), self.ang.w.copy()
        return d

    def load(self, d):
        for k, v in self.ttc.items():
            v.theta = np.asarray(d[f"theta_{k}"]).copy(); v.w = np.asarray(d[f"w_{k}"]).copy()
        if self.velocity:
            self.lin.w = np.asarray(d["w_lin"]).copy(); self.ang.w = np.asarray(d["w_ang"]).copy()
