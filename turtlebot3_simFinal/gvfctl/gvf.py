"""Neural safety GVF (+ optional future-velocity GVFs) learned from replay.

Safety, for each horizon gamma_h and action a (paper: action-conditioned Q^pi(s,a), TD(0)):
    Q_h(s,a) = E[(1-g) sum_t g^t c_{t+1}],   c = 1 if min LiDAR > d_min + v(a)*tau else 0
    target   y = (1-g) c + g (1-term) sum_a' pi(a'|a) Q_h^targ(s', a')
    pi(a'|a) = persistent "keep doing what I'm doing" policy (no IS, no max).
    term = collision only; a step-limit truncation still bootstraps.
    Sigmoid heads, BCE (or MSE ablation), Adam, Polyak target network.

Velocity (Part 2), cumulants c = v(a)/v_max, |w(a)|/w_max and w(a)/w_max (signed,
for the mirror-consistency check), horizon H -> g_v = 1 - 1/H:
    y = (1-g_v) c + g_v (1-term) sum_a' pi_ctrl(a'|s', a) V^targ(s', a')
    pi_ctrl = (1-eps) * onehot(controller(s', prev=a)) + eps/n   (expected SARSA)
    The controller is evaluated with TARGET-network predictions, so the target
    policy tracks the improving controller without chasing the online net.
    Ablation: pi_ctrl -> persistent policy.

Mirror augmentation: every sampled transition is also used mirrored
(beams permuted, turn index mirrored). Symmetrised prediction:
Q_sym(s,a) = (Q(s,a) + Q(m s, m a)) / 2 (signed turn-rate head: difference).
"""
from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import Optional, Tuple

import numpy as np

from .actions import ActionSpace
from .controllers import SteerCfg, SpeedCfg, controller_batch
from .features import FEATURES, mirror_ranges
from .nn import MLP, sigmoid


@dataclass
class GVFConfig:
    horizons: Tuple[float, ...] = (0.90, 0.97)
    persist_p: float = 0.1
    loss: str = "bce"                 # bce | mse
    mirror: bool = True
    features: str = "rich"
    hidden: Tuple[int, ...] = (128, 128)
    lr: float = 1e-3
    polyak: float = 0.005
    batch: int = 64
    replay: int = 100_000
    learning_starts: int = 1000
    d_min: float = 0.25               # Part 1: c = min range > 0.25
    headway_tau: float = 0.0          # Part 2: c = min range > d_min + v*tau  (d_min 0.2)
    velocity: bool = False
    vel_H: int = 20
    vel_shared: bool = True
    vel_target: str = "controller"    # controller | persistent
    vel_eps: float = 0.05

    def to_dict(self):
        d = asdict(self)
        d["horizons"] = list(self.horizons); d["hidden"] = list(self.hidden)
        return d


class Replay:
    def __init__(self, capacity: int, seed: int, fields: dict):
        self.cap = capacity
        self.buf = {k: np.zeros((capacity,) + tuple(shape), dtype=dt) for k, (shape, dt) in fields.items()}
        self.i = 0
        self.full = False
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return self.cap if self.full else self.i

    def add(self, **kw):
        for k, v in kw.items():
            self.buf[k][self.i] = v
        self.i = (self.i + 1) % self.cap
        self.full = self.full or self.i == 0

    def sample(self, n: int) -> dict:
        idx = self.rng.integers(0, len(self), size=n)
        return {k: v[idx] for k, v in self.buf.items()}


def safety_cumulant(min_range: float, v: float, cfg: GVFConfig) -> float:
    return float(min_range > cfg.d_min + v * cfg.headway_tau)


class GVFModel:
    def __init__(self, cfg: GVFConfig, aspace: ActionSpace, seed: int = 0,
                 steer: Optional[SteerCfg] = None, speed: Optional[SpeedCfg] = None):
        self.cfg, self.A = cfg, aspace
        self.feat, in_dim = FEATURES[cfg.features]
        self.H = len(cfg.horizons)
        self.g = np.asarray(cfg.horizons, np.float32)
        n = aspace.n
        self.n_s = self.H * n
        self.n_v = 3 * n if cfg.velocity else 0
        self.g_v = 1.0 - 1.0 / cfg.vel_H
        self.steer, self.speed = steer or SteerCfg(), speed or SpeedCfg()
        sizes = lambda out: [in_dim, *cfg.hidden, out]
        if cfg.velocity and not cfg.vel_shared:
            self.net = MLP(sizes(self.n_s), seed)
            self.vnet = MLP(sizes(self.n_v), seed + 7919)
        else:
            self.net = MLP(sizes(self.n_s + self.n_v), seed)
            self.vnet = None
        self.tnet = MLP(self.net.sizes, seed); self.tnet.copy_from(self.net)
        self.tvnet = None
        if self.vnet is not None:
            self.tvnet = MLP(self.vnet.sizes, seed); self.tvnet.copy_from(self.vnet)
        self.P = aspace.persistent_policy(cfg.persist_p).astype(np.float32)
        self.updates = 0
        # velocity scaling: outputs are normalised by v_max / w_max
        self.v_scale = np.array([aspace.v_max, aspace.w_max, aspace.w_max], np.float32)

    # ------------------------------------------------------------------ io
    def _outs(self, X, target: bool):
        net = self.tnet if target else self.net
        vnet = self.tvnet if target else self.vnet
        o = net.forward(X)
        s = o[:, :self.n_s]
        if not self.cfg.velocity:
            return s, None
        v = vnet.forward(X) if vnet is not None else o[:, self.n_s:]
        return s, v

    def _shape(self, s, v):
        n = self.A.n
        Q = sigmoid(s).reshape(-1, self.H, n)
        V = v.reshape(-1, 3, n) if v is not None else None
        return Q, V

    def _sym(self, Q1, V1, Q2, V2):
        m = self.A.mirror
        Q = 0.5 * (Q1 + Q2[..., m])
        V = None
        if V1 is not None:
            V = 0.5 * (V1 + V2[..., m])
            V[:, 2] = 0.5 * (V1[:, 2] - V2[:, 2][..., m])     # signed turn rate flips
        return Q, V

    def predict(self, ranges: np.ndarray, target: bool = False, symmetric: Optional[bool] = None):
        """Q: (B, H, n) in [0,1]; V: (B, 3, n) normalised [lin/v_max, |w|/w_max, w/w_max]."""
        ranges = np.atleast_2d(ranges)
        symmetric = self.cfg.mirror if symmetric is None else symmetric
        if not symmetric:
            return self._shape(*self._outs(self.feat(ranges), target))
        B = len(ranges)
        X = self.feat(np.concatenate([ranges, mirror_ranges(ranges)]))
        Q, V = self._shape(*self._outs(X, target))
        return self._sym(Q[:B], None if V is None else V[:B], Q[B:], None if V is None else V[B:])

    def decision_inputs(self, ranges, target=False):
        """(Qmin (B,n), Vlin m/s (B,n) or None, Vabs rad/s (B,n) or None)."""
        Q, V = self.predict(ranges, target=target)
        Qmin = Q.min(1)
        if V is None:
            return Qmin, None, None
        return Qmin, V[:, 0] * self.A.v_max, np.clip(V[:, 1], 0, None) * self.A.w_max

    # --------------------------------------------------------------- update
    def targets(self, b: dict):
        """Augmented batch + TD targets. Returns (s, a, y (N,H), yv (N,3) or None)."""
        cfg, A = self.cfg, self.A
        s, sp, a = b["s"], b["sp"], b["a"].astype(int)
        c, term = b["c"].astype(np.float32), b["term"].astype(np.float32)
        if cfg.mirror:
            s = np.concatenate([s, mirror_ranges(s)])
            sp = np.concatenate([sp, mirror_ranges(sp)])
            a = np.concatenate([a, A.mirror[a]])
            c = np.concatenate([c, c]); term = np.concatenate([term, term])
        N = len(a)
        ar = np.arange(N)
        Qn, Vn = self._shape(*self._outs(self.feat(sp), target=True))
        if cfg.mirror:          # rows [:h] are s', rows [h:] are mirror(s'): symmetrise both halves
            h = N // 2
            Va_, Vb_ = (None, None) if Vn is None else (Vn[:h], Vn[h:])
            Qa, Va = self._sym(Qn[:h], Va_, Qn[h:], Vb_)
            Qb, Vb = self._sym(Qn[h:], Vb_, Qn[:h], Va_)
            Qn = np.concatenate([Qa, Qb])
            Vn = None if Vn is None else np.concatenate([Va, Vb])
        Pa = self.P[a]
        boot = (Qn * Pa[:, None, :]).sum(-1)                               # (N, H)
        y = (1 - self.g)[None] * c[:, None] + self.g[None] * (1 - term)[:, None] * boot
        yv = None
        if cfg.velocity:
            n = A.n
            cv = np.stack([A.v[a] / A.v_max, np.abs(A.w[a]) / A.w_max, A.w[a] / A.w_max], 1).astype(np.float32)
            if cfg.vel_target == "controller":
                act = controller_batch(Qn.min(1), Vn[:, 0] * A.v_max, np.clip(Vn[:, 1], 0, None) * A.w_max,
                                       a, A, self.steer, self.speed)
                pi = np.full((N, n), cfg.vel_eps / n, np.float32)
                pi[ar, act] += 1 - cfg.vel_eps
            else:
                pi = Pa
            vboot = (Vn * pi[:, None, :]).sum(-1)                          # (N, 3)
            yv = (1 - self.g_v) * cv + self.g_v * (1 - term)[:, None] * vboot
        return s, a, y, yv

    def update(self, b: dict) -> dict:
        cfg, A = self.cfg, self.A
        s, a, y, yv = self.targets(b)
        N, n = len(a), A.n
        ar = np.arange(N)
        X = self.feat(s)
        out = self.net.forward(X, cache=True)
        logit = out[:, :self.n_s].reshape(N, self.H, n)[ar, :, a]          # (N, H)
        p = sigmoid(logit)
        d = np.zeros_like(out)
        if cfg.loss == "bce":
            g = (p - y) / N
            loss_s = float(-np.mean(y * np.log(p + 1e-7) + (1 - y) * np.log(1 - p + 1e-7)))
        else:
            g = 2 * (p - y) * p * (1 - p) / N
            loss_s = float(np.mean((p - y) ** 2))
        for h in range(self.H):
            d[ar, h * n + a] = g[:, h]
        loss_v = 0.0
        if cfg.velocity:
            if self.vnet is None:
                vout = out[:, self.n_s:].reshape(N, 3, n)[ar, :, a]
                gv = (vout - yv) / N
                for k in range(3):
                    d[ar, self.n_s + k * n + a] = gv[:, k]
            else:
                vo = self.vnet.forward(X, cache=True)
                vout = vo.reshape(N, 3, n)[ar, :, a]
                gv = (vout - yv) / N
                dv = np.zeros_like(vo)
                for k in range(3):
                    dv[ar, k * n + a] = gv[:, k]
                self.vnet.adam(self.vnet.backward(dv), cfg.lr)
                self.tvnet.polyak_from(self.vnet, cfg.polyak)
            loss_v = float(np.mean((vout - yv) ** 2))
        self.net.adam(self.net.backward(d), cfg.lr)
        self.tnet.polyak_from(self.net, cfg.polyak)
        self.updates += 1
        return {"loss_safety": loss_s, "loss_vel": loss_v}

    # ------------------------------------------------------------ persistence
    def state(self) -> dict:
        d = {f"net_{k}": v for k, v in self.net.state().items()}
        if self.vnet is not None:
            d.update({f"vnet_{k}": v for k, v in self.vnet.state().items()})
        return d

    def load(self, d) -> None:
        self.net.load({k[4:]: d[k] for k in d if k.startswith("net_")})
        self.tnet.copy_from(self.net)
        if self.vnet is not None:
            self.vnet.load({k[5:]: d[k] for k in d if k.startswith("vnet_")})
            self.tvnet.copy_from(self.vnet)
