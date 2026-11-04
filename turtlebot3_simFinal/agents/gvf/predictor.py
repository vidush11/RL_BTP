"""Action-conditioned, policy-based GVF predictors q^pi(s, a; theta) -- paper
Sec. III-B and Algorithm 1, implemented exactly:

  replay memory D (capacity n); minibatch m < n; store (s, a, c, gamma', s', a')
  with a' ~ pi(. | s', a) sampled AT COLLECTION TIME;
  y = c + gamma' * q(s', a'; theta)    with the CURRENT theta, no gradient
  through y (semi-gradient TD(0), lambda = 0);  loss (y - q(s, a; theta))^2 on the
  taken action's output only.  No importance sampling, no target network
  (optional target network = clearly labelled deviation, off by default).

One network per gamma (paper: separate estimator per horizon). Each network has
two output groups of |A| units: front safety and speed (paper: "one for each
cumulant"; sharing the trunk across the two cumulants of the SAME gamma is our
interpretation of "one network per gamma" -- DECISIONS.md).

Cumulants (scaled by (1 - gamma) so predictions lie in [0, 1]):
  front safety c_f = info["safety"]["c_front"], forced to 0 on the collision step
  speed        c_v = v_{t+1} / v_max
gamma' = 0 on a true termination (collision), gamma otherwise -- a time-limit
truncation BOOTSTRAPS (H1). Because c_f = 0 whenever gamma' = 0, the paper's
"(1 - gamma_{t+1})" scaling and the constant "(1 - gamma)" scaling give the
same targets here.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, Optional, Sequence

import numpy as np

from gvfctl.nn import MLP


@dataclass
class PredictorConfig:
    gammas: Sequence[float] = (0.95, 0.975, 0.983)
    hidden: Sequence[int] = (128, 128)
    lr: float = 5e-4
    batch: int = 64
    replay: int = 100_000
    learning_starts: int = 2_000
    train_every: int = 4                  # env steps per gradient step (same ratio as DQN)
    target_network: bool = False          # DEVIATION if True (paper uses none)
    target_tau: float = 0.005

    def to_dict(self):
        d = asdict(self); d["gammas"] = list(self.gammas); d["hidden"] = list(self.hidden)
        return d


class GVFReplay:
    """Algorithm 1 memory: (s, a, c_f, c_v, gamma'-mask, s', a')."""

    def __init__(self, capacity: int, obs_dim: int, seed: int):
        self.cap = int(capacity)
        self.s = np.zeros((self.cap, obs_dim), np.float32)
        self.sp = np.zeros((self.cap, obs_dim), np.float32)
        self.a = np.zeros(self.cap, np.int16)
        self.ap = np.zeros(self.cap, np.int16)
        self.cf = np.zeros(self.cap, np.float32)
        self.cv = np.zeros(self.cap, np.float32)
        self.cont = np.zeros(self.cap, np.float32)     # 0 on termination, 1 otherwise (truncation = 1)
        self.i, self.full = 0, False
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return self.cap if self.full else self.i

    def add(self, s, a, cf, cv, terminated, sp, ap):
        i = self.i
        self.s[i], self.a[i], self.cf[i], self.cv[i] = s, a, cf, cv
        self.cont[i] = 0.0 if terminated else 1.0
        self.sp[i], self.ap[i] = sp, ap
        self.i = (i + 1) % self.cap
        self.full = self.full or self.i == 0

    def sample(self, m: int):
        idx = self.rng.integers(0, len(self), size=m)
        return (self.s[idx], self.a[idx].astype(int), self.cf[idx], self.cv[idx],
                self.cont[idx], self.sp[idx], self.ap[idx].astype(int))

    def state(self) -> Dict[str, np.ndarray]:
        n = len(self)
        return {"s": self.s[:n], "sp": self.sp[:n], "a": self.a[:n], "ap": self.ap[:n], "cf": self.cf[:n],
                "cv": self.cv[:n], "cont": self.cont[:n], "i": np.int64(self.i), "full": np.bool_(self.full)}

    def load(self, d) -> None:
        n = len(d["a"])
        for k in ("s", "sp", "a", "ap", "cf", "cv", "cont"):
            getattr(self, k)[:n] = d[k]
        self.i, self.full = int(d["i"]), bool(d["full"])


def td_targets(cf, cv, cont, q_next_f, q_next_v, gamma: float):
    """y = (1-g) c + g' q(s', a') with g' = g * cont  (cont = 0 on termination)."""
    g_next = gamma * cont
    return (1 - gamma) * cf + g_next * q_next_f, (1 - gamma) * cv + g_next * q_next_v


class GVFNet:
    """One network = one gamma; outputs [front safety (|A|), speed (|A|)]."""

    def __init__(self, obs_dim: int, n_actions: int, gamma: float, cfg: PredictorConfig, seed: int):
        self.gamma, self.n, self.cfg = float(gamma), n_actions, cfg
        self.net = MLP([obs_dim, *cfg.hidden, 2 * n_actions], seed)
        self.tnet = None
        if cfg.target_network:
            self.tnet = MLP(self.net.sizes, seed); self.tnet.copy_from(self.net)
        self.updates = 0

    def predict(self, obs: np.ndarray):
        out = self.net.forward(np.atleast_2d(obs))
        return out[:, :self.n], out[:, self.n:]           # (B, |A|) safety, (B, |A|) speed

    def update(self, batch) -> float:
        s, a, cf, cv, cont, sp, ap = batch
        B = len(a); ar = np.arange(B)
        nxt = (self.tnet or self.net).forward(sp)          # current params (paper) unless deviation
        yf, yv = td_targets(cf, cv, cont, nxt[ar, ap], nxt[ar, self.n + ap], self.gamma)
        out = self.net.forward(s, cache=True)
        ef = out[ar, a] - yf
        ev = out[ar, self.n + a] - yv
        d = np.zeros_like(out)
        d[ar, a] = 2 * ef / B
        d[ar, self.n + a] = 2 * ev / B
        self.net.adam(self.net.backward(d), self.cfg.lr)
        if self.tnet is not None:
            self.tnet.polyak_from(self.net, self.cfg.target_tau)
        self.updates += 1
        return float(np.mean(ef ** 2) + np.mean(ev ** 2))


class GVFBank:
    """All gammas, trained from one shared replay memory."""

    def __init__(self, obs_dim: int, n_actions: int, cfg: PredictorConfig, seed: int,
                 gammas: Optional[Sequence[float]] = None):
        self.cfg = cfg
        self.gammas = [float(g) for g in (gammas or cfg.gammas)]
        self.nets = {g: GVFNet(obs_dim, n_actions, g, cfg, seed + 101 * k) for k, g in enumerate(self.gammas)}
        self.replay = GVFReplay(cfg.replay, obs_dim, seed + 7)
        self.n = n_actions

    def maybe_train(self, step: int) -> Optional[float]:
        if len(self.replay) < self.cfg.learning_starts or step % self.cfg.train_every:
            return None
        batch = self.replay.sample(self.cfg.batch)
        return float(np.mean([net.update(batch) for net in self.nets.values()]))

    def predict(self, obs):
        """{gamma: (safety (B,|A|), speed (B,|A|))}."""
        return {g: net.predict(obs) for g, net in self.nets.items()}

    def state(self) -> Dict[str, np.ndarray]:
        d = {}
        for g, net in self.nets.items():
            for k, v in net.net.state().items():
                d[f"g{g}_{k}"] = v
            for k, (m, vv) in enumerate(zip(net.net._m, net.net._v)):
                d[f"g{g}_adam_m{k}"], d[f"g{g}_adam_v{k}"] = m, vv
            d[f"g{g}_adam_t"] = np.int64(net.net.t)
        return d

    def load(self, d) -> None:
        for g, net in self.nets.items():
            net.net.load({k.split("_", 1)[1]: d[k] for k in d if k.startswith(f"g{g}_p")})
            for k in range(len(net.net._m)):
                net.net._m[k][...] = d[f"g{g}_adam_m{k}"]; net.net._v[k][...] = d[f"g{g}_adam_v{k}"]
            net.net.t = int(d[f"g{g}_adam_t"])

    def frozen_copy(self) -> "GVFBank":
        """Stage 2: a frozen snapshot of the current predictors (no replay)."""
        c = GVFBank.__new__(GVFBank)
        c.cfg, c.gammas, c.n, c.replay = self.cfg, list(self.gammas), self.n, None
        c.nets = {}
        for g, net in self.nets.items():
            nn = GVFNet.__new__(GVFNet)
            nn.gamma, nn.n, nn.cfg, nn.tnet, nn.updates = net.gamma, net.n, net.cfg, None, net.updates
            nn.net = MLP(net.net.sizes, 0); nn.net.copy_from(net.net)
            c.nets[g] = nn
        return c
