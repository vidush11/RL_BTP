"""Rule-based predictive controller (paper Alg. 3, then Alg. 2 with the speed GVF).

The paper adjusts continuous throttle/brake; this is the discrete analogue for
accel actions (DECISIONS.md):
  a_hat = a_prev;  q = q_front(s, a_hat)          (gamma chosen on validation)
  if q < beta1:   a = argmax_a q_front(s, a) over enabled actions (ties -> STOP)
                  -- "a sweep needs only |A| predictions"
  elif q > beta2: speed control (Alg. 2): v_hat = q_speed(s, a_prev) * v_max
                  (or the current speed if use_speed_gvf is False)
                  v_hat < v_target - tol -> FORWARD;  v_hat > v_target + tol -> STOP
                  otherwise keep a_prev
  else:           keep a_prev   (beta1 <= beta2: hysteresis)
Predictions come from a FROZEN predictor snapshot (stage 2 of the paper).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass
class ControllerConfig:
    gamma: float = 0.975
    beta1: float = 0.5
    beta2: float = 0.8
    use_speed_gvf: bool = True
    v_target: float = 0.2
    v_tol: float = 0.01          # v_target + v_tol must stay < v_max (0.22) or the slow-down branch never fires

    def to_dict(self):
        return asdict(self)


class PaperController:
    name = "paper-gvf"

    def __init__(self, bank, action_names, cfg: ControllerConfig, v_max: float):
        self.bank, self.names, self.cfg, self.v_max = bank, list(action_names), cfg, float(v_max)
        self.net = bank.nets[float(cfg.gamma)]
        self.FWD, self.STOP = self.names.index("FORWARD"), self.names.index("STOP")
        self.tie = np.zeros(len(self.names)); self.tie[self.STOP] = 1e-6
        self.reset()

    def reset(self):
        self.a_prev = self.STOP          # at rest: "keep doing" = coast
        self.last_q = None

    def act(self, obs, info) -> int:
        c = self.cfg
        qf, qv = self.net.predict(obs)
        qf, qv = qf[0], qv[0]
        q = float(qf[self.a_prev])
        if q < c.beta1:
            a = int(np.argmax(qf + self.tie))
        elif q > c.beta2:
            v_hat = float(qv[self.a_prev]) * self.v_max if c.use_speed_gvf else float(info["linear_velocity"])
            if v_hat < c.v_target - c.v_tol:
                a = self.FWD
            elif v_hat > c.v_target + c.v_tol:
                a = self.STOP
            else:
                a = self.a_prev
        else:
            a = self.a_prev
        self.last_q = qf
        self.a_prev = a
        return a
