"""Predictive controller driven by the GIVEN GVFs (RL_BTP demons, unedited),
learned on this simulator through the Horde adapter. Same structure as the
paper controller, but with the RL_BTP questions:

  tto  = q_steps-to-obstacle(o, FORWARD)   steps until near an obstacle if I keep driving
  tts  = q_steps-to-stop(o, STOP)          steps needed to coast to a stop
  bump = q_bump-soon(o, FORWARD)           discounted chance of bumping if I keep driving
  slack = tto - tts
  danger (slack < beta1 or bump > bump_max):
        a = argmax over enabled actions other than FORWARD of
            q_avoid-front-obstacles[ctrl](o, a)          (ties -> STOP)
  slack > beta2 and v < v_target - tol:   FORWARD
  otherwise:                              keep a_prev   (hysteresis)
The Horde is a frozen snapshot (theta copied) at every evaluation point.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass
class HordeControllerConfig:
    beta1: float = 5.0         # steps of slack
    beta2: float = 15.0
    bump_max: float = 0.5
    v_target: float = 0.2
    v_tol: float = 0.01          # v_target + v_tol must stay < v_max (0.22) or the slow-down branch never fires

    def to_dict(self):
        return asdict(self)


def frozen_horde(horde):
    h = horde.__class__.__new__(horde.__class__)
    h.demons = horde.demons
    h.theta = horde.theta.copy()
    return h


class HordeController:
    name = "given-gvf-horde"

    def __init__(self, horde, adapter, action_names, cfg: HordeControllerConfig):
        self.h, self.ad, self.names, self.cfg = horde, adapter, list(action_names), cfg
        dn = [d["name"] for d in horde.demons]
        self.i_tto, self.i_tts = dn.index("steps-to-obstacle"), dn.index("steps-to-stop")
        self.i_bump, self.i_avoid = dn.index("bump-soon"), dn.index("avoid-front-obstacles [ctrl]")
        A = adapter.Action
        self.rl = [int(A[n]) for n in self.names]
        self.FWD, self.STOP = self.names.index("FORWARD"), self.names.index("STOP")
        self.rl_fwd, self.rl_stop = int(A.FORWARD), int(A.STOP)
        self.reset()

    def reset(self):
        self.a_prev = self.STOP
        self.last = None

    def act(self, obs, info) -> int:
        c = self.cfg
        Q = self.h.q(self.ad.sensors(info))
        tto, tts = float(Q[self.i_tto, self.rl_fwd]), float(Q[self.i_tts, self.rl_stop])
        bump = float(Q[self.i_bump, self.rl_fwd])
        slack = tto - tts
        if slack < c.beta1 or bump > c.bump_max:
            cand = [i for i in range(len(self.names)) if i != self.FWD]
            vals = [Q[self.i_avoid, self.rl[i]] + (1e-6 if i == self.STOP else 0.0) for i in cand]
            a = cand[int(np.argmax(vals))]
        elif slack > c.beta2 and float(info["linear_velocity"]) < c.v_target - c.v_tol:
            a = self.FWD
        else:
            a = self.a_prev
        self.last = {"tto": tto, "tts": tts, "bump": bump, "slack": slack}
        self.a_prev = a
        return a
