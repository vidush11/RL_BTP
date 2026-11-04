"""Horde adapter (A6): this simulator's LiDAR + info -> RL_BTP's 9-sensor vector
[IR_F, IR_L, IR_R, LIGHT_L, LIGHT_R, VEL, ANG_VEL, BUMP, COMPASS], so that the
unedited RL_BTP Horde (third_party/rl_btp) can learn its demons here.

* IR_F/IR_L/IR_R: the LiDAR beams nearest 0, +1 and -1 rad, squashed with
  exp(-d/0.6) exactly as RL_BTP's prox(). Unlike RL_BTP, which sees only the
  arena walls, these rays also hit obstacles.
* LIGHT_L/R: info["light"] (two sensors at +/-1 rad, RL_BTP formula), else 0.
* VEL / ANG_VEL: robot twist (m/s, rad/s). BUMP: 1.0 on the collision step.
  COMPASS: cos(theta).
No sensor noise is added; RL_BTP adds N(0, 0.005), which this adapter leaves out.
"""
from __future__ import annotations

import importlib
import math
import sys
from pathlib import Path

import numpy as np

RLBTP_DIR = Path(__file__).resolve().parents[2] / "third_party" / "rl_btp"


def load_rlbtp():
    """Import the unedited RL_BTP modules (env, features, horde, demons)."""
    if str(RLBTP_DIR) not in sys.path:
        sys.path.insert(0, str(RLBTP_DIR))
    return {name: importlib.import_module(name) for name in ("env", "features", "horde", "demons")}


class HordeAdapter:
    def __init__(self, env):
        ang = np.mod(np.asarray(env.unwrapped.lidar.angles) + math.pi, 2 * math.pi) - math.pi
        self.idx = [int(np.argmin(np.abs(np.angle(np.exp(1j * (ang - t)))))) for t in (0.0, 1.0, -1.0)]
        self.idx_angles = [float(ang[i]) for i in self.idx]
        u = env.unwrapped
        self.names = list(u.action_names) if u.accel_mode else None
        rl = load_rlbtp()
        self.Action = rl["env"].Action
        self.N_ACTIONS = rl["env"].N_ACTIONS

    def sensors(self, info) -> np.ndarray:
        r = np.asarray(info["ranges_m"], float)
        ir = np.exp(-r[self.idx] / 0.6)
        lt = info.get("light", (0.0, 0.0))
        return np.array([ir[0], ir[1], ir[2], lt[0], lt[1], float(info["linear_velocity"]),
                         float(info["angular_velocity"]), 1.0 if info["collision"] else 0.0,
                         math.cos(info["pose"][2])], dtype=np.float64)

    def rlbtp_action(self, env_action: int) -> int:
        """Env action index -> RL_BTP Action value (names match exactly)."""
        return int(self.Action[self.names[int(env_action)]])

    def env_action(self, rlbtp_action: int):
        """RL_BTP Action -> env index, or None if that action is disabled here."""
        name = self.Action(int(rlbtp_action)).name
        return self.names.index(name) if name in self.names else None
