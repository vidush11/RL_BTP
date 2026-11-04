"""Prediction-based reward for Experiment D (LiDAR-only DQN, GVF-shaped reward).

    r_t = r_task,t + lam * r_GVF,t

r_task (explicit terminal policy, computed from the env's outcome flags, not
from the env's own shaped reward):
    collision            -> collision_penalty   (takes precedence over success)
    goal reached         -> goal_reward
    step-limit truncation-> step_reward + timeout_penalty
    otherwise            -> step_reward

r_GVF (default "potential"): potential-based shaping (Ng, Harada & Russell 1999)

    r_GVF = gamma * Phi(s') - Phi(s),     Phi(s') := 0 if s' is terminal

Phi(s) is a bounded SAFETY potential built from the three directional-TTC GVF
predictions (forward/left/right probe policies):

    Phi(s) = agg_d tanh( clip(TTC_d(s), 0, cap) / scale )  in [0, 1)

Why potential-based: summed over an episode the shaping telescopes to
gamma^T Phi(s_T) - Phi(s_0), so the agent cannot farm reward by circling in a
safe area (the brief's reward-hacking concern); it only redistributes credit
toward states the GVFs judge safer. With a fixed Phi it provably leaves the
optimal policy unchanged; here Phi is learned concurrently, so that guarantee
holds only approximately (documented in DESIGN_EXPERIMENT_D.md).

Semantics (do not over-interpret): each TTC_d is the expected number of steps
until the full LiDAR scan reads <= 0.25 m IF the robot followed fixed probe
policy d from s. It is a state-level safety heuristic, NOT the outcome of the
DQN's chosen action and NOT a calibrated probability. Never-colliding
(censored/unbounded) futures saturate tanh -> Phi ~ 1 ("safe"), which is the
intended reading and keeps the reward bounded. Non-finite predictions map to
Phi contribution 0 and are counted, never hidden.

"absolute" shaping (r_GVF = Phi(s'), 0 if terminal) is an ABLATION only: it is
not potential-based and is expected to invite safe-circling reward hacking.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict, fields
from typing import NamedTuple, Optional

import numpy as np


@dataclass
class PredictionRewardConfig:
    goal_reward: float = 100.0
    collision_penalty: float = -50.0
    step_reward: float = 0.0
    timeout_penalty: float = 0.0
    lam: float = 1.0                  # weight of the GVF shaping term
    gamma: Optional[float] = None     # None -> use the DQN gamma (required for potential shaping)
    shaping: str = "potential"        # "potential" (main) | "absolute" (ablation)
    aggregate: str = "mean"           # mean | max | min | forward
    ttc_scale: float = 50.0           # steps; tanh(TTC/scale)
    ttc_cap: float = 400.0            # steps; bound before tanh
    prediction_only: bool = False     # ABLATION: drop goal/collision/step task reward

    @classmethod
    def from_dict(cls, d: dict) -> "PredictionRewardConfig":
        valid = {f.name for f in fields(cls)}
        unknown = set(d or {}) - valid
        if unknown:
            raise ValueError(f"unknown reward keys {sorted(unknown)}")
        cfg = cls(**(d or {}))
        if cfg.shaping not in ("potential", "absolute"):
            raise ValueError("reward.shaping must be 'potential' or 'absolute'")
        if cfg.aggregate not in ("mean", "max", "min", "forward"):
            raise ValueError("reward.aggregate must be mean|max|min|forward")
        if cfg.ttc_scale <= 0 or cfg.ttc_cap <= 0:
            raise ValueError("ttc_scale and ttc_cap must be > 0")
        return cfg

    def to_dict(self) -> dict:
        return asdict(self)


class RewardBreakdown(NamedTuple):
    total: float
    task: float
    gvf: float          # unweighted r_GVF (total = task + lam * gvf)
    phi_s: float
    phi_sp: float


class PredictionReward:
    def __init__(self, cfg: PredictionRewardConfig, dqn_gamma: float):
        self.cfg = cfg
        self.gamma = float(cfg.gamma if cfg.gamma is not None else dqn_gamma)
        self.nonfinite_count = 0

    # ---- Phi ----------------------------------------------------------------
    def potential(self, raw_ttc) -> float:
        """Bounded safety potential in [0, 1) from raw TTC predictions
        (forward, left, right), in steps."""
        ttc = np.asarray(raw_ttc, dtype=np.float64).reshape(-1)
        bad = ~np.isfinite(ttc)
        if bad.any():
            self.nonfinite_count += int(bad.sum())
            ttc = np.where(bad, 0.0, ttc)
        s = np.tanh(np.clip(ttc, 0.0, self.cfg.ttc_cap) / self.cfg.ttc_scale)
        agg = self.cfg.aggregate
        if agg == "forward":
            return float(s[0])
        return float({"mean": s.mean, "max": s.max, "min": s.min}[agg]())

    # ---- reward -------------------------------------------------------------
    def task_reward(self, collision: bool, success: bool, truncated: bool) -> float:
        c = self.cfg
        if collision:                       # precedence: collision over success
            return c.collision_penalty
        if success:
            return c.goal_reward
        return c.step_reward + (c.timeout_penalty if truncated else 0.0)

    def compute(self, phi_s: float, phi_sp: float, *, collision: bool, success: bool,
                terminated: bool, truncated: bool) -> RewardBreakdown:
        """Reward for ONE transition. phi_s / phi_sp must both be computed from
        s_t and s_{t+1} with the SAME (pre-update) GVF weights; nothing here
        looks beyond s_{t+1} or the outcome of this step."""
        c = self.cfg
        phi_sp_eff = 0.0 if terminated else float(phi_sp)
        if c.shaping == "potential":
            g = self.gamma * phi_sp_eff - float(phi_s)
        else:  # "absolute" ablation
            g = phi_sp_eff
        task = 0.0 if c.prediction_only else self.task_reward(collision, success, truncated)
        total = task + c.lam * g
        if not np.isfinite(total):          # cannot happen with bounded Phi; guard anyway
            self.nonfinite_count += 1
            total, g = task, 0.0
        return RewardBreakdown(float(total), float(task), float(g),
                               float(phi_s), float(phi_sp_eff))
