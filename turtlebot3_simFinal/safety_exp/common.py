"""Shared pieces of the safety-task experiment: seeds, env construction, the
single metric function used for EVERY agent (D4), and deterministic evaluation.

Seed ranges (disjoint, asserted):
  training env streams      seed (0, 1, 2; tuning runs 100+)
  TEST    1,000,000+        final 1,000-episode test (fresh, never used before)
  HARD    1,500,000+        final harder-layout test (5 obstacles)
  VAL     2,000,000+        tuning / threshold selection ONLY
  CURVE   3,000,000+        the FIXED 100 learning-curve episodes (same for every
                            method and every evaluation point)
  ABLATE  4,000,000+        post-hoc analysis only (gamma/beta ablation, prediction
                            quality); never used to select anything
"""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from simulator import TurtleBot3LidarEnv, load_config

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "safety_task.yaml"
BASES = {"test": 1_000_000, "hard": 1_500_000, "val": 2_000_000, "curve": 3_000_000, "ablate": 4_000_000}
SPAN = 500_000
NEAR_MISS = 0.25          # m: a step with min LiDAR range below this counts as a near-miss


def seeds(kind: str, n: int) -> List[int]:
    return [BASES[kind] + i for i in range(n)]


def assert_kind(seed_list, kind: str):
    lo = BASES[kind]
    bad = [s for s in seed_list if not lo <= s < lo + SPAN]
    if bad:
        raise AssertionError(f"{kind} evaluation used out-of-range seeds {bad[:3]}")


def sim_config(overrides: Optional[dict] = None, config_path=CONFIG) -> dict:
    cfg = load_config(config_path).to_dict()
    for sec, vals in (overrides or {}).items():
        if isinstance(vals, dict):
            cfg[sec].update(vals)
        else:
            cfg[sec] = vals
    return cfg


def make_env(overrides: Optional[dict] = None) -> TurtleBot3LidarEnv:
    return TurtleBot3LidarEnv(load_config(sim_config(overrides)))


HARD = {"world": {"random_obstacle_count": 5}}


# ------------------------------------------------------------- evaluation
def run_episode(env, policy, seed: int, trace: bool = False) -> Dict:
    obs, info = env.reset(seed=int(seed))
    policy.reset()
    dt = env.cfg.motion.dt * env.cfg.motion.action_repeat
    v_target = env.cfg.reward.v_target
    steps, ret, dist, vsum, wsum, near, minr = 0, 0.0, 0.0, 0.0, 0.0, 0, float("inf")
    qtrace = []
    while True:
        a = policy.act(obs, info)
        if trace:
            qtrace.append(_policy_signal(policy, a))
        obs, r, te, tr, info = env.step(a)
        steps += 1
        ret += r
        v = float(info["linear_velocity"])
        vsum += v; dist += v * dt; wsum += abs(float(info["angular_velocity"]))
        m = float(info["min_range"])
        minr = min(minr, m); near += m < NEAR_MISS
        if te or tr:
            break
    rec = {"seed": int(seed), "steps": steps, "collided": bool(te and info["collision"]),
           "truncated": bool(tr and not te), "task_return": ret, "speed_ratio": vsum / steps / v_target,
           "distance": dist, "min_range": minr, "near_miss_frac": near / steps, "mean_abs_w": wsum / steps}
    if trace:
        rec["trace"] = qtrace
    return rec


def _policy_signal(policy, a):
    if getattr(policy, "last_q", None) is not None:
        return float(np.asarray(policy.last_q)[policy.a_prev if hasattr(policy, "a_prev") else a])
    if getattr(policy, "last", None):
        return float(policy.last["slack"])
    return None


def evaluate(policy, seed_list: List[int], kind: str, overrides: Optional[dict] = None,
             trace: bool = False) -> List[Dict]:
    assert_kind(seed_list, kind)
    env = make_env(overrides)
    return [run_episode(env, policy, s, trace) for s in seed_list]


def summarize(recs: List[Dict]) -> Dict[str, float]:
    """D4 metrics -- the same function for every agent. Survival is always
    reported together with speed (a stopped robot never crashes)."""
    R = recs
    mr = np.array([r["min_range"] for r in R])
    return {"n": len(R),
            "collision_free_rate": float(np.mean([not r["collided"] for r in R])),
            "collision_rate": float(np.mean([r["collided"] for r in R])),
            "mean_steps": float(np.mean([r["steps"] for r in R])),
            "mean_task_return": float(np.mean([r["task_return"] for r in R])),
            "speed_ratio": float(np.mean([r["speed_ratio"] for r in R])),
            "distance": float(np.mean([r["distance"] for r in R])),
            "near_miss_frac": float(np.mean([r["near_miss_frac"] for r in R])),
            "min_range_p05": float(np.percentile(mr, 5)), "min_range_p50": float(np.median(mr)),
            "mean_abs_w": float(np.mean([r["mean_abs_w"] for r in R]))}
