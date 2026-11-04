"""Evaluation metrics and cross-seed aggregation."""
from __future__ import annotations

from typing import Dict, List

import numpy as np


def episode_outcome(reward_sum: float, length: int, last_info: dict,
                    truncated: bool) -> Dict[str, float]:
    goal = bool(last_info.get("goal_reached", False))
    collision = bool(last_info.get("collision", False))
    return {
        "reward": float(reward_sum),
        "length": float(length),
        "success": float(goal),
        "collision": float(collision and not goal),
        "timeout": float(truncated and not goal and not collision),
    }


def aggregate_episodes(outcomes: List[Dict[str, float]]) -> Dict[str, float]:
    keys = ("reward", "length", "success", "collision", "timeout")
    out = {}
    for k in keys:
        vals = np.array([o[k] for o in outcomes], float)
        out[f"{k}_mean"] = float(vals.mean()) if vals.size else float("nan")
    # rates are means of the 0/1 indicators
    out["success_rate"] = out.pop("success_mean")
    out["collision_rate"] = out.pop("collision_mean")
    out["timeout_rate"] = out.pop("timeout_mean")
    out["n_episodes"] = len(outcomes)
    return out


def aggregate_seeds(per_seed: List[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    """mean/std across seeds for every scalar metric."""
    if not per_seed:
        return {}
    keys = [k for k, v in per_seed[0].items() if isinstance(v, (int, float))]
    out = {}
    for k in keys:
        vals = np.array([s[k] for s in per_seed], float)
        out[k] = {"mean": float(np.nanmean(vals)), "std": float(np.nanstd(vals)),
                  "n": int(np.sum(~np.isnan(vals)))}
    return out
