"""Environment registry, seed discipline and action-table plumbing.

Seed ranges (disjoint; asserted wherever they are consumed):
  training env seeds        < 1,000,000   (main 0-9, tuning 100+)
  TEST episodes             1,000,000+    final evaluation ONLY
  VALIDATION episodes       2,000,000+    tuning / threshold selection ONLY
  CURVE episodes            3,000,000+    learning-curve checkpoints (never used for selection)
  QUALITY states            4,000,000+    GVF prediction-quality Monte-Carlo
"""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Optional

import numpy as np
import yaml

from simulator import TurtleBot3LidarEnv, load_config
from .actions import ActionSpace, SPEC5, SPEED15

ROOT = Path(__file__).resolve().parent.parent

TEST_BASE, VAL_BASE, CURVE_BASE, QUALITY_BASE = 1_000_000, 2_000_000, 3_000_000, 4_000_000
_RANGES = {"test": (TEST_BASE, VAL_BASE), "val": (VAL_BASE, CURVE_BASE),
           "curve": (CURVE_BASE, QUALITY_BASE), "quality": (QUALITY_BASE, 5_000_000)}

ENVS = {
    "train":    ("configs/obstacles_train.yaml", {}),
    "dense":    ("configs/obstacles_dense.yaml", {}),
    "empty":    ("configs/default_random_spawn.yaml", {}),
    "corridor": ("configs/obstacles_corridor.yaml", {}),
    "clutter":  ("configs/obstacles_clutter_extreme.yaml", {}),
    "noise001": ("configs/obstacles_train.yaml", {"lidar": {"noise_std": 0.01}}),
    "noise003": ("configs/obstacles_train.yaml", {"lidar": {"noise_std": 0.03}}),
    "gazebo":   ("configs/gazebo_compat.yaml", {}),
    "gazebo_train": ("configs/obstacles_train.yaml", {"__actions_from__": "configs/gazebo_compat.yaml"}),
}
ENV_INDEX = {n: i for i, n in enumerate(ENVS)}
PART_ENVS = {1: list(ENVS), 2: [n for n in ENVS if not n.startswith("gazebo")]}


def action_space(part: int, persist_p: float = 0.1) -> ActionSpace:
    return ActionSpace(SPEC5 if part == 1 else SPEED15, persist_p=persist_p)


def _deep_merge(a: dict, b: dict) -> dict:
    out = copy.deepcopy(a)
    for k, v in b.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _yaml(path) -> dict:
    return yaml.safe_load((ROOT / path).read_text()) or {}


def env_config(name: str, part: int, cap: Optional[int] = 800) -> dict:
    path, over = ENVS[name]
    cfg = _yaml(path)
    over = dict(over)
    src = over.pop("__actions_from__", None)
    cfg = _deep_merge(cfg, over)
    if part == 2:
        if name.startswith("gazebo"):
            raise ValueError("gazebo envs have a 5-action table; Part 2 uses 15 actions")
        cfg = _deep_merge(cfg, _yaml("configs/speed_actions.yaml"))
    if src:
        cfg = _deep_merge(cfg, {"motion": {"actions": _yaml(src)["motion"]["actions"]}})
    cfg = _deep_merge(cfg, {"termination": {"max_episode_steps": cap}})
    cfg.pop("seed", None)
    return cfg


def make_env(name: str, part: int, cap: Optional[int] = 800):
    """Returns (env, remap) where remap[agent_action] = env action index."""
    env = TurtleBot3LidarEnv(load_config(env_config(name, part, cap)))
    remap = action_space(part).remap_to(env.actions.table)
    return env, remap


def episode_seeds(kind: str, env_name: str, n: int, start: int = 0):
    base = _RANGES[kind][0] + 10_000 * ENV_INDEX[env_name]
    seeds = [base + start + i for i in range(n)]
    assert_seeds(seeds, kind)
    return seeds


def assert_seeds(seeds, kind: str) -> None:
    lo, hi = _RANGES[kind]
    bad = [s for s in seeds if not (lo <= s < hi)]
    if bad:
        raise AssertionError(f"{kind} evaluation used out-of-range seeds {bad[:3]}")


def assert_training_seed(seed: int) -> None:
    if not (0 <= seed < TEST_BASE):
        raise AssertionError(f"training seed {seed} overlaps evaluation seed ranges")
