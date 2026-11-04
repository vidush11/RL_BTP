"""Record golden seeded trajectories from the ORIGINAL simulator (before the
safety-task extension). Run once on the baseline commit; the regression test
tests/test_golden_regression.py asserts the current code reproduces them
bit-for-bit (H2: default behaviour must be unchanged).

    python tests/golden/make_golden.py      # writes tests/golden/golden_v1.pkl
"""
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from simulator import TurtleBot3LidarEnv, load_config  # noqa: E402

CASES = {
    "builtin_defaults": None,
    "default_yaml": "configs/default.yaml",
    "gazebo_compat": "configs/gazebo_compat.yaml",
    "obstacles": "configs/obstacles.yaml",
    "goal_nav": "configs/goal_nav.yaml",
    "obstacles_train": "configs/obstacles_train.yaml",
    "fixed_goal_nav": "configs/fixed_goal_nav.yaml",
    "noise_random_obstacles": {"world": {"randomize_obstacles": True, "random_obstacle_count": 4},
                               "spawn": {"mode": "random"}, "lidar": {"noise_std": 0.02}},
    "action_repeat_short_cap": {"motion": {"action_repeat": 3}, "termination": {"max_episode_steps": 40},
                                "spawn": {"mode": "random"}},
    "no_lidar_rule": {"termination": {"collision_lidar_threshold": None}, "spawn": {"mode": "random"}},
}


def _cfg(c):
    if c is None or isinstance(c, dict):
        return load_config(c)
    return load_config(ROOT / c)


def record(case_cfg, n_episodes=3, max_steps=200, seed=1234):
    env = TurtleBot3LidarEnv(_cfg(case_cfg))
    rng = np.random.default_rng(seed)
    eps = []
    for e in range(n_episodes):
        obs, info = env.reset(seed=seed + e)
        steps = [{"obs": obs.copy(), "info": _copy(info)}]
        for _ in range(max_steps):
            a = int(rng.integers(env.action_space.n))
            obs, r, te, tr, info = env.step(a)
            steps.append({"a": a, "obs": obs.copy(), "r": r, "te": te, "tr": tr, "info": _copy(info)})
            if te or tr:
                break
        eps.append(steps)
    return {"obs_space": (env.observation_space.shape, str(env.observation_space.dtype)),
            "n_actions": env.action_space.n, "episodes": eps}


def _copy(info):
    out = {}
    for k, v in info.items():
        out[k] = v.copy() if isinstance(v, np.ndarray) else v
    return out


if __name__ == "__main__":
    golden = {name: record(c) for name, c in CASES.items()}
    out = Path(__file__).with_name("golden_v1.pkl")
    with open(out, "wb") as fh:
        pickle.dump({"cases": CASES, "golden": golden}, fh)
    n = sum(len(s) for g in golden.values() for s in g["episodes"])
    print(f"wrote {out} ({len(golden)} configs, {n} recorded steps)")
