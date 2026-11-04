#!/usr/bin/env python3
"""Throughput benchmark.  Prints a table and (optionally) writes JSON.

    python benchmarks/benchmark_environment.py
    python benchmarks/benchmark_environment.py --steps 50000 --envs 1 2 4 8 --json bench.json

Measures
  1. single-environment steps/s (random actions, auto-reset)
  2. episode throughput (episodes/s) and mean episode length
  3. LiDAR ray-cast time per call (empty arena and with obstacles)
  4. several environment instances: in-process SyncVectorEnv (sequential, one core)
     and AsyncVectorEnv (one process per env), aggregate steps/s

Numbers depend entirely on the machine; the result prints the CPU count so runs
on a laptop and on an Omni node can be compared.
"""
import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from simulator import TurtleBot3LidarEnv, make_vec_env  # noqa: E402
from simulator.lidar import Lidar  # noqa: E402
from simulator.world import World  # noqa: E402
from simulator.config import LidarConfig, WorldConfig  # noqa: E402

LAYOUT = [[0.6, 0.6, 1.0, 1.0], [-1.2, -1.0, -0.8, -0.6], [0.5, -1.4, 0.7, -0.4],
          [-1.0, 0.8, -0.2, 1.0]]


def bench_single(n_steps, cfg):
    env = TurtleBot3LidarEnv(cfg)
    env.reset(seed=0)
    rng = np.random.default_rng(0)
    acts = rng.integers(0, 5, size=n_steps).tolist()
    episodes, ep_steps = 0, 0
    t0 = time.perf_counter()
    for a in acts:
        _, _, te, tr, _ = env.step(a)
        ep_steps += 1
        if te or tr:
            episodes += 1
            env.reset()
    dt = time.perf_counter() - t0
    return {"steps_per_s": n_steps / dt, "episodes_per_s": episodes / dt,
            "mean_episode_len": n_steps / max(episodes, 1), "seconds": dt}


def bench_lidar(n_calls, obstacles):
    world = World(WorldConfig(obstacles=obstacles))
    lidar = Lidar(LidarConfig())
    rng = np.random.default_rng(0)
    poses = rng.uniform(-1.5, 1.5, size=(n_calls, 3))
    t0 = time.perf_counter()
    for x, y, th in poses:
        lidar.cast(world, x, y, th)
    return (time.perf_counter() - t0) / n_calls * 1e6          # microseconds per cast


def bench_vec(n_envs, steps_per_env, cfg, asynchronous):
    vec = make_vec_env(n_envs, cfg, asynchronous=asynchronous)
    vec.reset(seed=0)
    rng = np.random.default_rng(0)
    acts = rng.integers(0, 5, size=(steps_per_env, n_envs))
    t0 = time.perf_counter()
    for t in range(steps_per_env):
        vec.step(acts[t])
    dt = time.perf_counter() - t0
    vec.close()
    return n_envs * steps_per_env / dt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--lidar-calls", type=int, default=20000)
    ap.add_argument("--envs", type=int, nargs="+", default=[1, 2, 4])
    ap.add_argument("--skip-async", action="store_true")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    res = {"machine": {"platform": platform.platform(), "python": platform.python_version(),
                       "numpy": np.__version__, "cpu_count": os.cpu_count()}}
    print(f"machine: {res['machine']}")

    res["single_empty"] = bench_single(args.steps, None)
    res["single_obstacles"] = bench_single(args.steps, {"world": {"obstacles": LAYOUT}})
    for k in ("single_empty", "single_obstacles"):
        r = res[k]
        print(f"{k:18s}: {r['steps_per_s']:8.0f} steps/s | {r['episodes_per_s']:6.1f} episodes/s "
              f"| mean episode {r['mean_episode_len']:.0f} steps")

    res["lidar_us_empty"] = bench_lidar(args.lidar_calls, [])
    res["lidar_us_4_obstacles"] = bench_lidar(args.lidar_calls, LAYOUT)
    print(f"LiDAR cast (24 beams): {res['lidar_us_empty']:.1f} us empty, "
          f"{res['lidar_us_4_obstacles']:.1f} us with 4 obstacles")

    res["vec"] = {}
    per_env = max(args.steps // 4, 1000)
    for n in args.envs:
        row = {"sync": bench_vec(n, per_env, {"world": {"obstacles": LAYOUT}}, False)}
        if not args.skip_async and n > 1:
            row["async"] = bench_vec(n, per_env, {"world": {"obstacles": LAYOUT}}, True)
        res["vec"][n] = row
        print(f"{n:2d} envs: sync {row['sync']:8.0f} steps/s total"
              + (f" | async {row['async']:8.0f} steps/s total" if "async" in row else ""))

    if args.json:
        Path(args.json).write_text(json.dumps(res, indent=2))
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
