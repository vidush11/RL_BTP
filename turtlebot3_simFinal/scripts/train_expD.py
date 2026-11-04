#!/usr/bin/env python3
"""Train Experiment D (LiDAR-only DQN, GVF-shaped reward) for one seed.

    python scripts/train_expD.py --seed 0 --out runs_expD/D_s0
    # lambda = 0 control (same everything, no GVF shaping):
    python scripts/train_expD.py --seed 0 --out runs_expD/D0_s0 \
        --set reward.lam=0 --set experiment_id=D0_lidar_only_terminal_reward
    # fixed, LiDAR-identifiable scenario:
    python scripts/train_expD.py --config configs/dqn/experiment_d_fixed.yaml --seed 0 --out runs_expD_fixed/D_s0

Auto-resumes from <out>/agent.pkl (Q-net, target net, optimizer, replay buffer,
GVF weights, step and episode counters). Writes run_metadata.json,
episodes.jsonl (one record per episode), history.json (periodic greedy evals),
summary.json, gvf_latest.npz. Never touches A/B/C runs.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rl.experiment_d import load_d_config, parse_set, train_d
from simulator import hpc


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(ROOT / "configs" / "dqn" / "experiment_d.yaml"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=None, help="override total_env_steps")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                    help="config override, e.g. reward.lam=0 (repeatable)")
    ap.add_argument("--no-resume", action="store_true")
    args = ap.parse_args()

    overrides = parse_set(args.set)
    if args.steps is not None:
        overrides["total_env_steps"] = args.steps
    exp = load_d_config(args.config, overrides)
    hpc.configure_threads(1)
    hpc.write_run_metadata(args.out, config=exp, seed=args.seed, agent=exp["experiment_id"])
    s = train_d(exp, args.seed, args.out, resume=not args.no_resume)
    f = s["final"] or {}
    print(f"[train_expD] {s['experiment_id']} seed={s['seed']} steps={s['env_steps']} "
          f"episodes={s['episodes']} in_dim={s['in_dim']} wall={s['wall_time_s']}s")
    if f:
        print(f"[train_expD] final greedy eval: success={f['success_rate']:.2f} "
              f"collision={f['collision_rate']:.2f} timeout={f['timeout_rate']:.2f} "
              f"task_return={f['task_return_mean']:.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
