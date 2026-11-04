#!/usr/bin/env python3
"""Train ONE experimental condition (Agent A, B, or C).

    python scripts/train_dqn.py --mode A --seed 0 --out runs_dqn/A_s0
    python scripts/train_dqn.py --mode B --seed 0 --out runs_dqn/B_s0 \
        --gvf-checkpoint runs_dqn/gvf_pretrained.npz
    python scripts/train_dqn.py --mode C --seed 0 --out runs_dqn/C_s0 \
        --gvf-checkpoint runs_dqn/gvf_pretrained.npz

A = LiDAR-only, B = frozen-GVF, C = adaptive-GVF. B and C require the SAME
pretrained GVF checkpoint. Auto-resumes from <out>/agent.pkl (replay, target
net, optimizer and epsilon step all restored). Baseline runs are never touched.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rl.experiment import load_exp_config, train
from simulator import hpc


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["A", "B", "C"], required=True)
    ap.add_argument("--config", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--gvf-checkpoint", default=None)
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--steps", type=int, default=None, help="override total_env_steps")
    args = ap.parse_args()

    if args.mode in ("B", "C") and not args.gvf_checkpoint:
        ap.error("--gvf-checkpoint is required for modes B and C")

    exp = load_exp_config(args.config)
    if args.steps is not None:
        exp["total_env_steps"] = args.steps
    hpc.configure_threads(1)
    hpc.write_run_metadata(args.out, config=exp, seed=args.seed, agent=f"dqn_{args.mode}",
                           extra={"gvf_checkpoint": args.gvf_checkpoint})
    summary = train(args.mode, exp, args.seed, args.out,
                    gvf_ckpt=args.gvf_checkpoint, resume=not args.no_resume)
    print(f"[train_dqn] done: mode={summary['mode']} seed={summary['seed']} "
          f"in_dim={summary['in_dim']} steps={summary['env_steps']} "
          f"wall={summary['wall_time_s']}s")
    if summary["final"]:
        f = summary["final"]
        print(f"[train_dqn] final eval: success={f['success_rate']:.2f} "
              f"collision={f['collision_rate']:.2f} reward={f['reward_mean']:.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
