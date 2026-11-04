#!/usr/bin/env python3
"""Run the full A/B/C comparison across seeds and aggregate the results.

    python scripts/run_comparison.py --seeds 0 1 2 3 4 \
        --gvf-checkpoint runs_dqn/gvf_pretrained.npz --out runs_dqn/compare

Trains all three agents with the SAME DQN budget, config, eval scenarios and
PAIRED seeds, then writes comparison.json (per-seed + cross-seed mean/std) and a
summary table. Pretrain the GVF checkpoint first (scripts/pretrain_gvf.py).
Use --modes to run a subset. Does not overwrite existing agent runs unless
--fresh is given (it resumes them).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rl.experiment import load_exp_config, train
from rl import metrics as M


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--modes", nargs="+", default=["A", "B", "C"])
    ap.add_argument("--gvf-checkpoint", default=None)
    ap.add_argument("--out", default="runs_dqn/compare")
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--fresh", action="store_true")
    args = ap.parse_args()

    exp = load_exp_config(args.config)
    if args.steps is not None:
        exp["total_env_steps"] = args.steps
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    per_seed = {m: [] for m in args.modes}
    for seed in args.seeds:
        for mode in args.modes:
            gvf = args.gvf_checkpoint if mode in ("B", "C") else None
            run_dir = out / f"{mode}_s{seed}"
            print(f"\n===== mode {mode} seed {seed} -> {run_dir} =====")
            summ = train(mode, exp, seed, str(run_dir), gvf_ckpt=gvf,
                         resume=not args.fresh)
            if summ["final"]:
                per_seed[mode].append({**summ["final"], "wall_time_s": summ["wall_time_s"],
                                       "gvf_updates": summ["gvf_updates"]})

    agg = {m: M.aggregate_seeds(per_seed[m]) for m in args.modes}
    comparison = {"config": exp, "seeds": args.seeds, "per_seed": per_seed, "aggregate": agg}
    (out / "comparison.json").write_text(json.dumps(comparison, indent=2))

    print("\n================ CROSS-SEED SUMMARY (mean +/- std) ================")
    cols = ["success_rate", "collision_rate", "reward_mean", "length_mean", "timeout_rate"]
    hdr = "mode | " + " | ".join(f"{c:>14}" for c in cols)
    print(hdr); print("-" * len(hdr))
    for m in args.modes:
        a = agg[m]
        row = f"  {m}  | " + " | ".join(
            (f"{a[c]['mean']:.3f}+/-{a[c]['std']:.3f}" if c in a else " n/a").rjust(14)
            for c in cols)
        print(row)
    print(f"\nwrote {out/'comparison.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
