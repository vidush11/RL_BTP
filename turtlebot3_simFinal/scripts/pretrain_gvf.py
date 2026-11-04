#!/usr/bin/env python3
"""Pretrain the 3 directional-TTC GVF demons for Agents B and C.

Documented behaviour policy: uniform random over the 5 actions (b=0.2), the same
off-policy data source the Horde demons were designed for. Trains in the
obstacle scenario so the left/right TTC demons are well-posed (bounded). Saves
ONE checkpoint (Horde .npz keys + feature normalization stats) that B freezes
and C initialises from. Pretraining interactions are reported separately from
the DQN budget.

    python scripts/pretrain_gvf.py --config configs/experiment.yaml \
        --episodes 2000 --seed 0 --out runs_dqn/gvf_pretrained.npz
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rl.experiment import load_exp_config, make_env
from rl.gvf_predictor import GVFPredictor
from simulator import hpc


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--episodes", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    exp = load_exp_config(args.config)
    g = exp["gvf"]
    env = make_env(exp)
    pred = GVFPredictor(frozen=False, obstacle_threshold=g["obstacle_threshold"],
                        alpha_theta=g["alpha_theta"], alpha_w=g["alpha_w"],
                        trace_lambda=g["trace_lambda"], clip_z=g["clip_z"])
    rng = np.random.default_rng(args.seed)
    env.reset(seed=args.seed)

    raw_preds = []
    interactions = 0
    for ep in range(args.episodes):
        obs, info = env.reset()
        lidar_t = np.asarray(obs, float)
        pred.reset_traces()
        while True:
            a = int(rng.integers(0, 5)); bprob = 1.0 / 5
            raw_preds.append(pred.raw_predict(lidar_t))
            nobs, r, term, trunc, info = env.step(a)
            interactions += 1
            lidar_tp1 = np.asarray(nobs, float)
            pred.update(lidar_t, a, lidar_tp1, bool(term or trunc), bprob)
            lidar_t = lidar_tp1
            if term or trunc:
                break
        if (ep + 1) % max(1, args.episodes // 10) == 0:
            th = pred.theta_snapshot()
            print(f"[pretrain] ep {ep+1}/{args.episodes}  interactions={interactions}  "
                  f"max|theta|={np.abs(th).max():.2f}  nonfinite={pred.nonfinite_count}")

    raw = np.asarray(raw_preds, float)
    mean = raw.mean(axis=0); std = raw.std(axis=0)
    pred.set_normalization(mean, std)

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    payload = {"episode": np.int64(args.episodes),
               "feat_mean": mean, "feat_std": np.where(std > 1e-6, std, 1.0),
               "pretrain_interactions": np.int64(interactions),
               "behaviour_policy": "uniform_random_b0.2",
               "config": exp["sim_config"]}
    st = pred.get_state()
    for k in ("theta_forward", "w_forward", "theta_left", "w_left",
              "theta_right", "w_right"):
        payload[k] = st[k]
    hpc.atomic_savez(out, **payload)
    print(f"[pretrain] saved {out}")
    print(f"[pretrain] feature means {np.round(mean,2)} stds {np.round(std,2)}  "
          f"interactions={interactions} (reported separately from DQN budget)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
