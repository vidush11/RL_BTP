#!/usr/bin/env python3
"""Validation-seed tuning with the SAME training budget for both methods (C, D2).

    python -m safety_exp.tune --out runs_safety/tuning --workers 10

1. 6 GVF predictor/policy configs and 6 DQN configs, 3,000 training episodes
   each (seed 100), curve on VALIDATION seeds (--val-mode, never the test or
   curve seeds).
2. Controller thresholds (eval-only, validation seeds) for every GVF run:
   paper controller gamma x (beta1, beta2) x use_speed_gvf; given-GVF (RL_BTP)
   controller (beta1, beta2) x bump_max.
3. Selection metric for every method: mean task return on validation episodes.
Writes <out>/tuning.json (consumed by safety_exp.run --tuned).
"""
from __future__ import annotations

import argparse
import itertools
import json
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from .common import evaluate, make_env, seeds, summarize

TUNE_EPISODES, TUNE_SEED, N_VAL_GRID, N_VAL_DQN = 3000, 100, 50, 100
import os
if os.environ.get("SAFETY_SMOKE"):          # identical code paths, tiny sizes (debugging only)
    TUNE_EPISODES, N_VAL_GRID, N_VAL_DQN = 30, 2, 3

GVF_CONFIGS = [{"predictor": {"lr": lr}, "policy": {"eps_pi": e}} for lr in (3e-4, 1e-3) for e in (0.05, 0.1, 0.2)]
DQN_CONFIGS = [{"dqn": d} for d in (
    dict(lr=3e-4, gamma=0.99, target_update=1000, eps_decay_frac=0.2),
    dict(lr=1e-3, gamma=0.99, target_update=1000, eps_decay_frac=0.2),
    dict(lr=3e-4, gamma=0.95, target_update=1000, eps_decay_frac=0.2),
    dict(lr=1e-3, gamma=0.95, target_update=1000, eps_decay_frac=0.2),
    dict(lr=5e-4, gamma=0.99, target_update=250, eps_decay_frac=0.1),
    dict(lr=5e-4, gamma=0.99, target_update=4000, eps_decay_frac=0.3))]

PAPER_GRID = [dict(gamma=g, beta1=b1, beta2=b2, use_speed_gvf=s)
              for g in (0.95, 0.975, 0.983)
              for b1, b2 in ((0.3, 0.5), (0.4, 0.6), (0.5, 0.7), (0.5, 0.8), (0.6, 0.8), (0.7, 0.9))
              for s in (True, False)]
HORDE_GRID = [dict(beta1=b1, beta2=b2, bump_max=bm)
              for b1, b2 in ((0, 8), (0, 15), (0, 30), (3, 8), (3, 15), (3, 30), (8, 15), (8, 30))
              for bm in (0.3, 0.6, 1.0)]


def _flags(cfg):
    out = []
    for sec, kv in cfg.items():
        for k, v in kv.items():
            out += ["--set", f"{sec}.{k}={v}"]
    return out


def _train(job):
    method, idx, cfg, out = job
    run = Path(out) / f"{method}_{idx}"
    if (run / "summary.json").exists():
        return str(run)
    cmd = [sys.executable, "-m", "safety_exp.run", "--method", method, "--seed", str(TUNE_SEED),
           "--episodes", str(TUNE_EPISODES), "--eval-every", str(min(1000, TUNE_EPISODES)),
           "--eval-episodes", str(min(50, N_VAL_DQN)),
           "--val-mode", "--out", str(run)] + _flags(cfg)
    with open(run.with_suffix(".log"), "w") as fh:
        subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, check=True)
    return str(run)


def _load_gvf(run: Path):
    from agents.gvf.predictor import GVFBank, PredictorConfig
    from agents.horde.adapter import HordeAdapter, load_rlbtp
    from agents.horde.controller import frozen_horde
    c = json.loads((run / "config.json").read_text())["cfgs"]["predictor"]
    env = make_env()
    pc = PredictorConfig(**{**c, "gammas": tuple(c["gammas"]), "hidden": tuple(c["hidden"])})
    bank = GVFBank(env.observation_space.shape[0], env.action_space.n, pc, 0)
    bank.load(np.load(run / "predictors.npz"))
    rl = load_rlbtp()
    horde = rl["horde"].Horde(rl["demons"].DEMONS)
    horde.theta = np.load(run / "horde.npz")["theta"]
    return env, bank.frozen_copy(), frozen_horde(horde), HordeAdapter(env)


def _grid(run_dir):
    from agents.gvf.controller import ControllerConfig, PaperController
    from agents.horde.controller import HordeController, HordeControllerConfig
    run = Path(run_dir)
    env, bank, horde, ad = _load_gvf(run)
    vs = seeds("val", N_VAL_GRID)
    paper = []
    for g in PAPER_GRID:
        s = summarize(evaluate(PaperController(bank, env.action_names, ControllerConfig(**g), env.cfg.motion.v_max),
                               vs, "val"))
        paper.append({**g, **{k: s[k] for k in ("mean_task_return", "collision_free_rate", "speed_ratio")}})
    horde_rows = []
    for g in HORDE_GRID:
        s = summarize(evaluate(HordeController(horde, ad, env.action_names, HordeControllerConfig(**g)), vs, "val"))
        horde_rows.append({**g, **{k: s[k] for k in ("mean_task_return", "collision_free_rate", "speed_ratio")}})
    return run_dir, paper, horde_rows


def _dqn_score(run_dir):
    from agents.dqn.dqn import DQNPolicy
    from gvfctl.nn import MLP
    run = Path(run_dir)
    env = make_env()
    c = json.loads((run / "config.json").read_text())["cfgs"]["dqn"]
    q = MLP([env.observation_space.shape[0], *c["hidden"], env.action_space.n], 0)
    q.load(np.load(run / "dqn.npz"))
    s = summarize(evaluate(DQNPolicy(q), seeds("val", N_VAL_DQN), "val"))
    return run_dir, s


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=10)
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    jobs = [("gvf", i, c, out) for i, c in enumerate(GVF_CONFIGS)] + [("dqn", i, c, out) for i, c in enumerate(DQN_CONFIGS)]
    with ProcessPoolExecutor(args.workers) as ex:
        runs = list(ex.map(_train, jobs))
    gvf_runs = [r for r in runs if Path(r).name.startswith("gvf")]
    dqn_runs = [r for r in runs if Path(r).name.startswith("dqn")]
    with ProcessPoolExecutor(args.workers) as ex:
        grids = list(ex.map(_grid, gvf_runs))
        dqn_scores = list(ex.map(_dqn_score, dqn_runs))

    table = {"gvf": [], "dqn": []}
    best_gvf = None
    for (run, paper, horde_rows), cfg in zip(grids, GVF_CONFIGS):
        bp = max(paper, key=lambda r: r["mean_task_return"])
        bh = max(horde_rows, key=lambda r: r["mean_task_return"])
        row = {"run": run, "config": cfg, "best_paper": bp, "best_horde": bh, "paper_grid": paper, "horde_grid": horde_rows}
        table["gvf"].append(row)
        if best_gvf is None or bp["mean_task_return"] > best_gvf["best_paper"]["mean_task_return"]:
            best_gvf = row
    best_dqn = None
    for (run, s), cfg in zip(dqn_scores, DQN_CONFIGS):
        table["dqn"].append({"run": run, "config": cfg, "val": s})
        if best_dqn is None or s["mean_task_return"] > best_dqn["val"]["mean_task_return"]:
            best_dqn = {"run": run, "config": cfg, "val": s}
    strip = lambda r: {k: v for k, v in r.items() if k not in ("mean_task_return", "collision_free_rate", "speed_ratio")}
    tuned = {"predictor": best_gvf["config"]["predictor"], "policy": best_gvf["config"]["policy"],
             "controller": strip(best_gvf["best_paper"]), "horde_controller": strip(best_gvf["best_horde"]),
             "dqn": best_dqn["config"]["dqn"],
             "selection": {"metric": "mean task return on validation seeds",
                           "gvf_run": best_gvf["run"], "dqn_run": best_dqn["run"]},
             "table": table}
    (out / "tuning.json").write_text(json.dumps(tuned, indent=2))
    print("GVF configs (best paper-controller validation return | best given-GVF controller):")
    for r in table["gvf"]:
        print(f"  {r['config']}  paper {r['best_paper']['mean_task_return']:.1f} "
              f"(cf {r['best_paper']['collision_free_rate']:.2f})  horde {r['best_horde']['mean_task_return']:.1f}")
    print("DQN configs (validation return):")
    for r in table["dqn"]:
        print(f"  {r['config']}  {r['val']['mean_task_return']:.1f} (cf {r['val']['collision_free_rate']:.2f})")
    print(f"selected: {json.dumps({k: tuned[k] for k in ('predictor', 'policy', 'controller', 'horde_controller', 'dqn')})}")


if __name__ == "__main__":
    main()
