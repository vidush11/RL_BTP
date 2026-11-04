#!/usr/bin/env python3
"""Evaluate A, B, C and D checkpoints on the SAME evaluation scenarios and
report TASK OUTCOMES only (success / collision / timeout / episode length).

Returns are deliberately not compared: D's reward (terminal + GVF shaping)
differs from A/B/C's env reward, so their scales are not commensurable.

    python scripts/compare_agents.py --abc-dir runs_dqn/compare --d-dir runs_expD \
        --eval-seed 20000 --eval-episodes 100 --out runs_expD/compare_ABCD.json

Expects <abc-dir>/<A|B|C>_s<seed>/agent.pkl and <d-dir>/<label>_s<seed>/agent.pkl
(every D-style run under --d-dir is reported under its directory label, e.g. D
and D0). Refuses to compare if the environment or task differ.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rl import experiment as X
from rl.experiment_d import evaluate_d, load_d_checkpoint

KEYS = ("success_rate", "collision_rate", "timeout_rate", "length_mean")


def _runs(dir_: Path):
    for p in sorted(dir_.glob("*_s*/agent.pkl")):
        m = re.match(r"(.+)_s(\d+)$", p.parent.name)
        if m:
            yield m.group(1), int(m.group(2)), p


def evaluate_all(abc_dir, abc_config, d_dir, eval_seed, eval_episodes):
    abc_exp = X.load_exp_config(abc_config)
    results = defaultdict(list)
    env_key = None
    if abc_dir and Path(abc_dir).is_dir():
        env_key = (abc_exp["sim_config"], abc_exp["task"])
        env = X.make_env(abc_exp)
        for label, seed, path in _runs(Path(abc_dir)):
            if label not in ("A", "B", "C"):
                continue
            agent = X.build_agent(label, abc_exp, None, seed)
            agent.load(path, with_replay=False)
            agg, _ = X.evaluate(agent, env, eval_episodes, eval_seed)
            results[label].append({k: agg[k] for k in KEYS} | {"seed": seed})
    if d_dir and Path(d_dir).is_dir():
        for label, seed, path in _runs(Path(d_dir)):
            agent, gvf, rm, d_exp = load_d_checkpoint(str(path), seed=seed)
            key = (d_exp["sim_config"], d_exp["task"])
            if env_key is not None and key != env_key:
                raise SystemExit(f"refusing to compare: {label} uses env {key}, "
                                 f"A/B/C use {env_key}")
            env = X.make_env(d_exp)
            agg = evaluate_d(agent, gvf, rm, env, eval_episodes, eval_seed)
            results[label].append({k: agg[k] for k in KEYS} | {"seed": seed})
    return dict(results)


def summarize(results):
    out = {}
    for label, rows in results.items():
        out[label] = {k: {"mean": float(np.mean([r[k] for r in rows])),
                          "std": float(np.std([r[k] for r in rows])),
                          "n_seeds": len(rows)} for k in KEYS}
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--abc-dir", default=None)
    ap.add_argument("--abc-config", default=str(ROOT / "configs" / "dqn" / "experiment.yaml"))
    ap.add_argument("--d-dir", default=None)
    ap.add_argument("--eval-seed", type=int, default=20000)
    ap.add_argument("--eval-episodes", type=int, default=100)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    res = evaluate_all(args.abc_dir, args.abc_config, args.d_dir,
                       args.eval_seed, args.eval_episodes)
    summ = summarize(res)
    print(f"Task outcomes on identical eval scenarios (seed {args.eval_seed}, "
          f"{args.eval_episodes} greedy episodes per checkpoint; mean +/- std over seeds)")
    print(f"{'agent':>32} | {'n':>2} | " + " | ".join(f"{k:>16}" for k in KEYS))
    for label in sorted(summ):
        s = summ[label]
        print(f"{label:>32} | {s[KEYS[0]]['n_seeds']:>2} | " + " | ".join(
            f"{s[k]['mean']:7.3f}+/-{s[k]['std']:5.3f}".rjust(16) for k in KEYS))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps({"per_seed": res, "summary": summ,
                                              "eval_seed": args.eval_seed,
                                              "eval_episodes": args.eval_episodes}, indent=2))
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
