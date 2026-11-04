#!/usr/bin/env python3
"""Plot Experiment D learning curves (mean +/- std across seeds) for one or more
run groups, e.g. D vs its lambda=0 control.

    python scripts/plot_expD.py --group D=runs_expD/D_s* --group D0=runs_expD/D0_s* \
        --out runs_expD/plots

Panels: goal success, collision, timeout, episode length, task return, shaped
return (greedy evals from history.json); forward-TTC GVF error vs independent
ground truth (with its mean-predictor baseline); DQN loss and GVF |TD error|
(training episodes from episodes.jsonl, binned by env steps). Each legend entry
states the number of seeds; the title states eval episodes per point.
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

EVAL_PANELS = [("success_rate", "Goal success rate"),
               ("collision_rate", "Collision rate"),
               ("timeout_rate", "Timeout rate"),
               ("length_mean", "Episode length (steps)"),
               ("task_return_mean", "Task return (terminal rewards)"),
               ("shaped_return_mean", "Shaped return (task + lam*GVF)"),
               ("gvf_fwd_mae", "Forward-TTC GVF MAE (steps)")]


def _load_group(pattern):
    runs = []
    for d in sorted(glob.glob(pattern)):
        d = Path(d)
        hp = d / "history.json"
        if not hp.exists():
            continue
        eps = []
        ep = d / "episodes.jsonl"
        if ep.exists():
            eps = [json.loads(l) for l in ep.read_text().splitlines() if l.strip()]
        runs.append({"history": json.loads(hp.read_text()), "episodes": eps})
    return runs


def _eval_curve(runs, key):
    n = min(len(r["history"]) for r in runs)
    x = np.array([runs[0]["history"][i]["env_step"] for i in range(n)])
    v = np.array([[r["history"][i].get(key, np.nan) for i in range(n)] for r in runs], float)
    return x, np.nanmean(v, 0), np.nanstd(v, 0)


def _binned(runs, key, idx=None, bins=20):
    allx = [e["env_step"] for r in runs for e in r["episodes"]]
    if not allx:
        return None
    edges = np.linspace(0, max(allx), bins + 1)
    per_seed = []
    for r in runs:
        xs = np.array([e["env_step"] for e in r["episodes"]])
        vals = []
        for e in r["episodes"]:
            v = e.get(key)
            if v is not None and idx is not None:
                v = v[idx]
            vals.append(np.nan if v is None else float(v))
        vals = np.array(vals)
        row = [np.nanmean(vals[(xs > lo) & (xs <= hi)]) if np.any((xs > lo) & (xs <= hi)) else np.nan
               for lo, hi in zip(edges[:-1], edges[1:])]
        per_seed.append(row)
    a = np.array(per_seed, float)
    return 0.5 * (edges[:-1] + edges[1:]), np.nanmean(a, 0), np.nanstd(a, 0)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--group", action="append", required=True, metavar="LABEL=GLOB")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    groups = {}
    for g in args.group:
        label, _, pat = g.partition("=")
        runs = _load_group(pat)
        if runs:
            groups[label] = runs
        else:
            print(f"warning: no runs for {label} ({pat})")
    if not groups:
        return 1

    palette = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    color = {label: palette[i % len(palette)] for i, label in enumerate(groups)}
    fig, axes = plt.subplots(3, 3, figsize=(17, 13))
    axes = axes.ravel()
    n_eval = None
    for ax, (key, title) in zip(axes, EVAL_PANELS):
        for label, runs in groups.items():
            x, m, s = _eval_curve(runs, key)
            c = color[label]
            ax.plot(x, m, lw=2, color=c, label=f"{label} (n={len(runs)} seeds)")
            ax.fill_between(x, m - s, m + s, color=c, alpha=0.15)
            n_eval = runs[0]["history"][0].get("n_episodes", n_eval)
            if key == "gvf_fwd_mae":
                _, bm, _ = _eval_curve(runs, "gvf_fwd_mean_baseline_mae")
                ax.plot(x, bm, ls="--", lw=1, color=c, alpha=0.7)
        if key.endswith("_rate"):
            ax.set_ylim(-0.05, 1.05)
        if key == "gvf_fwd_mae":
            ax.plot([], [], ls="--", color="gray", label="dashed: mean-predictor baseline")
        ax.set_title(title); ax.set_xlabel("environment steps"); ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    for ax, (key, idx, title) in zip(axes[7:], (("dqn_loss_mean", None, "DQN Huber loss (training)"),
                                                ("gvf_abs_td_mean", 0, "Forward-TTC GVF |TD error| (training)"))):
        for label, runs in groups.items():
            b = _binned(runs, key, idx)
            if b is None:
                continue
            x, m, s = b
            ax.plot(x, m, lw=2, color=color[label], label=f"{label} (n={len(runs)})")
            ax.fill_between(x, m - s, m + s, color=color[label], alpha=0.15)
        ax.set_title(title); ax.set_xlabel("environment steps"); ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    fig.suptitle(f"Experiment D learning curves — greedy evals of {n_eval} episodes per point; "
                 f"shaded = +/-1 std across seeds", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    p = out / "expD_learning_curves.png"
    fig.savefig(p, dpi=105)
    print("wrote", p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
