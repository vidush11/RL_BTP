#!/usr/bin/env python3
"""Plot A/B/C learning curves (success + collision rate vs env steps) and a
summary bar chart, averaged across seeds, from a run_comparison output dir.

    python scripts/plot_comparison.py --dir runs_dqn/compare --out runs_dqn/compare/plots

Reads <dir>/<MODE>_s<seed>/history.json for every mode/seed present.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LABELS = {"A": "A LiDAR-only", "B": "B frozen-GVF", "C": "C adaptive-GVF"}
COLORS = {"A": "tab:gray", "B": "tab:blue", "C": "tab:green"}


def load(dir_: Path):
    runs = defaultdict(list)  # mode -> list of history lists
    for p in sorted(dir_.glob("*_s*/history.json")):
        mode = p.parent.name.split("_s")[0]
        runs[mode].append(json.loads(p.read_text()))
    return runs


def curve(histories, key):
    steps = [h["env_step"] for h in histories[0]]
    vals = np.array([[h[key] for h in run] for run in histories], float)  # (seeds, points)
    return np.array(steps), vals.mean(0), vals.std(0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    d = Path(args.dir); out = Path(args.out or d / "plots"); out.mkdir(parents=True, exist_ok=True)
    runs = load(d)
    if not runs:
        print(f"no history.json under {d}"); return 1

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for key, ax, title in (("success_rate", axes[0], "Goal success rate"),
                           ("collision_rate", axes[1], "Collision rate")):
        for mode in ("A", "B", "C"):
            if mode not in runs:
                continue
            x, m, s = curve(runs, key) if False else curve(runs[mode], key)
            ax.plot(x, m, label=LABELS[mode], color=COLORS[mode], lw=2)
            ax.fill_between(x, m - s, m + s, color=COLORS[mode], alpha=0.15)
        ax.set_title(title); ax.set_xlabel("environment steps"); ax.set_ylabel("rate")
        ax.grid(alpha=0.3); ax.legend()
    fig.suptitle("GVF-enhanced DQN: A vs B vs C (mean +/- std across seeds)")
    fig.tight_layout(); fig.savefig(out / "learning_curves.png", dpi=110)
    print("wrote", out / "learning_curves.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
