"""Aggregate job JSONs -> figures (PNG, 150 dpi), raw CSV, REPORT.md.

    python -m gvfctl.report --root results/<timestamp>

Every number in the report is computed here from the job files. Comparative
statements are generated from the data (including where a baseline ties or
wins). A hand-written conclusion can be supplied in <root>/CONCLUSION.md.
"""
from __future__ import annotations

import argparse
import csv
import json
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from . import stats as S
from .envs import PART_ENVS, action_space, make_env

METHODS = ["gvf", "dqnA", "dqnB", "hand", "demons-default", "demons-fixed"]
LABEL = {"gvf": "GVF controller", "dqnA": "DQN (A, standard)", "dqnB": "DQN (B, strong)",
         "hand": "Hand rule (C)", "demons-default": "Orig. demons (D, default)",
         "demons-fixed": "Orig. demons (D, fixed)", "big-dqnB": "DQN (B) 1M steps"}
COLOR = {"gvf": "#1f77b4", "dqnA": "#ff7f0e", "dqnB": "#d62728", "hand": "#7f7f7f",
         "demons-default": "#9467bd", "demons-fixed": "#8c564b", "big-dqnB": "#e377c2"}
DPI = 150


# ================================================================== loading
def load(root: Path):
    jobs = {}
    for p in sorted((root / "jobs").glob("*.json")):
        jobs[p.stem] = json.loads(p.read_text())
    return jobs


def group(jobs, kind, part):
    g = defaultdict(list)
    for j in jobs.values():
        if j.get("kind") == kind and j.get("part") == part:
            key = j["method"] if kind != "big" else "big-dqnB"
            g[key].append(j)
    for k in g:
        g[k].sort(key=lambda j: j["seed"])
    return g


def per_seed(runs, env, metric, section="final"):
    out = []
    for j in runs:
        v = j.get(section, {}).get(env, {}).get(metric)
        out.append(np.nan if v is None else float(v))
    return np.array(out, float)


def fmt(m, h, digits=3):
    if np.isnan(m):
        return "n/a"
    return f"{m:.{digits}f}" if np.isnan(h) else f"{m:.{digits}f} ± {h:.{digits}f}"


def ci_str(x, digits=3):
    m, h, n = S.mean_ci(x)
    return fmt(m, h, digits)


# ================================================================== curves
def curve_matrix(runs, key):
    if not runs or not runs[0].get("curve"):
        return None, None
    steps = [c["step"] for c in runs[0]["curve"]]
    M = np.array([[c.get(key, np.nan) for c in j["curve"]] for j in runs if len(j["curve"]) == len(steps)], float)
    return np.array(steps), M


def sample_efficiency(runs, key="survival"):
    steps, M = curve_matrix(runs, key)
    if M is None:
        return {}
    def first(th):
        out = []
        for row in M:
            idx = np.flatnonzero(row >= th)
            out.append(steps[idx[0]] if len(idx) else np.nan)
        return np.array(out, float)
    res = {"steps": steps}
    for th in (0.8, 0.9):
        f = first(th)
        res[f"to_{th}"] = (np.nanmean(f) if np.any(~np.isnan(f)) else np.nan, int(np.sum(~np.isnan(f))), len(f))
    res["aulc"] = M.mean(1)                                   # mean over checkpoints (normalised AULC)
    for s in (20_000, 50_000, 100_000, 300_000, 1_000_000):
        if s in set(steps):
            res[f"at_{s}"] = M[:, list(steps).index(s)]
    last5 = M[:, -5:]
    res["last5_mean"], res["last5_worst"] = last5.mean(1), last5.min(1)
    res["std_last10"] = M[:, -10:].std(1)
    res["collapses"] = np.sum(np.diff(M, axis=1) < -0.3, axis=1)
    return res


# ================================================================== figures
def fig_curves(G, part, out, key="survival", title=None):
    fig, ax = plt.subplots(figsize=(8, 5))
    for m in METHODS:
        runs = G.get(m, [])
        if m == "hand" and runs:
            v = per_seed(runs, "train", key if key != "mean_distance" else "mean_distance")
            ax.axhline(np.nanmean(v), color=COLOR[m], ls="--", label=f"{LABEL[m]} (no training)")
            continue
        steps, M = curve_matrix(runs, key)
        if M is None or not len(M):
            continue
        mean = M.mean(0)
        half = np.array([S.mean_ci(M[:, i])[1] for i in range(M.shape[1])]) if len(M) > 1 else np.zeros_like(mean)
        ax.plot(steps, mean, color=COLOR[m], lw=2, label=f"{LABEL[m]} (n={len(M)})")
        ax.fill_between(steps, mean - half, mean + half, color=COLOR[m], alpha=0.15)
    ax.set_xlabel("environment steps"); ax.set_ylabel(key.replace("_", " "))
    ax.set_title(title or f"Part {part}: learning curves (40 curve-seed episodes per point, 95% CI over seeds)")
    ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out, dpi=DPI); plt.close(fig)


def fig_bars(G, part, envs, metric, out, ylabel):
    methods = [m for m in METHODS + ["big-dqnB"] if G.get(m)]
    fig, ax = plt.subplots(figsize=(max(8, 1.6 * len(envs) * len(methods) / 3), 5))
    w = 0.8 / len(methods)
    for i, m in enumerate(methods):
        for k, e in enumerate(envs):
            x = per_seed(G[m], e, metric)
            if np.all(np.isnan(x)):
                continue
            mm, h, n = S.mean_ci(x)
            pos = k + (i - len(methods) / 2 + 0.5) * w
            ax.bar(pos, mm, w, color=COLOR[m], yerr=None if np.isnan(h) else h, capsize=2,
                   label=LABEL[m] if k == 0 else None, alpha=0.85)
            ax.scatter(np.full(len(x), pos), x, s=6, color="k", zorder=3)
    ax.set_xticks(range(len(envs))); ax.set_xticklabels(envs, rotation=20)
    ax.set_ylabel(ylabel); ax.set_title(f"Part {part}: {ylabel} (bars: mean ± 95% CI over seeds; dots: seeds)")
    ax.grid(alpha=0.3, axis="y"); ax.legend(fontsize=7, ncol=2)
    fig.tight_layout(); fig.savefig(out, dpi=DPI); plt.close(fig)


def fig_km(G, part, env, out):
    fig, ax = plt.subplots(figsize=(8, 5))
    for m in METHODS:
        runs = G.get(m, [])
        t, e = [], []
        for j in runs:
            u = j.get("uncapped", {}).get(env)
            if u:
                t += u["times"]; e += u["events"]
        if not t:
            continue
        xs, ys = S.kaplan_meier(t, e)
        xs = xs + [max(t)]; ys = ys + [ys[-1]]
        ax.step(xs, ys, where="post", color=COLOR[m], lw=2,
                label=f"{LABEL[m]} ({len(t)} eps, {100 * np.mean(e):.0f}% collided)")
    ax.set_xscale("symlog", linthresh=100); ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("steps until first collision (censored at 20,000)"); ax.set_ylabel("fraction without collision")
    ax.set_title(f"Part {part}: Kaplan-Meier, run-until-collision, env '{env}'")
    ax.grid(alpha=0.3); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(out, dpi=DPI); plt.close(fig)


def fig_stability(G, part, out, key="survival"):
    fig, ax = plt.subplots(figsize=(8, 5))
    for m in ("gvf", "dqnA", "dqnB"):
        steps, M = curve_matrix(G.get(m, []), key)
        if M is None:
            continue
        for row in M:
            ax.plot(steps, row, color=COLOR[m], alpha=0.35, lw=1)
        ax.plot([], [], color=COLOR[m], label=f"{LABEL[m]} (one line per seed)")
    ax.set_xlabel("environment steps"); ax.set_ylabel(key); ax.grid(alpha=0.3); ax.legend(fontsize=8)
    ax.set_title(f"Part {part}: per-seed learning curves (stability)")
    fig.tight_layout(); fig.savefig(out, dpi=DPI); plt.close(fig)


def fig_ablation(rows, out, title, ylabel):
    fig, ax = plt.subplots(figsize=(max(7, 0.7 * len(rows)), 4.5))
    for i, (name, x) in enumerate(rows):
        mm, h, n = S.mean_ci(x)
        ax.bar(i, mm, yerr=None if np.isnan(h) else h, capsize=3, color="#1f77b4" if name != "main" else "#2ca02c")
        ax.scatter(np.full(len(x), i), x, s=8, color="k", zorder=3)
    ax.set_xticks(range(len(rows))); ax.set_xticklabels([r[0] for r in rows], rotation=30, ha="right")
    ax.set_ylabel(ylabel); ax.set_title(title); ax.grid(alpha=0.3, axis="y")
    fig.tight_layout(); fig.savefig(out, dpi=DPI); plt.close(fig)


def fig_safety_quality(root, runs, part, out):
    """Scatter (seed 0 model, recomputed) + calibration (all seeds) + TTC trace."""
    from .gvf import GVFConfig, GVFModel
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    j0 = runs[0]
    key = "mc__p1" if part == 1 else f"mc__p2__{j0['config']['headway_tau']}"
    cache = root / "cache" / f"{key}.pkl"
    if cache.exists() and (root / "agents" / f"{j0['job']}.npz").exists():
        d = pickle.load(open(cache, "rb"))
        c = dict(j0["config"]); c["horizons"] = tuple(c["horizons"]); c["hidden"] = tuple(c["hidden"])
        A = action_space(part)
        m = GVFModel(GVFConfig(**c), A, seed=0)
        m.load(np.load(root / "agents" / f"{j0['job']}.npz"))
        R = np.array([s["ranges"] for s in d["states"]])
        Q = m.predict(R)[0]
        for h, g in enumerate(c["horizons"]):
            axes[0].scatter(d["mc"][..., h].ravel(), np.transpose(Q, (0, 2, 1))[..., h].ravel(), s=3, alpha=0.3,
                            label=f"gamma={g}")
        axes[0].plot([0, 1], [0, 1], "k--", lw=1)
        axes[0].set_xlabel("Monte-Carlo safety (persistent policy)"); axes[0].set_ylabel("GVF prediction")
        axes[0].set_title(f"seed {j0['seed']}: {len(R)} states x {A.n} actions"); axes[0].legend(fontsize=8)
    for j in runs:
        sq = j.get("quality", {}).get("safety")
        if isinstance(sq, dict):
            for h, (gk, r) in enumerate(sq.items()):
                cal = r["calibration"]
                axes[1].plot([b["pred"] for b in cal], [b["mc"] for b in cal], "o-", ms=3, alpha=0.4,
                             color=["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"][h % 4],
                             label=gk if j is runs[0] else None)
    axes[1].plot([0, 1], [0, 1], "k--", lw=1); axes[1].set_xlabel("predicted (bin mean)")
    axes[1].set_ylabel("Monte-Carlo (bin mean)"); axes[1].set_title("calibration, all seeds"); axes[1].legend(fontsize=8)
    tr = j0.get("quality", {}).get("ttc_trace")
    if tr:
        true = np.array(tr["true"]); P = np.array(tr["pred"])
        axes[2].plot(true, "k", lw=2, label="true steps to collision")
        for h, g in enumerate(tr["horizons"]):
            axes[2].plot(P[:, h], lw=1.5, label=f"ttc_from_safety, gamma={g}")
        axes[2].axhline(true.mean(), color="gray", ls="--", label="mean baseline")
        axes[2].set_xlabel("step"); axes[2].set_ylabel("steps"); axes[2].set_ylim(0, true.max() * 1.6)
        axes[2].set_title("straight into a wall (seed 0)"); axes[2].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(out, dpi=DPI); plt.close(fig)


def fig_velocity(runs, out):
    names = [("linear_velocity", "future linear velocity (/v_max)"),
             ("abs_angular_velocity", "future |turn rate| (/w_max)")]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    j0 = runs[0]
    vq = j0.get("quality", {}).get("velocity")
    if vq:
        for ax, (k, t) in zip(axes, names):
            sc = vq[k]["scatter"]
            ax.scatter(sc["true"], sc["pred"], s=8, alpha=0.5, label=f"GVF (MAE {vq[k]['gvf']['mae']:.3f})")
            ax.scatter(sc["true"], sc["persistence"], s=8, alpha=0.5, marker="x",
                       label=f"persistence (MAE {vq[k]['persistence']['mae']:.3f})")
            ax.axhline(np.mean(sc["true"]), color="gray", ls="--", label=f"mean (MAE {vq[k]['mean']['mae']:.3f})")
            ax.axhline(0, color="k", ls=":", label=f"zero (MAE {vq[k]['zero']['mae']:.3f})")
            lo, hi = min(sc["true"]), max(sc["true"])
            ax.plot([lo, hi], [lo, hi], "k--", lw=1)
            ax.set_xlabel("Monte-Carlo ground truth (controller target policy)"); ax.set_ylabel("prediction")
            ax.set_title(f"{t}, seed {j0['seed']}"); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(out, dpi=DPI); plt.close(fig)


def fig_pareto(G, out):
    fig, ax = plt.subplots(figsize=(7, 5))
    for m, lab in (("gvf", "v_target sweep"), ("dqnA", "speed-bias sweep"), ("dqnB", "speed-bias sweep")):
        runs = [j for j in G.get(m, []) if j.get("pareto")]
        if not runs:
            continue
        keys = list(runs[0]["pareto"])
        x = [np.mean([j["pareto"][k]["collision_rate"] for j in runs]) for k in keys]
        y = [np.mean([j["pareto"][k]["mean_distance"] for j in runs]) for k in keys]
        ax.plot(x, y, "o-", color=COLOR[m], label=f"{LABEL[m]} ({lab}, n={len(runs)})")
        for k, xi, yi in zip(keys, x, y):
            ax.annotate(k, (xi, yi), fontsize=6)
    ax.set_xlabel("collision rate (800-step episodes)"); ax.set_ylabel("mean distance per episode (m)")
    ax.set_title("Part 2: speed / safety trade-off (up-left is better)"); ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out, dpi=DPI); plt.close(fig)


def fig_speed_usage(G, out):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    methods = [m for m in METHODS if G.get(m)]
    A = action_space(2)
    w = 0.8 / max(1, len(methods))
    for i, m in enumerate(methods):
        F = np.array([j["final"]["train"]["speed_frac"] for j in G[m]])
        ax.bar(np.arange(A.n_speed) + (i - len(methods) / 2 + 0.5) * w, F.mean(0), w, color=COLOR[m], label=LABEL[m])
    ax.set_xticks(range(A.n_speed)); ax.set_xticklabels([f"{v} m/s" for v in A.speeds])
    ax.set_ylabel("fraction of steps"); ax.set_title("Part 2: speed-level usage (train env)")
    ax.legend(fontsize=7); ax.grid(alpha=0.3, axis="y")
    fig.tight_layout(); fig.savefig(out, dpi=DPI); plt.close(fig)


def fig_trajectories(G, part, out):
    from .envs import episode_seeds
    methods = [m for m in ("gvf", "dqnA", "dqnB", "hand", "demons-fixed") if G.get(m)]
    fig, axes = plt.subplots(len(methods), 6, figsize=(18, 3 * len(methods)), squeeze=False)
    seeds = episode_seeds("test", "train", 6)
    env, _ = make_env("train", part)
    for r, m in enumerate(methods):
        trajs = G[m][0].get("trajectories", {}).get("train", [])
        for c in range(6):
            ax = axes[r][c]
            env.reset(seed=seeds[c])
            for (x0, y0, x1, y1) in env.world.obstacles:
                ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, color="0.6"))
            if c < len(trajs):
                T = np.array(trajs[c])
                sc = ax.scatter(T[:, 0], T[:, 1], c=T[:, 2], s=2, cmap="viridis", vmin=0.05, vmax=0.22)
                ax.plot(T[0, 0], T[0, 1], "g^", ms=6); ax.plot(T[-1, 0], T[-1, 1], "rx", ms=6)
            ax.set_xlim(-2, 2); ax.set_ylim(-2, 2); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
            if c == 0:
                ax.set_ylabel(LABEL[m], fontsize=8)
    fig.suptitle(f"Part {part}: example test trajectories (seed-0 agents; colour = speed m/s; red x = end)")
    fig.tight_layout(); fig.savefig(out, dpi=110); plt.close(fig)


def fig_heatmap(runs, out):
    acc = defaultdict(list)
    for j in runs:
        rows = j.get("val_tuning", {}).get("steer_grid", [])
        best = defaultdict(float)
        for r in rows:
            k = (r["beta_lo"], r["beta_hi"])
            best[k] = max(best[k], r["metric"])
        for k, v in best.items():
            acc[k].append(v)
    if not acc:
        return False
    los = sorted({k[0] for k in acc}); his = sorted({k[1] for k in acc})
    M = np.full((len(los), len(his)), np.nan)
    for (lo, hi), v in acc.items():
        M[los.index(lo), his.index(hi)] = np.mean(v)
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    im = ax.imshow(M, origin="lower", cmap="viridis")
    for i in range(len(los)):
        for k in range(len(his)):
            if not np.isnan(M[i, k]):
                ax.text(k, i, f"{M[i, k]:.3f}", ha="center", va="center", fontsize=7, color="w")
    ax.set_xticks(range(len(his))); ax.set_xticklabels(his); ax.set_yticks(range(len(los))); ax.set_yticklabels(los)
    ax.set_xlabel("beta_hi"); ax.set_ylabel("beta_lo"); fig.colorbar(im, label="validation survival (best delta)")
    ax.set_title(f"Part 1 threshold sensitivity (validation seeds, mean of {len(runs)} agents)")
    fig.tight_layout(); fig.savefig(out, dpi=DPI); plt.close(fig)
    return True


# ================================================================== report
def table(header, rows):
    s = "| " + " | ".join(header) + " |\n|" + "---|" * len(header) + "\n"
    return s + "".join("| " + " | ".join(str(c) for c in r) + " |\n" for r in rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    args = ap.parse_args()
    root = Path(args.root)
    figs = root / "figures"; figs.mkdir(parents=True, exist_ok=True)
    jobs = load(root)
    md, summary_rows, findings = [], [], []
    md.append(f"# GVF predictive controller vs DQN — TurtleBot3 LiDAR navigation\n")
    md.append(f"Results root: `{root}` · {len(jobs)} job files · generated by `python -m gvfctl.report`.\n")
    prov = next((j.get("provenance") for j in jobs.values() if j.get("provenance")), {})
    md.append(f"Code sha256 (gvfctl/*.py): `{prov.get('code_sha256')}` · git: not a git repo · "
              f"Python {prov.get('python')} · NumPy {prov.get('numpy')} · gymnasium {prov.get('gymnasium')} · "
              f"{prov.get('platform')}\n")
    setup = root / "SETUP.md"
    if not setup.exists():
        setup = Path(__file__).resolve().parent / "SETUP.md"
    if setup.exists():
        md.append(setup.read_text() + "\n")

    # ---------------------------------------------------------------- tuning
    md.append("## DQN tuning (validation seeds 2,000,000+, 100k steps, tuning seeds 100–101)\n")
    for part in (1, 2):
        for fam in "AB":
            rows = [j for j in jobs.values() if j.get("kind") == "tune-dqn" and j["part"] == part and j["family"] == fam]
            if not rows:
                continue
            by = defaultdict(list)
            for r in rows:
                by[r["grid_index"]].append(r)
            T = []
            for gi in sorted(by):
                c = by[gi][0]["config"]
                T.append([gi, c["gamma"], c["lr"], c["target_update"], c["reward_scale"], c["turn_penalty"],
                          f"{np.mean([r['val_metric'] for r in by[gi]]):.3f}", len(by[gi])])
            best = max(T, key=lambda r: (float(r[6]), -r[0]))
            md.append(f"**Part {part}, family {fam}** (metric: {'survival' if part == 1 else 'distance (m)'}; chosen: grid {best[0]})\n\n")
            md.append(table(["grid", "gamma", "lr", "target_update", "reward_scale", "turn_penalty", "val metric", "n"], T) + "\n")
    tau_rows = [j for j in jobs.values() if j.get("kind") == "tune-tau"]
    if tau_rows:
        by = defaultdict(list)
        for r in tau_rows:
            by[r["tau"]].append(r["val_metric"])
        md.append("**Part 2 headway tau** (validation distance): " +
                  ", ".join(f"tau={t}: {np.mean(v):.3f} (n={len(v)})" for t, v in sorted(by.items())) + "\n")

    csv_rows = []
    for part in (1, 2):
        G = group(jobs, "main", part)
        G.update(group(jobs, "big", part))
        if not G:
            continue
        envs = PART_ENVS[part]
        prim = "survival" if part == 1 else "mean_distance"
        md.append(f"\n## Part {part}: {'steering only (5 actions, 0.15 m/s)' if part == 1 else 'steering + speed (15 actions)'}\n")
        # -------- headline
        H = []
        for m in METHODS + ["big-dqnB"]:
            runs = G.get(m, [])
            if not runs:
                continue
            n_ep = runs[0]["final"]["train"]["n_episodes"]
            unc = np.array([j.get("uncapped", {}).get("train", {}).get("median_ttc", np.nan) for j in runs], float)
            row = [LABEL[m], len(runs), n_ep, ci_str(per_seed(runs, "train", "survival")),
                   ci_str(per_seed(runs, "train", "success")),
                   ci_str(unc, 0) if not np.all(np.isnan(unc)) else "n/a"]
            if part == 2:
                row += [ci_str(per_seed(runs, "train", "mean_distance"), 2),
                        ci_str(per_seed(runs, "train", "collisions_per_100m"), 2),
                        ci_str(per_seed(runs, "train", "mean_speed"), 3)]
            H.append(row)
            summary_rows.append((part, LABEL[m], np.nanmean(per_seed(runs, "train", "survival")),
                                 np.nanmean(per_seed(runs, "train", "mean_distance"))))
            for j in runs:
                for e in envs:
                    f = j["final"].get(e)
                    if f:
                        csv_rows.append({"part": part, "method": m, "seed": j["seed"], "env": e,
                                         **{k: f[k] for k in ("survival", "success", "mean_distance", "mean_speed",
                                                              "collisions_per_100m", "mean_abs_w", "reversals_per100",
                                                              "area_cells", "min_clearance", "frac_within_025")}})
        hdr = ["method", "n seeds", "episodes/seed", "survival (steps/800)", "success", "median TTC uncapped (steps)"]
        if part == 2:
            hdr += ["distance/episode (m)", "collisions /100 m", "mean speed (m/s)"]
        md.append("### Headline (train layout, test seeds; mean ± 95% CI over seeds, Student-t)\n\n" + table(hdr, H))
        md.append("\nHand rule is deterministic (no training seed): n = 1, no CI.\n")

        # -------- statistics
        md.append("\n### Statistical tests: GVF controller vs each baseline (unit = training seed; paired by seed)\n\n")
        ST = []
        gv = G.get("gvf", [])
        metrics = [("survival", "final", "train"), ("success", "final", "train"), ("median_ttc", "uncapped", "train")]
        if part == 2:
            metrics += [("mean_distance", "final", "train"), ("distance_before_collision", "uncapped", "train")]
        for b in ["dqnA", "dqnB", "hand", "demons-default", "demons-fixed", "big-dqnB"]:
            if not gv or not G.get(b):
                continue
            for met, sec, e in metrics:
                a = per_seed(gv, e, met, sec); bb = per_seed(G[b], e, met, sec)
                if np.all(np.isnan(a)) or np.all(np.isnan(bb)):
                    continue
                a, bb = a[~np.isnan(a)], bb[~np.isnan(bb)]
                c = S.compare(a, bb)
                bs = c.get("bootstrap", {})
                sig = (bs.get("p", 1) < 0.05)
                p_w = c.get("welch", c.get("one_sample_t", {})).get("p", np.nan)
                ST.append([LABEL[b], met, f"{c['mean_a']:.3f}", f"{c['mean_b']:.3f}",
                           f"{bs.get('mean_diff', np.nan):+.3f} [{bs.get('ci_lo', np.nan):+.3f}, {bs.get('ci_hi', np.nan):+.3f}]",
                           f"{bs.get('p', np.nan):.4f}", f"{p_w:.4f}",
                           f"{c.get('mann_whitney', {}).get('p', np.nan):.4f}", f"{c.get('cohen_d', np.nan):.2f}",
                           f"{len(a)} vs {len(bb)}", "yes" if sig else "**no**"])
                if met in ("survival", "mean_distance", "success"):
                    if c["mean_b"] >= c["mean_a"]:
                        findings.append(f"Part {part}: {LABEL[b]} is **equal or better** than the GVF controller on {met} "
                                        f"({c['mean_b']:.3f} vs {c['mean_a']:.3f}).")
                    elif not sig:
                        findings.append(f"Part {part}: GVF vs {LABEL[b]} on {met}: difference "
                                        f"{c['mean_a'] - c['mean_b']:+.3f} is **not significant** (bootstrap p={bs.get('p', np.nan):.3f}).")
        md.append(table(["baseline", "metric", "GVF", "baseline", "diff (bootstrap 95% CI)", "p boot", "p Welch/1-sample t",
                         "p Mann-Whitney", "Cohen d", "n seeds", "p<0.05"], ST))
        md.append("\nBootstrap: 10,000 paired resamples of per-seed differences. Mann-Whitney: normal approximation "
                  "with tie and continuity correction (n=10 per group). Cohen's d uses the pooled SD. "
                  "Against the deterministic hand rule, a one-sample t-test is used.\n")

        # -------- sample efficiency & stability
        SE, ST2 = [], []
        for m in ["gvf", "dqnA", "dqnB", "demons-default", "demons-fixed", "big-dqnB"]:
            runs = G.get(m, [])
            if not runs:
                continue
            se = sample_efficiency(runs, "survival" if part == 1 else "mean_distance")
            if not se:
                continue
            if part == 1:
                t8, k8, n8 = se["to_0.8"]; t9, k9, n9 = se["to_0.9"]
                to8 = f"{t8:,.0f} ({k8}/{n8})" if k8 else f"never (0/{n8})"
                to9 = f"{t9:,.0f} ({k9}/{n9})" if k9 else f"never (0/{n9})"
            else:
                to8 = to9 = "—"
            SE.append([LABEL[m], to8, to9, ci_str(se["aulc"])] +
                      [ci_str(se[f"at_{s}"]) if f"at_{s}" in se else "—" for s in (20_000, 50_000, 100_000, 300_000)])
            ST2.append([LABEL[m], ci_str(se["last5_mean"]), ci_str(se["last5_worst"]), ci_str(se["std_last10"]),
                        f"{int(np.sum(se['collapses']))} (seeds with ≥1: {int(np.sum(se['collapses'] > 0))}/{len(runs)})"])
        unit = "survival" if part == 1 else "distance (m)"
        md.append(f"\n### Sample efficiency (learning-curve {unit}; curve seeds 3,000,000+)\n\n")
        md.append(table(["method", "steps to survival ≥0.8 (seeds reaching)", "≥0.9", "AULC (mean over checkpoints)",
                         "@20k", "@50k", "@100k", "@300k"], SE))
        md.append(f"\n### Stability (curve {unit})\n\n")
        md.append(table(["method", "mean of last 5 checkpoints", "worst of last 5", "std over last 10",
                         "collapse events (drop > 0.3)"], ST2))
        md.append("\nCollapse threshold 0.3 is in survival units; for Part 2 distance it is 0.3 m.\n")

        # -------- OOD / robustness
        md.append(f"\n### Robustness / out-of-distribution (test seeds; {prim}; mean ± 95% CI)\n\n")
        R = []
        for m in METHODS + ["big-dqnB"]:
            runs = G.get(m, [])
            if runs:
                R.append([LABEL[m]] + [ci_str(per_seed(runs, e, prim), 3 if part == 1 else 2) for e in envs])
        md.append(table(["method"] + envs, R))
        if part == 1:
            md.append("\n`gazebo` = configs/gazebo_compat.yaml exactly (empty arena, fixed spawn, so all episodes are one "
                      "trajectory); `gazebo_train` = training layout with the Gazebo action table (0.20 m/s, "
                      "reversed indices). Agents choose physical (v, omega); the env index is remapped by omega. "
                      "Controller thresholds were re-tuned on validation seeds of each gazebo env; DQN unchanged.\n")
        md.append("\n### Collisions per 100 m and uncapped time-to-first-collision (median, steps; censored at 20,000)\n\n")
        U = []
        for m in METHODS:
            runs = G.get(m, [])
            if runs:
                U.append([LABEL[m]] + [ci_str(np.array([j.get("uncapped", {}).get(e, {}).get("median_ttc", np.nan)
                                                         for j in runs], float), 0) for e in envs])
        md.append(table(["method"] + envs, U))

        # -------- behaviour
        md.append("\n### Behaviour (train layout, test seeds)\n\n")
        B = []
        for m in METHODS:
            runs = G.get(m, [])
            if runs:
                B.append([LABEL[m]] + [ci_str(per_seed(runs, "train", k), 3) for k in
                                       ("mean_abs_w", "reversals_per100", "area_cells", "min_clearance",
                                        "frac_within_025", "speed_changes_per100")])
        md.append(table(["method", "mean |turn rate|", "reversals /100 steps", "area (0.25 m cells)",
                         "min clearance (m)", "frac. time < 0.25 m", "speed changes /100 steps"], B))
        if part == 2:
            md.append("\nMinimum-clearance-vs-speed (mean per-step min range at each speed level, train): " + "; ".join(
                f"{LABEL[m]}: " + ", ".join("—" if v is None else f"{v:.2f}" for v in
                                           np.nanmean(np.array([[np.nan if x is None else x for x in j['final']['train']['clearance_by_speed']]
                                                                for j in G[m]], float), 0).tolist())
                for m in METHODS if G.get(m)) + " (levels 0.05/0.12/0.22 m/s).\n")

        # -------- compute
        md.append("\n### Compute\n\n")
        C = []
        for m in METHODS + ["big-dqnB"]:
            runs = G.get(m, [])
            if not runs:
                continue
            upd = np.mean([j["train_meta"].get("updates", 0) for j in runs])
            wall = np.array([j["train_meta"].get("wall_s", np.nan) for j in runs], float)
            bud = np.array([j.get("budget", 0) for j in runs], float)
            per100k = wall / np.maximum(bud, 1) * 1e5
            C.append([LABEL[m], f"{upd:,.0f}", ci_str(per100k, 1) if bud.max() > 0 else "—",
                      ci_str(np.array([j.get("inference_ms_per_decision", np.nan) for j in runs]), 4),
                      ci_str(np.array([j.get("wall_s", np.nan) for j in runs]) / 60, 1)])
        md.append(table(["method", "gradient updates", "train wall s /100k steps (incl. curve evals)",
                         "inference ms /decision", "job wall min (train+all evals)"], C))
        md.append("\nOne process per job on a 10-core Apple M4 laptop with up to 10 jobs in parallel (contended timings).\n")

        # -------- prediction quality
        gq = G.get("gvf", [])
        if gq and any(isinstance(j.get("quality", {}).get("safety"), dict) for j in gq):
            md.append("\n### Safety-GVF prediction quality vs Monte-Carlo ground truth\n\n")
            QT = []
            for m in ("gvf", "demons-default", "demons-fixed"):
                runs = [j for j in G.get(m, []) if isinstance(j.get("quality", {}).get("safety"), dict)]
                if not runs:
                    continue
                for gk in runs[0]["quality"]["safety"]:
                    r = [j["quality"]["safety"][gk] for j in runs]
                    QT.append([LABEL[m], gk, ci_str([x["mae"] for x in r]), ci_str([x["mean_baseline_mae"] for x in r]),
                               ci_str([x["rmse"] for x in r]), ci_str([x["r2"] for x in r])])
            n_states = gq[0]["quality"].get("safety_n_states")
            md.append(f"{n_states} states × all actions × 8 rollouts under the persistent target policy "
                      f"(QUALITY seeds 4,000,000+). Original demons are scored after TTC→Q conversion.\n\n")
            md.append(table(["predictor", "horizon", "MAE", "mean-baseline MAE", "RMSE", "R²"], QT))
            if part == 1 and gq[0]["quality"].get("ttc_trace"):
                TT = []
                for m in ("gvf", "demons-default", "demons-fixed"):
                    runs = [j for j in G.get(m, []) if j.get("quality", {}).get("ttc_trace")]
                    if not runs:
                        continue
                    for gk in ("gamma_0.9", "gamma_0.97"):
                        TT.append([LABEL[m], gk, ci_str([j["quality"]["ttc_trace"][gk]["mae"] for j in runs], 1),
                                   ci_str([j["quality"]["ttc_trace"][gk]["mae_last40"] for j in runs], 1)])
                t0 = gq[0]["quality"]["ttc_trace"]
                md.append("\n**Straight-into-wall TTC trace** (ttc_from_safety on the straight action vs true steps "
                          f"to collision; trace length {len(t0['true'])} steps): mean-baseline MAE {t0['mean_baseline_mae']:.1f} "
                          f"(last 40: {t0['mean_baseline_mae_last40']:.1f}), zero-baseline MAE {t0['zero_baseline_mae']:.1f} "
                          f"(last 40: {t0['zero_baseline_mae_last40']:.1f}).\n\n")
                md.append(table(["predictor", "horizon", "MAE (whole trace)", "MAE (last 40 steps)"], TT))
            fig_safety_quality(root, gq, part, figs / f"p{part}_safety_quality.png")
            md.append(f"\n![safety quality](figures/p{part}_safety_quality.png)\n")
        if part == 2 and gq and any(j.get("quality", {}).get("velocity") for j in gq):
            md.append("\n### Velocity-GVF prediction quality (Monte-Carlo under the controller target policy)\n\n")
            VT = []
            runs = [j for j in gq if j.get("quality", {}).get("velocity")]
            for k in ("linear_velocity", "abs_angular_velocity", "signed_angular_velocity"):
                vq = [j["quality"]["velocity"][k] for j in runs]
                for b in ("gvf", "persistence", "mean", "zero"):
                    VT.append([k, b, ci_str([x[b]["mae"] for x in vq], 4), ci_str([x[b]["rmse"] for x in vq], 4),
                               ci_str([x[b]["r2"] for x in vq], 3)])
                nb = sum(x["beats_persistence"] for x in vq)
                findings.append(f"Part 2: velocity GVF ({k}) beats the persistence predictor (lower MAE and RMSE) "
                                f"in **{nb}/{len(vq)}** seeds.")
            md.append(table(["quantity (normalised)", "predictor", "MAE", "RMSE", "R²"], VT))
            mc = [j["quality"]["velocity"]["mirror_consistency"] for j in runs]
            md.append(f"\nMirror consistency of the raw signed-turn-rate head: mean |V(s,a)+V(ms,ma)| = "
                      f"{np.mean([x['signed_w_mean_abs_sum'] for x in mc]):.4f} vs mean |V| = "
                      f"{np.mean([x['signed_w_mean_abs'] for x in mc]):.4f}.\n")
            fig_velocity(runs, figs / "p2_velocity_quality.png")
            md.append("\n![velocity quality](figures/p2_velocity_quality.png)\n")

        # -------- failure analysis
        md.append("\n### Failure analysis (collision categories, test episodes, train + dense layouts, all seeds)\n\n")
        FA = []
        from .evaluate import CATEGORIES
        for m in METHODS:
            runs = G.get(m, [])
            if not runs:
                continue
            cnt = defaultdict(int)
            for j in runs:
                for e in ("train", "dense"):
                    for c, v in j["final"].get(e, {}).get("failure_categories", {}).items():
                        cnt[c] += v
            FA.append([LABEL[m]] + [cnt[c] for c in CATEGORIES] + [sum(cnt.values())])
        md.append(table(["method"] + list(CATEGORIES) + ["total"], FA))
        md.append("\nRaw logs (last 30 scans, actions, speeds, Q values) for up to 50 collision episodes per agent "
                  "and layout are in each job JSON under `fail_logs`.\n")

        # -------- figures
        fig_curves(G, part, figs / f"p{part}_curves.png", key="survival")
        md.append(f"\n![curves](figures/p{part}_curves.png)\n")
        if part == 2:
            fig_curves(G, part, figs / "p2_curves_distance.png", key="mean_distance",
                       title="Part 2: distance per episode during training")
            md.append("\n![curves distance](figures/p2_curves_distance.png)\n")
        fig_bars(G, part, envs, prim, figs / f"p{part}_final_ood.png",
                 "survival" if part == 1 else "distance per episode (m)")
        md.append(f"\n![final and OOD](figures/p{part}_final_ood.png)\n")
        if part == 2:
            fig_bars(G, part, envs, "collision_rate", figs / "p2_collision_ood.png", "collision rate")
            md.append("\n![collision OOD](figures/p2_collision_ood.png)\n")
        for e in ("train", "dense"):
            fig_km(G, part, e, figs / f"p{part}_km_{e}.png")
            md.append(f"\n![KM {e}](figures/p{part}_km_{e}.png)\n")
        fig_stability(G, part, figs / f"p{part}_stability.png", "survival" if part == 1 else "mean_distance")
        md.append(f"\n![stability](figures/p{part}_stability.png)\n")
        fig_trajectories(G, part, figs / f"p{part}_trajectories.png")
        md.append(f"\n![trajectories](figures/p{part}_trajectories.png)\n")
        if part == 1 and fig_heatmap(G.get("gvf", []), figs / "p1_threshold_heatmap.png"):
            md.append("\n![threshold heatmap](figures/p1_threshold_heatmap.png)\n")
        if part == 2:
            fig_pareto(G, figs / "p2_pareto.png")
            md.append("\n![pareto](figures/p2_pareto.png)\n")
            fig_speed_usage(G, figs / "p2_speed_usage.png")
            md.append("\n![speed usage](figures/p2_speed_usage.png)\n")
            pr = [j for j in G.get("gvf", []) if j.get("pareto")]
            dq = [j for j in G.get("dqnB", []) + G.get("dqnA", []) if j.get("pareto")]
            if pr and dq:
                gpts = [(np.mean([j["pareto"][k]["collision_rate"] for j in pr]), np.mean([j["pareto"][k]["mean_distance"] for j in pr]))
                        for k in pr[0]["pareto"]]
                dpts = [(np.mean([j["pareto"][k]["collision_rate"] for j in dq]), np.mean([j["pareto"][k]["mean_distance"] for j in dq]))
                        for k in dq[0]["pareto"]]
                dominated = sum(any(gc <= dc and gd >= dd and (gc < dc or gd > dd) for gc, gd in gpts) for dc, dd in dpts)
                findings.append(f"Part 2 Pareto: {dominated}/{len(dpts)} DQN sweep points are dominated by at least one "
                                f"GVF v_target point (lower-or-equal collision rate and higher-or-equal distance).")

        # -------- ablations
        A_ = group(jobs, "abl", part)
        if A_:
            main_ref = [j for j in G.get("gvf", []) if j["seed"] < 5]
            rows_s, rows_d = [("main", per_seed(main_ref, "train", prim))], [("main", per_seed(main_ref, "dense", prim))]
            AT = [["main (seeds 0-4)", len(main_ref), ci_str(rows_s[0][1]), ci_str(rows_d[0][1])]]
            for v, runs in sorted(A_.items()):
                rows_s.append((v, per_seed(runs, "train", prim))); rows_d.append((v, per_seed(runs, "dense", prim)))
                AT.append([v, len(runs), ci_str(rows_s[-1][1]), ci_str(rows_d[-1][1])])
                a = rows_s[-1][1]; b = rows_s[0][1]
                if len(a) == len(b) and len(a) > 1:
                    bs = S.paired_bootstrap(a, b)
                    AT[-1].append(f"{bs['mean_diff']:+.3f} (p={bs['p']:.3f})")
            AT[0].append("—")
            md.append(f"\n### Ablations (5 seeds each; {prim}; test seeds)\n\n")
            md.append(table(["variant", "n", "train", "dense", "diff vs main (train, paired bootstrap)"], AT))
            fig_ablation(rows_s, figs / f"p{part}_ablations.png", f"Part {part} ablations (train layout)", prim)
            md.append(f"\n![ablations](figures/p{part}_ablations.png)\n")
        if part == 2 and any(j.get("speed_ablation") for j in G.get("gvf", [])):
            runs = [j for j in G["gvf"] if j.get("speed_ablation")]
            SA = []
            for mode in ("fixed", "safety", "lin", "ang", "both"):
                SA.append([mode] + [ci_str([j["speed_ablation"][mode][e][k] for j in runs], 3)
                                    for e in ("train", "dense") for k in ("mean_distance", "collision_rate")])
            md.append("\n### Speed-controller / velocity-GVF ablations (decision-time, same trained networks, 10 seeds)\n\n")
            md.append(table(["speed mode", "train distance (m)", "train collision", "dense distance (m)", "dense collision"], SA))
            md.append("\nfixed = middle speed level; safety = safety Q only (v_hat := current speed, no turn cap); "
                      "lin = + linear-velocity GVF; ang = + angular-velocity GVF cap only; both = full controller.\n")

    # ---------------------------------------------------------------- findings
    md.append("\n## Data-generated findings (where baselines tie or win, non-significant differences)\n\n")
    md.append("".join(f"- {f}\n" for f in findings) if findings else "- none generated\n")
    concl = root / "CONCLUSION.md"
    md.append("\n" + (concl.read_text() if concl.exists() else "## Conclusion\n\n(pending: written after reviewing the results)\n"))
    (root / "REPORT.md").write_text("\n".join(md))
    if csv_rows:
        with open(root / "per_seed_metrics.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(csv_rows[0])); w.writeheader(); w.writerows(csv_rows)
    print("\nSUMMARY (train layout, test seeds, mean over seeds)")
    print(f"{'part':>4}  {'method':28s} {'survival':>9} {'distance':>9}")
    for part, lab, s, d in summary_rows:
        print(f"{part:>4}  {lab:28s} {s:9.3f} {d:9.2f}")
    print(f"\nREPORT: {root / 'REPORT.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
