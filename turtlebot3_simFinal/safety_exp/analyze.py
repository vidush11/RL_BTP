#!/usr/bin/env python3
"""Analysis and D5 figures for the safety-task experiment.

    python -m safety_exp.analyze --runs runs_safety --out runs_safety/analysis

Reads every <runs>/<group>_s<seed>/ directory written by safety_exp.run
(groups: gvf, dqn, and optional ablations such as gvf_b24 / dqn_b24) and writes:

  curves.png / curves.json        D4 learning curves vs episodes AND env steps
                                  (mean over seeds, 95% bootstrap CI over seeds)
  final.json / final.md           final 1,000-episode TEST and HARD tables
                                  (hierarchical bootstrap: seeds, then episodes)
  pre_collision.png               q_hat (paper) / slack (given GVFs) in the 30 steps
                                  before every test collision
  fig4_wall_predictions.png       Fig. 4 analogue: q_hat_front per gamma vs c_f, the
                                  realised return and the Monte-Carlo pi-return
  fig5_wall_approach.png          Fig. 5 analogue: approach a wall from several speeds
  prediction_quality.json/.md     paper GVFs AND the given RL_BTP demons vs
                                  Monte-Carlo returns from the same states
  ablation_gamma_beta.json/.md    eval-only gamma x (beta1, beta2) table
  beams.md                        24- vs 72-beam runs (if present)

Everything post-hoc uses the ABLATE seed range (4,000,000+) and selects nothing.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from .common import evaluate, make_env, seeds, summarize

COLORS = {"paper": "#2a78d6", "horde": "#eb6834", "dqn": "#1baf7a"}
LABELS = {"paper": "paper GVF controller", "horde": "given GVFs (RL_BTP Horde)", "dqn": "DQN"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
METRICS = (("collision_free_rate", "collision-free rate"), ("speed_ratio", "mean speed / v_target"),
           ("mean_task_return", "task return"))
WALL = {"obstacles": np.zeros((0, 4)), "spawn": (-1.0, 0.0, 0.0)}    # facing the x = +2 wall, 3 m away


# ------------------------------------------------------------------ helpers
def boot_ci(x, n=5000, seed=0):
    x = np.asarray(x, float)
    if len(x) < 2:
        return float(x.mean()), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    m = x[rng.integers(0, len(x), (n, len(x)))].mean(1)
    return float(x.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def hier_boot(per_seed, n=5000, seed=0):
    """Resample seeds, then episodes within each seed (95% CI of the pooled mean)."""
    per_seed = [np.asarray(p, float) for p in per_seed]
    rng = np.random.default_rng(seed)
    point = float(np.mean([p.mean() for p in per_seed]))
    if len(per_seed) == 1 and len(per_seed[0]) < 2:
        return point, float("nan"), float("nan")
    out = np.empty(n)
    for b in range(n):
        ss = rng.integers(0, len(per_seed), len(per_seed))
        out[b] = np.mean([per_seed[s][rng.integers(0, len(per_seed[s]), len(per_seed[s]))].mean() for s in ss])
    return point, float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def fmt(t, p=2):
    m, lo, hi = t
    return f"{m:.{p}f}" if np.isnan(lo) else f"{m:.{p}f} [{lo:.{p}f}, {hi:.{p}f}]"


def discover(root: Path):
    groups = defaultdict(dict)
    for d in sorted(root.iterdir()):
        m = re.fullmatch(r"(.+)_s(\d+)", d.name)
        if m and d.is_dir() and (d / "curve.json").exists():
            groups[m.group(1)][int(m.group(2))] = d
    return groups


def _style(ax):
    ax.grid(True, color=GRID, lw=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=8)


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "axes.labelcolor": INK2, "text.color": INK})
    return plt


def load_gvf(run: Path):
    """Frozen paper GVF bank + frozen Horde from a finished gvf run."""
    from agents.gvf.predictor import GVFBank, PredictorConfig
    from agents.horde.adapter import HordeAdapter, load_rlbtp
    from agents.horde.controller import frozen_horde
    conf = json.loads((run / "config.json").read_text())
    c, sim = conf["cfgs"]["predictor"], conf.get("sim_overrides", {})
    env = make_env(sim)
    pc = PredictorConfig(**{**c, "gammas": tuple(c["gammas"]), "hidden": tuple(c["hidden"])})
    bank = GVFBank(env.observation_space.shape[0], env.action_space.n, pc, 0)
    bank.load(np.load(run / "predictors.npz"))
    rl = load_rlbtp()
    horde = rl["horde"].Horde(rl["demons"].DEMONS)
    horde.theta = np.load(run / "horde.npz")["theta"]
    return env, bank.frozen_copy(), frozen_horde(horde), HordeAdapter(env), conf["cfgs"], sim


# ------------------------------------------------------------ learning curves
def curves(groups, out: Path, plt):
    series = {}
    for grp, method_keys in (("gvf", ("paper", "horde")), ("dqn", ("dqn",))):
        runs = groups.get(grp, {})
        if not runs:
            continue
        cs = [json.loads((d / "curve.json").read_text()) for d in runs.values()]
        L = min(len(c) for c in cs)
        for k in method_keys:
            ep = np.array([cs[0][i]["episode"] for i in range(L)])
            st = np.array([np.mean([c[i]["env_steps"] for c in cs]) for i in range(L)])
            rows = {}
            for m, _ in METRICS:
                vals = np.array([[c[i][k][m] for c in cs] for i in range(L)])      # (L, seeds)
                rows[m] = [boot_ci(v, seed=i) for i, v in enumerate(vals)]
            series[k] = {"episode": ep.tolist(), "env_steps": st.tolist(), "n_seeds": len(cs), **rows}
    (out / "curves.json").write_text(json.dumps(series, indent=1))
    if not series:
        return series
    fig, axes = plt.subplots(2, 3, figsize=(13, 6.8), constrained_layout=True)
    for row, xkey, xlabel in ((0, "episode", "training episodes"), (1, "env_steps", "training env steps")):
        for col, (m, title) in enumerate(METRICS):
            ax = axes[row, col]
            for k, s in series.items():
                x = np.array(s[xkey]); y = np.array(s[m])
                ax.plot(x, y[:, 0], color=COLORS[k], lw=2, label=f"{LABELS[k]} (n={s['n_seeds']})")
                if not np.isnan(y[:, 1]).all():
                    ax.fill_between(x, y[:, 1], y[:, 2], color=COLORS[k], alpha=0.18, lw=0)
            ax.set_title(title, color=INK, loc="left"); ax.set_xlabel(xlabel); _style(ax)
    axes[0, 0].legend(frameon=False, fontsize=8, loc="lower right")
    fig.suptitle("Learning curves on the fixed 100 curve episodes (mean, 95% bootstrap CI over seeds)",
                 x=0.01, ha="left", color=INK)
    fig.savefig(out / "curves.png", dpi=140); plt.close(fig)
    return series


# ---------------------------------------------------------------- final tables
FINAL_KEYS = (("collided", "collision rate"), ("speed_ratio", "speed / v_target"), ("task_return", "task return"),
              ("distance", "distance (m)"), ("near_miss_frac", "near-miss frac"), ("steps", "episode steps"))


def final_tables(groups, out: Path):
    res, md = {}, []
    for grp in sorted(groups):
        sums = {s: json.loads((d / "summary.json").read_text())
                for s, d in groups[grp].items() if (d / "summary.json").exists()}
        sums = {s: v for s, v in sums.items() if "final" in v}
        if not sums:
            continue
        for meth in next(iter(sums.values()))["final"]:
            for kind in ("test", "hard"):
                eps = [sums[s]["final"][meth][kind]["episodes"] for s in sorted(sums)]
                row = {"seeds": sorted(sums), "n_episodes": int(sum(len(e) for e in eps)),
                       "env_steps": [sums[s]["env_steps"] for s in sorted(sums)],
                       "wall_h": [round(sums[s]["wall_s"] / 3600, 2) for s in sorted(sums)]}
                for key, _ in FINAL_KEYS:
                    row[key] = hier_boot([[float(e[key]) for e in ep] for ep in eps], seed=hash(key) % 1000)
                res[f"{grp}/{meth}/{kind}"] = row
    (out / "final.json").write_text(json.dumps(res, indent=1))
    for kind in ("test", "hard"):
        md.append(f"\n### {'TEST (3 obstacles, seeds 1,000,000+)' if kind == 'test' else 'HARD (5 obstacles, seeds 1,500,000+)'}\n")
        md.append("| run group / controller | seeds | episodes | " + " | ".join(l for _, l in FINAL_KEYS) + " |")
        md.append("|---" * (3 + len(FINAL_KEYS)) + "|")
        for name, r in res.items():
            if name.endswith("/" + kind):
                md.append(f"| {name.rsplit('/', 1)[0]} | {','.join(map(str, r['seeds']))} | {r['n_episodes']} | "
                          + " | ".join(fmt(r[k], 3 if k in ('collided', 'near_miss_frac') else 2)
                                       for k, _ in FINAL_KEYS) + " |")
    md.append("\nValues: mean [95% hierarchical bootstrap CI: seeds, then episodes]. "
              "With one seed the CI is over episodes only.")
    (out / "final.md").write_text("\n".join(md))
    return res


# --------------------------------------------------- pre-collision signals
def pre_collision(groups, out: Path, plt):
    runs = groups.get("gvf", {})
    data = {"paper": [], "horde": []}
    beta = {}
    for s, d in runs.items():
        sm = json.loads((d / "summary.json").read_text()) if (d / "summary.json").exists() else {}
        for k in data:
            if "final" in sm:
                data[k] += sm["final"][k]["test"]["pre_collision_signal"]
        if "cfgs" in sm:
            beta = {"paper": (sm["cfgs"]["controller"]["beta1"], sm["cfgs"]["controller"]["beta2"]),
                    "horde": (sm["cfgs"]["horde_controller"]["beta1"], sm["cfgs"]["horde_controller"]["beta2"])}
    if not any(data.values()):
        return {}
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), constrained_layout=True)
    stats = {}
    for ax, (k, ylabel) in zip(axes, (("paper", "q_hat_front(s, a_prev)"), ("horde", "slack = tto - tts (steps)"))):
        tr = [t for t in data[k] if t and all(v is not None for v in t)]
        if not tr:
            ax.set_visible(False); continue
        M = np.full((len(tr), 30), np.nan)
        for i, t in enumerate(tr):
            M[i, 30 - len(t):] = t
        x = np.arange(-29, 1)
        for row in M[:40]:
            ax.plot(x, row, color=COLORS[k], alpha=0.12, lw=0.8)
        mean = np.nanmean(M, 0)
        ax.plot(x, mean, color=COLORS[k], lw=2.2, label=f"mean over {len(tr)} collisions")
        for b, ls in zip(beta.get(k, ()), ("--", ":")):
            ax.axhline(b, color=INK2, ls=ls, lw=1)
        ax.set_xlabel("steps relative to the collision step (0)"); ax.set_ylabel(ylabel)
        ax.set_title(LABELS[k], loc="left"); _style(ax); ax.legend(frameon=False, fontsize=8)
        stats[k] = {"n_collisions": len(tr), "mean_by_offset": mean.tolist()}
    fig.suptitle("Controller signal in the 30 steps before test collisions (dashed beta1, dotted beta2)",
                 x=0.01, ha="left")
    fig.savefig(out / "pre_collision.png", dpi=140); plt.close(fig)
    return stats


# -------------------------------------------- Monte-Carlo ground truth (A5)
def _disc(c, g):
    return float(np.dot((1 - g) * np.asarray(c), g ** np.arange(len(c)))) if len(c) else 0.0


def mc_paper(env, state, a0, pi, gammas, n, horizon, rng):
    """E_pi[sum_k (1-g) g^k c_{t+k+1}] from (state, a0) for both cumulants, every gamma."""
    vmax = env.cfg.motion.v_max
    st = dict(state, elapsed=0, needs_reset=False)
    Gf = {g: [] for g in gammas}; Gv = {g: [] for g in gammas}
    for _ in range(n):
        env.set_state(st)
        a, cf, cv = a0, [], []
        for _t in range(horizon):
            _, _, te, _, info = env.step(a)
            col = te and info["collision"]
            cf.append(0.0 if col else info["safety"]["c_front"]); cv.append(info["linear_velocity"] / vmax)
            if te:
                break
            a = pi.sample(a, rng)
        for g in gammas:
            Gf[g].append(_disc(cf, g)); Gv[g].append(_disc(cv, g))
    return {g: (float(np.mean(Gf[g])), float(np.mean(Gv[g]))) for g in gammas}


def mc_horde(env, state, adapter, demon, env_action, cap=1000):
    """The demon's return G = r(o') + gamma(o') G(o') under its fixed policy,
    ending at env termination (deterministic env -> one rollout)."""
    env.set_state(dict(state, elapsed=0, needs_reset=False))
    G, disc = 0.0, 1.0
    for _ in range(cap):
        _, _, te, _, info = env.step(env_action)
        o2 = adapter.sensors(info)
        G += disc * float(demon["r"](o2))
        disc *= float(demon["gamma"](o2))
        if te or disc < 1e-4:
            break
    return G


def fig4(run: Path, out: Path, plt, n_mc=24, every=2):
    from agents.gvf.policies import PolicyConfig, TargetPolicy
    env, bank, *_rest, cfgs, _ = load_gvf(run)
    names, FWD = env.action_names, env.action_names.index("FORWARD")
    pi = TargetPolicy(len(names), False, PolicyConfig(**cfgs["policy"]))
    rng = np.random.default_rng(4_000_000)
    gammas = sorted(bank.gammas)
    obs, info = env.reset(seed=4_000_000, options=WALL)
    rows = []
    while True:
        st = env.get_state()
        q = {g: float(bank.nets[g].predict(obs)[0][0, FWD]) for g in gammas}
        t = len(rows)
        mc = mc_paper(env, st, FWD, pi, gammas, n_mc, 400, rng) if t % every == 0 else None
        env.set_state(st)
        obs, _, te, tr, info = env.step(FWD)
        rows.append({"t": t, "q": q, "mc": mc, "c_f": 0.0 if te and info["collision"] else info["safety"]["c_front"],
                     "wall_dist": float(env.world.clearance(*info["pose"][:2])), "v": info["linear_velocity"]})
        if te or tr:
            break
    cf = np.array([r["c_f"] for r in rows])
    for i, r in enumerate(rows):                         # realised return of the always-FORWARD trajectory
        r["realised"] = {g: _disc(cf[i:], g) for g in gammas}
    show = [g for g in (0.95, 0.975, 0.983) if g in gammas] or gammas[:3]
    fig, axes = plt.subplots(1, len(show), figsize=(4.4 * len(show), 3.6), constrained_layout=True, sharey=True)
    axes = np.atleast_1d(axes)
    t = np.array([r["t"] for r in rows])
    for ax, g in zip(axes, show):
        ax.step(t, cf, where="post", color="#9a9893", lw=1, label="c_f (cumulant)")
        ax.plot(t, [r["realised"][g] for r in rows], color=INK2, lw=1.2, ls="--", label="realised return (always FORWARD)")
        tm = [r["t"] for r in rows if r["mc"]]
        ax.plot(tm, [r["mc"][g][0] for r in rows if r["mc"]], color=INK, lw=1.6, label="Monte-Carlo return under pi")
        ax.plot(t, [r["q"][g] for r in rows], color=COLORS["paper"], lw=2.2, label="q_hat_front(s, FORWARD)")
        ax.set_title(f"gamma = {g}", loc="left"); ax.set_xlabel("step (driving FORWARD from rest)"); _style(ax)
    axes[0].set_ylabel("front safety"); axes[0].legend(frameon=False, fontsize=7, loc="lower left")
    fig.suptitle(f"Fig. 4 analogue: approach a wall 3 m ahead (run {run.name}; collision at step {len(rows) - 1})",
                 x=0.01, ha="left")
    fig.savefig(out / "fig4_wall_predictions.png", dpi=140); plt.close(fig)
    json.dump(rows, open(out / "fig4_wall_predictions.json", "w"), default=float)
    err = {g: float(np.sqrt(np.mean([(r["q"][g] - r["mc"][g][0]) ** 2 for r in rows if r["mc"]]))) for g in gammas}
    return {"steps_to_collision": len(rows) - 1, "rmse_vs_mc": err}


def fig5(run: Path, out: Path, plt, speeds=(0.0, 0.1, 0.2, 0.22), T=250):
    from agents.gvf.controller import ControllerConfig, PaperController
    from agents.horde.controller import HordeController, HordeControllerConfig
    env, bank, horde, ad, cfgs, _ = load_gvf(run)
    names = env.action_names
    ctrls = {"paper": lambda: PaperController(bank, names, ControllerConfig(**cfgs["controller"]), env.cfg.motion.v_max),
             "horde": lambda: HordeController(horde, ad, names, HordeControllerConfig(**cfgs["horde_controller"]))}
    shades = ["#9ec5f2", "#5d9ee8", "#2a78d6", "#174a8c"]
    fig, axes = plt.subplots(2, 2, figsize=(11, 6.4), constrained_layout=True, sharex=True)
    res = {}
    for row, (k, mk) in enumerate(ctrls.items()):
        for v0, col in zip(speeds, shades):
            pol = mk(); pol.reset()
            obs, info = env.reset(seed=4_000_001, options={**WALL, "velocity": (v0, 0.0)})
            dist, vel, collided = [], [], False
            for _ in range(T):
                obs, _, te, tr, info = env.step(pol.act(obs, info))
                dist.append(float(env.world.clearance(*info["pose"][:2]) - env.cfg.robot.collision_radius))
                vel.append(float(info["linear_velocity"]))
                if te or tr:
                    collided = bool(te and info["collision"]); break
            axes[row, 0].plot(dist, color=col, lw=1.8, label=f"v0 = {v0}")
            axes[row, 1].plot(vel, color=col, lw=1.8, label=f"v0 = {v0}")
            res[f"{k}/v0={v0}"] = {"collided": collided, "min_clearance_m": min(dist), "steps": len(dist),
                                   "mean_speed": float(np.mean(vel))}
        axes[row, 0].set_ylabel(f"{LABELS[k]}\nclearance to nearest wall (m)")
        axes[row, 1].set_ylabel("speed (m/s)")
        for ax in axes[row]:
            _style(ax)
    axes[0, 0].legend(frameon=False, fontsize=8)
    for ax in axes[1]:
        ax.set_xlabel("step")
    fig.suptitle("Fig. 5 analogue: drive at a wall 3 m ahead from different initial speeds (frozen final predictors)",
                 x=0.01, ha="left")
    fig.savefig(out / "fig5_wall_approach.png", dpi=140); plt.close(fig)
    return res


# ----------------------------------------------------- prediction quality
def _pq_worker(job):
    run, ep_seeds, n_mc, per_ep = job
    from agents.gvf.policies import BehaviourPolicy, PolicyConfig, TargetPolicy
    env, bank, horde, ad, cfgs, _ = load_gvf(Path(run))
    names = env.action_names
    pcfg = PolicyConfig(**cfgs["policy"])
    mu, pi = BehaviourPolicy(len(names), False, pcfg), TargetPolicy(len(names), False, pcfg)
    rng = np.random.default_rng(ep_seeds[0])
    demons = {d["name"]: d for d in horde.demons}
    fixed = {"steps-to-obstacle": "FORWARD", "steps-to-stop": "STOP", "light-if-forward": "FORWARD",
             "bump-soon": "FORWARD"}
    dn = [d["name"] for d in horde.demons]
    rows = []
    for s in ep_seeds:
        obs, info = env.reset(seed=int(s))
        traj, a_prev = [], names.index("STOP")
        while True:                                       # behaviour-policy trajectory (states the predictors were trained on)
            a = mu.sample(a_prev, rng)
            traj.append((env.get_state(), obs, info, a))
            obs, _, te, tr, info = env.step(a)
            if te or tr:
                break
            a_prev = a
        for i in sorted(rng.choice(len(traj), min(per_ep, len(traj)), replace=False)):
            st, o, inf, a = traj[i]
            mc = mc_paper(env, st, a, pi, bank.gammas, n_mc, 300, rng)
            r = {"paper": {str(g): {"q_f": float(bank.nets[g].predict(o)[0][0, a]), "mc_f": mc[g][0],
                                    "q_v": float(bank.nets[g].predict(o)[1][0, a]), "mc_v": mc[g][1]}
                           for g in bank.gammas}, "horde": {}}
            Q = horde.q(ad.sensors(inf))
            for name, act in fixed.items():
                r["horde"][name] = {"q": float(Q[dn.index(name), int(ad.Action[act])]),
                                    "mc": mc_horde(env, st, ad, demons[name], names.index(act))}
            rows.append(r)
    return rows


def _pq_stats(q, y):
    q, y = np.asarray(q), np.asarray(y)
    rmse = float(np.sqrt(np.mean((q - y) ** 2)))
    sd = float(np.std(y))
    corr = float(np.corrcoef(q, y)[0, 1]) if sd > 0 and np.std(q) > 0 else float("nan")
    return {"rmse": rmse, "mae": float(np.mean(np.abs(q - y))), "target_mean": float(y.mean()), "target_sd": sd,
            "nrmse": rmse / sd if sd > 0 else float("nan"), "pearson_r": corr, "n": int(len(y))}


def prediction_quality(run: Path, out: Path, n_states_eps=48, per_ep=5, n_mc=16, workers=8):
    eps = seeds("ablate", 100 + n_states_eps)[100:]
    chunks = [(str(run), eps[i::workers], n_mc, per_ep) for i in range(workers)]
    with ProcessPoolExecutor(workers) as ex:
        rows = [r for part in ex.map(_pq_worker, chunks) for r in part]
    res = {"run": run.name, "n_states": len(rows), "paper": {}, "horde": {}}
    for g in rows[0]["paper"]:
        for head, (qk, mk) in (("front_safety", ("q_f", "mc_f")), ("speed", ("q_v", "mc_v"))):
            res["paper"][f"gamma={g}/{head}"] = _pq_stats([r["paper"][g][qk] for r in rows],
                                                          [r["paper"][g][mk] for r in rows])
    for name in rows[0]["horde"]:
        res["horde"][name] = _pq_stats([r["horde"][name]["q"] for r in rows], [r["horde"][name]["mc"] for r in rows])
    (out / "prediction_quality.json").write_text(json.dumps(res, indent=1))
    md = [f"Prediction quality on {len(rows)} states sampled from behaviour-policy episodes "
          f"(ABLATE seeds; frozen final predictors of {run.name}). Paper GVFs: {n_mc} Monte-Carlo rollouts "
          "under pi per state. Given demons: one deterministic rollout under their fixed policy.\n",
          "| predictor | target mean | target sd | RMSE | NRMSE (RMSE/sd) | Pearson r |", "|---|---|---|---|---|---|"]
    for grp in ("paper", "horde"):
        for k, s in res[grp].items():
            md.append(f"| {'paper ' if grp == 'paper' else 'given demon '}{k} | {s['target_mean']:.3f} | "
                      f"{s['target_sd']:.3f} | {s['rmse']:.3f} | {s['nrmse']:.2f} | {s['pearson_r']:.2f} |")
    (out / "prediction_quality.md").write_text("\n".join(md))
    return res


# ---------------------------------------------------------- gamma/beta table
BETAS = ((0.3, 0.5), (0.4, 0.6), (0.5, 0.7), (0.5, 0.8), (0.6, 0.8), (0.7, 0.9))


def _abl_worker(job):
    run, g, n = job
    from agents.gvf.controller import ControllerConfig, PaperController
    env, bank, *_r, cfgs, sim = load_gvf(Path(run))
    out = []
    for b1, b2 in BETAS:
        cc = ControllerConfig(**{**cfgs["controller"], "gamma": g, "beta1": b1, "beta2": b2})
        s = summarize(evaluate(PaperController(bank, env.action_names, cc, env.cfg.motion.v_max),
                               seeds("ablate", n), "ablate", sim))
        out.append({"gamma": g, "beta1": b1, "beta2": b2, **s})
    return out


def ablation(run: Path, out: Path, n=100, workers=8):
    _, bank, *_ = load_gvf(run)
    with ProcessPoolExecutor(workers) as ex:
        rows = [r for part in ex.map(_abl_worker, [(str(run), g, n) for g in sorted(bank.gammas)]) for r in part]
    (out / "ablation_gamma_beta.json").write_text(json.dumps(rows, indent=1))
    md = [f"Eval-only gamma x (beta1, beta2) ablation, frozen final predictors of {run.name}, {n} ABLATE episodes "
          "each (no selection is made from this table).\n",
          "| gamma | beta1 | beta2 | collision-free | speed / v_target | task return | near-miss frac |",
          "|---|---|---|---|---|---|---|"]
    for r in rows:
        md.append(f"| {r['gamma']} | {r['beta1']} | {r['beta2']} | {r['collision_free_rate']:.2f} | "
                  f"{r['speed_ratio']:.2f} | {r['mean_task_return']:.1f} | {r['near_miss_frac']:.3f} |")
    (out / "ablation_gamma_beta.md").write_text("\n".join(md))
    return rows


def beams(final, out: Path):
    md = ["| controller | beams | seed | test collision rate | test speed / v_target | test task return |",
          "|---|---|---|---|---|---|"]
    found = False
    for grp, nb in (("gvf", 72), ("gvf_b24", 24), ("dqn", 72), ("dqn_b24", 24)):
        for key, r in final.items():
            g, meth, kind = key.split("/")
            if g == grp and kind == "test":
                found |= grp.endswith("b24")
                md.append(f"| {meth} | {nb} | {','.join(map(str, r['seeds']))} | {fmt(r['collided'], 3)} | "
                          f"{fmt(r['speed_ratio'])} | {fmt(r['task_return'])} |")
    if found:
        (out / "beams.md").write_text("\n".join(md))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", default="runs_safety")
    ap.add_argument("--out", default=None)
    ap.add_argument("--probe-run", default=None, help="gvf run for Fig. 4/5, prediction quality, ablation "
                                                      "(default: gvf_s0)")
    ap.add_argument("--skip-mc", action="store_true", help="skip the Monte-Carlo pieces (Fig. 4, prediction quality)")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    root = Path(args.runs); out = Path(args.out or root / "analysis"); out.mkdir(parents=True, exist_ok=True)
    plt = _plt()
    groups = discover(root)
    report = {"groups": {g: sorted(v) for g, v in groups.items()}}
    report["curves"] = bool(curves(groups, out, plt))
    final = final_tables(groups, out)
    report["pre_collision"] = {k: v["n_collisions"] for k, v in pre_collision(groups, out, plt).items()}
    beams(final, out)
    probe = Path(args.probe_run) if args.probe_run else groups.get("gvf", {}).get(0)
    if probe and (probe / "predictors.npz").exists():
        report["fig5"] = fig5(probe, out, plt)
        report["ablation_rows"] = len(ablation(probe, out, workers=args.workers))
        if not args.skip_mc:
            report["fig4"] = fig4(probe, out, plt)
            pq = prediction_quality(probe, out, workers=args.workers)
            report["prediction_quality_states"] = pq["n_states"]
    (out / "analysis_index.json").write_text(json.dumps(report, indent=1, default=str))
    print(json.dumps(report, indent=1, default=str))


if __name__ == "__main__":
    main()
