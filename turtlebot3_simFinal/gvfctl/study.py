"""Job definitions and the per-job pipeline (one process per job; resumable).

    python -m gvfctl.study list --phase 1|2          # print job ids
    python -m gvfctl.study run JOB_ID --root results/<ts> [--budget 300000]

Job id grammar (double-underscore separated):
  tune-dqn__p{1,2}__{A,B}__{grid index}__s{seed}   DQN grid, VALIDATION seeds, 100k steps
  tune-tau__p2__{0.5,1.0}__s{seed}                 headway tau for the Part 2 safety GVF
  mc__p1 | mc__p2__{tau}                           cached Monte-Carlo safety targets
  main__p{1,2}__{gvf|dqnA|dqnB|hand|demons-default|demons-fixed}__s{seed}
  abl__p{1,2}__{variant}__s{seed}
  big__p{1,2}__dqnB__s{seed}                       1M steps
A job writes <root>/jobs/<id>.json and is skipped if that file already exists.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import pickle
import platform
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

from .controllers import HandCfg, SpeedCfg, SteerCfg, as_dict
from .dqn import family
from .envs import PART_ENVS, ROOT, action_space, episode_seeds
from .evaluate import (ControllerPolicy, DQNPolicy, HandPolicy, determinism_check, run_episodes,
                       summarize, survival_data)
from .gvf import GVFConfig
from . import quality as Qm
from . import train as T

TRAIN_SEEDS = list(range(10))
ABL_SEEDS = list(range(5))
BIG_SEEDS = list(range(3))
TUNE_SEEDS = (100, 101)
TUNE_BUDGET = 100_000
N_VAL, N_TEST, N_UNCAPPED, UNCAPPED_CAP = 40, 200, 100, 20_000
N_QSTATES, K_ROLLOUTS = 300, 8
if os.environ.get("GVFCTL_SMOKE"):           # identical code paths, tiny sizes (debugging only)
    N_VAL, N_TEST, N_UNCAPPED, UNCAPPED_CAP, TUNE_BUDGET = 4, 6, 4, 1500, 3000
    N_QSTATES, K_ROLLOUTS = 12, 2

DQN_GRID = {
    1: [dict(gamma=g, lr=lr, target_update=tu, reward_scale=rs)
        for g in (0.95, 0.99) for lr in (3e-4, 1e-3) for tu in (500, 2000) for rs in (1.0, 0.1)],
}
DQN_GRID[2] = [dict(d, turn_penalty=tp) for d in DQN_GRID[1] for tp in (0.0, 0.1)]
# Extension appended after phase 1 found every winner at the grid edge (lowest lr,
# shortest target period): extend beyond that edge so DQN is not handicapped.
_EXT = [dict(gamma=g, lr=lr, target_update=tu, reward_scale=1.0)
        for g in (0.95, 0.99) for lr, tu in ((1e-4, 250), (1e-4, 500), (3e-4, 250))]
DQN_GRID[1] = DQN_GRID[1] + _EXT
DQN_GRID[2] = DQN_GRID[2] + [dict(d, turn_penalty=0.0) for d in _EXT]

ABLATIONS = {
    1: {"nomirror": {"mirror": False}, "h090": {"horizons": (0.90,)}, "h095": {"horizons": (0.95,)},
        "h097": {"horizons": (0.97,)}, "h099": {"horizons": (0.99,)}, "p00": {"persist_p": 0.0},
        "p03": {"persist_p": 0.3}, "randwalk": {"__behaviour": "randomwalk"}, "mse": {"loss": "mse"},
        "b50k": {"__budget": 50_000}, "b100k": {"__budget": 100_000}},
    2: {"H10": {"vel_H": 10}, "H40": {"vel_H": 40}, "separate": {"vel_shared": False},
        "veltarget-persistent": {"vel_target": "persistent"}},
}
MAIN_METHODS = ("gvf", "dqnA", "dqnB", "hand", "demons-default", "demons-fixed")


# ============================================================ job listing ===
def list_jobs(phase: int):
    jobs = []
    if phase == 1:
        for part in (1, 2):
            for fam in "AB":
                for i in range(len(DQN_GRID[part])):
                    jobs += [f"tune-dqn__p{part}__{fam}__{i}__s{s}" for s in TUNE_SEEDS]
        jobs += [f"tune-tau__p2__{tau}__s{s}" for tau in (0.5, 1.0) for s in TUNE_SEEDS]
        jobs += ["mc__p1", "mc__p2__0.5", "mc__p2__1.0"]
    elif phase == 2:                         # longest jobs first (better load balancing)
        jobs += [f"big__p{part}__dqnB__s{s}" for part in (1, 2) for s in BIG_SEEDS]
        for m in MAIN_METHODS:
            for part in (2, 1):
                seeds = [0] if m == "hand" else TRAIN_SEEDS
                jobs += [f"main__p{part}__{m}__s{s}" for s in seeds]
        for part in (2, 1):
            jobs += [f"abl__p{part}__{v}__s{s}" for v in ABLATIONS[part] for s in ABL_SEEDS]
    return jobs


# =========================================================== utilities ===
def _code_hash():
    h = hashlib.sha256()
    for p in sorted((ROOT / "gvfctl").glob("*.py")):
        h.update(p.read_bytes())
    return h.hexdigest()[:16]


def _provenance(args):
    import gymnasium
    files = {q.name: hashlib.sha256(q.read_bytes()).hexdigest()[:12] for q in sorted((ROOT / "gvfctl").glob("*.py"))}
    return {"code_sha256": _code_hash(), "file_sha256": files, "git_sha": None, "python": sys.version.split()[0],
            "numpy": np.__version__, "gymnasium": gymnasium.__version__,
            "platform": platform.platform(), "args": vars(args)}


def _read_jobs(root: Path, prefix: str):
    out = []
    for p in sorted((root / "jobs").glob(prefix + "*.json")):
        try:
            out.append(json.loads(p.read_text()))
        except Exception:
            pass
    return out


def best_dqn_config(root: Path, part: int, fam: str):
    rows = _read_jobs(root, f"tune-dqn__p{part}__{fam}__")
    if not rows:
        return 0, DQN_GRID[part][0], "no tuning results found; using grid[0]"
    by = {}
    for r in rows:
        by.setdefault(r["grid_index"], []).append(r["val_metric"])
    means = {i: float(np.mean(v)) for i, v in by.items()}
    best = max(means, key=lambda i: (means[i], -i))
    return best, DQN_GRID[part][best], {str(i): m for i, m in sorted(means.items())}


def best_tau(root: Path):
    rows = _read_jobs(root, "tune-tau__p2__")
    if not rows:
        return 0.5, "no tau results; default 0.5"
    by = {}
    for r in rows:
        by.setdefault(r["tau"], []).append(r["val_metric"])
    means = {t: float(np.mean(v)) for t, v in by.items()}
    return max(means, key=lambda t: (means[t], -t)), means


def _metric(part, summ):
    return summ["survival"] if part == 1 else summ["mean_distance"]


def _val(policy, part, env="train", n=N_VAL):
    return summarize(run_episodes(policy, env, part, episode_seeds("val", env, n)))


def _compact(recs):
    return [{"steps": r["steps"], "collided": r["collided"], "distance": round(r["distance"], 4),
             "category": r["category"]} for r in recs]


# ============================================================ tuning ======
def tune_steer(src, A, part, speed: SpeedCfg, env="train", grid="full"):
    if grid == "full":
        pairs = [(lo, hi) for lo in (0.4, 0.5, 0.6, 0.7) for hi in (0.6, 0.7, 0.8, 0.9) if lo <= hi]
        deltas = (0.02, 0.05, 0.1)
    else:
        pairs, deltas = [(0.5, 0.7), (0.6, 0.8), (0.7, 0.9)], (0.05, 0.1)
    rows = []
    for (lo, hi), d in itertools.product(pairs, deltas):
        cfg = SteerCfg(lo, hi, d)
        s = _val(ControllerPolicy(src, A, cfg, speed), part, env)
        rows.append({"beta_lo": lo, "beta_hi": hi, "delta": d, "metric": _metric(part, s),
                     "survival": s["survival"], "distance": s["mean_distance"]})
    return _pick(rows, SteerCfg(), lambda r: SteerCfg(r["beta_lo"], r["beta_hi"], r["delta"])), rows


def tune_speed(src, A, steer: SteerCfg, grid="full"):
    vt = (0.12, 0.22); lo = (0.4, 0.6); hi = (0.7, 0.85); kw = (0.0, 0.5)
    if grid != "full":
        lo, kw = (0.5,), (0.0, 0.5)
    rows = []
    for v_t, bl, bh, k in itertools.product(vt, lo, hi, kw):
        cfg = SpeedCfg(v_target=v_t, beta_v_lo=bl, beta_v_hi=bh, k_w=k)
        s = _val(ControllerPolicy(src, A, steer, cfg), 2)
        rows.append({"v_target": v_t, "beta_v_lo": bl, "beta_v_hi": bh, "k_w": k,
                     "metric": s["mean_distance"], "survival": s["survival"]})
    return _pick(rows, SpeedCfg(), lambda r: SpeedCfg(v_target=r["v_target"], beta_v_lo=r["beta_v_lo"],
                                                       beta_v_hi=r["beta_v_hi"], k_w=r["k_w"])), rows


def _pick(rows, default, make):
    best = max(r["metric"] for r in rows)
    ties = [r for r in rows if r["metric"] >= best - 1e-12]
    dd = as_dict(default)
    for r in ties:                               # prefer the documented default when tied
        if all(r.get(k) == v for k, v in dd.items() if k in r):
            return make(r)
    return make(ties[0])


def tune_hand(A, part):
    rows = []
    turn = [(dt, dh) for dt in (0.4, 0.5, 0.7, 0.9) for dh in (0.25, 0.3, 0.4) if dh < dt]
    speeds = [(1.0, 0.5)] if part == 1 else [(df, dm) for df in (0.8, 1.2) for dm in (0.4, 0.6)]
    for (dt, dh), (df, dm) in itertools.product(turn, speeds):
        cfg = HandCfg(dt, dh, df, dm)
        s = _val(HandPolicy(A, cfg), part)
        rows.append({**as_dict(cfg), "metric": _metric(part, s)})
    best = max(rows, key=lambda r: r["metric"])
    return HandCfg(best["d_turn"], best["d_hard"], best["d_fast"], best["d_mid"]), rows


# ============================================================ evaluation ==
def full_evaluation(policy_factory, part, A, out, envs=None, uncapped=True, fail_log=50, retune=None):
    """policy_factory(env_name) -> Policy (allows per-env retuned thresholds)."""
    envs = envs or PART_ENVS[part]
    p0 = policy_factory("train")
    out["determinism_ok"] = determinism_check(p0, "train", part, episode_seeds("test", "train", 10))
    R = np.random.default_rng(0).standard_normal((1, 24)) * 0 + 1.5
    t = time.perf_counter()
    for _ in range(200):
        p0.reset(1); p0.act(R, np.array([0]))
    out["inference_ms_per_decision"] = (time.perf_counter() - t) / 200 * 1e3
    out["final"], out["final_episodes"], out["fail_logs"], out["trajectories"] = {}, {}, {}, {}
    for e in envs:
        pol = policy_factory(e)
        recs = run_episodes(pol, e, part, episode_seeds("test", e, N_TEST),
                            fail_log=fail_log if e in ("train", "dense") else 0,
                            traj=6 if e == "train" else 0)
        out["final"][e] = summarize(recs)
        out["final_episodes"][e] = _compact(recs)
        logs = [r["fail_log"] for r in recs if "fail_log" in r]
        if logs:
            out["fail_logs"][e] = logs
        tr = [r["traj"] for r in recs if "traj" in r]
        if tr:
            out["trajectories"][e] = tr
    if uncapped:
        out["uncapped"] = {}
        for e in envs:
            recs = run_episodes(policy_factory(e), e, part, episode_seeds("test", e, N_UNCAPPED, start=5000),
                                cap=UNCAPPED_CAP, n_par=N_UNCAPPED)
            times, events = survival_data(recs, UNCAPPED_CAP)
            s = summarize(recs)
            out["uncapped"][e] = {"times": times, "events": events,
                                  "mean_ttc": s["mean_steps"], "median_ttc": s["median_ttc_steps"],
                                  "collision_rate": s["collision_rate"], "mean_distance": s["mean_distance"],
                                  "distance_before_collision": s["distance_before_collision"],
                                  "censored_frac": float(np.mean([not ev for ev in events]))}


# ============================================================ job bodies ==
def _gvf_cfg(part, root, overrides=None):
    cfg = GVFConfig()
    if part == 2:
        tau, _ = best_tau(root)
        cfg = replace(cfg, velocity=True, d_min=0.2, headway_tau=float(tau))
    for k, v in (overrides or {}).items():
        if not k.startswith("__"):
            cfg = replace(cfg, **{k: v})
    return cfg


def _load_mc(root, part, cfg):
    key = "mc__p1" if part == 1 else f"mc__p2__{cfg.headway_tau}"
    path = root / "cache" / f"{key}.pkl"
    if not path.exists():
        return None, None
    with open(path, "rb") as fh:
        d = pickle.load(fh)
    return d["states"], d["mc"]


def run_job(job: str, root: Path, budget: int, log=print) -> dict:
    parts = job.split("__")
    kind = parts[0]
    out = {"job": job, "kind": kind}
    t0 = time.perf_counter()

    if kind == "mc":
        part = int(parts[1][1]); tau = float(parts[2]) if len(parts) > 2 else 0.0
        A = action_space(part)
        cfg = GVFConfig() if part == 1 else GVFConfig(velocity=True, d_min=0.2, headway_tau=tau)
        states = Qm.sample_states(part, A, n=N_QSTATES)
        mc = Qm.mc_safety(states, A, cfg.horizons, cfg.d_min, cfg.headway_tau, cfg.persist_p, K=K_ROLLOUTS)
        (root / "cache").mkdir(parents=True, exist_ok=True)
        with open(root / "cache" / f"{job}.pkl", "wb") as fh:
            pickle.dump({"states": states, "mc": mc, "cfg": cfg.to_dict()}, fh)
        out.update(part=part, tau=tau, n_states=len(states), mc_mean=mc.mean(axis=(0, 1)).tolist())

    elif kind == "tune-dqn":
        part, fam, gi, seed = int(parts[1][1]), parts[2], int(parts[3]), int(parts[4][1:])
        A = action_space(part)
        dcfg = family(fam, **DQN_GRID[part][gi])
        dqn, curve, meta = T.train_dqn(part, dcfg, TUNE_BUDGET, seed, log=log)
        s = _val(DQNPolicy(dqn, A), part)
        out.update(part=part, family=fam, grid_index=gi, seed=seed, config=dcfg.to_dict(),
                   curve=curve, train_meta=meta, val=s, val_metric=_metric(part, s))

    elif kind == "tune-tau":
        part, tau, seed = 2, float(parts[2]), int(parts[3][1:])
        A = action_space(2)
        cfg = GVFConfig(velocity=True, d_min=0.2, headway_tau=tau)
        model, curve, meta = T.train_gvf(2, cfg, TUNE_BUDGET, seed, log=log)
        s = _val(ControllerPolicy(model, A, SteerCfg(), SpeedCfg()), 2)
        out.update(part=2, tau=tau, seed=seed, curve=curve, train_meta=meta, val=s, val_metric=s["mean_distance"])

    elif kind in ("main", "abl", "big"):
        part, method, seed = int(parts[1][1]), parts[2], int(parts[3][1:])
        A = action_space(part)
        out.update(part=part, method=method, seed=seed)
        light = kind == "abl"
        if kind == "big":
            budget = 5000 if os.environ.get("GVFCTL_SMOKE") else 1_000_000
        if kind == "abl":
            ov = ABLATIONS[part][method]
            budget = min(ov.get("__budget", budget), budget) if os.environ.get("GVFCTL_SMOKE") else ov.get("__budget", budget)
            behaviour = ov.get("__behaviour", "mixture")
            cfg = _gvf_cfg(part, root, ov)
            out["ablation"] = {k: (list(v) if isinstance(v, tuple) else v) for k, v in ov.items()}
            model, curve, meta = T.train_gvf(part, cfg, budget, seed, behaviour=behaviour, log=log)
            src_kind, src = "gvf", model
            out["config"] = cfg.to_dict()
        elif method == "gvf":
            cfg = _gvf_cfg(part, root)
            model, curve, meta = T.train_gvf(part, cfg, budget, seed, log=log)
            src_kind, src = "gvf", model
            out["config"] = cfg.to_dict()
        elif method.startswith("demons"):
            variant = method.split("-")[1]
            src, curve, meta = T.train_demons(part, variant, budget, seed, log=log)
            src_kind = "demons"
            out["config"] = {"variant": variant}
        elif method.startswith("dqn"):
            fam = method[-1]
            gi, grid_cfg, tuning = best_dqn_config(root, part, fam)
            dcfg = family(fam, **grid_cfg)
            src, curve, meta = T.train_dqn(part, dcfg, budget, seed, log=log)
            src_kind = "dqn"
            out["config"] = dcfg.to_dict(); out["dqn_tuning"] = {"chosen_index": gi, "val_means": tuning}
        elif method == "hand":
            src_kind, src, curve, meta = "hand", None, [], {"updates": 0, "wall_s": 0}
        else:
            raise ValueError(job)
        out["budget"] = budget if src_kind != "hand" else 0
        out["curve"], out["train_meta"] = curve, meta

        # ---- validation tuning (validation seeds only) -------------------
        if src_kind in ("gvf", "demons"):
            steer, steer_rows = tune_steer(src, A, part, SpeedCfg(), grid="small" if light else "full")
            speed, speed_rows = SpeedCfg(), []
            if part == 2:
                speed, speed_rows = tune_speed(src, A, steer, grid="small" if light else "full")
            out["val_tuning"] = {"steer": as_dict(steer), "speed": as_dict(speed),
                                 "steer_grid": steer_rows, "speed_grid": speed_rows}
            retuned = {}
            if part == 1 and not light:
                for ge in ("gazebo", "gazebo_train"):
                    st_g, _ = tune_steer(src, A, part, speed, env=ge, grid="small")
                    retuned[ge] = st_g
                out["val_tuning"]["retuned"] = {k: as_dict(v) for k, v in retuned.items()}
            factory = lambda e: ControllerPolicy(src, A, retuned.get(e, steer), speed)
        elif src_kind == "dqn":
            factory = lambda e: DQNPolicy(src, A)
        else:
            hcfg, rows = tune_hand(A, part)
            out["val_tuning"] = {"hand": as_dict(hcfg), "grid": rows}
            factory = lambda e: HandPolicy(A, hcfg)

        # ---- test evaluation ----------------------------------------------
        envs = ["train", "dense"] if light else (PART_ENVS[part] if kind == "main" else PART_ENVS[part])
        full_evaluation(factory, part, A, out, envs=envs, uncapped=(kind == "main"),
                        fail_log=0 if light else 50)
        if kind == "big":
            out.pop("fail_logs", None); out.pop("trajectories", None)

        # ---- prediction quality ---------------------------------------------
        if kind == "main" and src_kind in ("gvf", "demons"):
            q = {}
            gcfg = src.cfg if src_kind == "gvf" else _gvf_cfg(part, root)
            states, mc = _load_mc(root, part, gcfg)
            if states is not None:
                R = np.array([s["ranges"] for s in states])
                Qp = np.transpose(Qm._q_all(src, R), (0, 2, 1))     # (S, n, H)
                q["safety"] = Qm.safety_quality(Qp, mc, gcfg.horizons)
                q["safety_n_states"] = len(states)
            else:
                q["safety"] = "MC cache missing (run phase 1 mc jobs)"
            if part == 1:
                q["ttc_trace"] = Qm.ttc_trace(src, A, gcfg.horizons)
            if part == 2 and src_kind == "gvf" and states is not None:
                q["velocity"] = Qm.velocity_quality(src, A, states, K=K_ROLLOUTS)
            out["quality"] = q

        # ---- Part 2 extras -----------------------------------------------------
        if kind == "main" and part == 2:
            seeds = episode_seeds("test", "train", N_TEST)
            if src_kind == "gvf":
                ab = {}
                for mode in ("fixed", "safety", "lin", "ang", "both"):
                    sp = replace(speed, mode=mode)
                    ab[mode] = {e: summarize(run_episodes(ControllerPolicy(src, A, steer, sp), e, 2,
                                                          episode_seeds("test", e, N_TEST)))
                                for e in ("train", "dense")}
                out["speed_ablation"] = ab
                par = {}
                for vt in (0.06, 0.09, 0.12, 0.16, 0.22):
                    s = summarize(run_episodes(ControllerPolicy(src, A, steer, replace(speed, v_target=vt)),
                                               "train", 2, seeds[:100]))
                    par[str(vt)] = {k: s[k] for k in ("mean_distance", "collision_rate", "survival", "mean_speed")}
                out["pareto"] = par
            if src_kind == "dqn":
                par = {}
                for b in (-0.5, -0.25, 0.0, 0.25, 0.5):
                    s = summarize(run_episodes(DQNPolicy(src, A, speed_bias=b), "train", 2, seeds[:100]))
                    par[str(b)] = {k: s[k] for k in ("mean_distance", "collision_rate", "survival", "mean_speed")}
                out["pareto"] = par

        # ---- save trained parameters --------------------------------------------
        if src is not None:
            (root / "agents").mkdir(parents=True, exist_ok=True)
            np.savez_compressed(root / "agents" / f"{job}.npz", **src.state())
    else:
        raise ValueError(f"unknown job kind: {job}")
    out["wall_s"] = round(time.perf_counter() - t0, 1)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    l = sub.add_parser("list"); l.add_argument("--phase", type=int, required=True)
    r = sub.add_parser("run"); r.add_argument("job"); r.add_argument("--root", required=True)
    r.add_argument("--budget", type=int, default=300_000)
    args = ap.parse_args()
    if args.cmd == "list":
        print("\n".join(list_jobs(args.phase)))
        return 0
    root = Path(args.root)
    path = root / "jobs" / f"{args.job}.json"
    if path.exists():
        print(f"skip {args.job} (exists)")
        return 0
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    out = run_job(args.job, root, args.budget, log=lambda *a: print(*a, flush=True))
    out["provenance"] = _provenance(args)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(out, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))
    os.replace(tmp, path)
    print(f"done {args.job} in {out['wall_s']}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
