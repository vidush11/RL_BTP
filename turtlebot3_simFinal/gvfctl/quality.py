"""Prediction quality against Monte-Carlo ground truth.

States: >= 300 states visited by a smooth random walk in the training layout
(QUALITY seed range), each saved as (pose, obstacle layout, scan, previous action)
and restored exactly through env.reset(options={"spawn", "obstacles"}).

Safety: for every state x every action x K rollouts, start with that action and
follow the persistent target policy; G_h = (1-g_h) sum_t g_h^t c_{t+1}, stopped
at collision (gamma = 0). Truncated at T_max where g_max^T_max < 1e-3.

Velocity (Part 2): from each state the controller picks a0 (with the state's
previous action); K rollouts of the epsilon-softened controller (the velocity
GVF's own target policy) give G = (1-g_v) sum_t g_v^t c(a_t), stopped at
collision. Baselines: mean, zero, persistence (future = current commanded value).
"""
from __future__ import annotations

import math
from typing import List, Optional

import numpy as np

from simulator import TurtleBot3LidarEnv, load_config
from .actions import ActionSpace
from .controllers import controller_batch, ttc_from_safety
from .envs import QUALITY_BASE, env_config, make_env
from .train import random_walk_step


def sample_states(part: int, A: ActionSpace, n: int = 300, every: int = 7, seed: int = QUALITY_BASE):
    env, _ = make_env("train", part, 800)
    rng = np.random.default_rng(seed)
    states, ep = [], 0
    while len(states) < n:
        _, info = env.reset(seed=seed + ep); ep += 1
        prev = A.default_action()
        for t in range(800):
            x, y, th = info["pose"]
            if t % every == 3 and env.world.clearance(x, y) >= 0.16:
                states.append({"pose": (float(x), float(y), float(th)),
                               "obstacles": env.world.obstacles.copy(),
                               "ranges": np.asarray(info["ranges_m"], float).copy(), "prev": int(prev)})
                if len(states) >= n:
                    break
            a = random_walk_step(prev, A, rng)
            _, _, term, trunc, info = env.step(a)
            prev = a
            if term or trunc:
                break
    return states


def _rollout_env(part):
    return TurtleBot3LidarEnv(load_config(env_config("train", part, cap=None)))


def mc_safety(states, A: ActionSpace, horizons, d_min: float, tau: float, persist_p: float,
              K: int = 8, seed: int = QUALITY_BASE + 777, n_par: int = 64) -> np.ndarray:
    """MC targets, shape (S, n_actions, H)."""
    g = np.asarray(horizons, float)
    T_max = int(math.ceil(math.log(1e-3) / math.log(g.max())))
    P = A.persistent_policy(persist_p)
    jobs = [(i, a, k) for i in range(len(states)) for a in range(A.n) for k in range(K)]
    out = np.zeros((len(states), A.n, len(g)))
    rng = np.random.default_rng(seed)
    envs = [_rollout_env(2 if A.n_speed > 1 else 1) for _ in range(n_par)]
    q = list(range(len(jobs)))
    slot = [None] * n_par

    def start(s):
        if not q:
            slot[s] = None; return
        j = q.pop(0); i, a, _ = jobs[j]
        envs[s].reset(seed=0, options={"spawn": states[i]["pose"], "obstacles": states[i]["obstacles"]})
        slot[s] = {"job": j, "a": a, "t": 0, "G": np.zeros(len(g))}

    for s in range(n_par):
        start(s)
    while any(x is not None for x in slot):
        for s in range(n_par):
            st = slot[s]
            if st is None:
                continue
            a = st["a"]
            _, _, term, _, info = envs[s].step(int(a))
            c = float(info["ranges_m"].min() > d_min + A.v[a] * tau)
            st["G"] += (1 - g) * g ** st["t"] * c
            st["t"] += 1
            if term or st["t"] >= T_max:
                i, a0, _ = jobs[st["job"]]
                out[i, a0] += st["G"] / K
                start(s)
            else:
                st["a"] = int(rng.choice(A.n, p=P[a]))
    return out


def _reg(pred, target):
    pred, target = np.asarray(pred, float).ravel(), np.asarray(target, float).ravel()
    err = pred - target
    ss = float(np.sum((target - target.mean()) ** 2))
    return {"mae": float(np.mean(np.abs(err))), "rmse": float(np.sqrt(np.mean(err ** 2))),
            "r2": float(1 - np.sum(err ** 2) / ss) if ss > 1e-12 else float("nan")}


def safety_quality(Q_pred: np.ndarray, mc: np.ndarray, horizons) -> dict:
    """Q_pred, mc: (S, n, H)."""
    out = {}
    for h, g in enumerate(horizons):
        p, t = Q_pred[..., h].ravel(), mc[..., h].ravel()
        r = _reg(p, t)
        r["mean_baseline_mae"] = float(np.mean(np.abs(t - t.mean())))
        r["mean_baseline_rmse"] = float(np.sqrt(np.mean((t - t.mean()) ** 2)))
        bins = np.clip((p * 10).astype(int), 0, 9)
        r["calibration"] = [{"pred": float(p[bins == b].mean()), "mc": float(t[bins == b].mean()),
                             "n": int((bins == b).sum())} for b in range(10) if (bins == b).any()]
        out[f"gamma_{g}"] = r
    return out


def ttc_trace(src, A: ActionSpace, horizons, start=(-1.3, 0.0, 0.0)) -> dict:
    """Drive straight into the +x wall of the empty arena; compare ttc_from_safety
    (straight action) with true remaining steps to collision."""
    env = TurtleBot3LidarEnv(load_config({"termination": {"max_episode_steps": None}}))
    _, info = env.reset(seed=0, options={"spawn": start})
    a = int(A.grid[0, A.straight_turn])
    preds, steps = [], 0
    while True:
        Q = _q_all(src, info["ranges_m"][None])[0]                 # (H, n)
        preds.append([float(ttc_from_safety(Q[h, a], g)) for h, g in enumerate(horizons)])
        _, _, term, trunc, info = env.step(a); steps += 1
        if term or trunc or steps > 5000:
            break
    true = np.arange(steps, 0, -1, dtype=float)
    P = np.array(preds)
    res = {"true": true.tolist(), "pred": P.tolist(), "horizons": list(horizons)}
    for h, g in enumerate(horizons):
        e = P[:, h] - true
        res[f"gamma_{g}"] = {"mae": float(np.mean(np.abs(e))), "mae_last40": float(np.mean(np.abs(e[-40:])))}
    res["mean_baseline_mae"] = float(np.mean(np.abs(true - true.mean())))
    res["zero_baseline_mae"] = float(np.mean(true))
    res["mean_baseline_mae_last40"] = float(np.mean(np.abs(true[-40:] - true.mean())))
    res["zero_baseline_mae_last40"] = float(np.mean(true[-40:]))
    return res


def _q_all(src, ranges):
    """(B, H, n) safety predictions from a GVFModel or OriginalDemons."""
    if hasattr(src, "predict"):
        return src.predict(ranges)[0]
    ttc = np.clip(src.ttc_pred(ranges), 0, 1e6)
    return 1.0 - src.g[None, :, None] ** ttc[:, None, :]


def velocity_quality(model, A: ActionSpace, states, K: int = 8, seed: int = QUALITY_BASE + 999,
                     n_par: int = 64) -> dict:
    """MC ground truth for the velocity GVF under its target policy (softened controller)."""
    gv, eps = model.g_v, model.cfg.vel_eps
    T_max = int(math.ceil(math.log(1e-3) / math.log(gv)))
    R0 = np.array([s["ranges"] for s in states])
    prev0 = np.array([s["prev"] for s in states])
    Qm, Vl, Va = model.decision_inputs(R0)
    a0 = controller_batch(Qm, Vl, Va, prev0, A, model.steer, model.speed)
    _, V = model.predict(R0)
    ar = np.arange(len(states))
    pred = V[ar, :, a0]                                                    # (S, 3) normalised
    cfun = lambda a: np.array([A.v[a] / A.v_max, abs(A.w[a]) / A.w_max, A.w[a] / A.w_max])
    rng = np.random.default_rng(seed)
    envs = [_rollout_env(2) for _ in range(n_par)]
    jobs = [(i, k) for i in range(len(states)) for k in range(K)]
    q = list(range(len(jobs)))
    G = np.zeros((len(states), 3))
    slot = [None] * n_par

    def start(s):
        if not q:
            slot[s] = None; return
        j = q.pop(0); i, _ = jobs[j]
        envs[s].reset(seed=0, options={"spawn": states[i]["pose"], "obstacles": states[i]["obstacles"]})
        slot[s] = {"i": i, "a": int(a0[i]), "t": 0, "G": np.zeros(3)}

    for s in range(n_par):
        start(s)
    while any(x is not None for x in slot):
        live = [s for s in range(n_par) if slot[s] is not None]
        nxt = {}
        for s in live:
            st = slot[s]
            _, _, term, _, info = envs[s].step(st["a"])
            st["G"] += (1 - gv) * gv ** st["t"] * cfun(st["a"])
            st["t"] += 1
            if term or st["t"] >= T_max:
                G[st["i"]] += st["G"] / K
                start(s)
            else:
                nxt[s] = info["ranges_m"]
        if nxt:
            ss = list(nxt)
            Rb = np.array([nxt[s] for s in ss])
            prevb = np.array([slot[s]["a"] for s in ss])
            qm, vl, va = model.decision_inputs(Rb)
            ac = controller_batch(qm, vl, va, prevb, A, model.steer, model.speed)
            for s, a in zip(ss, ac):
                slot[s]["a"] = int(rng.integers(A.n)) if rng.random() < eps else int(a)
    persist = np.array([cfun(a) for a in a0])
    names = ("linear_velocity", "abs_angular_velocity", "signed_angular_velocity")
    out = {}
    for k, nm in enumerate(names):
        t = G[:, k]
        res = {"gvf": _reg(pred[:, k], t), "mean": _reg(np.full_like(t, t.mean()), t),
               "zero": _reg(np.zeros_like(t), t), "persistence": _reg(persist[:, k], t)}
        res["beats_persistence"] = bool(res["gvf"]["mae"] < res["persistence"]["mae"]
                                        and res["gvf"]["rmse"] < res["persistence"]["rmse"])
        res["scale"] = float(A.v_max if k == 0 else A.w_max)
        res["scatter"] = {"pred": pred[:, k].tolist(), "true": t.tolist(), "persistence": persist[:, k].tolist()}
        out[nm] = res
    # mirror consistency of the RAW (unsymmetrised) heads
    from .features import mirror_ranges
    _, Vr = model.predict(R0, symmetric=False)
    _, Vm = model.predict(mirror_ranges(R0), symmetric=False)
    ma = A.mirror[a0]
    out["mirror_consistency"] = {
        "signed_w_mean_abs_sum": float(np.mean(np.abs(Vr[ar, 2, a0] + Vm[ar, 2, ma]))),
        "signed_w_mean_abs": float(np.mean(np.abs(Vr[ar, 2, a0]))),
        "linear_mean_abs_diff": float(np.mean(np.abs(Vr[ar, 0, a0] - Vm[ar, 0, ma])))}
    return out
