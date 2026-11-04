#!/usr/bin/env python3
"""Controlled single-demon experiments that REUSE the agent's exact demon
classes (DirectionalTTCGVF, FixedHorizonGVF from gvf_horde_agent.py) without
editing any original file. Lets us change ONE hyperparameter at a time and
measure learned vs mean-baseline vs LS-capacity, to back the diagnosis with
evidence before recommending any change to the agent.

    python scripts/gvf_experiment.py forward_ttc --episodes 300 --alpha-theta 0.05
    python scripts/gvf_experiment.py linear_velocity --episodes 300 --no-importance-sampling

Deterministic given --seed. Writes nothing unless --out is given.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from simulator import TurtleBot3LidarEnv, load_config
from simulator.ros_compat import install, uninstall

FORWARD, LEFT, RIGHT = 2, 1, 3
LID, ACT = 24, 5
FEAT = LID * ACT


def _import_demons(cfg):
    """Install the ROS shim (so gvf_horde_agent's top-level imports resolve),
    then import its demon classes. Returns (DirectionalTTCGVF, FixedHorizonGVF)."""
    env = TurtleBot3LidarEnv(cfg)
    install(env, seed=0)
    import importlib
    mod = importlib.import_module("gvf_horde_agent")
    return mod.DirectionalTTCGVF, mod.FixedHorizonGVF


def phi(ranges, a):
    f = np.zeros((ACT, LID)); f[a] = np.clip(np.asarray(ranges) / 3.5, 0, 1)
    return f.ravel()


def ls_capacity_forward(cfg, seed, n_eps):
    """Best achievable MAE for forward steps-to-termination with these features."""
    env = TurtleBot3LidarEnv(cfg); env.reset(seed=seed)
    X, y = [], []
    for _ in range(n_eps):
        obs, info = env.reset(); feats = []
        step = 0
        while True:
            feats.append(phi(info["ranges_m"], FORWARD)); step += 1
            obs, r, term, trunc, info = env.step(FORWARD)
            if term or trunc:
                break
        if term and info["collision"]:
            X.extend(feats); y.extend(range(step, 0, -1))
    X, y = np.asarray(X), np.asarray(y, float)
    theta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return float(np.mean(np.abs(X @ theta - y)))


def eval_forward_ttc(theta, cfg, seed, n_eps):
    env = TurtleBot3LidarEnv(cfg); env.reset(seed=seed)
    preds, targs = [], []
    collided = 0
    for _ in range(n_eps):
        obs, info = env.reset(); ep = []; step = 0
        while True:
            ep.append(float(np.dot(theta, phi(info["ranges_m"], FORWARD)))); step += 1
            obs, r, term, trunc, info = env.step(FORWARD)
            if term or trunc:
                break
        if term and info["collision"]:
            collided += 1
            preds.extend(ep); targs.extend(range(step, 0, -1))
    targs = np.asarray(targs, float); preds = np.asarray(preds)
    mae = float(np.mean(np.abs(preds - targs)))
    base = float(np.mean(np.abs(targs - targs.mean())))
    return {"mae": mae, "mean_baseline_mae": base, "collided": collided,
            "max_abs_theta": float(np.abs(theta).max())}


def run_forward_ttc(args):
    cfg = load_config(args.config)
    Directional, _ = _import_demons(cfg)
    demon = Directional(FEAT, ACT, FORWARD, slice(0, 24),
                        args.alpha_theta, args.alpha_w, args.trace_lambda,
                        0.25, "ttc_forward")
    env = TurtleBot3LidarEnv(cfg); env.reset(seed=args.seed)
    rng = np.random.default_rng(args.seed)
    for _ in range(args.episodes):
        obs, info = env.reset(); demon.reset_trace()
        state = info["ranges_m"]
        while True:
            a = int(rng.integers(0, ACT)); b = 1.0 / ACT
            phi_t = phi(state, a); lidar_t = np.asarray(state)
            obs, r, term, trunc, info = env.step(a)
            nxt = info["ranges_m"]; done = term or trunc
            phi_tn = phi(nxt, FORWARD)
            demon.update(phi_t, phi_tn, lidar_t, np.asarray(nxt), a, b, done)
            state = nxt
            if done:
                break
    res = eval_forward_ttc(demon.theta, cfg, args.eval_seed, args.eval_episodes)
    res["capacity_ls_mae"] = ls_capacity_forward(cfg, args.eval_seed, args.eval_episodes)
    return res


def run_linear_velocity(args):
    cfg = load_config(args.config)
    _, FixedHorizon = _import_demons(cfg)
    demon = FixedHorizon(FEAT, args.horizon, args.alpha_w_horizon, args.rho_clip)
    env = TurtleBot3LidarEnv(cfg); env.reset(seed=args.seed)
    rng = np.random.default_rng(args.seed)
    eps = 0.05
    def pi_prob(a):
        return (1 - eps) + eps / ACT if a == FORWARD else eps / ACT
    for _ in range(args.episodes):
        obs, info = env.reset(); demon.reset_window()
        state = info["ranges_m"]
        while True:
            a = int(rng.integers(0, ACT)); b = 1.0 / ACT
            rho = 1.0 if args.no_importance_sampling else pi_prob(a) / b
            phi_t = phi(state, a)
            obs, r, term, trunc, info = env.step(a)
            demon.push_and_maybe_update(phi_t, rho, info["linear_velocity"])
            state = info["ranges_m"]
            if term or trunc:
                break
    # evaluate: predict vs true linear velocity H steps later (eps-greedy-forward)
    env = TurtleBot3LidarEnv(cfg); env.reset(seed=args.eval_seed)
    preds, trues = [], []
    for _ in range(args.eval_episodes):
        obs, info = env.reset(); pend = []
        while True:
            a = int(env.np_random.integers(0, ACT)) if env.np_random.random() < eps else FORWARD
            pend.append(phi(info["ranges_m"], a))
            obs, r, term, trunc, info = env.step(a)
            if len(pend) > args.horizon:
                preds.append(float(np.dot(demon.w, pend.pop(0)))); trues.append(info["linear_velocity"])
            if term or trunc:
                break
    preds, trues = np.asarray(preds), np.asarray(trues)
    rmse = float(np.sqrt(np.mean((preds - trues) ** 2)))
    base = float(np.sqrt(np.mean((trues - trues.mean()) ** 2)))
    return {"rmse": rmse, "mean_baseline_rmse": base, "max_abs_w": float(np.abs(demon.w).max()),
            "mean_prediction": float(preds.mean()), "target_mean": float(trues.mean())}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("demon", choices=["forward_ttc", "linear_velocity"])
    ap.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    ap.add_argument("--episodes", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval-seed", type=int, default=42)
    ap.add_argument("--eval-episodes", type=int, default=20)
    ap.add_argument("--alpha-theta", type=float, default=0.005)
    ap.add_argument("--alpha-w", type=float, default=0.0001)
    ap.add_argument("--trace-lambda", type=float, default=0.4)
    ap.add_argument("--alpha-w-horizon", type=float, default=0.001)
    ap.add_argument("--rho-clip", type=float, default=50.0)
    ap.add_argument("--horizon", type=int, default=20)
    ap.add_argument("--no-importance-sampling", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    try:
        res = run_forward_ttc(args) if args.demon == "forward_ttc" else run_linear_velocity(args)
    finally:
        uninstall()
    res["demon"] = args.demon
    res["hyperparams"] = {k: getattr(args, k) for k in
                          ("episodes", "alpha_theta", "alpha_w", "trace_lambda",
                           "alpha_w_horizon", "rho_clip", "horizon",
                           "no_importance_sampling", "seed")}
    print(json.dumps(res, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
