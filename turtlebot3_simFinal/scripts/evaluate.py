#!/usr/bin/env python3
"""Evaluate saved Horde checkpoints with INDEPENDENT ground truth.

Reads training checkpoints read-only and writes results to a SEPARATE
``--out-dir`` -- it never touches the training run's files.  For every
checkpoint it reports, per demon, metrics computed against a ground truth the
simulator actually knows, instead of the training agent's
``np.arange(steps,0,-1)`` target (which silently treats a step-limit
*truncation* as a collision countdown -- see the Phase-7 notes in the README).

Metrics per checkpoint (written to ``metrics.json`` + printed):
  * TTC_forward / TTC_left / TTC_right: mean absolute error of the demon's
    prediction against the TRUE remaining-steps-to-collision, measured by
    rolling out that demon's deterministic target policy.  Episodes that end by
    the step limit (no real collision) are reported as **censored** and excluded
    from the TTC error, not scored as if they were collisions.  Zero- and
    mean-predictor baselines are reported on the same episodes.
  * Future linear/angular velocity: RMSE/MAE of the prediction against the
    velocity ACTUALLY measured ``horizon`` steps later, under the demons' shared
    epsilon-greedy-forward target policy.  Mean-predictor baseline included.
  * A behavior-policy importance-sampling diagnostic (pure arithmetic): the
    distribution of the velocity demons' 19-step rho product, showing why those
    weights barely move.

This drives the Gymnasium env directly (no ROS shim), so it is fully
deterministic given ``--seed`` and needs no agent file.  ``--with-agent-eval``
additionally runs your UNMODIFIED gvf_horde_agent_evaluate.py via run_agent.py
into the same out-dir for comparison.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from simulator import TurtleBot3LidarEnv, load_config, hpc  # noqa: E402

# Action-index convention shared with the agents.
FORWARD_ACTION, LEFT_ACTION, RIGHT_ACTION = 2, 1, 3
LIDAR_SIZE, ACTION_SIZE = 24, 5
DIRECTIONS = {"ttc_forward": (FORWARD_ACTION, "theta_forward"),
              "ttc_left": (LEFT_ACTION, "theta_left"),
              "ttc_right": (RIGHT_ACTION, "theta_right")}


def feature_vector(ranges_m: np.ndarray, action: int) -> np.ndarray:
    """phi(s, a): normalised lidar placed in the action's block (exactly the
    agents' get_feature_vector)."""
    phi = np.zeros((ACTION_SIZE, LIDAR_SIZE), dtype=np.float64)
    phi[action, :] = np.clip(np.asarray(ranges_m, dtype=np.float64) / 3.5, 0.0, 1.0)
    return phi.ravel()


def _mae(pred, targ):
    pred, targ = np.asarray(pred, float), np.asarray(targ, float)
    return float(np.mean(np.abs(pred - targ))) if pred.size else float("nan")


def ttc_ground_truth(env: TurtleBot3LidarEnv, theta: np.ndarray, action: int,
                     num_episodes: int):
    """Roll out the deterministic `action` policy; for episodes that end in a
    REAL collision, collect (prediction, true_remaining_steps) pairs.  Returns a
    dict with MAE over collided episodes and censored (truncated) count."""
    preds, targs = [], []
    collided = censored = 0
    for _ in range(num_episodes):
        obs, info = env.reset()
        ep_preds = []
        steps = 0
        terminated = truncated = False
        while True:
            phi = feature_vector(info["ranges_m"], action)
            p = float(np.dot(theta, phi))
            ep_preds.append(p if np.isfinite(p) else 0.0)
            steps += 1
            obs, r, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break
        if terminated and info["collision"]:
            collided += 1
            true_ttc = np.arange(steps, 0, -1, dtype=np.float64)  # valid: real collision
            preds.extend(ep_preds)
            targs.extend(true_ttc.tolist())
        else:
            censored += 1  # step-limit truncation -> NOT a collision; excluded
    targs_arr = np.asarray(targs, float)
    baseline_zero = _mae(np.zeros_like(targs_arr), targs_arr) if targs_arr.size else float("nan")
    baseline_mean = (_mae(np.full_like(targs_arr, targs_arr.mean()), targs_arr)
                     if targs_arr.size else float("nan"))
    return {
        "mae": _mae(preds, targs),
        "baseline_zero_mae": baseline_zero,
        "baseline_mean_mae": baseline_mean,
        "collided_episodes": collided,
        "censored_episodes": censored,
        "n_pairs": int(targs_arr.size),
    }


def velocity_ground_truth(env: TurtleBot3LidarEnv, w_lin: np.ndarray,
                          w_ang: np.ndarray, horizon: int, num_episodes: int,
                          eps: float = 0.05):
    """Epsilon-greedy-forward rollout; compare each demon's prediction at phi_t
    to the velocity ACTUALLY measured `horizon` steps later."""
    lin_pred, lin_true, ang_pred, ang_true = [], [], [], []
    for _ in range(num_episodes):
        obs, info = env.reset()
        pending = []  # (phi_t,)
        while True:
            a = (int(env.np_random.integers(0, ACTION_SIZE))
                 if env.np_random.random() < eps else FORWARD_ACTION)
            phi = feature_vector(info["ranges_m"], a)
            pending.append(phi)
            obs, r, terminated, truncated, info = env.step(a)
            # the velocity realised after this step:
            lin_now, ang_now = info["linear_velocity"], info["angular_velocity"]
            if len(pending) > horizon:
                old = pending.pop(0)
                lin_pred.append(float(np.dot(w_lin, old))); lin_true.append(lin_now)
                ang_pred.append(float(np.dot(w_ang, old))); ang_true.append(ang_now)
            if terminated or truncated:
                break

    def stats(pred, true):
        pred, true = np.asarray(pred, float), np.asarray(true, float)
        if not pred.size:
            return {"rmse": float("nan"), "mae": float("nan"),
                    "baseline_mean_rmse": float("nan"), "n": 0}
        rmse = float(np.sqrt(np.mean((pred - true) ** 2)))
        base = float(np.sqrt(np.mean((true - true.mean()) ** 2)))
        return {"rmse": rmse, "mae": _mae(pred, true),
                "baseline_mean_rmse": base, "n": int(pred.size)}

    return {"future_linear_velocity": stats(lin_pred, lin_true),
            "future_angular_velocity": stats(ang_pred, ang_true)}


def rho_product_diagnostic(horizon: int = 20, eps: float = 0.05,
                           clip: float = 50.0, trials: int = 200_000,
                           seed: int = 0):
    """Monte-Carlo distribution of the velocity demons' (horizon-1)-step
    importance-weight product under uniform-random behavior.  Explains why the
    velocity-demon weights stay near zero."""
    rng = np.random.default_rng(seed)
    mu = 1.0 / ACTION_SIZE
    pi_fwd = (1.0 - eps) + eps / ACTION_SIZE
    pi_other = eps / ACTION_SIZE
    n = horizon - 1
    acts = rng.integers(0, ACTION_SIZE, size=(trials, n))
    rho = np.where(acts == FORWARD_ACTION, pi_fwd / mu, pi_other / mu)
    prod = np.clip(rho.prod(axis=1), 0.0, clip)
    return {"mean": float(prod.mean()), "median": float(np.median(prod)),
            "frac_gt_1e-3": float(np.mean(prod > 1e-3)),
            "frac_eq_0_to_1e-6": float(np.mean(prod < 1e-6)),
            "max": float(prod.max()), "clip": clip, "horizon": horizon}


def evaluate_checkpoint(path: Path, cfg, num_episodes: int, horizon: int,
                        seed: int) -> dict:
    data = np.load(path)
    out = {"checkpoint": str(path), "episode": int(data["episode"])}
    # one env, re-seeded per section for determinism & comparability
    for name, (action, key) in DIRECTIONS.items():
        env = TurtleBot3LidarEnv(cfg)
        env.reset(seed=seed)
        out[name] = ttc_ground_truth(env, data[key], action, num_episodes)
    env = TurtleBot3LidarEnv(cfg)
    env.reset(seed=seed + 1)
    out.update(velocity_ground_truth(
        env, data["w_future_linear_velocity"], data["w_future_angular_velocity"],
        horizon, num_episodes))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint-dir", required=True,
                    help="directory of gvf_weights_ep_*.npz from a training run")
    ap.add_argument("--out-dir", default=None,
                    help="where to write metrics.json (default: <checkpoint-dir>/../eval)")
    ap.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--num-eval-episodes", type=int, default=20)
    ap.add_argument("--horizon", type=int, default=20)
    ap.add_argument("--only-latest", action="store_true",
                    help="evaluate only the newest checkpoint")
    ap.add_argument("--with-agent-eval", action="store_true",
                    help="also run the unmodified gvf_horde_agent_evaluate.py for comparison")
    args = ap.parse_args()

    ckpt_dir = Path(args.checkpoint_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else ckpt_dir.parent / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = load_config(Path(args.config).expanduser().resolve())

    checkpoints = hpc.scan_checkpoints(ckpt_dir)
    if not checkpoints:
        print(f"[evaluate] no valid checkpoints in {ckpt_dir}", file=sys.stderr)
        return 1
    if args.only_latest:
        checkpoints = [checkpoints[-1]]
    print(f"[evaluate] {len(checkpoints)} checkpoint(s) from {ckpt_dir}")
    print(f"[evaluate] independent ground truth, config={Path(args.config).name}, "
          f"seed={args.seed}, episodes/checkpoint={args.num_eval_episodes}")

    t0 = time.perf_counter()
    results = []
    for ep, path in checkpoints:
        r = evaluate_checkpoint(path, cfg, args.num_eval_episodes, args.horizon, args.seed)
        results.append(r)
        f = r["ttc_forward"]
        v = r["future_linear_velocity"]
        print(f"  ep {ep:>6}: TTC_fwd MAE {f['mae']:.2f} "
              f"(zero {f['baseline_zero_mae']:.2f}, mean {f['baseline_mean_mae']:.2f}; "
              f"{f['collided_episodes']} collided / {f['censored_episodes']} censored) | "
              f"FutLinVel RMSE {v['rmse']:.3f} (mean-base {v['baseline_mean_rmse']:.3f})")

    diag = rho_product_diagnostic(horizon=args.horizon)
    payload = {
        "checkpoint_dir": str(ckpt_dir),
        "config": Path(args.config).name,
        "seed": args.seed,
        "num_eval_episodes": args.num_eval_episodes,
        "horizon": args.horizon,
        "elapsed_s": round(time.perf_counter() - t0, 1),
        "rho_product_diagnostic": diag,
        "checkpoints": results,
    }
    out_path = out_dir / "metrics.json"
    hpc.write_json_atomic(out_path, payload)
    print(f"\n[evaluate] rho-product (19-step, uniform behavior): mean={diag['mean']:.3f} "
          f"median={diag['median']:.2e} frac>1e-3={diag['frac_gt_1e-3']*100:.3f}%  "
          f"-> velocity demons barely update")
    print(f"[evaluate] wrote {out_path}")

    if args.with_agent_eval:
        eval_agent = ROOT / "gvf_horde_agent_evaluate.py"
        print(f"\n[evaluate] also running unmodified {eval_agent.name} for comparison")
        subprocess.run([sys.executable, str(ROOT / "scripts" / "run_agent.py"),
                        str(eval_agent), "--config", args.config, "--seed", str(args.seed),
                        "--run-dir", str(out_dir / "agent_eval"),
                        "--param", f"checkpoint_dir:={ckpt_dir}",
                        "--param", f"num_eval_episodes:={args.num_eval_episodes}"],
                       check=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
