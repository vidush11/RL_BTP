"""Experiment D: LiDAR-only DQN whose REWARD (not state) is shaped by GVFs that
learn concurrently from the same interaction stream.

Contract (enforced + tested):
  * DQN state = the env's 24-dim normalised LiDAR observation, nothing else
    (no goal features, no GVF outputs). Replay stores exactly that vector.
  * Per transition, in this order:
      1. env.step(a) -> s', terminated, truncated, info   (env flags untouched)
      2. Phi(s), Phi(s') from the CURRENT (pre-update) GVF weights
      3. r = r_task + lam * r_GVF                          (rl/prediction_reward.py)
      4. replay.add(s, a, r, s', terminated, truncated)   (BEFORE any reset;
         the stored reward is never recomputed later)
      5. GVF update on (s, a, s') with its own GQ(lambda) objective
      6. DQN update from replay (its own TD objective)
  * Episodes are strictly episodic: collision (geometric swept-disc contact or
    the LiDAR < 0.15 m rule) or goal -> terminated; step limit -> truncated.
    The env refuses to step after termination; the loop resets only after the
    final transition is stored and the episode is logged.

Reuses: simulator env, rl.agent.DQNAgent (mode "A", goal features OFF), rl.dqn,
rl.replay, rl.gvf_predictor (the Horde directional-TTC demons). Agents A/B/C
are untouched.
"""
from __future__ import annotations

import importlib.util
import json
import pickle
import time
from pathlib import Path
from typing import Optional

import numpy as np

from simulator import hpc
from .agent import DQNAgent
from .encoder import StateEncoder
from .experiment import DEFAULTS as _ABC_DEFAULTS, _dqn_cfg, epsilon_at, make_env
from .gvf_predictor import GVFPredictor, FORWARD
from .prediction_reward import PredictionReward, PredictionRewardConfig

EXPERIMENT_ID = "D_lidar_only_gvf_reward"

DEFAULTS_D = {
    "experiment_id": EXPERIMENT_ID,
    # Same env, task, budget, DQN and eval protocol as A/B/C by default.
    "sim_config": _ABC_DEFAULTS["sim_config"],
    "task": _ABC_DEFAULTS["task"],
    "total_env_steps": _ABC_DEFAULTS["total_env_steps"],
    "learning_starts": _ABC_DEFAULTS["learning_starts"],
    "train_freq": _ABC_DEFAULTS["train_freq"],
    "eval_interval": _ABC_DEFAULTS["eval_interval"],
    "eval_episodes": _ABC_DEFAULTS["eval_episodes"],
    "eval_seed": _ABC_DEFAULTS["eval_seed"],
    "epsilon_start": _ABC_DEFAULTS["epsilon_start"],
    "epsilon_end": _ABC_DEFAULTS["epsilon_end"],
    "epsilon_decay_steps": _ABC_DEFAULTS["epsilon_decay_steps"],
    "dqn": dict(_ABC_DEFAULTS["dqn"]),
    "replay_capacity": _ABC_DEFAULTS["replay_capacity"],
    "gvf": {"obstacle_threshold": 0.25, "alpha_theta": 0.1, "alpha_w": 1e-4,
            "trace_lambda": 0.4, "init_checkpoint": None,
            # DIAGNOSTIC ABLATION ONLY: true = stationary (pretrained) GVF, no
            # concurrent learning. The main experiment keeps this false.
            "frozen": False},
    "reward": PredictionRewardConfig().to_dict(),
    "gvf_eval_episodes": 10,
}


# --------------------------------------------------------------------------- #
# config
# --------------------------------------------------------------------------- #
def _merge(base: dict, user: dict) -> dict:
    out = json.loads(json.dumps(base))
    for k, v in (user or {}).items():
        if k not in out:
            raise ValueError(f"unknown experiment-D config key '{k}'")
        if isinstance(v, dict) and isinstance(out[k], dict):
            out[k].update(v)
        else:
            out[k] = v
    return out


def load_d_config(path: Optional[str] = None, overrides: Optional[dict] = None) -> dict:
    user = {}
    if path:
        import yaml
        user = yaml.safe_load(Path(path).read_text()) or {}
    cfg = _merge(DEFAULTS_D, user)
    if overrides:
        cfg = _merge(cfg, overrides)
    PredictionRewardConfig.from_dict(cfg["reward"])   # validate early
    return cfg


def parse_set(items) -> dict:
    """['reward.lam=0', 'total_env_steps=500'] -> nested dict (YAML-typed)."""
    import yaml
    out: dict = {}
    for item in items or []:
        key, _, val = item.partition("=")
        node = out
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = yaml.safe_load(val)
    return out


# --------------------------------------------------------------------------- #
# builders
# --------------------------------------------------------------------------- #
def build_d(exp: dict, seed: int):
    """LiDAR-only DQN agent + concurrent GVF learner + reward model."""
    encoder = StateEncoder(use_goal=False)            # 24 LiDAR values only
    agent = DQNAgent("A", n_actions=5, encoder=encoder, dqn_cfg=_dqn_cfg(exp),
                     predictor=None, replay_capacity=exp["replay_capacity"], seed=seed)
    g = exp["gvf"]
    gvf = GVFPredictor(frozen=bool(g.get("frozen", False)),
                       obstacle_threshold=g["obstacle_threshold"],
                       alpha_theta=g["alpha_theta"], alpha_w=g["alpha_w"],
                       trace_lambda=g["trace_lambda"])
    if g.get("init_checkpoint"):
        gvf.load_checkpoint(g["init_checkpoint"])
    rm = PredictionReward(PredictionRewardConfig.from_dict(exp["reward"]),
                          dqn_gamma=exp["dqn"]["gamma"])
    return agent, gvf, rm


# --------------------------------------------------------------------------- #
# one transition (the unit the tests pin down)
# --------------------------------------------------------------------------- #
def process_transition(agent: DQNAgent, gvf: GVFPredictor, rm: PredictionReward,
                       base, action, next_base, info, terminated, truncated,
                       behaviour_prob):
    """Steps 2-5 of the module contract. Returns (RewardBreakdown, td_errors,
    raw TTC predictions at s)."""
    collision = bool(info.get("collision", False))
    success = bool(info.get("goal_reached", False))
    pred_s = gvf.raw_predict(base)            # pre-update weights ...
    pred_sp = gvf.raw_predict(next_base)      # ... same weights for s'
    rb = rm.compute(rm.potential(pred_s), rm.potential(pred_sp),
                    collision=collision, success=success,
                    terminated=bool(terminated), truncated=bool(truncated))
    agent.observe(base, action, rb.total, next_base, terminated, truncated)
    # GVF termination = real termination only (a time-limit cut is not the end
    # of the predicted future); the episode's traces are reset at the next start.
    deltas = gvf.update(base, action, next_base, bool(terminated), behaviour_prob)
    return rb, deltas, pred_s


# --------------------------------------------------------------------------- #
# evaluation
# --------------------------------------------------------------------------- #
def _evaluate_module():
    spec = importlib.util.spec_from_file_location(
        "tb3_evaluate_d", Path(__file__).resolve().parent.parent / "scripts" / "evaluate.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def evaluate_d(agent: DQNAgent, gvf: GVFPredictor, rm: PredictionReward, env,
               n_episodes: int, seed: int) -> dict:
    """Greedy evaluation on fixed seeds; GVF read-only (never updated here).
    Primary outputs are TASK outcomes; shaped return is reported separately."""
    rows = []
    for ep in range(n_episodes):
        obs, info = env.reset(seed=seed + ep)
        base = agent.encoder.base_obs(obs, info)
        task_ret = shaped_ret = 0.0
        length = 0
        while True:
            a = agent.act(base, epsilon=0.0)
            nobs, _r, terminated, truncated, info = env.step(a)
            nbase = agent.encoder.base_obs(nobs, info)
            rb = rm.compute(rm.potential(gvf.raw_predict(base)),
                            rm.potential(gvf.raw_predict(nbase)),
                            collision=bool(info.get("collision")),
                            success=bool(info.get("goal_reached")),
                            terminated=terminated, truncated=truncated)
            task_ret += rb.task; shaped_ret += rb.total; length += 1
            base = nbase
            if terminated or truncated:
                break
        coll = bool(info.get("collision"))
        succ = bool(info.get("goal_reached")) and not coll
        rows.append({"success": float(succ), "collision": float(coll),
                     "timeout": float(truncated and not terminated),
                     "length": float(length), "task_return": task_ret,
                     "shaped_return": shaped_ret,
                     "final_goal_distance": float(info.get("goal_distance", np.nan))})
    agg = {f"{k}_mean": float(np.nanmean([r[k] for r in rows])) for k in rows[0]}
    agg["success_rate"] = agg.pop("success_mean")
    agg["collision_rate"] = agg.pop("collision_mean")
    agg["timeout_rate"] = agg.pop("timeout_mean")
    agg["n_episodes"] = n_episodes
    return agg


def gvf_forward_error(gvf: GVFPredictor, exp: dict, n_episodes: int, seed: int) -> dict:
    """Forward-TTC prediction error against independent ground truth (episodes
    that end in a real collision under the forward probe policy; others are
    censored and excluded). Note the ~7-step definitional offset between the
    GVF's 0.25 m cutoff and the collision-time target (DIAGNOSIS_GVF.md)."""
    if n_episodes <= 0:
        return {}
    ev = _evaluate_module()
    env = make_env(exp)
    env.reset(seed=seed)
    r = ev.ttc_ground_truth(env, gvf.demons["forward"].theta, FORWARD, n_episodes)
    return {"gvf_fwd_mae": r["mae"], "gvf_fwd_mean_baseline_mae": r["baseline_mean_mae"],
            "gvf_fwd_collided": r["collided_episodes"], "gvf_fwd_censored": r["censored_episodes"]}


# --------------------------------------------------------------------------- #
# training
# --------------------------------------------------------------------------- #
def _truncate_episode_log(path: Path, max_step: int) -> None:
    """On resume, drop episodes logged after the checkpoint (they will be re-run)."""
    if not path.exists():
        return
    keep = [l for l in path.read_text().splitlines()
            if l.strip() and json.loads(l)["env_step"] <= max_step]
    path.write_text("".join(l + "\n" for l in keep))


def train_d(exp: dict, seed: int, out_dir: str, resume: bool = True, log=print) -> dict:
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    env = make_env(exp)
    eval_env = make_env(exp)
    agent, gvf, rm = build_d(exp, seed)
    assert agent.in_dim == env.observation_space.shape[0] == agent.encoder.base_dim, \
        "Experiment D's DQN must see the raw LiDAR observation only"

    ckpt = out / "agent.pkl"
    ep_log = out / "episodes.jsonl"
    step = episode = 0
    history = []
    if resume and ckpt.exists():
        meta = agent.load(ckpt, with_replay=True)
        with open(ckpt, "rb") as fh:
            extra = pickle.load(fh).get("extra", {})
        gvf.load_state(extra["gvf"])
        step = int(meta["env_steps"]); episode = int(extra.get("episode_index", 0))
        if (out / "history.json").exists():
            history = json.loads((out / "history.json").read_text())
        _truncate_episode_log(ep_log, step)
        log(f"[{exp['experiment_id']} s{seed}] resumed at env_step {step}, episode {episode}")
    elif ep_log.exists():
        ep_log.unlink()                       # fresh run: no stale episodes

    rng = np.random.default_rng(seed + 7 + step)
    total, bs = exp["total_env_steps"], exp["dqn"]["batch_size"]
    t0 = time.perf_counter()
    obs, info = env.reset(seed=seed if step == 0 else seed * 7919 + step)
    base = agent.encoder.base_obs(obs, info)
    gvf.reset_traces()

    def _new_acc():
        return {"len": 0, "task": 0.0, "gvf": 0.0, "total": 0.0, "phi": 0.0,
                "ttc": np.zeros(3), "absdelta": np.zeros(3), "n_delta": 0, "losses": []}
    acc = _new_acc()

    with open(ep_log, "a") as logfh:
        while step < total:
            eps = epsilon_at(step, exp)
            if step < exp["learning_starts"]:
                a, bprob = int(rng.integers(0, 5)), 1.0 / 5
            else:
                a = agent.act(base, eps); bprob = agent._last_bprob
            nobs, _env_r, terminated, truncated, ninfo = env.step(a)
            nbase = agent.encoder.base_obs(nobs, ninfo)
            rb, deltas, pred_s = process_transition(agent, gvf, rm, base, a, nbase,
                                                    ninfo, terminated, truncated, bprob)
            step += 1
            acc["len"] += 1; acc["task"] += rb.task; acc["gvf"] += rb.gvf
            acc["total"] += rb.total; acc["phi"] += rb.phi_s
            acc["ttc"] += np.nan_to_num(pred_s)
            if deltas is not None:
                acc["absdelta"] += np.abs(deltas); acc["n_delta"] += 1
            if step >= exp["learning_starts"] and step % exp["train_freq"] == 0:
                loss = agent.learn(bs)
                if loss is not None:
                    acc["losses"].append(loss)

            if terminated or truncated:
                coll = bool(ninfo.get("collision"))
                n = max(acc["len"], 1)
                rec = {"experiment_id": exp["experiment_id"], "seed": seed,
                       "episode": episode, "env_step": step, "length": acc["len"],
                       "return_total": acc["total"], "return_task": acc["task"],
                       "gvf_reward_sum": acc["gvf"],
                       "gvf_reward_weighted": exp["reward"]["lam"] * acc["gvf"],
                       "success": bool(ninfo.get("goal_reached")) and not coll,
                       "collision": coll, "collision_type": ninfo.get("collision_type"),
                       "timeout": bool(truncated and not terminated),
                       "final_goal_distance": ninfo.get("goal_distance"),
                       "phi_mean": acc["phi"] / n,
                       "ttc_pred_mean": (acc["ttc"] / n).tolist(),
                       "gvf_abs_td_mean": (acc["absdelta"] / max(acc["n_delta"], 1)).tolist(),
                       "dqn_loss_mean": float(np.mean(acc["losses"])) if acc["losses"] else None,
                       "epsilon": eps, "elapsed_s": round(time.perf_counter() - t0, 2)}
                logfh.write(json.dumps(rec) + "\n"); logfh.flush()
                episode += 1
                obs, info = env.reset()
                base = agent.encoder.base_obs(obs, info)
                gvf.reset_traces()
                acc = _new_acc()
            else:
                base = nbase

            if step % exp["eval_interval"] == 0 or step == total:
                ev = evaluate_d(agent, gvf, rm, eval_env, exp["eval_episodes"], exp["eval_seed"])
                ev.update(gvf_forward_error(gvf, exp, exp["gvf_eval_episodes"],
                                            exp["eval_seed"] + 50000))
                ev.update({"env_step": step, "episode": episode, "epsilon": eps,
                           "gvf_theta_max": float(np.abs(gvf.theta_snapshot()).max()),
                           "gvf_nonfinite": gvf.nonfinite_count,
                           "reward_nonfinite": rm.nonfinite_count,
                           "wall_time_s": round(time.perf_counter() - t0, 1)})
                history.append(ev)
                agent.save(ckpt, episode=episode, env_steps=step,
                           extra={"gvf": gvf.get_state(), "episode_index": episode,
                                  "experiment_id": exp["experiment_id"], "seed": seed,
                                  "config": exp})
                st = gvf.get_state()
                hpc.atomic_savez(out / "gvf_latest.npz", episode=np.int64(episode),
                                 **{k: st[k] for k in ("theta_forward", "w_forward",
                                                       "theta_left", "w_left",
                                                       "theta_right", "w_right")})
                hpc.write_json_atomic(out / "history.json", history)
                log(f"[{exp['experiment_id']} s{seed}] step {step}/{total} eps={eps:.2f} "
                    f"success={ev['success_rate']:.2f} collision={ev['collision_rate']:.2f} "
                    f"timeout={ev['timeout_rate']:.2f} task_ret={ev['task_return_mean']:.1f} "
                    f"shaped_ret={ev['shaped_return_mean']:.1f}")

    summary = {"experiment_id": exp["experiment_id"], "seed": seed, "env_steps": step,
               "episodes": episode, "wall_time_s": round(time.perf_counter() - t0, 1),
               "in_dim": agent.in_dim, "final": history[-1] if history else None}
    hpc.write_json_atomic(out / "summary.json", summary)
    return summary


def load_d_checkpoint(path: str, exp: Optional[dict] = None, seed: int = 0):
    """Rebuild a trained D agent (+ its GVF) from agent.pkl for evaluation."""
    with open(path, "rb") as fh:
        state = pickle.load(fh)
    exp = exp or state["extra"]["config"]
    agent, gvf, rm = build_d(exp, seed)
    agent.load(path, with_replay=False)
    gvf.load_state(state["extra"]["gvf"])
    return agent, gvf, rm, exp
