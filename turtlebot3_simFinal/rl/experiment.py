"""Shared DQN training + evaluation used by all three agents (A/B/C).

One env, one DQN implementation, one reward/termination path -> no duplicated
logic across agents. The only per-agent difference is the input dimension and
(for B/C) the GVF features; everything else is identical, which is what makes
the comparison fair.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

import numpy as np

from simulator import TurtleBot3LidarEnv, load_config
from .agent import DQNAgent
from .dqn import DQNConfig
from .encoder import StateEncoder
from .gvf_predictor import GVFPredictor
from . import metrics as M

DEFAULTS = {
    "sim_config": "configs/obstacles_train.yaml",
    "task": "goal_navigation",          # or "obstacle_avoidance"
    "total_env_steps": 60000,
    "learning_starts": 1000,
    "train_freq": 1,
    "eval_interval": 10000,
    "eval_episodes": 50,
    "eval_seed": 10000,
    "epsilon_start": 1.0,
    "epsilon_end": 0.05,
    "epsilon_decay_steps": 30000,
    "dqn": {"hidden": [64, 64], "gamma": 0.99, "lr": 1e-3, "batch_size": 64,
            "target_update": 500, "huber_delta": 1.0, "grad_clip": 10.0},
    "replay_capacity": 100000,
    "gvf": {"obstacle_threshold": 0.25, "alpha_theta": 0.1, "alpha_w": 1e-4,
            "trace_lambda": 0.4, "clip_z": 5.0},
}


def load_exp_config(path: Optional[str]) -> dict:
    cfg = json.loads(json.dumps(DEFAULTS))  # deep copy
    if path:
        import yaml
        user = yaml.safe_load(Path(path).read_text()) or {}
        for k, v in user.items():
            if isinstance(v, dict) and isinstance(cfg.get(k), dict):
                cfg[k].update(v)
            else:
                cfg[k] = v
    return cfg


def make_env(exp_cfg: dict) -> TurtleBot3LidarEnv:
    path = Path(exp_cfg["sim_config"])
    if not path.is_absolute() and not path.exists():   # allow running from any cwd
        path = Path(__file__).resolve().parent.parent / path
    sim = load_config(path)
    sim.task.mode = exp_cfg["task"]
    return TurtleBot3LidarEnv(sim)


def _dqn_cfg(exp_cfg: dict) -> DQNConfig:
    d = exp_cfg["dqn"]
    return DQNConfig(hidden=tuple(d["hidden"]), gamma=d["gamma"], lr=d["lr"],
                     batch_size=d["batch_size"], target_update=d["target_update"],
                     huber_delta=d["huber_delta"], grad_clip=d["grad_clip"])


def build_predictor(mode: str, exp_cfg: dict, gvf_ckpt: Optional[str]) -> Optional[GVFPredictor]:
    if mode == "A":
        return None
    g = exp_cfg["gvf"]
    pred = GVFPredictor(frozen=(mode == "B"), obstacle_threshold=g["obstacle_threshold"],
                        alpha_theta=g["alpha_theta"], alpha_w=g["alpha_w"],
                        trace_lambda=g["trace_lambda"], clip_z=g["clip_z"])
    if gvf_ckpt:
        data = np.load(gvf_ckpt)
        pred.load_checkpoint(gvf_ckpt)
        if "feat_mean" in data.files:
            pred.set_normalization(data["feat_mean"], data["feat_std"])
    return pred


def build_agent(mode: str, exp_cfg: dict, gvf_ckpt: Optional[str], seed: int) -> DQNAgent:
    encoder = StateEncoder(use_goal=(exp_cfg["task"] == "goal_navigation"))
    predictor = build_predictor(mode, exp_cfg, gvf_ckpt)
    return DQNAgent(mode, n_actions=5, encoder=encoder, dqn_cfg=_dqn_cfg(exp_cfg),
                    predictor=predictor, replay_capacity=exp_cfg["replay_capacity"], seed=seed)


def epsilon_at(step: int, exp_cfg: dict) -> float:
    s0, s1 = exp_cfg["epsilon_start"], exp_cfg["epsilon_end"]
    frac = min(1.0, step / max(1, exp_cfg["epsilon_decay_steps"]))
    return s0 + frac * (s1 - s0)


def evaluate(agent: DQNAgent, env: TurtleBot3LidarEnv, n_episodes: int, seed: int):
    """Greedy (epsilon=0) evaluation on fixed seeds. GVF predictors are used
    read-only (never updated during evaluation)."""
    outcomes = []
    for ep in range(n_episodes):
        obs, info = env.reset(seed=seed + ep)
        base = agent.encoder.base_obs(obs, info)
        rsum, length, truncated = 0.0, 0, False
        last_info = info
        while True:
            a = agent.act(base, epsilon=0.0)
            obs, r, terminated, truncated, info = env.step(a)
            base = agent.encoder.base_obs(obs, info)
            rsum += r; length += 1; last_info = info
            if terminated or truncated:
                break
        outcomes.append(M.episode_outcome(rsum, length, last_info, truncated))
    return M.aggregate_episodes(outcomes), outcomes


def train(mode: str, exp_cfg: dict, seed: int, out_dir: str,
          gvf_ckpt: Optional[str] = None, resume: bool = True) -> dict:
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    env = make_env(exp_cfg)
    eval_env = make_env(exp_cfg)
    agent = build_agent(mode, exp_cfg, gvf_ckpt, seed)

    ckpt_path = out / "agent.pkl"
    start_step = 0
    history = []
    if resume and ckpt_path.exists():
        meta = agent.load(ckpt_path, with_replay=True)
        start_step = int(meta["env_steps"])
        hp = out / "history.json"
        if hp.exists():
            history = json.loads(hp.read_text())
        print(f"[train {mode}] resumed at env_step {start_step}")

    rng = np.random.default_rng(seed + 7)
    total = exp_cfg["total_env_steps"]
    bs = exp_cfg["dqn"]["batch_size"]
    t0 = time.perf_counter()
    gvf_updates = 0

    obs, info = env.reset(seed=seed)
    base = agent.encoder.base_obs(obs, info)
    agent.on_episode_start()
    ep_r, ep_len, losses = 0.0, 0, []

    step = start_step
    while step < total:
        eps = epsilon_at(step, exp_cfg)
        if step < exp_cfg["learning_starts"]:
            a = int(rng.integers(0, 5))
            agent._last_bprob = 1.0 / 5
        else:
            a = agent.act(base, eps)
        nobs, r, terminated, truncated, ninfo = env.step(a)
        nbase = agent.encoder.base_obs(nobs, ninfo)
        agent.observe(base, a, r, nbase, terminated, truncated)
        if mode == "C":
            gvf_updates += 1
        base = nbase; ep_r += r; ep_len += 1; step += 1

        if step >= exp_cfg["learning_starts"] and step % exp_cfg["train_freq"] == 0:
            loss = agent.learn(bs)
            if loss is not None:
                losses.append(loss)

        if terminated or truncated:
            obs, info = env.reset()
            base = agent.encoder.base_obs(obs, info)
            agent.on_episode_start()
            ep_r, ep_len = 0.0, 0

        if step % exp_cfg["eval_interval"] == 0 or step == total:
            agg, _ = evaluate(agent, eval_env, exp_cfg["eval_episodes"], exp_cfg["eval_seed"])
            rec = {"env_step": step, "epsilon": eps,
                   "mean_loss": float(np.mean(losses[-1000:])) if losses else None,
                   "gvf_updates": gvf_updates,
                   "gvf_nonfinite": (agent.predictor.nonfinite_count if agent.predictor else 0),
                   **agg}
            history.append(rec)
            agent.save(ckpt_path, episode=0, env_steps=step,
                       extra={"mode": mode, "seed": seed})
            (out / "history.json").write_text(json.dumps(history, indent=2))
            print(f"[train {mode} s{seed}] step {step}/{total} eps={eps:.2f} "
                  f"success={agg['success_rate']:.2f} collision={agg['collision_rate']:.2f} "
                  f"reward={agg['reward_mean']:.1f} len={agg['length_mean']:.0f}")

    wall = time.perf_counter() - t0
    summary = {"mode": mode, "seed": seed, "env_steps": step,
               "gvf_updates": gvf_updates, "wall_time_s": round(wall, 1),
               "final": history[-1] if history else None,
               "in_dim": agent.in_dim, "base_dim": agent.encoder.base_dim}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary
