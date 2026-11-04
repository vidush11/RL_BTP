#!/usr/bin/env python3
"""Train + evaluate one method for one seed on the safety task.

    python -m safety_exp.run --method gvf --seed 0 --episodes 30000 --out runs_safety/gvf_s0
    python -m safety_exp.run --method dqn --seed 0 --episodes 30000 --out runs_safety/dqn_s0
    (add --tuned runs_safety/tuning.json to use the validation-selected settings)

gvf: paper stage 1 -- the behaviour policy mu drives; the paper GVF predictors
     (Algorithm 1) AND the given RL_BTP Horde demons learn off-policy from the
     same transitions. Stage 2 -- every `eval_every` episodes both are frozen and
     drive their controllers on the fixed 100 CURVE episodes (learning curves).
dqn: standard DQN on the task reward.
Both: same env config, observation, actions, episode budget, evaluation.
Atomic checkpoints every `ckpt_every` episodes; re-running resumes from the
newest valid one. Final: 1,000 TEST episodes + 1,000 HARD (5-obstacle) episodes.
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import time
from pathlib import Path

import numpy as np

from agents.dqn.dqn import DQNAgent, DQNConfig
from agents.gvf.controller import ControllerConfig, PaperController
from agents.gvf.policies import BehaviourPolicy, PolicyConfig, TargetPolicy
from agents.gvf.predictor import GVFBank, PredictorConfig
from agents.horde.adapter import HordeAdapter, load_rlbtp
from agents.horde.controller import HordeController, HordeControllerConfig, frozen_horde
from simulator import hpc
from .common import HARD, evaluate, make_env, seeds, summarize

SWEEP_GAMMAS = (0.90, 0.95, 0.967)        # horizon-matched to dt = 0.1 s (B1)


def _save_ckpt(path: Path, state: dict):
    tmp = path.with_suffix(".tmp")
    with open(tmp, "wb") as fh:
        pickle.dump(state, fh, protocol=pickle.HIGHEST_PROTOCOL)
        fh.flush(); os.fsync(fh.fileno())
    os.replace(tmp, path)


def _load_ckpt(path: Path):
    try:
        with open(path, "rb") as fh:
            return pickle.load(fh)
    except Exception:
        return None


def _final_eval(policy_factory, out, n_test, trace=True):
    res = {}
    for kind, ov in (("test", None), ("hard", HARD)):
        recs = evaluate(policy_factory(), seeds(kind, n_test), kind, ov, trace=trace)
        res[kind] = {"summary": summarize(recs),
                     "episodes": [{k: v for k, v in r.items() if k != "trace"} for r in recs],
                     "pre_collision_signal": [r["trace"][-30:] for r in recs if r["collided"] and "trace" in r][:200]}
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--method", choices=["gvf", "dqn"], required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--episodes", type=int, default=30_000)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--eval-episodes", type=int, default=100)
    ap.add_argument("--n-test", type=int, default=1000)
    ap.add_argument("--ckpt-every", type=int, default=2000)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tuned", default=None, help="JSON with validation-selected settings")
    ap.add_argument("--sweep-gammas", action="store_true", help="also train the horizon-matched gammas")
    ap.add_argument("--set", action="append", default=[], help="override, e.g. predictor.lr=1e-3")
    ap.add_argument("--sim", action="append", default=[], help="simulator override, e.g. lidar.num_beams=24")
    ap.add_argument("--val-mode", action="store_true", help="tuning run: curve on VALIDATION seeds, no final test")
    args = ap.parse_args()

    import yaml
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    tuned = json.loads(Path(args.tuned).read_text()) if args.tuned else {}
    cfgs = {"predictor": PredictorConfig().to_dict(), "policy": PolicyConfig().__dict__.copy(),
            "controller": ControllerConfig().to_dict(), "horde_controller": HordeControllerConfig().to_dict(),
            "dqn": DQNConfig().to_dict()}
    for k in cfgs:
        cfgs[k].update(tuned.get(k, {}))
    for item in args.set:
        key, _, val = item.partition("="); sec, field = key.split(".")
        cfgs[sec][field] = yaml.safe_load(val)
    sim_over = {}
    for item in args.sim:
        key, _, val = item.partition("="); sec, field = key.split(".")
        sim_over.setdefault(sec, {})[field] = yaml.safe_load(val)
    if args.sweep_gammas:
        cfgs["predictor"]["gammas"] = sorted(set(cfgs["predictor"]["gammas"]) | set(SWEEP_GAMMAS))
    hpc.configure_threads(1)
    (out / "config.json").write_text(json.dumps({"args": vars(args), "cfgs": cfgs, "sim_overrides": sim_over}, indent=2))
    hpc.write_run_metadata(out, config={"cfgs": cfgs, "sim_overrides": sim_over}, seed=args.seed, agent=args.method)

    env = make_env(sim_over)
    names = env.action_names
    n = env.action_space.n
    obs_dim = env.observation_space.shape[0]
    v_max = env.cfg.motion.v_max
    rng = np.random.default_rng(args.seed + 99_991)
    curve_kind = "val" if args.val_mode else "curve"
    curve_seeds = seeds(curve_kind, args.eval_episodes)

    pcfg = PredictorConfig(**{**cfgs["predictor"], "gammas": tuple(cfgs["predictor"]["gammas"]),
                              "hidden": tuple(cfgs["predictor"]["hidden"])})
    dcfg = DQNConfig(**{**cfgs["dqn"], "hidden": tuple(cfgs["dqn"]["hidden"])})
    ccfg = ControllerConfig(**cfgs["controller"])
    hcfg = HordeControllerConfig(**cfgs["horde_controller"])

    if args.method == "gvf":
        pol_cfg = PolicyConfig(**cfgs["policy"])
        mu, pi = BehaviourPolicy(n, False, pol_cfg), TargetPolicy(n, False, pol_cfg)
        bank = GVFBank(obs_dim, n, pcfg, args.seed)
        rl = load_rlbtp()
        horde = rl["horde"].Horde(rl["demons"].DEMONS)
        adapter = HordeAdapter(env)
    else:
        agent = DQNAgent(obs_dim, n, dcfg, args.seed)

    ck_path = out / "checkpoint.pkl"
    episodes = steps = 0
    curve, ep_log = [], []
    t_start = time.perf_counter(); wall_prev = 0.0
    ck = _load_ckpt(ck_path)
    if ck is not None:
        episodes, steps, curve, ep_log, wall_prev = ck["episodes"], ck["steps"], ck["curve"], ck["ep_log"], ck["wall"]
        rng.bit_generator.state = ck["rng"]
        if args.method == "gvf":
            bank.load(ck["bank"]); bank.replay.load(ck["replay"]); bank.replay.rng.bit_generator.state = ck["replay_rng"]
            horde.theta, horde.w, horde.e = ck["horde"]
        else:
            agent.q.load(ck["q"]); agent.t.load(ck["t"]); agent.replay.load(ck["replay"])
            agent.replay.rng.bit_generator.state = ck["replay_rng"]; agent.updates = ck["updates"]
            for k, (m, v) in enumerate(ck["adam"]):
                agent.q._m[k][...], agent.q._v[k][...] = m, v
            agent.q.t = ck["adam_t"]
        print(f"[run] resumed at episode {episodes}, env step {steps}", flush=True)

    def wall():
        return wall_prev + time.perf_counter() - t_start

    def do_eval():
        rec = {"episode": episodes, "env_steps": steps, "wall_s": round(wall(), 1)}
        if args.method == "gvf":
            fb = bank.frozen_copy()
            rec["paper"] = summarize(evaluate(PaperController(fb, names, ccfg, v_max), curve_seeds, curve_kind, sim_over))
            rec["horde"] = summarize(evaluate(HordeController(frozen_horde(horde), adapter, names, hcfg),
                                              curve_seeds, curve_kind, sim_over))
        else:
            rec["dqn"] = summarize(evaluate(agent.frozen_policy(), curve_seeds, curve_kind, sim_over))
        recent = ep_log[-args.eval_every:]
        rec["train_mean_len"] = float(np.mean([e[0] for e in recent])) if recent else None
        rec["train_mean_return"] = float(np.mean([e[1] for e in recent])) if recent else None
        curve.append(rec)
        hpc.write_json_atomic(out / "curve.json", curve)
        msg = " ".join(f"{k}: cf={rec[k]['collision_free_rate']:.2f} spd={rec[k]['speed_ratio']:.2f} "
                       f"ret={rec[k]['mean_task_return']:.0f}" for k in ("paper", "horde", "dqn") if k in rec)
        print(f"[{args.method} s{args.seed}] ep {episodes} steps {steps} wall {rec['wall_s']}s | {msg}", flush=True)

    def checkpoint():
        st = {"episodes": episodes, "steps": steps, "curve": curve, "ep_log": ep_log, "wall": wall(),
              "rng": rng.bit_generator.state}
        if args.method == "gvf":
            st.update(bank=bank.state(), replay=bank.replay.state(), replay_rng=bank.replay.rng.bit_generator.state,
                      horde=(horde.theta, horde.w, horde.e))
        else:
            st.update(q=agent.q.state(), t=agent.t.state(), replay=agent.replay.state(),
                      replay_rng=agent.replay.rng.bit_generator.state, updates=agent.updates,
                      adam=list(zip(agent.q._m, agent.q._v)), adam_t=agent.q.t)
        _save_ckpt(ck_path, st)

    if episodes == 0 and not curve:
        do_eval()                                          # untrained starting point
    obs, info = env.reset(seed=args.seed if episodes == 0 else args.seed * 1_000_003 + episodes)
    a_prev = names.index("STOP")
    o9 = adapter.sensors(info) if args.method == "gvf" else None
    if args.method == "gvf":
        horde.e[:] = 0.0
    ep_len, ep_ret = 0, 0.0
    while episodes < args.episodes:
        if args.method == "gvf":
            a = mu.sample(a_prev, rng)
            p_mu = float(mu.probs(a_prev)[a])
        else:
            eps = agent.epsilon(episodes, args.episodes)
            a = int(rng.integers(n)) if rng.random() < eps else agent.greedy(obs)
        obs2, r, te, tr, info2 = env.step(a)
        steps += 1; ep_len += 1; ep_ret += r
        if args.method == "gvf":
            cf = 0.0 if (te and info2["collision"]) else float(info2["safety"]["c_front"])
            bank.replay.add(obs, a, cf, float(info2["linear_velocity"]) / v_max, te, obs2, pi.sample(a, rng))
            bank.maybe_train(steps)
            o9b = adapter.sensors(info2)
            horde.update(o9, adapter.rlbtp_action(a), o9b, p_mu)
            o9 = o9b
        else:
            agent.replay.add(obs, a, r, obs2, te)
            agent.maybe_train(steps)
        if te or tr:
            episodes += 1
            ep_log.append((ep_len, ep_ret, bool(te)))
            ep_len, ep_ret = 0, 0.0
            obs, info = env.reset()
            a_prev = names.index("STOP")
            if args.method == "gvf":
                o9 = adapter.sensors(info); horde.e[:] = 0.0
            if episodes % args.eval_every == 0 or episodes == args.episodes:
                do_eval()
            if episodes % args.ckpt_every == 0:
                checkpoint()
        else:
            obs, a_prev = obs2, a

    # ---------------- final artefacts -------------------------------------
    summary = {"method": args.method, "seed": args.seed, "episodes": episodes, "env_steps": steps,
               "wall_s": round(wall(), 1), "cfgs": cfgs}
    if args.method == "gvf":
        np.savez_compressed(out / "predictors.npz", **bank.state())
        np.savez_compressed(out / "horde.npz", theta=horde.theta, w=horde.w)
        if not args.val_mode:
            fb = bank.frozen_copy(); fh = frozen_horde(horde)
            summary["final"] = {"paper": _final_eval(lambda: PaperController(fb, names, ccfg, v_max), out, args.n_test),
                                "horde": _final_eval(lambda: HordeController(fh, adapter, names, hcfg), out, args.n_test)}
    else:
        np.savez_compressed(out / "dqn.npz", **agent.q.state())
        if not args.val_mode:
            summary["final"] = {"dqn": _final_eval(lambda: agent.frozen_policy(), out, args.n_test, trace=False)}
    hpc.write_json_atomic(out / "summary.json", summary)
    np.save(out / "episode_log.npy", np.array(ep_log, dtype=np.float64))
    print(f"[run] done: {episodes} episodes, {steps} env steps, {summary['wall_s']} s", flush=True)


if __name__ == "__main__":
    main()
