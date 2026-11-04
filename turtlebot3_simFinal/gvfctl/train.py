"""Training loops. Identical across methods: env 'train', step budget, one
gradient update per env step after `learning_starts` (UTD 1, batch 64), replay
100k, learning-curve evaluation (greedy, 40 fixed CURVE-seed episodes) at
20k-step intervals plus 50k and 100k.

GVF behaviour: segments of geometric length (mean 30 steps); each segment is a
smooth random walk over turn rates (and speed levels in Part 2) with probability
eps, else the controller. eps decays 1.0 -> 0.1 over the first 30% of steps.
"""
from __future__ import annotations

import time
from typing import Callable, Optional

import numpy as np

from .actions import ActionSpace
from .controllers import SteerCfg, SpeedCfg, controller_batch
from .demons import OriginalDemons
from .dqn import DQN, DQNConfig
from .envs import action_space, assert_training_seed, episode_seeds, make_env
from .evaluate import ControllerPolicy, DQNPolicy, run_episodes, summarize
from .gvf import GVFConfig, GVFModel, Replay, safety_cumulant

F32 = np.float32


def eval_points(budget: int):
    pts = set(range(20_000, budget + 1, 20_000)) | {p for p in (50_000, 100_000) if p <= budget} | {budget}
    return sorted(pts)


def curve_point(policy, part: int, n: int = 40) -> dict:
    s = summarize(run_episodes(policy, "train", part, episode_seeds("curve", "train", n)))
    return {k: s[k] for k in ("survival", "success", "mean_distance", "mean_speed", "collision_rate")}


def random_walk_step(prev: int, A: ActionSpace, rng) -> int:
    t = A.turn_idx[prev] + rng.choice((-1, 0, 1), p=(0.2, 0.6, 0.2))
    s = A.speed_idx[prev]
    if A.n_speed > 1:
        s = s + rng.choice((-1, 0, 1), p=(0.05, 0.9, 0.05))
    return int(A.grid[int(np.clip(s, 0, A.n_speed - 1)), int(np.clip(t, 0, A.n_turn - 1))])


def _replay(cap, seed, extra):
    f = {"s": ((24,), F32), "sp": ((24,), F32), "a": ((), np.int16), "term": ((), F32)}
    f.update(extra)
    return Replay(cap, seed, f)


def train_gvf(part: int, gcfg: GVFConfig, budget: int, seed: int, steer: Optional[SteerCfg] = None,
              speed: Optional[SpeedCfg] = None, behaviour: str = "mixture", log: Callable = print,
              curve_n: int = 40):
    assert_training_seed(seed)
    steer, speed = steer or SteerCfg(), speed or SpeedCfg()
    A = action_space(part)
    model = GVFModel(gcfg, A, seed, steer, speed)
    env, _ = make_env("train", part, 800)
    rng = np.random.default_rng(seed + 12345)
    rep = _replay(gcfg.replay, seed + 1, {"c": ((), F32)})
    _, info = env.reset(seed=seed)
    r, prev = info["ranges_m"].astype(F32), A.default_action()
    seg_left, mode = 0, "rw"
    policy = ControllerPolicy(model, A, steer, speed)
    pts, curve, losses = set(eval_points(budget)), [], []
    t0 = time.perf_counter(); episodes = 0
    for t in range(budget):
        eps = 1.0 if behaviour == "randomwalk" else max(0.1, 1.0 - 0.9 * t / (0.3 * budget))
        if seg_left <= 0:
            seg_left = int(rng.geometric(1 / 30)); mode = "rw" if rng.random() < eps else "ctrl"
        seg_left -= 1
        if mode == "rw":
            a = random_walk_step(prev, A, rng)
        else:
            Qmin, Vl, Va = model.decision_inputs(r[None])
            a = int(controller_batch(Qmin, Vl, Va, np.array([prev]), A, steer, speed)[0])
        _, _, term, trunc, info = env.step(a)
        rp = info["ranges_m"].astype(F32)
        rep.add(s=r, sp=rp, a=a, c=safety_cumulant(float(rp.min()), A.v[a], gcfg), term=float(term))
        if len(rep) >= gcfg.learning_starts:
            losses.append(model.update(rep.sample(gcfg.batch)))
        if term or trunc:
            _, info = env.reset(); r, prev = info["ranges_m"].astype(F32), A.default_action()
            seg_left = 0; episodes += 1
        else:
            r, prev = rp, a
        if t + 1 in pts:
            cp = curve_point(policy, part, curve_n)
            cp.update(step=t + 1, wall_s=round(time.perf_counter() - t0, 1),
                      loss_safety=float(np.mean([l["loss_safety"] for l in losses[-2000:]])) if losses else None,
                      loss_vel=float(np.mean([l["loss_vel"] for l in losses[-2000:]])) if losses else None)
            curve.append(cp)
            log(f"[gvf p{part} s{seed}] {t + 1}/{budget} surv={cp['survival']:.3f} "
                f"dist={cp['mean_distance']:.2f} wall={cp['wall_s']}s")
    meta = {"updates": model.updates, "episodes": episodes, "wall_s": round(time.perf_counter() - t0, 1)}
    return model, curve, meta


def train_dqn(part: int, dcfg: DQNConfig, budget: int, seed: int, log: Callable = print, curve_n: int = 40):
    assert_training_seed(seed)
    A = action_space(part)
    dqn = DQN(dcfg, A, seed)
    env, _ = make_env("train", part, 800)
    rng = np.random.default_rng(seed + 12345)
    rep = _replay(dcfg.replay, seed + 1, {"r": ((), F32)})
    _, info = env.reset(seed=seed)
    r = info["ranges_m"].astype(F32)
    policy = DQNPolicy(dqn, A)
    pts, curve, losses = set(eval_points(budget)), [], []
    t0 = time.perf_counter(); episodes = 0
    for t in range(budget):
        eps = max(dcfg.eps_end, dcfg.eps_start - (dcfg.eps_start - dcfg.eps_end) * t / (dcfg.eps_frac * budget))
        a = int(rng.integers(A.n)) if rng.random() < eps else int(np.argmax(dqn.q_values(r)[0]))
        _, _, term, trunc, info = env.step(a)
        rp = info["ranges_m"].astype(F32)
        rep.add(s=r, sp=rp, a=a, r=dqn.reward(a), term=float(term))
        if len(rep) >= dcfg.learning_starts:
            losses.append(dqn.update(rep.sample(dcfg.batch))["loss"])
        if term or trunc:
            _, info = env.reset(); r = info["ranges_m"].astype(F32); episodes += 1
        else:
            r = rp
        if t + 1 in pts:
            cp = curve_point(policy, part, curve_n)
            cp.update(step=t + 1, wall_s=round(time.perf_counter() - t0, 1),
                      loss=float(np.mean(losses[-2000:])) if losses else None, epsilon=eps)
            curve.append(cp)
            log(f"[dqn p{part} s{seed}] {t + 1}/{budget} surv={cp['survival']:.3f} "
                f"dist={cp['mean_distance']:.2f} wall={cp['wall_s']}s")
    meta = {"updates": dqn.updates, "episodes": episodes, "wall_s": round(time.perf_counter() - t0, 1)}
    return dqn, curve, meta


def train_demons(part: int, variant: str, budget: int, seed: int, steer: Optional[SteerCfg] = None,
                 speed: Optional[SpeedCfg] = None, log: Callable = print, curve_n: int = 40):
    assert_training_seed(seed)
    A = action_space(part)
    dem = OriginalDemons(A, variant)
    env, _ = make_env("train", part, 800)
    rng = np.random.default_rng(seed + 12345)
    _, info = env.reset(seed=seed); r = info["ranges_m"]
    dem.reset_episode()
    policy = ControllerPolicy(dem, A, steer or SteerCfg(), speed or SpeedCfg())
    pts, curve = set(eval_points(budget)), []
    t0 = time.perf_counter(); episodes = 0
    for t in range(budget):
        a = int(rng.integers(A.n))                       # original Horde behaviour: uniform random
        _, _, term, trunc, info = env.step(a)
        rp = info["ranges_m"]
        dem.update(r, a, rp, bool(term or trunc))
        if term or trunc:
            _, info = env.reset(); r = info["ranges_m"]; dem.reset_episode(); episodes += 1
        else:
            r = rp
        if t + 1 in pts:
            cp = curve_point(policy, part, curve_n)
            cp.update(step=t + 1, wall_s=round(time.perf_counter() - t0, 1))
            curve.append(cp)
            log(f"[demons-{variant} p{part} s{seed}] {t + 1}/{budget} surv={cp['survival']:.3f}")
    meta = {"updates": budget, "episodes": episodes, "wall_s": round(time.perf_counter() - t0, 1)}
    return dem, curve, meta
