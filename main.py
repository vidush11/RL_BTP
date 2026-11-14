"""
Drive the robot with one sticky-random behaviour policy; every demon learns off-policy.

Every EVAL_EVERY steps:
- prediction demons: q-hat vs the Monte-Carlo return of their target policy (Figs 2-3)
- control demons: reward per step of their greedy policy vs a random policy (Fig 5),
  from EVAL_STARTS, the same fixed starts onpolicy.py uses

Author: Vidush Jindal(jindalv)
"""

import logging
import os

import matplotlib.pyplot as plt
import numpy as np

from demons import DEMONS, greedy
from env import N_ACTIONS, Action, Robot
from horde import Horde
from viz import Live

STEPS, STICK, SEED, EVAL_EVERY = 3_000_000, 0.8, 0, 25_000
N_CTRL, CTRL_STEPS = 100, 250  # control eval: number of starts, steps from each
VIZ, RESUME = False, True  # RESUME=False (or deleting CHK) trains from scratch
CHK, LOG, PER = "checkpoint.npz", "train_log.txt", "eval_per_start.npz"
CTRL = [i for i, d in enumerate(DEMONS) if d["pi"] is greedy and "follow" not in d]

# Shared evaluation, also imported by onpolicy.py.
sim_temp, rng, prev_action, starts = (
    Robot(seed=1105),
    np.random.default_rng(1234),
    Action.STOP,
    [],
)
for i in range(15_000):
    p = np.full(N_ACTIONS, (1 - STICK) / N_ACTIONS)
    p[prev_action] += STICK
    prev_action = Action(rng.choice(N_ACTIONS, p=p))
    sim_temp.step(prev_action)
    if i % 30 == 0:
        starts.append(sim_temp.get_state())
EVAL_STARTS = [starts[k] for k in rng.choice(len(starts), N_CTRL, replace=False)]

# random policy's reward per step from EVAL_STARTS (0 for prediction demons)
random_rewards = np.zeros(len(DEMONS))
for st in EVAL_STARTS:
    sim_temp.set_state(st)
    for _ in range(CTRL_STEPS):
        _obs = sim_temp.step(Action(rng.integers(N_ACTIONS)))
        random_rewards[CTRL] += [DEMONS[i]["r"](_obs) for i in CTRL]
random_rewards /= N_CTRL * CTRL_STEPS  # averaging the reward with steps, starts


def evaluate(horde, visit, seed, n=25, horizon=400):
    """Rows (name, kind, x, y, z) and each control demon's reward/step per start
    (len(CTRL) x N_CTRL). pred: predicted, actual, mean |error|.
    ctrl: learned reward/step, random reward/step (random_rewards), learned/random."""
    rng, sim, rows = np.random.default_rng(seed), Robot(seed=seed + 1), []
    per = np.zeros((len(CTRL), N_CTRL))
    for i, d in enumerate(DEMONS):
        if i in CTRL:
            for k, st in enumerate(EVAL_STARTS):
                sim.set_state(st)
                o = sim.obs()
                for _ in range(CTRL_STEPS):
                    o = sim.step(Action(np.argmax(horde.q(o)[i])))
                    per[CTRL.index(i), k] += d["r"](o) / CTRL_STEPS
            x, y = per[CTRL.index(i)].mean(), random_rewards[i]
            rows.append((d["name"], "ctrl", x, y, x / y if y > 1e-9 else np.inf))
            continue

        pred, actual = [], []
        for _ in range(n):
            st = visit[rng.integers(len(visit))]  # only where the behaviour went
            sim.set_state(st)
            o = sim.obs()
            if d["gamma"](o) == 0.0:
                continue  # already terminal
            q = horde.q(o)  # psi demons act by the policy of demon "follow"
            b = Action(rng.choice(N_ACTIONS, p=d["pi"](o, q[d.get("follow", i)])))
            pred.append(q[i][b])
            sim.set_state(st)  # roll out the target policy from the same state
            G, disc = 0.0, 1.0
            for _ in range(horizon):
                o = sim.step(b)
                G += disc * d["r"](o)
                g = d["gamma"](o)
                if g == 0.0:
                    G += disc * d["z"](o)
                    break
                disc *= g
                if disc < 1e-3:
                    break
                q = horde.q(o)[d.get("follow", i)]
                b = Action(rng.choice(N_ACTIONS, p=d["pi"](o, q)))
            actual.append(G)
        pred, actual = np.array(pred), np.array(actual)
        err = np.abs(pred - actual).mean()
        rows.append((d["name"], "pred", pred.mean(), actual.mean(), err))
    return rows, per


if __name__ == "__main__":
    resuming = RESUME and os.path.exists(CHK)
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[
            logging.FileHandler(LOG, mode="a" if resuming else "w"),
            logging.StreamHandler(),
        ],
    )
    log = logging.info

    horde, env = Horde(DEMONS), Robot(seed=SEED)
    start, prev, obs, visited = 0, Action.STOP, env.obs(), []
    if resuming:
        ck = np.load(CHK)
        assert ck["theta"].shape == horde.theta.shape, f"{CHK}: different demons"
        horde.theta, horde.w, horde.e = ck["theta"], ck["w"], ck["e"]
        start, prev, obs = int(ck["t"]), Action(int(ck["prev"])), ck["obs"]
        visited = list(ck["visited"])
        env = Robot(seed=SEED + start, friction=float(ck["friction"]))
        env.set_state(tuple(float(v) for v in ck["robot"]))
        log(f"Resumed from {CHK} at step {start:,}")
    rng = np.random.default_rng(SEED + start)
    live = Live() if VIZ else None
    per_t, per_all = [], []  # per-start history; a resume keeps what is already saved
    if resuming and os.path.exists(PER):
        per_t, per_all = list(np.load(PER)["t"]), list(np.load(PER)["scores"])

    for t in range(start + 1, STEPS + 1):
        p = np.full(N_ACTIONS, (1 - STICK) / N_ACTIONS)  # repeat prev with prob STICK
        p[prev] += STICK
        p /= p.sum()
        a = Action(rng.choice(N_ACTIONS, p=p))
        obs2 = env.step(a)
        delta = horde.update(obs, a, obs2, float(p[a]))
        obs, prev = obs2, a
        if t % 30 == 0:
            visited.append(env.get_state())
        if live:
            live.update(t, env.get_state(), delta)
        if t % EVAL_EVERY:
            continue

        rows, per = evaluate(horde, visited, SEED + t)
        per_t.append(t)
        per_all.append(per)
        # scores: (evals, control demons, starts)
        np.savez(PER, t=np.array(per_t), scores=np.array(per_all))
        if live:
            live.record(t, rows)
        log(f"\nsteps {t:,} | friction {env.friction}")
        log(
            "PREDICTION demons: the demon's prediction vs. the return actually observed"
        )
        log(f"  {'demon':<30}{'predicted':>10}{'actual':>11}{'avg error':>11}")
        for name, kind, x, y, z in rows:
            if kind == "pred":
                log(f"  {name:<30}{x:>10.2f}{y:>11.2f}{z:>11.2f}")
        log("CONTROL demons: reward per step, learned greedy policy vs. random policy")
        log(f"  {'demon':<30}{'learned':>10}{'random':>11}{'learned/random':>18}")
        for name, kind, x, y, z in rows:
            if kind == "ctrl":
                log(f"  {name:<30}{x:>10.2f}{y:>11.2f}{z:>17.2f}x")
        log(
            "  max|theta| per demon: "
            + " ".join(f"{v:.3g}" for v in np.abs(horde.theta).max(1))
        )

        # write to a temp file then rename, so a crash never leaves a half-written checkpoint
        np.savez(
            "checkpoint.tmp.npz",
            theta=horde.theta,
            w=horde.w,
            e=horde.e,
            robot=np.array(env.get_state()),
            friction=env.friction,
            t=t,
            prev=int(prev),
            obs=obs,
            visited=np.array(visited[-20000:]),
        )
        os.replace("checkpoint.tmp.npz", CHK)
        log(f"saved checkpoint -> {CHK}")

    if live:
        plt.ioff()
        plt.show()
