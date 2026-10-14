"""
Drive the robot with one behaviour policy; every demon learns off-policy.

Evaluation compares each demon's prediction q-hat against the actual return
obtained by following that demon's target policy from the same state. That
comparison is the bold-vs-thin line in Figures 2 and 3. This captures difference
between off policy and on policy

Control demons are scored differently: average cumulant per time step under the
learned policy, against a random-policy baseline (Figure 5).

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

STEPS, STICK, SEED, EVAL_EVERY = 450_0000, 0.8, 0, 25_000
VIZ = False
CHK = "checkpoint.npz"  # where training is saved; delete this file to start fresh
RESUME = True  # False = ignore any existing checkpoint and train from scratch
LOG = "train_log.txt"


def save_checkpoint(
    horde: Horde,
    env: Robot,
    t: int,
    prev: int,
    obs: np.ndarray,
    visited: list[tuple[float, ...]],
):
    """
    Save everything needed to continue training later, as plain numpy arrays.

    np.savez writes several named arrays into ONE .npz file (a zip of .npy files).
    for eg np.load(name)["parameter"] gives back exactly the array saved under that parameter.

    - theta, w, e: the learned weights of every demon (+ eligibility traces).
    - robot, friction: the Robot's physical state (x, y, th, v, w, bump) and floor.
    - t, prev, obs, visited: the training loop's own variables.

    """
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
        visited=np.array(visited),  # list of 6-tuples -> (N x 6) array
    )
    os.replace("checkpoint.tmp.npz", CHK)


def evaluate(
    horde: Horde,
    visit: list[tuple[float, float, float, float, float, float]],
    seed: int,
    n: int = 25,
    horizon: int = 400,
    n_ctrl: int = 10,
    ctrl_steps: int = 250,
):
    """
    For each demon: mean prediction vs mean Monte-Carlo return.

    States are sampled from what the BEHAVIOUR policy actually visited. A value
    function is only meaningful where the behaviour gave coverage, so testing on
    uniformly random states measures randomness.
    """
    rng = np.random.default_rng(seed)
    sim = Robot(seed=seed + 1)
    rows = []
    for i, d in enumerate(DEMONS):
        # a control demon is exactly one whose target policy is greedy(q-hat).
        # score its POLICY (cumulant per step vs random), not its value estimate
        if d["pi"] is greedy:
            learned, chance = [], []
            for _ in range(n_ctrl):
                st = visit[rng.integers(len(visit))]
                for use_learned in (True, False):  # same start, two policies
                    sim.set_state(st)
                    o, total = sim.obs(), 0.0
                    for _ in range(ctrl_steps):
                        a = (
                            Action(np.argmax(horde.q(o)[i]))
                            if use_learned
                            else Action(rng.integers(N_ACTIONS))
                        )
                        o = sim.step(a)
                        total += d["r"](o)  # cumulant, undiscounted
                        # mean over horizon
                    (learned if use_learned else chance).append(total / ctrl_steps)
            # mean over starting steps
            learned_rt, ch_rt = float(np.mean(learned)), float(np.mean(chance))
            rows.append(
                (
                    d["name"],
                    "ctrl",
                    learned_rt,
                    ch_rt,
                    learned_rt / ch_rt if ch_rt > 1e-9 else np.inf,
                )
            )
            continue

        pred, actual = [], []
        for _ in range(n):
            # visited state sampled randomly from all of them
            st = visit[rng.integers(len(visit))]
            sim.set_state(st)
            o = sim.obs()
            if d["gamma"](o) == 0.0:
                continue  # the last state before episode terminated

            # sample an action from the target policy -- pi may be stochastic
            qi = horde.q(o)[i]
            a = Action(rng.choice(N_ACTIONS, p=d["pi"](o, qi)))
            # append the target policy greedy action value integer as prediction
            pred.append(qi[a])

            sim.set_state(st)  # roll out target policy
            o, G, disc, b = (
                sim.obs(),
                0.0,
                1.0,
                a,
            )
            # G is return, we will use it for monte carlo return
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
                b = Action(rng.choice(N_ACTIONS, p=d["pi"](o, horde.q(o)[i])))
            actual.append(G)
        pred, actual = np.array(pred), np.array(actual)
        rows.append(
            (
                d["name"],
                "pred",
                pred.mean(),
                actual.mean(),
                np.abs(pred - actual).mean(),
            )
        )
    return rows


if __name__ == "__main__":
    rng = np.random.default_rng(SEED)
    env = Robot(seed=SEED)
    horde = Horde(DEMONS)
    obs, prev = env.obs(), Action.STOP
    visited = []  # states the behaviour actually reached
    start = 0  # steps already done; 0 for a fresh run, changed below when resuming
    resuming = RESUME and os.path.exists(CHK)

    # log to terminal and LOG; fresh run overwrites the file, resumed run appends
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[
            logging.FileHandler(LOG, mode="a" if resuming else "w"),
            logging.StreamHandler(),
        ],
    )
    log = logging.info

    if resuming:
        ck = np.load(CHK)
        assert (
            ck["theta"].shape == horde.theta.shape
        ), f"{CHK} was made with different demons/features; delete it to start fresh"
        horde.theta, horde.w, horde.e = ck["theta"], ck["w"], ck["e"]
        start = int(ck["t"])  # 0-d arrays -> plain python numbers
        prev, obs = Action(int(ck["prev"])), ck["obs"]
        visited = list(ck["visited"])  # back to a list so .append() works

        rng = np.random.default_rng(SEED + start)
        env = Robot(seed=SEED + start, friction=float(ck["friction"]))
        env.set_state(tuple(float(v) for v in ck["robot"]))
        log(f"Resumed from {CHK} at step {start:,}")

    live = Live() if VIZ else None

    for t in range(start + 1, STEPS + 1):
        # out behaviour policy is biased towards repeating the current taken action

        p = np.full(N_ACTIONS, (1 - STICK) / N_ACTIONS)
        p[prev] += STICK
        p /= p.sum()
        a = Action(rng.choice(N_ACTIONS, p=p))
        mu = float(p[a])
        prev = a

        obs2 = env.step(a)  # the next sensor readings
        delta = horde.update(obs, a, obs2, mu)  # All demons learn from this one step
        obs = obs2
        if t % 30 == 0:
            visited.append(env.get_state())
        if live:
            live.update(t, env.get_state(), delta)

        if t % EVAL_EVERY == 0:
            rows = evaluate(horde, visited, SEED + t)
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
            log(
                "CONTROL demons: reward per step, learned greedy policy vs. random policy"
            )
            log(f"  {'demon':<30}{'learned':>10}{'random':>11}{'learned/random':>18}")
            for name, kind, x, y, z in rows:
                if kind == "ctrl":
                    log(f"  {name:<30}{x:>10.2f}{y:>11.2f}{z:>17.2f}x")

            save_checkpoint(horde, env, t, prev, obs, visited)
            log(f"saved checkpoint -> {CHK}")

    if live:
        plt.ioff()
        plt.show()  # keep the window open at the end
