"""
On-policy light seeking with Sarsa(lambda), using the SAME features as the Horde.

Why this file exists: the seek-light control demon learns OFF-policy (the robot is
driven by a sticky random behaviour, the demon learns about its own greedy policy).
That needs importance ratios rho = pi/mu, which can be 25, and cuts the eligibility
trace whenever behaviour and greedy disagree. Here the robot is driven BY the policy
being learned (epsilon-greedy), so none of that is needed. Same phi(s,a), same
reward (light sensor), same gamma -> if this does much better, the off-policy
learning was the bottleneck; if not, the features are.

Sarsa(lambda), one step:
    the robot is in s, takes a (chosen epsilon-greedily), gets r and s'
    picks its next action a' epsilon-greedily from q(s', .)
    delta = r + gamma * q(s', a') - q(s, a)       TD error of the policy we follow
    e     = gamma * lambda * e + phi(s, a)        eligibility trace (no rho, never cut)
    theta = theta + alpha * delta * e
No secondary weights w: on-policy linear TD is stable on its own.

"""

import os
import sys

import numpy as np

from demons import DEMONS
from env import ARENA, N_ACTIONS, OBSTACLES, Action, Robot
from features import N_ACTIVE, N_FEATURES, phi
from main import CTRL_STEPS, EVAL_STARTS, random_rewards  # shared eval with main.py

STEPS, SEED, EVAL_EVERY = 3_000_000, 0, 25_000
GAMMA = 0.95  # same horizon as the control demons (~20 steps)
LAM = 0.9  # on-policy, so a long trace is safe (off-policy it blew up above ~0.7)
ALPHA = 0.05 / N_ACTIVE  # per active feature, same scaling as horde.py
EPS = 0.1  # exploration: 10% of the time take a random action
CHK = "checkpoint_onpolicy.npz"  # separate from the Horde's checkpoint.npz
PER = "eval_per_start.npz"  # every eval: reward/step from each of EVAL_STARTS
RESUME = True
# "episodic" on the command line: every EP_LEN steps jump to a random free spot and heading,
# so the robot keeps practising far from the light instead of parking next to it
EPISODIC, EP_LEN = "episodic" in sys.argv, 250
# which control demon to learn on-policy: start of its name, e.g. "seek", "avoid", "spin"
K = next(
    i
    for i, d in enumerate(DEMONS)
    if d["name"].startswith(sys.argv[1] if len(sys.argv) > 1 else "seek")
)


def q(theta: np.ndarray, obs: np.ndarray) -> np.ndarray:
    """q(s, a) for all 5 actions: sum of the weights of each action's active features."""
    return np.array([theta[phi(obs, a)].sum() for a in Action])


def evaluate(theta: np.ndarray, seed: int) -> np.ndarray:
    """
    Reward per step of the GREEDY learned policy (no exploration) from each of
    main.py's EVAL_STARTS: same starts, steps and sensor noise as the Horde's control
    demons, so the numbers are directly comparable.
    """
    sim, per = Robot(seed=seed + 1), np.zeros(len(EVAL_STARTS))
    for k, st in enumerate(EVAL_STARTS):
        sim.set_state(st)
        o = sim.obs()
        for _ in range(CTRL_STEPS):
            o = sim.step(Action(int(np.argmax(q(theta, o)))))
            per[k] += DEMONS[K]["r"](o) / CTRL_STEPS
    return per


if __name__ == "__main__":
    rng = np.random.default_rng(SEED)
    env = Robot(seed=SEED)
    theta = np.zeros(N_FEATURES)  # one weight vector: this is a single "demon"
    e = np.zeros(N_FEATURES)
    start = 0

    # ---- reload a previous run, same idea as main.py ----
    if RESUME and os.path.exists(CHK):
        ck = np.load(CHK)
        # weights only fit the features they were trained with (see main.py)
        assert (
            ck["theta"].shape == theta.shape
        ), f"{CHK} was made with different features; delete it to start fresh"
        theta, start = ck["theta"], int(ck["t"])
        rng = np.random.default_rng(SEED + start)
        env = Robot(seed=SEED + start)
        env.set_state(tuple(float(v) for v in ck["robot"]))
        print(f"Resumed from {CHK} at step {start:,}")
        # e starts at zero again: a trace only matters for the last few steps anyway

    # first action, chosen epsilon-greedily
    obs = env.obs()
    a = (
        int(np.argmax(q(theta, obs)))
        if rng.random() > EPS
        else int(rng.integers(N_ACTIONS))
    )

    print(
        f"{DEMONS[K]['name']} on-policy: reward per step, learned greedy policy vs. random policy"
    )
    print(f"  {'step':>10}{'learned':>10}{'random':>11}{'learned/random':>18}")
    per_t, per_all = [], []  # per-start history; a resume keeps what is already saved
    if start and os.path.exists(PER):
        per_t, per_all = list(np.load(PER)["t"]), list(np.load(PER)["scores"])
    for t in range(start + 1, STEPS + 1):
        obs2 = env.step(Action(a))
        r = DEMONS[K]["r"](obs2)  # the same cumulant as the Horde demon

        # next action from the SAME epsilon-greedy policy: this is what makes it on-policy
        q2 = q(theta, obs2)
        a2 = int(np.argmax(q2)) if rng.random() > EPS else int(rng.integers(N_ACTIONS))

        idx = phi(obs, a)
        delta = r + GAMMA * q2[a2] - theta[idx].sum()
        e *= GAMMA * LAM  # older steps get less credit
        e[idx] += 1.0  # the step we just took gets full credit
        theta += ALPHA * delta * e

        obs, a = obs2, a2
        if EPISODIC and t % EP_LEN == 0:  # new episode; no update uses the jump itself
            while (
                True
            ):  # random free spot, 0.12 m from walls and pillars as in env.step
                x, y = rng.uniform(0.12, ARENA - 0.12, 2)
                gap = (
                    np.hypot(x - OBSTACLES[:, 0], y - OBSTACLES[:, 1]) - OBSTACLES[:, 2]
                )
                if gap.min() > 0.12:
                    break
            env.set_state((x, y, rng.uniform(-np.pi, np.pi), 0.0, 0.0, 0.0))
            obs, e = env.obs(), np.zeros(N_FEATURES)
            a = (
                int(np.argmax(q(theta, obs)))
                if rng.random() > EPS
                else int(rng.integers(N_ACTIONS))
            )

        if t % EVAL_EVERY == 0:
            per = evaluate(theta, SEED + t)
            per_t.append(t)
            per_all.append(per)
            np.savez(
                PER, t=np.array(per_t), scores=np.array(per_all)
            )  # (evals, starts)
            learned, chance = per.mean(), random_rewards[K]
            print(
                f"  {t:>10,}{learned:>10.2f}{chance:>11.2f}{learned / chance:>17.2f}x",
                flush=True,
            )
            np.savez(
                CHK,
                theta=theta,
                t=t,
                robot=np.array(env.get_state()),
            )
