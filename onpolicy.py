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

Author: Vidush Jindal(jindalv)
"""

import os

import numpy as np

from env import LIGHT_L, LIGHT_R, N_ACTIONS, Action, Robot
from features import N_ACTIVE, N_FEATURES, phi

STEPS, SEED, EVAL_EVERY = 500_000, 0, 25_000
GAMMA = 0.95  # same horizon as the seek-light demon (~20 steps)
LAM = 0.9  # on-policy, so a long trace is safe (off-policy it blew up above ~0.7)
ALPHA = 0.05 / N_ACTIVE  # per active feature, same scaling as horde.py
EPS = 0.1  # exploration: 10% of the time take a random action
CHK = "checkpoint_onpolicy.npz"  # separate from the Horde's checkpoint.npz
RESUME = True


def q(theta: np.ndarray, obs: np.ndarray) -> np.ndarray:
    """q(s, a) for all 5 actions: sum of the weights of each action's active features."""
    return np.array([theta[phi(obs, a)].sum() for a in Action])


def evaluate(theta: np.ndarray, visit: list, seed: int, n: int = 30, steps: int = 250):
    """
    Mean light per step of the GREEDY learned policy (no exploration) vs a uniformly
    random policy, both started from the same n visited states. Same measurement as
    the control demons in main.py, so the numbers are directly comparable.
    """
    rng = np.random.default_rng(seed)
    sim = Robot(seed=seed + 1)
    total = {"learned": 0.0, "random": 0.0}
    for _ in range(n):
        st = visit[rng.integers(len(visit))]
        for label in total:  # same start, two policies
            sim.set_state(st)
            o = sim.obs()
            for _ in range(steps):
                if label == "learned":
                    a = int(np.argmax(q(theta, o)))
                else:
                    a = int(rng.integers(N_ACTIONS))
                o = sim.step(Action(a))
                total[label] += o[LIGHT_L] + o[LIGHT_R]
    return total["learned"] / (n * steps), total["random"] / (n * steps)


if __name__ == "__main__":
    rng = np.random.default_rng(SEED)
    env = Robot(seed=SEED)
    theta = np.zeros(N_FEATURES)  # one weight vector: this is a single "demon"
    e = np.zeros(N_FEATURES)
    visited = []
    start = 0

    # ---- reload a previous run, same idea as main.py ----
    if RESUME and os.path.exists(CHK):
        ck = np.load(CHK)
        # weights only fit the features they were trained with (see main.py)
        assert ck["theta"].shape == theta.shape, (
            f"{CHK} was made with different features; delete it to start fresh"
        )
        theta, start, visited = ck["theta"], int(ck["t"]), list(ck["visited"])
        rng = np.random.default_rng(SEED + start)
        env = Robot(seed=SEED + start)
        env.set_state(tuple(float(v) for v in ck["robot"]))
        print(f"Resumed from {CHK} at step {start:,}")
        # e starts at zero again: a trace only matters for the last few steps anyway

    # first action, chosen epsilon-greedily
    obs = env.obs()
    a = int(np.argmax(q(theta, obs))) if rng.random() > EPS else int(rng.integers(N_ACTIONS))

    print("Light seen per step: learned greedy policy vs. random policy")
    print(f"  {'step':>10}{'learned':>10}{'random':>11}{'learned/random':>18}")
    for t in range(start + 1, STEPS + 1):
        obs2 = env.step(Action(a))
        r = obs2[LIGHT_L] + obs2[LIGHT_R]  # total light, same reward as seek-light

        # next action from the SAME epsilon-greedy policy: this is what makes it on-policy
        q2 = q(theta, obs2)
        a2 = int(np.argmax(q2)) if rng.random() > EPS else int(rng.integers(N_ACTIONS))

        idx = phi(obs, a)
        delta = r + GAMMA * q2[a2] - theta[idx].sum()
        e *= GAMMA * LAM  # older steps get less credit
        e[idx] += 1.0  # the step we just took gets full credit
        theta += ALPHA * delta * e

        obs, a = obs2, a2
        if t % 30 == 0:
            visited.append(env.get_state())

        if t % EVAL_EVERY == 0:
            learned, chance = evaluate(theta, visited, SEED + t)
            print(f"  {t:>10,}{learned:>10.2f}{chance:>11.2f}{learned / chance:>17.2f}x", flush=True)
            np.savez(
                CHK,
                theta=theta,
                t=t,
                robot=np.array(env.get_state()),
                visited=np.array(visited),
            )
