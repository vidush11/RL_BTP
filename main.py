"""
Drive the robot with one behaviour policy; every demon learns off-policy.

Evaluation compares each demon's prediction q-hat against the actual return
obtained by following that demon's target policy from the same state. That
comparison is the bold-vs-thin line in Figures 2 and 3. This captures difference
between off policy and on policy

Author: Vidush Jindal(jindalv)
"""

import numpy as np

from demons import DEMONS
from env import N_ACTIONS, Action, Robot
from horde import Horde

STEPS, STICK, SEED = 150_000, 0.8, 0


def evaluate(
    horde: Horde,
    visit: list[tuple[float, float, float, float, float, float]],
    seed: int,
    n: int = 25,
    horizon: int = 400,
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
        pred, actual = [], []
        for _ in range(n):
            # visited state sampled randomly from all of them
            st = visit[rng.integers(len(visit))]
            sim.set_state(st)
            o = sim.obs()
            if d["gamma"](o) == 0.0:
                continue  # the last state before episode terminated

            # which action had max probability according to stochastic target policy
            a = int(np.argmax(d["pi"](o, horde.q(o)[i])))
            # append the state action value integer as prediction
            pred.append(horde.q(o)[i][a])

            sim.set_state(st)  # roll out target policy
            o, G, disc = (
                sim.obs(),
                0.0,
                1.0,
            )  # G is return, we will use it for monte carlo return
            for _ in range(horizon):
                b = int(np.argmax(d["pi"](o, horde.q(o)[i])))
                o = sim.step(b)
                G += disc * d["r"](o)
                g = d["gamma"](o)
                if g == 0.0:
                    G += disc * d["z"](o)
                    break
                disc *= g
                if disc < 1e-3:
                    break
            actual.append(G)
        pred, actual = np.array(pred), np.array(actual)
        rows.append(
            (d["name"], pred.mean(), actual.mean(), np.abs(pred - actual).mean())
        )
    return rows


rng = np.random.default_rng(SEED)
env = Robot(seed=SEED)
horde = Horde(DEMONS)
obs, prev = env.obs(), Action.STOP
visited = []  # states the behaviour actually reached

for t in range(1, STEPS + 1):
    if t == 100_000:  # change the "floor surface" mid-run
        env.friction = 0.04
        print("\nThe floor was changed: friction 0.10 -> 0.04 (slippery)")

    # out behaviour policy is biased towards repeating the current taken action
    p = np.full(N_ACTIONS, (1 - STICK) / N_ACTIONS)
    p[prev] += STICK
    p /= p.sum()
    a = Action(rng.choice(N_ACTIONS, p=p))
    mu = float(p[a])
    prev = a

    obs2 = env.step(a)  # the next sensor readings
    horde.update(obs, a, obs2, mu)  # All demons learn from this one step
    obs = obs2
    if t % 30 == 0:
        visited.append(env.get_state())

    if t % 50_000 == 0:
        print(f"\nstep {t:,}   (friction={env.friction})")
        print(f"{'demon':<20}{'predicted':>11}{'actual':>10}{'MAE':>9}")
        for name, pr, ac, mae in evaluate(horde, visited, SEED + t):
            print(f"{name:<20}{pr:>11.2f}{ac:>10.2f}{mae:>9.2f}")
