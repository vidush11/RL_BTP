"""Drive the robot with one behaviour policy; every demon learns off-policy.

Evaluation compares each demon's prediction q-hat against the ACTUAL return
obtained by following that demon's target policy from the same state. That
comparison is the bold-vs-thin line in Figures 2 and 3, and it is the only
thing that will catch off-policy divergence.
"""

import numpy as np
from env import Robot, N_ACTIONS, STOP, ARENA
from horde import Horde
from demons import DEMONS

STEPS, STICK, SEED = 150_000, 0.8, 0


def evaluate(horde, visited, seed, n=25, horizon=400):
    """For each demon: mean prediction vs mean Monte-Carlo return.

    States are sampled from what the BEHAVIOUR policy actually visited. A value
    function is only meaningful where the behaviour gave coverage, so testing on
    uniformly random states measures extrapolation, not learning.
    """
    rng = np.random.default_rng(seed)
    sim = Robot(seed=seed + 1)
    rows = []
    for i, d in enumerate(DEMONS):
        pred, actual = [], []
        for _ in range(n):
            st = visited[rng.integers(len(visited))]
            sim.set_state(st)
            o = sim.obs()
            if d["gamma"](o) == 0.0:
                continue  # already terminated
            a = int(np.argmax(d["pi"](o, horde.q(o)[i])))
            pred.append(horde.q(o)[i][a])

            sim.set_state(st)  # roll out target policy
            o, G, disc = sim.obs(), 0.0, 1.0
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
obs, prev = env.obs(), STOP
visited = []  # states the behaviour actually reached

for t in range(1, STEPS + 1):
    if t == 100_000:  # change the "floor surface" mid-run
        env.friction = 0.04
        print("\n--- floor changed: friction 0.10 -> 0.04 (slippery) ---")

    # sticky-random behaviour policy: uniform, biased toward repeating
    p = np.full(N_ACTIONS, (1 - STICK) / N_ACTIONS)
    p[prev] += STICK
    p /= p.sum()
    a = int(rng.choice(N_ACTIONS, p=p))
    mu = float(p[a])
    prev = a

    obs2 = env.step(a)
    horde.update(obs, a, obs2, mu)  # ALL demons learn from this one step
    obs = obs2
    if t % 37 == 0:
        visited.append(env.get_state())

    if t % 50_000 == 0:
        print(f"\nstep {t:,}   (friction={env.friction})")
        print(f"{'demon':<20}{'predicted':>11}{'actual':>10}{'MAE':>9}")
        for name, pr, ac, mae in evaluate(horde, visited, SEED + t):
            print(f"{name:<20}{pr:>11.2f}{ac:>10.2f}{mae:>9.2f}")
