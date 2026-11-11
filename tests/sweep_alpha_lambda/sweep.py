"""
Parameter sweep for one control demon (alpha x lambda).

The behaviour policy does not depend on what the demons learn, so many copies of
the same demon with different (alpha, lambda) can learn side by side in ONE Horde
from ONE behaviour stream: every setting sees exactly the same experience.

usage (from this folder):  PYTHONPATH=.. python3 -u sweep.py seek 0
    first argument: start of the demon's name in demons.py, second: seed
"""

import sys

import numpy as np

from demons import DEMONS
from env import BUMP, LIGHT, N_ACTIONS, Action, Robot
from horde import Horde

STEPS, STICK, EVAL_EVERY = 1_000_000, 0.8, 250_000  # STICK same as main.py
ALPHAS, LAMS = (0.01, 0.03, 0.1, 0.3), (0.0, 0.3, 0.6, 0.9)
N_START, HORIZON = 100, 250

name, seed = sys.argv[1], int(sys.argv[2])
base = next(d for d in DEMONS if d["name"].startswith(name))
grid = [(al, lm) for al in ALPHAS for lm in LAMS]
horde = Horde([dict(base, alpha=al, lam=lm) for al, lm in grid])
print(f"sweep {base['name']} seed {seed}: {len(grid)} copies, beta {base['beta']}, STICK {STICK}")

rng = np.random.default_rng(seed)
env = Robot(seed=seed)
obs, prev, visited, starts = env.obs(), Action.STOP, [], None
results = []  # per eval: (copies x [reward/step, reached %, bumps per 1k]) + random row

for t in range(1, STEPS + 1):
    p = np.full(N_ACTIONS, (1 - STICK) / N_ACTIONS)  # same sticky behaviour as main.py
    p[prev] += STICK
    a = Action(rng.choice(N_ACTIONS, p=p))
    obs2 = env.step(a)
    horde.update(obs, a, obs2, float(p[a]))
    obs, prev = obs2, a
    if t % 30 == 0:
        visited.append(env.get_state())

    if t % EVAL_EVERY == 0:
        if starts is None:  # same starts at every eval, so evals are comparable over time
            pick = np.random.default_rng(seed + 99).integers(len(visited), size=N_START)
            starts = [visited[k] for k in pick]
        rows = []
        for i in list(range(len(grid))) + [None]:  # None = random policy baseline
            sim, act_rng = Robot(seed=seed + 1), np.random.default_rng(seed + 2)
            total = reached = bumps = 0.0
            for st in starts:
                sim.set_state(st)
                o, near = sim.obs(), False
                for _ in range(HORIZON):
                    if i is None:
                        o = sim.step(Action(act_rng.integers(N_ACTIONS)))
                    else:
                        o = sim.step(Action(np.argmax(horde.q(o)[i])))
                    total += base["r"](o)
                    bumps += o[BUMP]
                    near |= np.hypot(sim.x - LIGHT[0], sim.y - LIGHT[1]) < 0.5
                reached += near
            n = N_START * HORIZON
            rows.append([total / n, 100 * reached / N_START, 1000 * bumps / n])
        rows = np.array(rows)
        results.append(rows)
        print(f"\nstep {t:,}  (random: {rows[-1, 0]:.3f}/step, reached {rows[-1, 1]:.0f}%, bumps {rows[-1, 2]:.1f}/1k)")
        print(f"  {'alpha':>6}{'lam':>5}{'reward/step':>13}{'x random':>10}{'reached %':>11}{'bumps/1k':>10}{'max|theta|':>12}")
        for i, (al, lm) in enumerate(grid):
            r, rc, b = rows[i]
            print(f"  {al:>6}{lm:>5}{r:>13.3f}{r / rows[-1, 0]:>9.2f}x{rc:>11.0f}{b:>10.1f}{np.abs(horde.theta[i]).max():>12.2f}")
        sys.stdout.flush()
        np.savez(f"sweep_{name}_{seed}.npz", results=np.array(results), grid=np.array(grid))
