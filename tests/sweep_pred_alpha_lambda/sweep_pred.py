"""
Parameter sweep for one prediction demon (alpha x lambda), seed 0 only.

Like tests/sweep_alpha_lambda/sweep.py: every copy learns side by side in ONE Horde from ONE
sticky-random stream, so every setting sees exactly the same experience. The target policy
is a fixed action, so each state's true return does not change with learning: it is computed
once (mean of N_REP rollouts) and every eval scores mean |prediction - return| on the same states.

usage (from a folder inside the code dir):  PYTHONPATH=.. python3 -u sweep_pred.py steps-to-obstacle
"""

import sys

import numpy as np

from demons import DEMONS
from env import N_ACTIONS, Action, Robot
from horde import Horde

STEPS, STICK, EVAL_EVERY, SEED = 1_000_000, 0.8, 100_000, 0  # STICK same as main.py
ALPHAS, LAMS = (0.03, 0.1, 0.2, 0.3, 1.0), (0.0, 0.4, 0.7, 0.9)
N_START, N_REP, HORIZON = 100, 5, 400  # HORIZON same as main.evaluate

name = sys.argv[1]
base = next(d for d in DEMONS if d["name"] == name)
grid = [(al, lm) for al in ALPHAS for lm in LAMS]
horde = Horde([dict(base, alpha=al, lam=lm) for al, lm in grid])
print(f"sweep {name}: {len(grid)} copies, beta {base['beta']}, STICK {STICK}, seed {SEED}")

rng, env = np.random.default_rng(SEED), Robot(seed=SEED)
obs, prev, visited, starts = env.obs(), Action.STOP, [], None
results = []  # per eval: copies x [mean |error|, mean (prediction - return)]

for t in range(1, STEPS + 1):
    p = np.full(N_ACTIONS, (1 - STICK) / N_ACTIONS)  # same sticky behaviour as main.py
    p[prev] += STICK
    a = Action(rng.choice(N_ACTIONS, p=p))
    obs2 = env.step(a)
    horde.update(obs, a, obs2, float(p[a]))
    obs, prev = obs2, a
    if t % 30 == 0:
        visited.append(env.get_state())

    if t % EVAL_EVERY:
        continue
    if starts is None:  # fixed states and their true returns, picked at the first eval
        sim, starts, G = Robot(seed=SEED + 1), [], []
        for k in np.random.default_rng(SEED + 99).permutation(len(visited)):
            sim.set_state(visited[k])
            o = sim.obs()
            if base["gamma"](o) == 0.0:
                continue  # already terminal: nothing to predict
            b = Action(int(np.argmax(base["pi"](o, np.zeros(N_ACTIONS)))))  # the fixed action
            g_rep = []
            for _ in range(N_REP):
                sim.set_state(visited[k])
                ret, disc = 0.0, 1.0
                for _ in range(HORIZON):
                    o2 = sim.step(b)
                    ret += disc * base["r"](o2)
                    if base["gamma"](o2) == 0.0:
                        ret += disc * base["z"](o2)
                        break
                    disc *= base["gamma"](o2)
                    if disc < 1e-3:
                        break
                g_rep.append(ret)
            starts.append((o, b))
            G.append(np.mean(g_rep))
            if len(starts) == N_START:
                break
        G = np.array(G)
        print(f"{len(starts)} states, true return mean {G.mean():.2f}, sd {G.std():.2f}")

    pred = np.array([horde.q(o)[:, b] for o, b in starts]).T  # copies x states
    rows = np.stack([np.abs(pred - G).mean(1), (pred - G).mean(1)], 1)
    results.append(rows)
    print(f"\nstep {t:,}")
    print(f"  {'alpha':>6}{'lam':>5}{'mean |err|':>12}{'bias':>8}{'max|theta|':>12}")
    for i, (al, lm) in enumerate(grid):
        print(f"  {al:>6}{lm:>5}{rows[i, 0]:>12.3f}{rows[i, 1]:>8.2f}{np.abs(horde.theta[i]).max():>12.3g}")
    sys.stdout.flush()
    np.savez(f"sweep_pred_{name}.npz", results=np.array(results), grid=np.array(grid), G=G)
