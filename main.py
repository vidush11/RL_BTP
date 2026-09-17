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

import matplotlib.pyplot as plt
import numpy as np

from demons import DEMONS, greedy
from env import N_ACTIONS, Action, Robot
from horde import Horde
from viz import Live

STEPS, STICK, SEED, EVAL_EVERY = 150_0000, 0.8, 0, 25_000
VIZ = True


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
    live = Live() if VIZ else None

    for t in range(1, STEPS + 1):
        # if t == 100_000:  # change the "floor surface" mid-run
        #     env.friction = 0.04
        #     print("\nThe floor was changed: friction 0.10 -> 0.04 (slippery)")

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
            print(f"\nstep {t:,}   (friction={env.friction})")
            print(f"{'prediction demon':<24}{'predicted':>11}{'actual':>10}{'MAE':>9}")
            for name, kind, x, y, z in rows:
                if kind == "pred":
                    print(f"{name:<24}{x:>11.2f}{y:>10.2f}{z:>9.2f}")
            print(f"{'control demon':<24}{'learned':>11}{'random':>10}{'ratio':>9}")
            for name, kind, x, y, z in rows:
                if kind == "ctrl":
                    print(f"{name:<24}{x:>11.3f}{y:>10.3f}{z:>8.1f}x")

    if live:
        plt.ioff()
        plt.show()  # keep the window open at the end
