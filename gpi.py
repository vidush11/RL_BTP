"""
Implements the research paper: Successor Features for Transfer in Reinforcement Learning
GPI over various policies: we combine returns from an policy to immediately get q(s,a) values for that policy on new combine rewards without any new training
Then take argmax over all policies.
i.e GPI action = the action with the highest task value under any of the policies

Author: Vidush Jindal (jindalv)
"""

import logging

import numpy as np

from demons import DEMONS
from env import N_ACTIONS, Action, Robot
from horde import Horde
from main import CTRL_STEPS, EVAL_STARTS

SEEK, SPIN, AVOID = 4, 5, 6  # control demons in demons.py
REWARDS = ["light", "spin", "avoid", "velocity", "bump"]
# demon index: row = policy followed (seek, spin, avoid), column = reward (as in REWARDS)
VALUE = np.array(
    [
        [4, 7, 8, 9, 10],
        [11, 5, 12, 13, 14],
        [15, 16, 6, 17, 18],
    ]
)
TASKS = {  # weight per reward, in REWARDS order
    "seek + 0.25 avoid": [1, 0, 0.25, 0, 0],
    "seek, never bump": [1, 0, 0, 0, -1],
    "avoid + spin": [0, 1, 1, 0, 0],
    "drive forward, no bumps": [0, 0, 0, 1, -1],
}
POLICIES = ["random", "seek alone", "spin alone", "avoid alone", "local demons", "GPI"]

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[logging.FileHandler("gpi_log.txt", mode="w"), logging.StreamHandler()],
)
log = logging.info

checkpoint = np.load("checkpoint.npz")
horde = Horde(DEMONS)
horde.theta = checkpoint["theta"]
# demons for any polocy would have same reward functions hence 0
reward_fns = [DEMONS[k]["r"] for k in VALUE[0]]  # in REWARDS order
log(
    f"checkpoint step {int(checkpoint['t']):,} | {len(EVAL_STARTS)} starts x {CTRL_STEPS} steps"
)

for task, w in TASKS.items():
    w = np.array(w, dtype=float)
    log(f"\nTASK {task}: weights {dict(zip(REWARDS, w.tolist()))}")
    log(
        f"  {'policy':<14}{'task':>8}"
        + "".join(f"{r:>10}" for r in REWARDS)
        + "   (per step)"
    )

    for policy in POLICIES:
        # same action and sensor noise for every policy
        rng, sim = np.random.default_rng(0), Robot(seed=0)
        reward_sum = np.zeros(len(REWARDS))
        task_sum = 0.0
        for start in EVAL_STARTS:
            sim.set_state(start)
            obs = sim.obs()
            for _ in range(CTRL_STEPS):
                q = horde.q(obs)  # value of every action for every demon
                if policy == "random":
                    a = rng.integers(N_ACTIONS)
                elif policy == "seek alone":
                    a = np.argmax(q[SEEK])
                elif policy == "spin alone":
                    a = np.argmax(q[SPIN])
                elif policy == "avoid alone":
                    a = np.argmax(q[AVOID])
                # old way: add the control demons' own values
                elif policy == "local demons":
                    a = np.argmax(w[0] * q[SEEK] + w[1] * q[SPIN] + w[2] * q[AVOID])
                # GPI: each policy's value on the task (policies x actions), best of all
                else:
                    # each policy's rewards weighed by current task
                    task_value = (q[VALUE] * w[None, :, None]).sum(1)
                    # task_value-> policies X actions
                    a = np.argmax(task_value.max(0))
                obs = sim.step(Action(int(a)))
                # each of the 5 rewards this step, and what this action earned on the task
                rewards = np.array([r(obs) for r in reward_fns])
                reward_sum += rewards
                task_sum += w @ rewards

        n_steps = len(EVAL_STARTS) * CTRL_STEPS
        per_step = reward_sum / n_steps
        log(
            f"  {policy:<14}{task_sum / n_steps:>8.3f}"
            + "".join(f"{v:>10.3f}" for v in per_step)
        )
