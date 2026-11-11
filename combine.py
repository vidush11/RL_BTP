"""
Combining two control demons with NO new learning.

seek-light knows how to collect light, avoid-obstacles knows how to keep
space ahead and behind. We act greedily on a weighted sum of their action values,
    a = argmax_a  q_seek(s, a) + w * q_avoid(s, a)
and score every policy on the SAME cumulants the two demons were trained on
(seek-light's r and avoid-obstacles' r from demons.py), plus bumps and reaching the light.
w = 0 is seek-light alone; larger w cares more about free space.

"""

import logging

import numpy as np

from demons import DEMONS
from env import BUMP, LIGHT, N_ACTIONS, OBSTACLES, Action, Robot
from horde import Horde

CHK, LOG = "checkpoint.npz", "combine_log.txt"
N_STARTS, STEPS, SEED = 100, 250, 0
WEIGHTS = [0.0, 0.25, 0.4, 0.5, 0.6, 0.75, 1.0]  # finer around the sweet spot (0.5 at 3M)

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[logging.FileHandler(LOG, mode="w"), logging.StreamHandler()],
)
log = logging.info

# trained weights of all demons; we only use two of them
ck = np.load(CHK)
horde = Horde(DEMONS)
horde.theta = ck["theta"]
names = [d["name"] for d in DEMONS]
seek, avoid = names.index("seek-light [ctrl]"), names.index("avoid-obstacles [ctrl]")

# the same 100 start states (where the robot actually went) for every policy
rng = np.random.default_rng(SEED)
starts = ck["visited"][rng.choice(len(ck["visited"]), N_STARTS, replace=False)]

# (label, weight on q_seek, weight on q_avoid); random ignores the weights
policies = [("random", 0, 0)] + [(f"seek + {w}*avoid", 1, w) for w in WEIGHTS]
policies += [("avoid only", 0, 1)]

log(f"checkpoint step {int(ck['t']):,} | {N_STARTS} starts x {STEPS} steps each")
log(
    f"  {'policy':<20}{'seek r/step':>12}{'avoid r/step':>13}{'bumps/1k':>10}{'pillar/1k':>11}"
    f"{'final dist':>12}{'ever <0.5m':>11}{'ended <0.5m':>12}"
)
for label, ws, wa in policies:
    light = clear = bumps = pillar = dist = reached = ever = 0.0
    sim = Robot(seed=SEED)  # same sensor noise for every policy
    for st in starts:
        sim.set_state(tuple(st))
        o, near = sim.obs(), False
        for _ in range(STEPS):
            q = horde.q(o)  # (demons x actions)
            a = rng.integers(N_ACTIONS) if label == "random" else np.argmax(ws * q[seek] + wa * q[avoid])
            o = sim.step(Action(int(a)))

            light += DEMONS[seek]["r"](o)  # front light: what seek-light is trained on
            clear += DEMONS[avoid]["r"](o)  # 1 - max IR - bump: what avoid is trained on
            near |= np.hypot(*(LIGHT - [sim.x, sim.y])) < 0.5
            bumps += o[BUMP]
            # pillars are >0.7 m from walls, so a bump this close to a pillar is a pillar hit
            gap_to_pillar = np.hypot(sim.x - OBSTACLES[:, 0], sim.y - OBSTACLES[:, 1]) - OBSTACLES[:, 2]
            pillar += o[BUMP] * (gap_to_pillar.min() < 0.2)

        final_dist = np.hypot(*(LIGHT - [sim.x, sim.y]))
        dist += final_dist
        reached += final_dist < 0.5  # ended the run within 0.5 m of the light
        ever += near  # got within 0.5 m of the light at some point

    n = N_STARTS * STEPS
    log(
        f"  {label:<20}{light / n:>12.3f}{clear / n:>13.3f}{1000 * bumps / n:>10.1f}{1000 * pillar / n:>11.1f}"
        f"{dist / N_STARTS:>11.2f}m{100 * ever / N_STARTS:>10.0f}%{100 * reached / N_STARTS:>11.0f}%"
    )
