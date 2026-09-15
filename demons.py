"""The questions. Each demon is just four functions (pi, gamma, r, z).

This is the whole point of the paper: the learning machinery never changes, only
these four functions. Adding a new piece of knowledge means adding a dict here.
"""
import numpy as np
from env import (FORWARD, LEFT, RIGHT, STOP, N_ACTIONS,
                 IR_F, IR_L, IR_R, LIGHT_S, VEL, ANG_VEL, BUMP, COMPASS)


def fixed(a):                       # a prediction demon: always the same action
    p = np.zeros(N_ACTIONS); p[a] = 1.0
    return lambda obs, q, p=p: p


def greedy(obs, q):                 # a control demon: pi = greedy(q-hat)
    p = np.zeros(N_ACTIONS); p[int(np.argmax(q))] = 1.0
    return p


ZERO = lambda o: 0.0

DEMONS = [
    # "How many steps until I hit something, if I just keep driving forward?"
    # r=1 per step and gamma=0 at the wall makes the return a step count.
    dict(name="steps-to-obstacle", pi=fixed(FORWARD), r=lambda o: 1.0, z=ZERO,
         gamma=lambda o: 0.0 if o[IR_F] > 0.6 else 1.0,
         lam=0.4, alpha=0.3, beta=0.001),

    # "How many steps do I need to come to a stop?"  Same shape, different sensor.
    # Change env.friction mid-run and watch this one adapt (Figure 3).
    dict(name="steps-to-stop", pi=fixed(STOP), r=lambda o: 1.0, z=ZERO,
         gamma=lambda o: 0.0 if abs(o[VEL]) < 0.03 else 1.0,
         lam=0.4, alpha=0.3, beta=0.001),

    # "How much light will I accumulate if I drive forward?"  A discounted
    # sensor prediction -- the "nexting" flavour of GVF.
    dict(name="light-if-forward", pi=fixed(FORWARD), r=lambda o: o[LIGHT_S], z=ZERO,
         gamma=lambda o: 0.9, lam=0.4, alpha=0.2, beta=0.001),

    # "Am I about to bump?"  Cumulant is a binary event, so the value is roughly
    # a discounted probability of collision.
    dict(name="bump-soon", pi=fixed(FORWARD), r=lambda o: o[BUMP], z=ZERO,
         gamma=lambda o: 0.8, lam=0.4, alpha=0.2, beta=0.001),

    # Control demon: learn a policy that maximises the light sensor (Section 5.3).
    dict(name="seek-light [ctrl]", pi=greedy, r=lambda o: o[LIGHT_S], z=ZERO,
         gamma=lambda o: 0.95, lam=0.3, alpha=0.1, beta=0.01),

    # Control demon: maximise |angular velocity|, i.e. learn to spin (Section 5.2).
    dict(name="spin [ctrl]", pi=greedy, r=lambda o: abs(o[ANG_VEL]) / 3.0, z=ZERO,
         gamma=lambda o: 0.95, lam=0.3, alpha=0.1, beta=0.01),

    # Control demon: back away from walls, maximise open space in front.
    dict(name="avoid-wall [ctrl]", pi=greedy, r=lambda o: 1.0 - o[IR_F], z=ZERO,
         gamma=lambda o: 0.95, lam=0.3, alpha=0.1, beta=0.01),
]
