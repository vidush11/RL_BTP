"""
The questions. Each demon is just four functions (pi, gamma, rewards, terminal reward fn).

This is the essence of the paper: the learning machinery does not change, only
these four functions.
We can easily learn a new question or policy by adding a dictionary here

Author: Vidush Jindal(jindalv)
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from env import ANG_VEL, BUMP, IR_F, LIGHT_S, N_ACTIONS, VEL, Action


def fixed(a: Action) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    """
     a prediction demon: always the same action

    :param a: _description_
    :type a: Action
    :return: _description_
    :rtype: Callable[[np.ndarray, np.ndarray], np.ndarray]
    """

    p = np.zeros(N_ACTIONS)
    p[a] = 1.0
    return lambda obs, q: p


def greedy(obs: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Returns a greedy policy pi wrt state action values

    :param obs: observation of robot
    :type obs: np.ndarray
    :param q: the state action values
    :type q: np.ndarray
    :return: one hot vector with one at action with max state action value
    :rtype: np.ndarray
    """
    p = np.zeros(N_ACTIONS)
    p[int(np.argmax(q))] = 1.0
    return p


ZERO = lambda obs: 0.0

DEMONS = [
    # Prediction Demon: How many steps until I hit something, if I just keep driving forward?
    # r=1 per step and gamma=0 at the wall makes the return a step count.
    dict(
        name="steps-to-obstacle",
        pi=fixed(Action.FORWARD),
        r=lambda obs: 1.0,
        z=ZERO,
        # if the robot has collided then gamma should be zero that is no further rewards
        gamma=lambda obs: 0.0 if obs[IR_F] > 0.6 else 1.0,
        lam=0.4,
        alpha=0.3,
        beta=0.001,
    ),
    # Prediction Demon: How many steps do I need to come to a stop?
    # Change env.friction mid-run and watch this one adapt (Figure 3).
    dict(
        name="steps-to-stop",
        pi=fixed(Action.STOP),
        r=lambda obs: 1.0,
        z=ZERO,
        gamma=lambda obs: 0.0 if abs(obs[VEL]) < 0.03 else 1.0,
        lam=0.4,
        alpha=0.3,
        beta=0.001,
    ),
    # Prediction Demon: How much light will I accumulate if I drive forward?  A discounted
    # sensor prediction -- the "nexting" flavour of GVF.
    dict(
        name="light-if-forward",
        pi=fixed(Action.FORWARD),
        r=lambda obs: obs[LIGHT_S],
        z=ZERO,
        gamma=lambda o: 0.9,
        lam=0.4,
        alpha=0.2,
        beta=0.001,
    ),
    # Prediction Demon: Am I about to bump? If I keep running forward.
    # A discounted probability of collision.
    dict(
        name="bump-soon",
        pi=fixed(Action.FORWARD),
        r=lambda obs: obs[BUMP],
        z=ZERO,
        gamma=lambda o: 0.8,
        lam=0.4,
        alpha=0.2,
        beta=0.001,
    ),
    # Control demon: learn a policy that maximises the light sensor.
    dict(
        name="seek-light [ctrl]",
        pi=greedy,
        r=lambda obs: obs[LIGHT_S],
        z=ZERO,
        gamma=lambda o: 0.95,
        lam=0.3,
        alpha=0.1,
        beta=0.01,
    ),
    # Control demon: maximise angular velocity, i.e. learn to spin (Section 5.2).
    dict(
        name="spin [ctrl]",
        pi=greedy,
        r=lambda o: abs(o[ANG_VEL]) / 3.0,
        z=ZERO,
        gamma=lambda o: 0.95,
        lam=0.3,
        alpha=0.1,
        beta=0.01,
    ),
    # Control demon: back away from front obstacles, maximise open space in front.
    dict(
        name="avoid-front-obstacles [ctrl]",
        pi=greedy,
        r=lambda o: 1.0 - o[IR_F],
        z=ZERO,
        gamma=lambda o: 0.95,
        lam=0.3,
        alpha=0.1,
        beta=0.01,
    ),
]
