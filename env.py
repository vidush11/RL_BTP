"""
A simulated multi-sensor robot in a square arena.
Author: Vidush Jindal (jindalv)
"""

from __future__ import annotations

from enum import IntEnum

import numpy as np


class Action(IntEnum):
    """Actions that robot can take"""

    FORWARD = 0
    BACKWARD = 1
    LEFT = 2
    RIGHT = 3
    STOP = 4


N_ACTIONS = 5
ARENA = 3.0  # metres, square
LIGHT = np.array([0.4, 2.6])  # fixed light source in one corner

# observation layout -- index these constants, never raw numbers
IR_F, IR_L, IR_R, LIGHT_S, VEL, ANG_VEL, BUMP, COMPASS = range(8)
N_SENSORS = 8


class Robot:
    """Simulates a robot"""

    DT = 0.1  # 100 ms step, as in the paper

    def __init__(self, seed: int = 0, friction: float = 0.10):
        self.rng = np.random.default_rng(seed)
        self.friction = friction  # change this mid-run to switch "floor surface"
        self.x, self.y, self.th = ARENA / 2, ARENA / 2, 0.0
        self.v = self.w = self.bump = 0.0

    def get_state(self) -> tuple[float, float, float, float, float, float]:
        """Returns the state of robot

        :return: State of robot
        :rtype: tuple[float, float, float, float, float, float]
        """
        return (self.x, self.y, self.th, self.v, self.w, self.bump)

    def set_state(self, s: tuple[float, float, float, float, float, float]) -> None:
        """Sets the state to given state

        :param s: state to set to
        :type s: tuple[float, float, float, float, float, float]
        """
        self.x, self.y, self.th, self.v, self.w, self.bump = s

    def obs(self) -> np.ndarray:
        """
        Returns distance to the nearest wall along a ray, then squash into proximity (IR sensor)
        Real IR sensors return a large value when something is close and fall off nonlinearly.

        :return: the sensor data
        :rtype: np.ndarray
        """

        def prox(rel):
            dx, dy = np.cos(self.th + rel), np.sin(
                self.th + rel
            )  # dx and dy are unit vectors along self.th+rel direction
            t = np.inf
            if abs(dx) > 1e-9:
                t = min(t, ((ARENA - self.x) if dx > 0 else -self.x) / dx)
            if abs(dy) > 1e-9:
                t = min(t, ((ARENA - self.y) if dy > 0 else -self.y) / dy)
            return float(np.exp(-t / 0.6))
            # t is always positive

        d = LIGHT - np.array([self.x, self.y])
        dist = float(np.linalg.norm(d))
        # normalized dot product between robot's direction and light source
        # if the robot turns away then 0 irrespective of distance
        facing = max(
            0.0, float(np.cos(self.th) * d[0] + np.sin(self.th) * d[1]) / (dist + 1e-9)
        )
        o = np.array(
            [
                prox(0.0),  # sensor in robot's direction
                prox(1.0),  # sensor in robot's direction+ 1 rad
                prox(-1.0),  # sensor in robot's direction -1 rad
                facing / (1.0 + 0.5 * dist**2),  # light sensor
                self.v,
                self.w,
                self.bump,  # has the robot dumped into a wall
                np.cos(self.th),
            ]
        )
        o += self.rng.normal(0, 0.005, N_SENSORS)  # every sensor is noisy
        o[BUMP] = self.bump  # except the bump switch
        return o

    def step(self, a: Action) -> np.ndarray:
        """Steps the robot according to mentioned action

        :param a: one of the action from valid actions
        :type a: Action
        :return: the observation of robot
        :rtype: tuple[float, float, float, float, float, float]
        """
        if a == Action.FORWARD:
            self.v += 0.8 * self.DT
        elif a == Action.BACKWARD:
            self.v -= 0.8 * self.DT
        elif a == Action.LEFT:
            self.w += 3.0 * self.DT
        elif a == Action.RIGHT:
            self.w -= 3.0 * self.DT
        self.v *= 1 - self.friction  # STOP just coasts to a halt
        self.w *= 1 - self.friction
        self.v = float(np.clip(self.v, -0.6, 0.6))
        self.w = float(np.clip(self.w, -3.0, 3.0))

        self.th += self.w * self.DT
        nx = self.x + self.v * np.cos(self.th) * self.DT
        ny = self.y + self.v * np.sin(self.th) * self.DT
        m, self.bump = 0.12, 0.0
        if not (m < nx < ARENA - m and m < ny < ARENA - m):
            self.bump, self.v = 1.0, 0.0  # hit a wall, keep going
            nx, ny = np.clip(nx, m, ARENA - m), np.clip(ny, m, ARENA - m)
        self.x, self.y = float(nx), float(ny)
        return self.obs()
