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
# round pillars (x, y, radius); the first one sits between the centre and the light
OBSTACLES = np.array([[0.95, 2.05, 0.25], [2.1, 0.95, 0.3]])

# observation layout -- index these constants, never raw numbers
IR_F, IR_L, IR_R, IR_B = range(4)  # infrared: front, left, right, back
LIGHT_FL, LIGHT_FR, LIGHT_BL, LIGHT_BR = range(4, 8)  # light: the four diagonals
VEL, ANG_VEL, BUMP, COMPASS_X, COMPASS_Y = range(8, 13)
N_SENSORS = 13


class Robot:
    """Simulates a robot"""

    DT = 0.5  # 500 ms step, as in the paper's light-seeking experiment

    # friction is per step: 0.41 per 0.5 s step = same slowing as 0.10 per 0.1 s (1 - 0.9**5)
    def __init__(self, seed: int = 0, friction: float = 0.41):
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
            # ray-circle intersection: solve |p + t*d - c| = r for the nearest t > 0
            for cx, cy, r in OBSTACLES:
                fx, fy = self.x - cx, self.y - cy
                b = fx * dx + fy * dy
                disc = b * b - (fx * fx + fy * fy - r * r)
                if disc >= 0 and -b - np.sqrt(disc) > 0:
                    t = min(t, -b - np.sqrt(disc))
            return float(np.exp(-t / 0.6))
            # t is always positive

        d = LIGHT - np.array([self.x, self.y])
        dist = float(np.linalg.norm(d))

        u = d / (dist + 1e-9)  # unit vector towards the light
        brightness = 1.0 / (1.0 + 0.5 * dist**2)  # falls off with distance
        angles = np.array(
            [np.pi / 4, -np.pi / 4, 3 * np.pi / 4, -3 * np.pi / 4]
        )  # FL, FR, BL, BR

        facing = [
            max(0.0, float(np.cos(self.th + a) * u[0] + np.sin(self.th + a) * u[1]))
            for a in angles
        ]

        # a pillar blocks if the line passes through it and it lies between robot and light
        c = OBSTACLES[:, :2] - [self.x, self.y]  # pillar centres relative to the robot
        s = (
            c @ u
        )  # distance along the robot->light line to the point closest to each pillar
        perp = np.linalg.norm(
            c - s[:, None] * u, axis=1
        )  # distance from each pillar centre to the line

        brightness *= float(not np.any((perp < OBSTACLES[:, 2]) & (s > 0) & (s < dist)))
        # IR sensors have a beam, not a single ray (a real IR LED lights a cone): each
        # reading is a weighted average of 5 rays across +-40 degrees, strongest in the
        # middle.
        beam = np.radians([-40.0, -20.0, 0.0, 20.0, 40.0])
        w = np.cos(beam) / np.cos(beam).sum()
        o = np.array(
            [
                w @ [prox(k) for k in beam],  # IR front
                w @ [prox(np.pi / 2 + k) for k in beam],  # IR left
                w @ [prox(-np.pi / 2 + k) for k in beam],  # IR right
                w @ [prox(np.pi + k) for k in beam],  # IR back
                facing[0] * brightness,  # light front-left
                facing[1] * brightness,  # light front-right
                facing[2] * brightness,  # light back-left
                facing[3] * brightness,  # light back-right
                self.v,
                self.w,
                self.bump,  # has the robot bumped into a wall or pillar
                np.cos(self.th),
                np.sin(self.th),
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
        if not (m <= nx <= ARENA - m and m <= ny <= ARENA - m):
            self.bump, self.v = 1.0, 0.0  # hit a wall, keep going
            nx, ny = np.clip(nx, m, ARENA - m), np.clip(ny, m, ARENA - m)
        if np.any(
            np.hypot(nx - OBSTACLES[:, 0], ny - OBSTACLES[:, 1]) < OBSTACLES[:, 2] + m
        ):
            self.bump, self.v = 1.0, 0.0  # hit a pillar: blocked, stay where we were
            nx, ny = self.x, self.y
        self.x, self.y = float(nx), float(ny)
        return self.obs()
