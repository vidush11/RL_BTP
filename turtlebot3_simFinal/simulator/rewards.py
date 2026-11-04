"""Reward model.

Provenance: ``dqn_environment.py`` is a ROS node, so its reward helpers cannot
be imported without ROS.  The two pure functions below are *line-for-line ports*
of ``RLEnvironment.compute_directional_weights`` and
``RLEnvironment.compute_weighted_obstacle_reward`` (``tests/test_reward_parity.py``
extracts the originals from your file with ``ast`` and checks the ports against
them numerically), and ``goal_angle`` ports the angle logic of
``odom_sub_callback``.

How rewards work
----------------
Per step (``RewardModel.compute``)::

    r = per_step
        + yaw_coef      * (1 - 2*|goal_angle| / pi)      # goal_navigation only
        + obstacle_coef * weighted_obstacle_reward(...)  # front-sector term
        + progress_coef * (previous_goal_dist - goal_dist)  # goal_navigation only

On a terminal event the value is added (``terminal_mode: add``) or replaces the
step reward (``terminal_mode: replace``, what the commented-out Gazebo code did):

    collision -> ``collision_penalty``      goal reached -> ``goal_reward``

A collision takes priority over a goal on the same step.

Defaults (``obstacle_avoidance``) reproduce your live Gazebo code exactly:
``per_step = 1.0``, no other term, so the reward is 1.0 on every step including
the collision step and the episode return equals the episode length.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np

from .config import RewardConfig


def compute_directional_weights(relative_angles, max_weight: float = 10.0):
    """Port of ``RLEnvironment.compute_directional_weights``."""
    power = 6
    raw_weights = (np.cos(relative_angles)) ** power + 0.1
    scaled_weights = raw_weights * (max_weight / np.max(raw_weights))
    normalized_weights = scaled_weights / np.sum(scaled_weights)
    return normalized_weights


def compute_weighted_obstacle_reward(front_ranges, front_angles) -> float:
    """Port of ``RLEnvironment.compute_weighted_obstacle_reward`` (ranges in metres,
    angles in ``[0, 2*pi)`` as in a ROS scan)."""
    if len(front_ranges) == 0 or len(front_angles) == 0:
        return 0.0

    front_ranges = np.array(front_ranges)
    front_angles = np.array(front_angles)

    valid_mask = front_ranges <= 0.5
    if not np.any(valid_mask):
        return 0.0

    front_ranges = front_ranges[valid_mask]
    front_angles = front_angles[valid_mask]

    relative_angles = np.unwrap(front_angles)
    relative_angles[relative_angles > np.pi] -= 2 * np.pi

    weights = compute_directional_weights(relative_angles, max_weight=10.0)

    safe_dists = np.clip(front_ranges - 0.25, 1e-2, 3.5)
    decay = np.exp(-3.0 * safe_dists)

    weighted_decay = np.dot(weights, decay)

    return float(-(1.0 + 4.0 * weighted_decay))


def goal_angle(robot_x: float, robot_y: float, robot_theta: float,
               goal_x: float, goal_y: float) -> float:
    """Port of the goal-angle computation in ``odom_sub_callback``."""
    path_theta = math.atan2(goal_y - robot_y, goal_x - robot_x)
    angle = path_theta - robot_theta
    if angle > math.pi:
        angle -= 2 * math.pi
    elif angle < -math.pi:
        angle += 2 * math.pi
    return angle


class RewardModel:
    def __init__(self, cfg: RewardConfig, beam_angles_ros: np.ndarray):
        self.cfg = cfg
        a = np.asarray(beam_angles_ros, dtype=np.float64)
        # Same "front" sector as the existing code: [0, pi/2] U [3pi/2, 2pi].
        self._front_mask = ((a >= 0.0) & (a <= math.pi / 2)) | \
                           ((a >= 3 * math.pi / 2) & (a <= 2 * math.pi))
        self._front_angles = a[self._front_mask]

    def compute(self, ranges_m: np.ndarray, collision: bool, goal_reached: bool,
                goal_angle_rad: Optional[float] = None,
                progress: Optional[float] = None, speed: Optional[float] = None) -> float:
        c = self.cfg
        reward = c.per_step
        if c.speed_coef and speed is not None:          # opt-in safety-task reward
            reward += c.speed_coef * min(max(speed, 0.0), c.v_target) / c.v_target
        if c.yaw_coef and goal_angle_rad is not None:
            reward += c.yaw_coef * (1.0 - 2.0 * abs(goal_angle_rad) / math.pi)
        if c.obstacle_coef:
            reward += c.obstacle_coef * compute_weighted_obstacle_reward(
                ranges_m[self._front_mask], self._front_angles)
        if c.progress_coef and progress is not None:
            reward += c.progress_coef * progress

        terminal = None
        if collision:
            terminal = c.collision_penalty
        elif goal_reached:
            terminal = c.goal_reward
        if terminal is not None:
            reward = terminal if c.terminal_mode == "replace" else reward + terminal
        return float(reward)
