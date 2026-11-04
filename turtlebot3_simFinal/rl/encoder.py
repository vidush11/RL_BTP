"""State encoder: builds the DQN input from the env observation.

base observation (same for A, B, C -- fairness):
    normalised LiDAR (24)  [+ goal features (3): dist/diag, sin(angle), cos(angle)]
The goal features are added for a goal-navigation task because the raw env
observation is LiDAR-only (the goal is hidden in ``info``); they are given
IDENTICALLY to all three agents so the comparison stays fair (brief sec 8).

full DQN input:
    Agent A: base
    Agent B/C: concatenate(base, gvf_features(base_lidar))   (3 extra features)
"""
from __future__ import annotations

from typing import Optional

import numpy as np

LID = 24


class StateEncoder:
    def __init__(self, use_goal: bool, goal_dist_scale: float = 5.657):
        # 5.657 ~= diagonal of the 4x4 arena (max goal distance).
        self.use_goal = bool(use_goal)
        self.goal_dist_scale = float(goal_dist_scale)
        self.base_dim = LID + (3 if use_goal else 0)

    def base_obs(self, obs: np.ndarray, info: dict) -> np.ndarray:
        """Raw base observation stored in the replay buffer."""
        lidar = np.asarray(obs, dtype=np.float32).ravel()[:LID]
        if not self.use_goal:
            return lidar
        gdist = float(info.get("goal_distance", 0.0)) / self.goal_dist_scale
        gang = float(info.get("goal_angle", 0.0))
        return np.concatenate([lidar, [min(gdist, 1.0), np.sin(gang), np.cos(gang)]]).astype(np.float32)

    def lidar_of(self, base: np.ndarray) -> np.ndarray:
        """The LiDAR slice of a base observation (GVF predictor input)."""
        return np.asarray(base)[..., :LID]

    def full_dim(self, n_gvf_features: int) -> int:
        return self.base_dim + int(n_gvf_features)
