"""Gazebo-free, ROS-free, headless TurtleBot3 Burger LiDAR simulator."""

import gymnasium as _gym

from . import hpc
from .config import (GAZEBO_ACTION_TABLE, SPEC_ACTION_TABLE, ConfigError, SimConfig,
                     load_config)
from .environment import TurtleBot3LidarEnv
from .robot import InvalidActionError
from .vector import make_env_factory, make_vec_env

ENV_ID = "TurtleBot3Lidar-v0"
if ENV_ID not in _gym.registry:
    _gym.register(id=ENV_ID, entry_point="simulator.environment:TurtleBot3LidarEnv")

__all__ = [
    "ENV_ID", "TurtleBot3LidarEnv", "SimConfig", "ConfigError", "load_config",
    "InvalidActionError", "make_env_factory", "make_vec_env",
    "SPEC_ACTION_TABLE", "GAZEBO_ACTION_TABLE", "hpc",
]
