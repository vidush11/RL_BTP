"""Helpers for running several independent environments.

Each environment instance owns its own world, robot and ``numpy.random.Generator``
(``reset(seed=base_seed + i)`` gives instance ``i`` its own stream), so there is
no shared mutable state.  The factory receives a plain ``dict`` so it can be
pickled for ``spawn``-based multiprocessing (macOS) as well as ``fork`` (Linux).

Gymnasium >= 1.0 vector envs use "next-step" autoreset by default: the step
*after* an episode ends performs the reset and returns reward 0.  Pass
``autoreset_mode=gymnasium.vector.AutoresetMode.SAME_STEP`` to change that.
"""

from __future__ import annotations

from functools import partial
from typing import Any, Mapping, Optional, Union

import gymnasium as gym

from .config import SimConfig, load_config
from .environment import TurtleBot3LidarEnv


def make_env_factory(config: Union[None, str, Mapping[str, Any], SimConfig] = None):
    """Picklable zero-argument callable that builds one environment."""
    return partial(TurtleBot3LidarEnv, load_config(config).to_dict())


def make_vec_env(n_envs: int, config: Union[None, str, Mapping[str, Any], SimConfig] = None,
                 asynchronous: bool = False, **vector_kwargs):
    """``SyncVectorEnv`` (default, no processes) or ``AsyncVectorEnv`` of ``n_envs``
    independent environments.  Seed with ``vec.reset(seed=base_seed)``: instance
    ``i`` is seeded ``base_seed + i``."""
    if n_envs < 1:
        raise ValueError("n_envs must be >= 1")
    factory = make_env_factory(config)
    cls = gym.vector.AsyncVectorEnv if asynchronous else gym.vector.SyncVectorEnv
    return cls([factory for _ in range(n_envs)], **vector_kwargs)
