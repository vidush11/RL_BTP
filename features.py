"""
One tile-coded binary feature vector phi(s,a), SHARED by every demon.

Rather than one huge joint tiling over all 8 sensors (which would explode), we
tile small overlapping GROUPS of sensors and concatenate. This is inspired from what
the Horde paper did: "four overlapping joint tilings across three sensors".

Author: Vidush Jindal(jindalv)
"""

from __future__ import annotations

import numpy as np

from env import (
    ANG_VEL,
    BUMP,
    COMPASS,
    IR_F,
    IR_L,
    IR_R,
    LIGHT_S,
    N_ACTIONS,
    VEL,
    Action,
)

LOW = np.array([0.0, 0.0, 0.0, 0.0, -0.6, -3.0, 0.0, -1.0])
HIGH = np.array([1.0, 1.0, 1.0, 1.0, 0.6, 3.0, 1.0, 1.0])

# (sensor indices, bins per dimension, number of tilings)
GROUPS = [
    ((IR_F,), 16, 8),  # fine resolution: several demons key on it
    ((IR_L, IR_R), 8, 4),  # joint: lets a demon tell left from right
    ((LIGHT_S,), 16, 8),
    ((VEL,), 12, 8),
    ((ANG_VEL,), 8, 4),
    ((IR_F, VEL), 8, 4),  # joint: "how close AND how fast" matters
    ((IR_F, ANG_VEL), 8, 4),  # turning changes time-to-wall a lot
    ((LIGHT_S, COMPASS), 8, 4),
    ((BUMP,), 2, 1),
]

np.random.seed(0)

_BLOCKS, _base = [], 0
_rng = np.random
for sensors, _bins, _tilings in GROUPS:
    _idx = np.array(sensors)
    # width of each of the sensors divided into bins
    _w = (HIGH[_idx] - LOW[_idx]) / _bins
    # random tiling offsets for each of the sensors
    _offsets = _rng.uniform(0, 1, (_tilings, len(_idx))) * _w
    _BLOCKS.append((_idx, _bins, _tilings, _w, _offsets, _base))
    _base += _tilings * _bins ** len(_idx)

N_STATE = _base + 1  # +1 bias feature
N_FEATURES = N_ACTIONS * N_STATE  # features are offset per action
N_ACTIVE = sum(g[2] for g in GROUPS) + 1


def phi(obs: np.ndarray, a: Action) -> np.ndarray:
    """
    Indices of the active (1-features) of phi(s,a)
    We explicitly construct our features mathematically so that
    we do not have to construct a dense vector and select one indices.
    Last feature is always bias feature
    Always exactly N_ACTIVE of them.


    :param obs: observation from sensors
    :type obs: np.ndarray
    :param a: action that robot took
    :type a: Action
    :return: the int64 array of features that are active + bias
    :rtype: np.ndarray
    """

    out = np.empty(N_ACTIVE, dtype=np.int64)
    p = 0
    for idx, bins, tilings, w, offsets, base in _BLOCKS:
        # we get which bin it should be for each sensor, we have to cast it into int
        # since guassian noise in obs can make our indices negative, we clip them to 0,bins-1
        c = np.clip(((obs[idx] - LOW[idx] + offsets) / w), 0, bins - 1).astype(np.int64)
        # c is tilings X sensors
        flat = c[:, 0]
        for k in range(1, c.shape[1]):
            # we need base representation the inds can be bins**d
            flat = flat * bins + c[:, k]
        out[p : p + tilings] = base + np.arange(tilings) * bins ** len(idx) + flat
        p += tilings
    out[-1] = N_STATE - 1  # bias
    return out + a * N_STATE
