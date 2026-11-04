"""Observation features.

raw  (24): normalised ranges  min(d, 3.5) / 3.5      (standard DQN input)
rich (48): [raw, exp(-d / 0.3)]                         (GVF + strong-DQN input)
The proximity half gives resolution where safety is decided (d < ~1 m); both
halves are per-beam, so mirroring is the same beam permutation on each half.
"""
from __future__ import annotations

import numpy as np

from .actions import mirror_beams

MAX_RANGE = 3.5
N_BEAMS = 24
_MB = mirror_beams(N_BEAMS)


def raw_features(ranges_m: np.ndarray) -> np.ndarray:
    r = np.clip(np.asarray(ranges_m, dtype=np.float32), 0.0, MAX_RANGE)
    return r / np.float32(MAX_RANGE)


def rich_features(ranges_m: np.ndarray) -> np.ndarray:
    r = np.clip(np.asarray(ranges_m, dtype=np.float32), 0.0, MAX_RANGE)
    return np.concatenate([r / np.float32(MAX_RANGE), np.exp(-r / np.float32(0.3))], axis=-1)


def mirror_ranges(ranges_m: np.ndarray) -> np.ndarray:
    return np.asarray(ranges_m)[..., _MB]


FEATURES = {"raw": (raw_features, N_BEAMS), "rich": (rich_features, 2 * N_BEAMS)}
