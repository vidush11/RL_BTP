"""Rule-based predictive controllers (no search), all vectorised over a batch.

Steering (Part 1 and 2) -- uses Q[turn] = min over horizons of the safety GVF at
the current speed level, and the current (previous) turn:
  * straight and Q[straight] >= beta_lo                -> keep straight
  * turning  and Q[straight] >= beta_hi                -> go straight
  * turning  and Q[current turn] >= beta_lo            -> keep turning ("safe enough")
  * otherwise -> the gentlest turn whose Q is within delta of max(Q)
    (ties: higher Q, then the current turning direction).

Speed (Part 2; paper Alg. 2 + Alg. 3 hysteresis), after steering chose turn k,
with q = Q(k, current speed), v_hat / w_hat = predicted future linear speed /
|turn rate| for (k, current speed):
  q <  beta_v_lo : v_cmd = v_prev - alpha_decel * (1 - q)
  q >= beta_v_hi : v_cmd = v_prev + alpha_speed * clip(v_target - v_hat, e_min, e_max)
  else           : v_cmd = v_prev
  cap            : v_cmd <= v_max * (1 - k_w * w_hat / w_max)       (upcoming sharp turn)
  snap to the nearest speed level, at most one level change per step (comfort).
Ablation modes: fixed | safety (v_hat := v_prev, no w cap) | lin | ang | both.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np

from .actions import ActionSpace


@dataclass
class SteerCfg:
    beta_lo: float = 0.6
    beta_hi: float = 0.8
    delta: float = 0.05


@dataclass
class SpeedCfg:
    mode: str = "both"            # fixed | safety | lin | ang | both
    beta_v_lo: float = 0.5
    beta_v_hi: float = 0.8
    alpha_decel: float = 0.3
    alpha_speed: float = 1.0
    e_min: float = -0.2
    e_max: float = 0.2
    k_w: float = 0.5
    v_target: float = 0.22
    fixed_level: int = 1


@dataclass
class HandCfg:
    d_turn: float = 0.5           # front clearance below which we turn
    d_hard: float = 0.3           # ... and below which the turn is hard
    d_fast: float = 1.0           # Part 2: front clearance for top speed
    d_mid: float = 0.5            # Part 2: front clearance for middle speed


def as_dict(cfg) -> dict:
    return asdict(cfg)


# --------------------------------------------------------------------------- #
def steer_batch(Qt: np.ndarray, cur_turn: np.ndarray, aspace: ActionSpace, cfg: SteerCfg) -> np.ndarray:
    """Qt: (B, n_turn) safety per turn rate; cur_turn: (B,) current turn index."""
    B = Qt.shape[0]
    ar = np.arange(B)
    st = aspace.straight_turn
    q_st = Qt[:, st]
    q_cur = Qt[ar, cur_turn]
    is_st = cur_turn == st
    turns = aspace.turns
    cand = Qt >= Qt.max(1, keepdims=True) - cfg.delta
    same_dir = np.sign(turns)[None, :] == np.sign(turns[cur_turn])[:, None]
    key = np.abs(turns)[None, :] * 100.0 - Qt - 1e-3 * same_dir
    key = np.where(cand, key, np.inf)
    out = np.argmin(key, axis=1)
    keep_st = is_st & (q_st >= cfg.beta_lo)
    go_st = (~is_st) & (q_st >= cfg.beta_hi)
    keep_turn = (~is_st) & (~go_st) & (q_cur >= cfg.beta_lo)
    out = np.where(keep_st | go_st, st, out)
    out = np.where(keep_turn, cur_turn, out)
    return out.astype(int)


def speed_batch(q_cur, prev_lvl, v_hat, w_hat, aspace: ActionSpace, cfg: SpeedCfg) -> np.ndarray:
    speeds = aspace.speeds
    prev_lvl = np.asarray(prev_lvl, int)
    if cfg.mode == "fixed":
        target = np.full_like(prev_lvl, cfg.fixed_level)
        return np.clip(target, prev_lvl - 1, prev_lvl + 1)
    v_prev = speeds[prev_lvl]
    if cfg.mode in ("safety", "ang"):
        v_hat = v_prev
    v_cmd = v_prev.copy()
    dec = q_cur < cfg.beta_v_lo
    acc = q_cur >= cfg.beta_v_hi
    e = np.clip(cfg.v_target - v_hat, cfg.e_min, cfg.e_max)
    v_cmd = np.where(dec, v_prev - cfg.alpha_decel * (1.0 - q_cur), v_cmd)
    v_cmd = np.where(acc & ~dec, v_prev + cfg.alpha_speed * e, v_cmd)
    if cfg.mode in ("ang", "both"):
        cap = aspace.v_max * (1.0 - cfg.k_w * np.clip(w_hat, 0, None) / aspace.w_max)
        v_cmd = np.minimum(v_cmd, cap)
    lvl = np.argmin(np.abs(speeds[None, :] - v_cmd[:, None]), axis=1)
    return np.clip(lvl, prev_lvl - 1, prev_lvl + 1).astype(int)


def controller_batch(Qmin: np.ndarray, Vlin: np.ndarray, Vabs: np.ndarray, prev_a: np.ndarray,
                     aspace: ActionSpace, steer: SteerCfg, speed: SpeedCfg) -> np.ndarray:
    """Qmin: (B, n) min-over-horizons safety; Vlin (m/s), Vabs (rad/s): (B, n)
    predicted future speed / |turn rate| (may be None for Part 1); prev_a: (B,)."""
    prev_a = np.asarray(prev_a, int)
    B = len(prev_a)
    ar = np.arange(B)
    prev_lvl = aspace.speed_idx[prev_a]
    cur_turn = aspace.turn_idx[prev_a]
    Qt = Qmin[ar[:, None], aspace.grid[prev_lvl]]          # (B, n_turn) at current speed
    k = steer_batch(Qt, cur_turn, aspace, steer)
    if aspace.n_speed == 1:
        return aspace.grid[0, k]
    a_k = aspace.grid[prev_lvl, k]
    v_hat = Vlin[ar, a_k] if Vlin is not None else aspace.speeds[prev_lvl]
    w_hat = Vabs[ar, a_k] if Vabs is not None else np.zeros(B)
    lvl = speed_batch(Qmin[ar, a_k], prev_lvl, v_hat, w_hat, aspace, speed)
    return aspace.grid[lvl, k]


# --------------------------------------------------------------------------- #
_FRONT = np.array([22, 23, 0, 1, 2])
_LEFT = np.arange(3, 9)       # +45 .. +120 deg (CCW = left)
_RIGHT = np.arange(16, 22)    # -120 .. -45 deg


def hand_batch(ranges: np.ndarray, aspace: ActionSpace, cfg: HandCfg) -> np.ndarray:
    """Reactive turn-away rule (+ speed from front clearance in Part 2)."""
    r = np.asarray(ranges, float)
    front = r[:, _FRONT].min(1)
    left_free = r[:, _LEFT].mean(1) >= r[:, _RIGHT].mean(1)
    st = aspace.straight_turn
    hard = front < cfg.d_hard
    turn = front < cfg.d_turn
    k = np.full(len(r), st)
    k = np.where(turn & left_free, st + 1, k)
    k = np.where(turn & ~left_free, st - 1, k)
    k = np.where(hard & left_free, aspace.n_turn - 1, k)
    k = np.where(hard & ~left_free, 0, k)
    if aspace.n_speed == 1:
        return aspace.grid[0, k]
    lvl = np.where(front > cfg.d_fast, aspace.n_speed - 1, np.where(front > cfg.d_mid, 1, 0))
    return aspace.grid[lvl, k]


def ttc_from_safety(q, gamma: float) -> np.ndarray:
    """Read a safety prediction Q = (1-g) sum g^t c as steps until the first
    unsafe step, assuming c = 1 until then: Q = 1 - g^T  =>  T = log(1-Q)/log(g)."""
    q = np.clip(np.asarray(q, float), 0.0, 1.0 - 1e-9)
    return np.log1p(-q) / np.log(gamma)
