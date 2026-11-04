"""Policies and a batched, deterministic episode runner with full metrics.

Many env instances step in lockstep; the agent runs once per step on the stacked
scans. Every episode uses its own seed (reset(seed=...)), the policy carries no
randomness, and the model is never updated here -> evaluating twice with the
same seeds gives identical numbers (checked by determinism_check()).
"""
from __future__ import annotations

from collections import deque
from typing import Dict, List, Optional

import numpy as np

from .actions import ActionSpace
from .controllers import SteerCfg, SpeedCfg, HandCfg, controller_batch, hand_batch, _FRONT, _LEFT, _RIGHT
from .envs import make_env

CELL = 0.25


# =============================================================== policies ===
class Policy:
    uses_q = False

    def __init__(self, aspace: ActionSpace):
        self.A = aspace
        self.last_q = None

    def reset(self, n_slots: int):
        self.prev = np.full(n_slots, self.A.default_action(), dtype=int)

    def start(self, slot: int):
        self.prev[slot] = self.A.default_action()

    def act(self, ranges, slots):
        raise NotImplementedError


class ControllerPolicy(Policy):
    """GVF model or original demons + steering (+ speed) controller."""
    uses_q = True

    def __init__(self, src, aspace, steer: SteerCfg, speed: Optional[SpeedCfg] = None):
        super().__init__(aspace)
        self.src, self.steer, self.speed = src, steer, speed or SpeedCfg()

    def act(self, ranges, slots):
        Qmin, Vl, Va = self.src.decision_inputs(ranges)
        a = controller_batch(Qmin, Vl, Va, self.prev[slots], self.A, self.steer, self.speed)
        self.prev[slots] = a
        self.last_q = Qmin
        return a


class DQNPolicy(Policy):
    uses_q = True

    def __init__(self, dqn, aspace, speed_bias: float = 0.0):
        super().__init__(aspace)
        self.dqn, self.speed_bias = dqn, speed_bias

    def act(self, ranges, slots):
        q = self.dqn.q_values(ranges)
        self.last_q = q
        if self.speed_bias and self.A.n_speed > 1:
            span = q.max(1, keepdims=True) - q.min(1, keepdims=True)
            q = q + self.speed_bias * span * (self.A.speed_idx / (self.A.n_speed - 1))[None, :]
        a = np.argmax(q, 1)
        self.prev[slots] = a
        return a


class HandPolicy(Policy):
    def __init__(self, aspace, cfg: HandCfg):
        super().__init__(aspace)
        self.cfg = cfg

    def act(self, ranges, slots):
        a = hand_batch(ranges, self.A, self.cfg)
        self.prev[slots] = a
        return a


# ======================================================= failure analysis ===
CATEGORIES = ("dead_end", "narrow_gap", "too_fast", "head_on", "side_clip_turning", "other")


def classify_collision(scans: np.ndarray, ws: np.ndarray, speeds: np.ndarray, v_top: Optional[float]) -> str:
    final = scans[-1]
    b = int(np.argmin(final))
    ang = (b * 15 + 180) % 360 - 180
    front = abs(ang) <= 30
    turning = float(np.mean(np.abs(ws[-5:]))) > 0.5
    pre = scans[max(0, len(scans) - 10)]
    if pre[_FRONT].min() < 0.6 and pre[_LEFT].min() < 0.6 and pre[_RIGHT].min() < 0.6:
        return "dead_end"
    pre5 = scans[max(0, len(scans) - 5)]
    if pre5[_LEFT].min() < 0.4 and pre5[_RIGHT].min() < 0.4:
        return "narrow_gap"
    if v_top is not None and front and float(np.max(speeds[-5:])) >= v_top - 1e-9:
        return "too_fast"
    if front and not turning:
        return "head_on"
    if turning and not front:
        return "side_clip_turning"
    return "other"


# ================================================================ runner ===
def run_episodes(policy: Policy, env_name: str, part: int, seeds: List[int], cap: Optional[int] = 800,
                 n_par: int = 40, fail_log: int = 0, traj: int = 0, survival_cap: int = 800) -> List[dict]:
    """Run one episode per seed. Returns per-episode records (same order as seeds)."""
    A = policy.A
    n_par = max(1, min(n_par, len(seeds)))
    envs, remaps = zip(*[make_env(env_name, part, cap) for _ in range(n_par)])
    env_table = envs[0].actions.table
    dt = envs[0].cfg.motion.dt * envs[0].cfg.motion.action_repeat
    v_top = float(A.speeds.max()) if A.n_speed > 1 else None
    policy.reset(n_par)
    records: List[Optional[dict]] = [None] * len(seeds)
    queue = list(range(len(seeds)))
    slot_ep = [-1] * n_par
    S: List[dict] = [dict() for _ in range(n_par)]
    ranges = np.zeros((n_par, 24))
    n_failure_logged = 0

    def start(slot):
        nonlocal n_failure_logged
        if not queue:
            slot_ep[slot] = -1
            return
        k = queue.pop(0)
        slot_ep[slot] = k
        _, info = envs[slot].reset(seed=int(seeds[k]))
        ranges[slot] = info["ranges_m"]
        policy.start(slot)
        S[slot] = {"steps": 0, "dist": 0.0, "absw": 0.0, "rev": 0, "last_sgn": 0, "cells": set(),
                   "minc": np.inf, "sum_minc": 0.0, "within": 0, "lvl": np.zeros(A.n_speed),
                   "chg": 0, "last_lvl": None, "cs_sum": np.zeros(A.n_speed), "cs_n": np.zeros(A.n_speed),
                   "ring": deque(maxlen=30), "traj": [] if k < traj else None}

    for s in range(n_par):
        start(s)
    while True:
        active = [s for s in range(n_par) if slot_ep[s] >= 0]
        if not active:
            break
        act = policy.act(ranges[active], np.asarray(active))
        q = policy.last_q
        for j, s in enumerate(active):
            a = int(act[j]); ea = int(remaps[s][a])
            st = S[s]
            _, _, term, trunc, info = envs[s].step(ea)
            r = info["ranges_m"]
            v, w = env_table[ea]
            m = float(r.min())
            st["steps"] += 1
            st["dist"] += v * dt
            st["absw"] += abs(w)
            sg = int(np.sign(w))
            if sg != 0:
                if st["last_sgn"] != 0 and sg != st["last_sgn"]:
                    st["rev"] += 1
                st["last_sgn"] = sg
            x, y, _ = info["pose"]
            st["cells"].add((int(np.floor(x / CELL)), int(np.floor(y / CELL))))
            st["minc"] = min(st["minc"], m)
            st["sum_minc"] += m
            st["within"] += m < 0.25
            lvl = int(A.speed_idx[a])
            st["lvl"][lvl] += 1
            if st["last_lvl"] is not None and lvl != st["last_lvl"]:
                st["chg"] += 1
            st["last_lvl"] = lvl
            st["cs_sum"][lvl] += m; st["cs_n"][lvl] += 1
            st["ring"].append((ranges[s].copy(), a, v, w, None if q is None else np.asarray(q[j], float)))
            if st["traj"] is not None:
                st["traj"].append((float(x), float(y), float(v)))
            ranges[s] = r
            if term or trunc:
                coll = bool(term and info["collision"])
                n = st["steps"]
                rec = {"seed": int(seeds[slot_ep[s]]), "steps": n, "collided": coll,
                       "truncated": bool(trunc and not term),
                       "survival": min(n, survival_cap) / survival_cap,
                       "distance": st["dist"], "mean_speed": st["dist"] / (n * dt),
                       "mean_abs_w": st["absw"] / n, "reversals_per100": 100.0 * st["rev"] / n,
                       "area_cells": len(st["cells"]), "min_clearance": st["minc"],
                       "mean_min_clearance": st["sum_minc"] / n, "frac_within_025": st["within"] / n,
                       "speed_frac": (st["lvl"] / n).tolist(), "speed_changes_per100": 100.0 * st["chg"] / n,
                       "clear_by_speed_sum": st["cs_sum"].tolist(), "clear_by_speed_n": st["cs_n"].tolist(),
                       "category": None}
                if coll:
                    scans = np.array([e[0] for e in st["ring"]] + [r])
                    ws = np.array([e[3] for e in st["ring"]]); vs = np.array([e[2] for e in st["ring"]])
                    rec["category"] = classify_collision(scans, ws, vs, v_top)
                    if n_failure_logged < fail_log:
                        n_failure_logged += 1
                        rec["fail_log"] = {"scans": np.round(scans, 3).tolist(),
                                           "actions": [int(e[1]) for e in st["ring"]],
                                           "v": vs.tolist(), "w": ws.tolist(),
                                           "q": [None if e[4] is None else np.round(e[4], 3).tolist()
                                                 for e in st["ring"]]}
                if st["traj"] is not None:
                    rec["traj"] = st["traj"]
                records[slot_ep[s]] = rec
                start(s)
    return records


def summarize(records: List[dict]) -> Dict[str, float]:
    if not records:
        return {}
    R = records
    coll = np.array([r["collided"] for r in R], bool)
    steps = np.array([r["steps"] for r in R], float)
    dist = np.array([r["distance"] for r in R], float)
    out = {"n_episodes": len(R),
           "survival": float(np.mean([r["survival"] for r in R])),
           "success": float(np.mean(~coll)),
           "collision_rate": float(np.mean(coll)),
           "mean_steps": float(steps.mean()),
           "median_ttc_steps": float(np.median(steps)),        # censored episodes count at their cap
           "mean_distance": float(dist.mean()),
           "collisions_per_100m": float(100.0 * coll.sum() / max(dist.sum(), 1e-9)),
           "distance_before_collision": float(np.mean(dist[coll])) if coll.any() else float("nan")}
    for k in ("mean_speed", "mean_abs_w", "reversals_per100", "area_cells", "min_clearance",
              "mean_min_clearance", "frac_within_025", "speed_changes_per100"):
        out[k] = float(np.mean([r[k] for r in R]))
    out["min_clearance_p05"] = float(np.percentile([r["min_clearance"] for r in R], 5))
    out["speed_frac"] = np.mean([r["speed_frac"] for r in R], 0).tolist()
    cs = np.sum([r["clear_by_speed_sum"] for r in R], 0); cn = np.sum([r["clear_by_speed_n"] for r in R], 0)
    out["clearance_by_speed"] = [float(a / b) if b else None for a, b in zip(cs, cn)]
    cats = [r["category"] for r in R if r["category"]]
    out["failure_categories"] = {c: cats.count(c) for c in CATEGORIES}
    return out


def survival_data(records: List[dict], cap: int):
    """(time-to-first-collision steps, event observed) for Kaplan-Meier."""
    return ([r["steps"] for r in records], [bool(r["collided"]) for r in records])


def determinism_check(policy: Policy, env_name: str, part: int, seeds: List[int]) -> bool:
    a = run_episodes(policy, env_name, part, seeds)
    b = run_episodes(policy, env_name, part, seeds)
    key = lambda R: [(r["steps"], round(r["distance"], 9), r["collided"]) for r in R]
    ok = key(a) == key(b)
    assert ok, "evaluation is not deterministic"
    return ok
