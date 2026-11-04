"""Adapter around the existing directional-TTC GVF demons, exposing them as a
fixed set of predictive SAFETY features for the DQN (the paper's "perception as
prediction": front/side safety as policy-based GVF predictions).

Features (3): forward / left / right time-to-collision, each the demon's own
prediction theta_d . phi(s, a_d) under its deterministic target policy. The
degenerate velocity demons (constant linear speed, noise-like angular) are
excluded on purpose (see RESULTS_GVF_IMPROVED.md).

* frozen=True  (Agent B): predict only, weights never change.
* frozen=False (Agent C): update() runs the demons' own GQ(lambda) objective on
  the live transition -- NOT the DQN TD error.

Normalisation uses stats fit on TRAINING data only (set_normalization). Non-finite
or diverged predictions are clipped, never silently replaced with a plausible
mid-range value (they are clipped to the normalised range and counted).
"""
from __future__ import annotations

import importlib
from typing import Optional

import numpy as np

LID, ACT = 24, 5
FORWARD, LEFT, RIGHT = 2, 1, 3
_TARGETS = (("forward", FORWARD, "theta_forward", "w_forward"),
            ("left", LEFT, "theta_left", "w_left"),
            ("right", RIGHT, "theta_right", "w_right"))


def _demon_classes():
    """Import DirectionalTTCGVF, installing the ROS shim first so the agent's
    top-level imports resolve. Idempotent."""
    try:
        from simulator.ros_compat import install
        from simulator import TurtleBot3LidarEnv
        install(TurtleBot3LidarEnv(), seed=0)
    except Exception:
        pass
    mod = importlib.import_module("gvf_horde_agent")
    return mod.DirectionalTTCGVF


def _feature(norm_lidar: np.ndarray, action: int) -> np.ndarray:
    """phi(s, a): normalised lidar in the action's block (== the demon's own
    get_feature_vector, since the env observation is already clip(range/3.5))."""
    f = np.zeros((ACT, LID), dtype=np.float64)
    f[action] = np.clip(np.asarray(norm_lidar, dtype=np.float64), 0.0, 1.0)
    return f.ravel()


class GVFPredictor:
    FEATURE_NAMES = ("ttc_forward", "ttc_left", "ttc_right")
    n_features = 3

    def __init__(self, frozen: bool = True, obstacle_threshold: float = 0.25,
                 alpha_theta: float = 0.1, alpha_w: float = 1e-4,
                 trace_lambda: float = 0.4, clip_z: float = 5.0):
        Dir = _demon_classes()
        feat_dim = LID * ACT
        self.demons = {}
        for name, act, _tk, _wk in _TARGETS:
            self.demons[name] = Dir(feat_dim, ACT, act, slice(0, LID),
                                    alpha_theta, alpha_w, trace_lambda,
                                    obstacle_threshold, name=name)
        self.frozen = frozen
        self.clip_z = clip_z
        self._mean = np.zeros(self.n_features)
        self._std = np.ones(self.n_features)
        self.nonfinite_count = 0

    # ---- features ----------------------------------------------------------
    def raw_predict(self, norm_lidar: np.ndarray) -> np.ndarray:
        out = np.empty(self.n_features)
        for k, (name, act, _tk, _wk) in enumerate(_TARGETS):
            p = float(np.dot(self.demons[name].theta, _feature(norm_lidar, act)))
            if not np.isfinite(p):
                self.nonfinite_count += 1
                p = 0.0
            out[k] = p
        return out

    def features(self, norm_lidar: np.ndarray) -> np.ndarray:
        """Normalised (z-scored, clipped) predictive features for the DQN."""
        z = (self.raw_predict(norm_lidar) - self._mean) / self._std
        return np.clip(z, -self.clip_z, self.clip_z)

    def features_batch(self, norm_lidar_batch: np.ndarray) -> np.ndarray:
        return np.stack([self.features(x) for x in np.asarray(norm_lidar_batch)])

    # ---- adaptive learning (Agent C) --------------------------------------
    def reset_traces(self) -> None:
        for d in self.demons.values():
            d.reset_trace()

    def update(self, norm_lidar_t, action, norm_lidar_tp1, done,
               behaviour_prob) -> Optional[np.ndarray]:
        """Run the demons' own GQ(lambda) update on this transition. No-op when
        frozen. gamma needs ranges in METRES, recovered as norm_lidar * 3.5.

        Returns each demon's pre-update TD error (forward, left, right) for
        GVF-loss logging, or None when frozen. Callers that only need the update
        (Agents B/C) ignore the return value."""
        if self.frozen:
            return None
        lidar_t_m = np.asarray(norm_lidar_t, float) * 3.5
        lidar_tp1_m = np.asarray(norm_lidar_tp1, float) * 3.5
        phi_t = {name: _feature(norm_lidar_t, act) for name, act, _, _ in _TARGETS}
        deltas = np.empty(self.n_features)
        for k, (name, act, _tk, _wk) in enumerate(_TARGETS):
            d = self.demons[name]
            phi_target_next = _feature(norm_lidar_tp1, act)
            # same delta the demon computes internally (r=1, z=0):
            gamma_tp1 = 0.0 if done else d.compute_gamma(lidar_tp1_m)
            deltas[k] = (1.0 + gamma_tp1 * float(np.dot(d.theta, phi_target_next))
                         - float(np.dot(d.theta, phi_t[name])))
            d.update(phi_t[name], phi_target_next, lidar_t_m, lidar_tp1_m,
                     int(action), float(behaviour_prob), bool(done))
        return deltas

    # ---- normalisation (train data only) ----------------------------------
    def set_normalization(self, mean, std) -> None:
        self._mean = np.asarray(mean, float).reshape(self.n_features)
        std = np.asarray(std, float).reshape(self.n_features)
        self._std = np.where(std > 1e-6, std, 1.0)

    # ---- serialization -----------------------------------------------------
    def get_state(self) -> dict:
        d = {"mean": self._mean, "std": self._std, "frozen": self.frozen,
             "nonfinite_count": self.nonfinite_count}
        for name, _act, tk, wk in _TARGETS:
            d[tk] = self.demons[name].theta.copy()
            d[wk] = self.demons[name].w.copy()
        return d

    def load_state(self, d) -> None:
        for name, _act, tk, wk in _TARGETS:
            self.demons[name].theta = np.asarray(d[tk]).copy()
            self.demons[name].w = np.asarray(d[wk]).copy()
        if "mean" in d:
            self.set_normalization(d["mean"], d["std"])

    def load_checkpoint(self, path: str) -> None:
        """Load theta/w from a Horde combined .npz (the pretraining format)."""
        data = np.load(path)
        for name, _act, tk, wk in _TARGETS:
            self.demons[name].theta = np.asarray(data[tk]).copy()
            self.demons[name].w = np.asarray(data[wk]).copy()

    def theta_snapshot(self) -> np.ndarray:
        return np.concatenate([self.demons[n].theta for n, *_ in _TARGETS])
