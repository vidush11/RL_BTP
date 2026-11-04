"""The three experimental agents share this one class; ``mode`` selects A/B/C.

  A  LiDAR-only DQN            : features = base
  B  frozen-GVF DQN           : features = [base, frozen_gvf(base_lidar)]
  C  adaptive-GVF DQN         : features = [base, current_gvf(base_lidar)]

Only the input dimension differs between agents; the DQN hidden layers,
optimizer and hyperparameters are identical. For B and C the GVF features are
recomputed from the RAW base observation at minibatch time (never cached), so
Agent C never trains on stale features. Agent C additionally updates its GVF
demons on the live transition stream (their own GQ(lambda) objective), never
from the DQN TD error.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from .dqn import DQN, DQNConfig
from .encoder import StateEncoder
from .gvf_predictor import GVFPredictor
from .replay import ReplayBuffer


class DQNAgent:
    def __init__(self, mode: str, n_actions: int, encoder: StateEncoder,
                 dqn_cfg: DQNConfig, predictor: Optional[GVFPredictor] = None,
                 replay_capacity: int = 100_000, seed: int = 0):
        assert mode in ("A", "B", "C")
        if mode in ("B", "C"):
            assert predictor is not None, "B/C need a GVF predictor"
            assert predictor.frozen == (mode == "B"), "B->frozen, C->adaptive"
        self.mode = mode
        self.n_actions = n_actions
        self.encoder = encoder
        self.predictor = predictor
        n_gvf = predictor.n_features if predictor is not None else 0
        self.in_dim = encoder.full_dim(n_gvf)
        self.dqn = DQN(self.in_dim, n_actions, dqn_cfg, seed=seed)
        self.replay = ReplayBuffer(replay_capacity, encoder.base_dim, seed=seed + 1)
        self.rng = np.random.default_rng(seed + 2)
        self._last_bprob = 1.0 / n_actions

    # ---- feature construction ---------------------------------------------
    def _features(self, base: np.ndarray) -> np.ndarray:
        base = np.atleast_2d(np.asarray(base, float))
        if self.predictor is None:
            return base
        lidar = self.encoder.lidar_of(base)
        gvf = self.predictor.features_batch(lidar)
        return np.concatenate([base, gvf], axis=1)

    # ---- acting ------------------------------------------------------------
    def act(self, base: np.ndarray, epsilon: float) -> int:
        greedy = self.dqn.greedy_action(self._features(base))
        if self.rng.random() < epsilon:
            a = int(self.rng.integers(0, self.n_actions))
        else:
            a = greedy
        # behaviour prob of the chosen action under epsilon-greedy (for GVF rho)
        self._last_bprob = epsilon / self.n_actions + (1 - epsilon) * (a == greedy)
        return a

    # ---- experience --------------------------------------------------------
    def observe(self, base, action, reward, next_base, terminated, truncated) -> None:
        self.replay.add(base, action, reward, next_base, terminated, truncated)
        if self.mode == "C":  # adaptive GVF learns on the live stream
            done = bool(terminated or truncated)
            self.predictor.update(self.encoder.lidar_of(np.asarray(base)), action,
                                  self.encoder.lidar_of(np.asarray(next_base)),
                                  done, self._last_bprob)

    def on_episode_start(self) -> None:
        if self.mode == "C":
            self.predictor.reset_traces()

    # ---- learning ----------------------------------------------------------
    def learn(self, batch_size: int) -> Optional[float]:
        if len(self.replay) < batch_size:
            return None
        b = self.replay.sample(batch_size)
        feats = self._features(b.obs)        # current predictor for s
        next_feats = self._features(b.next_obs)  # ... and s' (same convention)
        return self.dqn.update(feats, b.action, b.reward, next_feats, b.terminated)

    # ---- checkpointing -----------------------------------------------------
    def save(self, path, episode=0, env_steps=0, extra=None) -> None:
        import pickle
        state = {"mode": self.mode, "episode": int(episode), "env_steps": int(env_steps),
                 "dqn": self.dqn.get_state(),
                 "predictor": self.predictor.get_state() if self.predictor else None,
                 "replay": self.replay.get_state(),
                 "rng": self.rng.bit_generator.state, "extra": extra or {}}
        with open(path, "wb") as fh:
            pickle.dump(state, fh)

    def load(self, path, with_replay=True) -> dict:
        import pickle
        with open(path, "rb") as fh:
            state = pickle.load(fh)
        self.dqn.load_state(state["dqn"])
        if self.predictor is not None and state.get("predictor") is not None:
            self.predictor.load_state(state["predictor"])
        if with_replay and state.get("replay") is not None:
            self.replay.load_state(state["replay"])
        try:
            self.rng.bit_generator.state = state["rng"]
        except Exception:
            pass
        return {"episode": state.get("episode", 0), "env_steps": state.get("env_steps", 0)}
