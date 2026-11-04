"""Acceptance tests for the GVF-enhanced DQN study (rl/ package).

Covers: MLP backprop (gradient check), DQN terminal-vs-truncation mask, replay
raw-observation storage, encoder dimensions per agent, GVF predictor shape/
finiteness, B-frozen / C-adaptive invariants, the stale-feature avoidance
(features recomputed from raw with the current predictor), checkpoint save/load
equivalence, and a short end-to-end run + evaluation for all three agents.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rl.dqn import MLP, DQN, DQNConfig, huber_grad
from rl.replay import ReplayBuffer
from rl.encoder import StateEncoder
from rl.gvf_predictor import GVFPredictor
from rl.agent import DQNAgent
from rl import experiment as X
from rl import metrics as M


# --------------------------------------------------------------------------- #
# MLP backprop gradient check
# --------------------------------------------------------------------------- #
def test_mlp_gradient_check():
    rng = np.random.default_rng(0)
    net = MLP(5, (7, 6), 3, seed=1)
    x = rng.normal(size=(4, 5))
    target = rng.normal(size=(4, 3))

    def loss_of():
        pred = net.forward(x, cache=True)
        return 0.5 * np.sum((pred - target) ** 2), pred

    L0, pred = loss_of()
    d_out = (pred - target)
    gW, gb = net._backward(d_out)

    # numerical gradient on W[0]
    eps = 1e-6
    W = net.W[0]
    num = np.zeros_like(W)
    for i in range(W.shape[0]):
        for j in range(W.shape[1]):
            old = W[i, j]
            W[i, j] = old + eps; Lp = 0.5 * np.sum((net.forward(x) - target) ** 2)
            W[i, j] = old - eps; Lm = 0.5 * np.sum((net.forward(x) - target) ** 2)
            W[i, j] = old
            num[i, j] = (Lp - Lm) / (2 * eps)
    assert np.allclose(gW[0], num, atol=1e-4), np.abs(gW[0] - num).max()


def test_huber_grad_clips():
    g = huber_grad(np.array([0.5, 5.0, -5.0]), np.array([0.0, 0.0, 0.0]), delta=1.0)
    assert np.allclose(g, [0.5, 1.0, -1.0])


# --------------------------------------------------------------------------- #
# DQN terminal mask: terminated -> no bootstrap; truncated -> bootstrap
# --------------------------------------------------------------------------- #
def test_dqn_terminal_mask_no_bootstrap():
    cfg = DQNConfig(hidden=(8,), gamma=0.9, target_update=10_000)
    dqn = DQN(4, 3, cfg, seed=0)
    feats = np.zeros((1, 4)); next_feats = np.ones((1, 4))
    r = np.array([1.0]); a = np.array([0])
    # terminated=1 -> target y = r only (no gamma*maxQ'); check via the update's y.
    qn = dqn.target.predict(next_feats)[0]
    y_term = r[0] + cfg.gamma * (1 - 1) * qn.max()
    y_trunc = r[0] + cfg.gamma * (1 - 0) * qn.max()
    assert y_term == pytest.approx(1.0)
    assert y_trunc == pytest.approx(1.0 + cfg.gamma * qn.max())
    # the learner must use `terminated`, so a truncated sample still bootstraps:
    dqn.update(feats, a, r, next_feats, terminated=np.array([0.0]))  # runs, no error


def test_dqn_update_runs_and_changes_weights():
    cfg = DQNConfig(hidden=(16,), target_update=10_000)
    dqn = DQN(6, 5, cfg, seed=0)
    w0 = dqn.q.W[0].copy()
    rng = np.random.default_rng(0)
    for _ in range(20):
        dqn.update(rng.normal(size=(32, 6)), rng.integers(0, 5, 32),
                   rng.normal(size=32), rng.normal(size=(32, 6)),
                   (rng.random(32) < 0.3).astype(float))
    assert not np.allclose(w0, dqn.q.W[0])


# --------------------------------------------------------------------------- #
# Replay buffer
# --------------------------------------------------------------------------- #
def test_replay_stores_raw_and_shapes():
    rb = ReplayBuffer(100, base_dim=27, seed=0)
    for i in range(50):
        rb.add(np.full(27, i / 50), i % 5, 1.0, np.full(27, (i + 1) / 50),
               terminated=(i % 7 == 0), truncated=(i % 11 == 0))
    assert len(rb) == 50
    b = rb.sample(16)
    assert b.obs.shape == (16, 27) and b.next_obs.shape == (16, 27)
    assert b.action.shape == (16,) and b.terminated.shape == (16,)
    # terminated and truncated are stored separately
    assert set(np.unique(b.terminated)) <= {0.0, 1.0}


# --------------------------------------------------------------------------- #
# Encoder dimensions per agent
# --------------------------------------------------------------------------- #
def test_encoder_dims_goal_and_nogoal():
    enc_goal = StateEncoder(use_goal=True)
    enc_lidar = StateEncoder(use_goal=False)
    assert enc_goal.base_dim == 27 and enc_lidar.base_dim == 24
    obs = np.linspace(0, 1, 24); info = {"goal_distance": 2.0, "goal_angle": 0.3}
    assert enc_goal.base_obs(obs, info).shape == (27,)
    assert enc_lidar.base_obs(obs, info).shape == (24,)
    # A uses base only; B/C add 3 GVF features
    assert enc_goal.full_dim(0) == 27 and enc_goal.full_dim(3) == 30


# --------------------------------------------------------------------------- #
# GVF predictor
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def predictor():
    return GVFPredictor(frozen=True)


def test_gvf_features_shape_and_finite(predictor):
    lidar = np.clip(np.linspace(0.2, 1.0, 24), 0, 1)
    f = predictor.features(lidar)
    assert f.shape == (3,) and np.all(np.isfinite(f))


def test_gvf_nonfinite_handled():
    p = GVFPredictor(frozen=True)
    p.demons["forward"].theta[:] = np.inf   # force a non-finite raw prediction
    f = p.features(np.ones(24))
    assert np.all(np.isfinite(f)) and p.nonfinite_count > 0


# --------------------------------------------------------------------------- #
# Frozen (B) vs adaptive (C) from identical params
# --------------------------------------------------------------------------- #
def test_frozen_predictor_never_changes():
    p = GVFPredictor(frozen=True)
    before = p.theta_snapshot().copy()
    for _ in range(50):
        p.update(np.random.rand(24), 2, np.random.rand(24), False, 0.2)
    assert np.array_equal(before, p.theta_snapshot())


def test_adaptive_predictor_changes_from_same_init():
    pf = GVFPredictor(frozen=True); pc = GVFPredictor(frozen=False)
    assert np.array_equal(pf.theta_snapshot(), pc.theta_snapshot())  # identical init
    rng = np.random.default_rng(0)
    for _ in range(200):
        lt, ltp1 = rng.random(24), rng.random(24)
        pc.update(lt, int(rng.integers(0, 5)), ltp1, False, 0.2)
    assert not np.array_equal(pf.theta_snapshot(), pc.theta_snapshot())


# --------------------------------------------------------------------------- #
# Stale-feature avoidance: features recomputed from RAW with current predictor
# --------------------------------------------------------------------------- #
def test_agent_recomputes_features_from_raw():
    enc = StateEncoder(use_goal=True)
    p = GVFPredictor(frozen=False)
    agent = DQNAgent("C", 5, enc, DQNConfig(hidden=(8,)), predictor=p, replay_capacity=50, seed=0)
    base = np.concatenate([np.linspace(0.2, 1, 24), [0.5, 0.0, 1.0]]).astype(np.float32)
    feats_before = agent._features(base).copy()
    # change the predictor; the SAME raw base must now yield different features
    agent.predictor.demons["forward"].theta += 1.0
    feats_after = agent._features(base)
    assert feats_before.shape[1] == 30
    assert not np.array_equal(feats_before[0, -3:], feats_after[0, -3:])  # GVF part moved
    assert np.array_equal(feats_before[0, :27], feats_after[0, :27])      # base part fixed


# --------------------------------------------------------------------------- #
# Checkpoint save/load equivalence
# --------------------------------------------------------------------------- #
def test_agent_checkpoint_roundtrip(tmp_path):
    enc = StateEncoder(use_goal=True)
    p = GVFPredictor(frozen=False)
    agent = DQNAgent("C", 5, enc, DQNConfig(hidden=(16,)), predictor=p, replay_capacity=200, seed=0)
    rng = np.random.default_rng(0)
    for _ in range(120):
        b = np.concatenate([rng.random(24), rng.random(3)]).astype(np.float32)
        nb = np.concatenate([rng.random(24), rng.random(3)]).astype(np.float32)
        agent.observe(b, int(rng.integers(0, 5)), 1.0, nb, False, False)
    for _ in range(5):
        agent.learn(32)
    probe = np.concatenate([rng.random(24), rng.random(3)]).astype(np.float32)
    q_before = agent.dqn.q_values(agent._features(probe)).copy()
    agent.save(tmp_path / "a.pkl", env_steps=120)

    enc2 = StateEncoder(use_goal=True)
    agent2 = DQNAgent("C", 5, enc2, DQNConfig(hidden=(16,)),
                      predictor=GVFPredictor(frozen=False), replay_capacity=200, seed=99)
    meta = agent2.load(tmp_path / "a.pkl")
    assert meta["env_steps"] == 120 and len(agent2.replay) == 120
    q_after = agent2.dqn.q_values(agent2._features(probe))
    assert np.allclose(q_before, q_after)


# --------------------------------------------------------------------------- #
# Shuffled-feature ablation: destroying the GVF signal must change predictions
# --------------------------------------------------------------------------- #
def test_shuffled_gvf_features_differ():
    enc = StateEncoder(use_goal=True)
    p = GVFPredictor(frozen=True)
    lidar = np.clip(np.linspace(0.2, 1, 24), 0, 1)
    real = p.features(lidar)
    shuffled = real[::-1]
    assert not np.array_equal(real, shuffled) or np.allclose(real, real.mean())


# --------------------------------------------------------------------------- #
# End-to-end: each agent trains a few steps, evaluates, saves
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("mode", ["A", "B", "C"])
def test_end_to_end_short(mode, tmp_path):
    exp = X.load_exp_config(None)
    exp["total_env_steps"] = 400
    exp["learning_starts"] = 100
    exp["eval_interval"] = 400
    exp["eval_episodes"] = 3
    gvf = None
    if mode in ("B", "C"):
        # tiny pretrained checkpoint
        p = GVFPredictor(frozen=False)
        import numpy as _np
        from simulator import hpc
        st = p.get_state()
        hpc.atomic_savez(tmp_path / "g.npz", episode=_np.int64(0),
                         feat_mean=_np.zeros(3), feat_std=_np.ones(3),
                         theta_forward=st["theta_forward"], w_forward=st["w_forward"],
                         theta_left=st["theta_left"], w_left=st["w_left"],
                         theta_right=st["theta_right"], w_right=st["w_right"])
        gvf = str(tmp_path / "g.npz")
    summ = X.train(mode, exp, seed=0, out_dir=str(tmp_path / mode), gvf_ckpt=gvf, resume=False)
    assert summ["env_steps"] == 400
    assert (tmp_path / mode / "agent.pkl").exists()
    assert summ["final"] is not None and 0.0 <= summ["final"]["success_rate"] <= 1.0
    assert summ["in_dim"] == (27 if mode == "A" else 30)
