"""Experiment D tests: strict episodic collision handling, LiDAR-only
observation isolation, prediction-reward correctness/temporal consistency, and
learning behaviour. All tests drive the real environment; collision handling is
never bypassed or weakened.
"""
from __future__ import annotations

import copy
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from simulator import TurtleBot3LidarEnv, load_config
from rl.experiment_d import (build_d, load_d_config, process_transition, train_d,
                             evaluate_d, load_d_checkpoint)
from rl.prediction_reward import PredictionReward, PredictionRewardConfig

FORWARD = 2


def _env(**over):
    base = {"task": {"mode": "goal_navigation"}}
    for k, v in over.items():
        base.setdefault(k, {}).update(v)
    return TurtleBot3LidarEnv(load_config(base))


def _drive_forward_until_done(env, max_steps=500):
    for _ in range(max_steps):
        obs, r, term, trunc, info = env.step(FORWARD)
        if term or trunc:
            return obs, term, trunc, info
    raise AssertionError("episode did not end")


def _tiny_exp(**over):
    exp = load_d_config(None, {"total_env_steps": 600, "learning_starts": 100,
                               "eval_interval": 600, "eval_episodes": 2,
                               "gvf_eval_episodes": 1, "replay_capacity": 5000})
    for k, v in over.items():
        if isinstance(v, dict):
            exp[k].update(v)
        else:
            exp[k] = v
    return exp


# =========================================================================== #
# Episodic collision behaviour
# =========================================================================== #
def test_wall_collision_terminates_episode():
    env = _env()
    env.reset(seed=0, options={"spawn": (1.6, 0.0, 0.0), "goal": (-1.5, -1.5)})
    _obs, term, trunc, info = _drive_forward_until_done(env)
    assert term and not trunc
    assert info["collision"] and not info["goal_reached"]
    assert info["pose"][0] < 2.0                     # never penetrated the wall


def test_wall_collision_geometric_only_terminates():
    """With the LiDAR proximity rule disabled, the authoritative geometric
    swept-disc check alone must end the episode at the wall."""
    env = _env(termination={"collision_lidar_threshold": None})
    env.reset(seed=0, options={"spawn": (1.6, 0.0, 0.0), "goal": (-1.5, -1.5)})
    _obs, term, _trunc, info = _drive_forward_until_done(env)
    assert term and info["collision"] and info["collision_type"] == "geometric"


def test_static_obstacle_collision_terminates_episode():
    env = _env(world={"obstacles": [[0.4, -0.3, 0.6, 0.3]]},
               termination={"collision_lidar_threshold": None})
    env.reset(seed=0, options={"spawn": (0.0, 0.0, 0.0), "goal": (-1.5, 1.5)})
    _obs, term, _trunc, info = _drive_forward_until_done(env)
    assert term and info["collision"] and info["collision_type"] == "geometric"
    assert info["pose"][0] < 0.4                     # stopped before the obstacle face


def test_collision_between_observation_times_is_caught():
    """1 m per step towards a 5 mm wall: endpoint-only checks would tunnel."""
    env = _env(motion={"actions": [[10.0, 0.0]] * 5},
               world={"obstacles": [[0.5, -0.5, 0.505, 0.5]]},
               termination={"collision_lidar_threshold": None})
    env.reset(seed=0, options={"spawn": (0.0, 0.0, 0.0), "goal": (-1.5, 1.5)})
    _obs, _r, term, _trunc, info = env.step(0)
    assert term and info["collision"] and info["pose"][0] < 0.5


def test_no_further_steps_after_termination():
    env = _env()
    env.reset(seed=0, options={"spawn": (1.6, 0.0, 0.0), "goal": (-1.5, -1.5)})
    _drive_forward_until_done(env)
    pose = env.robot.x, env.robot.y, env.robot.theta
    with pytest.raises(RuntimeError):
        env.step(FORWARD)
    assert (env.robot.x, env.robot.y, env.robot.theta) == pose


def test_collision_takes_precedence_over_goal():
    """Goal 4 cm ahead, wall 16 cm ahead: the first step is inside goal
    tolerance AND triggers the collision rule -> collision wins."""
    env = _env()
    env.reset(seed=0, options={"spawn": (1.84, 0.0, 0.0), "goal": (1.88, 0.0)})
    _obs, _r, term, _trunc, info = env.step(FORWARD)
    assert term and info["collision"] and not info["goal_reached"]
    rm = PredictionReward(PredictionRewardConfig(), dqn_gamma=0.99)
    assert rm.task_reward(collision=True, success=True, truncated=False) == -50.0


def test_goal_success_terminates_and_is_distinct_from_collision():
    env = _env()
    env.reset(seed=0, options={"spawn": (0.0, 0.0, 0.0), "goal": (0.25, 0.0)})
    _obs, term, _trunc, info = _drive_forward_until_done(env)
    assert term and info["goal_reached"] and not info["collision"]


def test_reset_permits_new_episode():
    env = _env()
    env.reset(seed=0, options={"spawn": (1.6, 0.0, 0.0), "goal": (-1.5, -1.5)})
    _drive_forward_until_done(env)
    obs, info = env.reset(seed=1)
    assert info["step"] == 0 and not info["collision"]
    env.step(FORWARD)                                # must not raise


def test_truncation_is_not_termination():
    env = _env(termination={"max_episode_steps": 5}, world={"obstacles": []})
    env.reset(seed=0, options={"spawn": (-1.5, 0.0, 0.0), "goal": (1.5, 1.5)})
    _obs, term, trunc, info = _drive_forward_until_done(env)
    assert trunc and not term and not info["collision"]


# =========================================================================== #
# Observation isolation: DQN sees LiDAR only
# =========================================================================== #
def test_dqn_input_is_lidar_only():
    exp = _tiny_exp()
    agent, gvf, rm = build_d(exp, seed=0)
    env = TurtleBot3LidarEnv(load_config(exp["sim_config"]))
    assert agent.in_dim == env.observation_space.shape[0] == 24
    assert agent.encoder.use_goal is False and agent.predictor is None
    assert agent.dqn.q.W[0].shape[0] == 24 and agent.dqn.target.W[0].shape[0] == 24
    obs, info = env.reset(seed=0)
    base = agent.encoder.base_obs(obs, info)
    assert base.shape == (24,) and np.array_equal(base, obs.astype(np.float32))


def test_replay_never_contains_gvf_predictions(tmp_path):
    exp = _tiny_exp()
    train_d(exp, seed=0, out_dir=str(tmp_path), resume=False, log=lambda *_: None)
    st = pickle.load(open(tmp_path / "agent.pkl", "rb"))
    obs = np.asarray(st["replay"]["obs"])
    assert obs.shape[1] == 24
    n = st["replay"]["i"]
    assert np.all((obs[:n] >= 0.0) & (obs[:n] <= 1.0))   # normalised LiDAR range only


def test_checkpoint_and_eval_keep_same_observation_contract(tmp_path):
    exp = _tiny_exp()
    train_d(exp, seed=0, out_dir=str(tmp_path), resume=False, log=lambda *_: None)
    agent, gvf, rm, exp2 = load_d_checkpoint(str(tmp_path / "agent.pkl"))
    assert agent.in_dim == 24 and agent.dqn.q.W[0].shape[0] == 24
    from rl.experiment import make_env
    agg = evaluate_d(agent, gvf, rm, make_env(exp2), 2, 123)
    assert 0.0 <= agg["success_rate"] <= 1.0


# =========================================================================== #
# Reward correctness
# =========================================================================== #
@pytest.fixture
def rm():
    return PredictionReward(PredictionRewardConfig(lam=2.0), dqn_gamma=0.9)


def test_reward_finite_under_extreme_predictions(rm):
    for pred in ([np.inf, -np.inf, np.nan], [1e300, -1e300, 0.0], [-5.0, 1e9, 3.0]):
        phi = rm.potential(pred)
        assert 0.0 <= phi < 1.0 + 1e-12
        for flags in ((True, False, True, False), (False, True, True, False),
                      (False, False, False, True), (False, False, False, False)):
            rb = rm.compute(phi, phi, collision=flags[0], success=flags[1],
                            terminated=flags[2], truncated=flags[3])
            assert np.isfinite(rb.total)
    assert rm.nonfinite_count > 0                    # counted, not hidden


def test_terminal_rewards(rm):
    rb_c = rm.compute(0.3, 0.8, collision=True, success=False, terminated=True, truncated=False)
    rb_g = rm.compute(0.3, 0.8, collision=False, success=True, terminated=True, truncated=False)
    assert rb_c.task == -50.0 and rb_g.task == 100.0
    # terminal next-state potential is zero -> shaping = -Phi(s)
    assert rb_c.gvf == pytest.approx(-0.3) and rb_c.phi_sp == 0.0


def test_reward_components_sum(rm):
    rb = rm.compute(0.2, 0.7, collision=False, success=False, terminated=False, truncated=False)
    assert rb.gvf == pytest.approx(0.9 * 0.7 - 0.2)
    assert rb.total == pytest.approx(rb.task + 2.0 * rb.gvf)


def test_potential_shaping_telescopes():
    """With gamma=1 and fixed predictions, shaping over an episode sums to
    Phi(s_T) - Phi(s_0): circling in a safe region cannot farm reward."""
    rm1 = PredictionReward(PredictionRewardConfig(gamma=1.0), dqn_gamma=0.99)
    phis = [0.9, 0.95, 0.9, 0.95, 0.9, 0.95]
    total = sum(rm1.compute(a, b, collision=False, success=False, terminated=False,
                            truncated=False).gvf for a, b in zip(phis[:-1], phis[1:]))
    assert total == pytest.approx(phis[-1] - phis[0])


def test_higher_predicted_ttc_means_higher_potential(rm):
    assert rm.potential([10, 10, 10]) < rm.potential([100, 100, 100])


def test_prediction_only_ablation_drops_task_reward():
    r = PredictionReward(PredictionRewardConfig(prediction_only=True), dqn_gamma=0.9)
    rb = r.compute(0.1, 0.5, collision=True, success=False, terminated=True, truncated=False)
    assert rb.task == 0.0 and rb.total == pytest.approx(-0.1)


def test_reward_uses_pre_update_gvf_and_is_stored_unchanged():
    """process_transition must score (s, s') with the GVF weights as they were
    BEFORE this transition's GVF update, store exactly that reward in replay,
    and only then update the GVF."""
    exp = _tiny_exp()
    agent, gvf, rm = build_d(exp, seed=0)
    for d in gvf.demons.values():                    # non-trivial weights
        d.theta[:] = np.linspace(0, 60, d.theta.size)
    snapshot = copy.deepcopy(gvf)
    rm_ref = PredictionReward(rm.cfg, dqn_gamma=exp["dqn"]["gamma"])
    rng = np.random.default_rng(0)
    s, sp = rng.uniform(0.1, 1, 24).astype(np.float32), rng.uniform(0.1, 1, 24).astype(np.float32)
    rb, deltas, _ = process_transition(agent, gvf, rm, s, FORWARD, sp,
                                       {"collision": False, "goal_reached": False},
                                       False, False, behaviour_prob=0.2)
    expect = rm_ref.compute(rm_ref.potential(snapshot.raw_predict(s)),
                            rm_ref.potential(snapshot.raw_predict(sp)),
                            collision=False, success=False, terminated=False, truncated=False)
    assert rb.total == pytest.approx(expect.total)
    assert agent.replay._rew[0] == pytest.approx(expect.total, rel=1e-6)
    assert not np.array_equal(snapshot.theta_snapshot(), gvf.theta_snapshot())  # updated after
    # the stored reward is not recomputed when the GVF later changes
    gvf.demons["forward"].theta += 100.0
    assert agent.replay._rew[0] == pytest.approx(expect.total, rel=1e-6)


# =========================================================================== #
# Learning behaviour
# =========================================================================== #
def test_short_run_updates_gvf_and_dqn_and_stores_terminals(tmp_path):
    exp = _tiny_exp()
    agent0, gvf0, _ = build_d(exp, seed=0)
    w0 = agent0.dqn.q.W[0].copy(); th0 = gvf0.theta_snapshot().copy()
    summ = train_d(exp, seed=0, out_dir=str(tmp_path), resume=False, log=lambda *_: None)
    assert summ["env_steps"] == 600 and summ["in_dim"] == 24
    agent, gvf, _rm, _ = load_d_checkpoint(str(tmp_path / "agent.pkl"), seed=0)
    assert not np.allclose(w0, agent.dqn.q.W[0])                 # DQN learned
    assert not np.array_equal(th0, gvf.theta_snapshot())          # GVF learned
    eps = [json.loads(l) for l in (tmp_path / "episodes.jsonl").read_text().splitlines()]
    st = pickle.load(open(tmp_path / "agent.pkl", "rb"))
    n = st["replay"]["i"]
    n_terminal_eps = sum(e["collision"] or e["success"] for e in eps)
    assert int(np.asarray(st["replay"]["term"])[:n].sum()) == n_terminal_eps
    for e in eps:                                                 # outcomes distinct & logged
        assert not (e["collision"] and e["success"])
        assert e["return_total"] == pytest.approx(
            e["return_task"] + exp["reward"]["lam"] * e["gvf_reward_sum"], abs=1e-6)


def test_replay_batch_trains_and_terminal_flag_masks_bootstrap(tmp_path):
    exp = _tiny_exp()
    agent, gvf, rm = build_d(exp, seed=0)
    env = TurtleBot3LidarEnv(load_config({"task": {"mode": "goal_navigation"}}))
    obs, info = env.reset(seed=0, options={"spawn": (1.6, 0.0, 0.0), "goal": (-1.5, -1.5)})
    base = agent.encoder.base_obs(obs, info)
    while True:
        nobs, _r, term, trunc, info = env.step(FORWARD)
        nbase = agent.encoder.base_obs(nobs, info)
        process_transition(agent, gvf, rm, base, FORWARD, nbase, info, term, trunc, 0.2)
        base = nbase
        if term or trunc:
            break
    last = (agent.replay._i - 1) % agent.replay.capacity
    assert agent.replay._term[last] == 1.0 and agent.replay._trunc[last] == 0.0
    while len(agent.replay) < 64:
        agent.replay.add(base, 0, 0.0, base, False, False)
    assert isinstance(agent.learn(32), float)


def test_checkpoint_resume_restores_state(tmp_path):
    exp = _tiny_exp()
    train_d(exp, seed=0, out_dir=str(tmp_path), resume=False, log=lambda *_: None)
    _a, gvf_saved, _r, _ = load_d_checkpoint(str(tmp_path / "agent.pkl"))
    exp2 = dict(exp, total_env_steps=900)
    summ = train_d(exp2, seed=0, out_dir=str(tmp_path), resume=True, log=lambda *_: None)
    assert summ["env_steps"] == 900
    hist = json.loads((tmp_path / "history.json").read_text())
    assert [h["env_step"] for h in hist] == [600, 900]
    eps = [json.loads(l) for l in (tmp_path / "episodes.jsonl").read_text().splitlines()]
    assert all(b["env_step"] > a["env_step"] for a, b in zip(eps, eps[1:]))


def test_gvf_concurrent_by_default_and_frozen_only_as_ablation():
    exp = _tiny_exp()
    assert exp["gvf"]["frozen"] is False                      # main experiment: concurrent
    _a, gvf, _r = build_d(exp, seed=0)
    assert gvf.frozen is False
    exp_f = _tiny_exp(gvf={"frozen": True})
    agent, gvf_f, rm = build_d(exp_f, seed=0)
    before = gvf_f.theta_snapshot().copy()
    s, sp = np.full(24, 0.5, np.float32), np.full(24, 0.4, np.float32)
    _rb, deltas, _ = process_transition(agent, gvf_f, rm, s, FORWARD, sp,
                                        {"collision": False, "goal_reached": False},
                                        False, False, behaviour_prob=0.2)
    assert deltas is None and np.array_equal(before, gvf_f.theta_snapshot())
