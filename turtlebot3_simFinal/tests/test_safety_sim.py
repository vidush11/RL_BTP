"""Simulator extensions for the safety task (A1-A8). All opt-in; the default
path is covered by tests/test_golden_regression.py."""
import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from simulator import ConfigError, TurtleBot3LidarEnv, load_config  # noqa: E402

INERTIAL = {"motion": {"dynamics": "inertial", "action_mode": "accel", "v_min": 0.0, "v_max": 0.22,
                       "w_max": 2.84, "enabled_actions": ["FORWARD", "LEFT", "RIGHT", "STOP"]},
            "termination": {"collision_lidar_threshold": None}}
F, L, R, S = 0, 1, 2, 3          # indices into enabled_actions above


def env_with(**over):
    cfg = {k: dict(v) for k, v in INERTIAL.items()}
    for k, v in over.items():
        cfg.setdefault(k, {}).update(v)
    return TurtleBot3LidarEnv(load_config(cfg))


# ------------------------------------------------------------------ A1
def test_step_before_reset_raises():
    with pytest.raises(RuntimeError):
        env_with().step(F)


def test_collision_terminates_on_the_contact_step():
    env = env_with()
    env.reset(seed=0, options={"spawn": (1.8, 0.0, 0.0), "velocity": (0.22, 0.0)})
    for t in range(100):
        _, r, term, trunc, info = env.step(F)
        if term:
            break
        assert not info["collision"]
    assert term and not trunc and info["collision"] and info["collision_type"] == "geometric"
    assert info["pose"][0] + 0.033 <= 2.0 + 1e-9          # blocked before penetrating
    assert info["linear_velocity"] == 0.0 and info["angular_velocity"] == 0.0
    with pytest.raises(RuntimeError):
        env.step(F)


def test_legacy_lidar_rule_still_available():
    env = env_with(termination={"collision_lidar_threshold": 0.15})
    env.reset(seed=0, options={"spawn": (1.7, 0.0, 0.0), "velocity": (0.22, 0.0)})
    while True:
        _, _, term, _, info = env.step(F)
        if term:
            break
    assert info["collision_type"] == "lidar"


# ------------------------------------------------------------------ A2
def test_default_observation_unchanged_and_channels_shape():
    assert TurtleBot3LidarEnv().observation_space.shape == (24,)
    env = TurtleBot3LidarEnv(load_config(ROOT / "configs" / "safety_task.yaml"))
    assert env.observation_space.shape == (3 * (72 + 2),)
    obs, _ = env.reset(seed=1)
    assert env.observation_space.contains(obs)
    fr = obs.reshape(3, 74)
    assert np.array_equal(fr[0], fr[2])                       # reset fills the stack
    obs2, *_ = env.step(F)
    fr2 = obs2.reshape(3, 74)
    assert np.array_equal(fr2[1], fr[2])                      # shifted by one frame
    assert fr2[2, 72] == pytest.approx(env.robot.v / 0.22, abs=1e-6)


def test_all_channels_and_compass_variants():
    env = env_with(observation={"channels": ["lidar", "velocity", "ang_velocity", "bump", "compass", "light"],
                                "compass": "cos_sin", "frame_stack": 2},
                   light={"enabled": True, "positions": [[1.0, 1.0]]})
    assert env.observation_space.shape == (2 * (24 + 1 + 1 + 1 + 2 + 2),)
    obs, _ = env.reset(seed=0, options={"spawn": (0.0, 0.0, 0.3)})
    f = obs.reshape(2, -1)[1]
    assert f[27] == pytest.approx(math.cos(0.3), abs=1e-6) and f[28] == pytest.approx(math.sin(0.3), abs=1e-6)


def test_bump_channel_is_one_on_collision():
    env = env_with(observation={"channels": ["lidar", "bump"]})
    env.reset(seed=0, options={"spawn": (1.8, 0.0, 0.0), "velocity": (0.22, 0.0)})
    while True:
        obs, _, term, _, info = env.step(F)
        if term:
            break
        assert obs[-1] == 0.0
    assert obs[-1] == 1.0


def test_unknown_keys_and_bad_combos_raise():
    for bad in ({"observation": {"foo": 1}}, {"safety": {"bogus": 1}}, {"light": {"x": 1}},
                {"motion": {"action_mode": "accel"}},                       # needs inertial
                {"observation": {"channels": ["radar"]}},
                {"observation": {"channels": ["light"]}}):                   # light disabled
        with pytest.raises(ConfigError):
            load_config(bad)


# ------------------------------------------------------------------ A4 dynamics
def test_friction_decay_acceleration_and_clipping():
    env = env_with()
    env.reset(seed=0, options={"spawn": (-1.5, 0.0, 0.0), "velocity": (0.2, 0.0)})
    env.step(S)
    assert env.robot.v == pytest.approx(0.2 * 0.9)                     # STOP = coast under friction
    env.reset(seed=0, options={"spawn": (-1.5, 0.0, 0.0)})
    env.step(F)
    assert env.robot.v == pytest.approx(0.8 * 0.1 * 0.9)               # accelerate, then friction
    env.step(L)
    assert env.robot.omega == pytest.approx(3.0 * 0.1 * 0.9)
    assert env.robot.v == pytest.approx(0.072 * 0.9)                   # turning does not accelerate
    env.reset(seed=0, options={"spawn": (-1.9 + 0.05, 0.0, 0.0)})
    env.reset(seed=0, options={"spawn": (-1.0, 0.0, 0.0)})
    for _ in range(30):
        env.step(F)
    assert env.robot.v == pytest.approx(0.22)                          # clipped at v_max
    for _ in range(30):
        env.step(L)
    assert env.robot.omega <= 2.84 + 1e-12


def test_backward_respects_v_min_and_mask():
    env = env_with(motion={"enabled_actions": ["FORWARD", "BACKWARD", "STOP"]})
    assert env.action_space.n == 3
    env.reset(seed=0, options={"spawn": (0.0, 0.0, 0.0)})
    env.step(1)
    assert env.robot.v == 0.0                                          # v_min = 0
    assert env_with().action_names == ["FORWARD", "LEFT", "RIGHT", "STOP"]


def test_set_friction_and_reset_option():
    env = env_with()
    env.reset(seed=0, options={"spawn": (-1.5, 0.0, 0.0), "velocity": (0.2, 0.0), "friction": 0.5})
    assert env.robot.friction == 0.5
    _, _, _, _, info = env.step(S)
    assert env.robot.v == pytest.approx(0.1) and info["friction"] == 0.5
    env.unwrapped.set_friction(0.0)
    env.step(S)
    assert env.robot.v == pytest.approx(0.1)


# ------------------------------------------------------------------ A3 light
def test_light_formula_and_occlusion():
    env = env_with(light={"enabled": True, "positions": [[1.0, 0.5]]})
    env.reset(seed=0, options={"spawn": (0.0, 0.0, 0.2)})
    dx, dy, th = 1.0, 0.5, 0.2
    dist = math.hypot(dx, dy)
    exp_l = max(0, math.cos(th + 1) * dx + math.sin(th + 1) * dy) / (dist + 1e-9) / (1 + 0.5 * dist ** 2)
    exp_r = max(0, math.cos(th - 1) * dx + math.sin(th - 1) * dy) / (dist + 1e-9) / (1 + 0.5 * dist ** 2)
    assert env.light_sensors() == pytest.approx((exp_l, exp_r))
    occ = env_with(light={"enabled": True, "positions": [[1.0, 0.5]], "occlusion": True},
                   world={"obstacles": [[0.4, 0.0, 0.6, 0.5]]})
    occ.reset(seed=0, options={"spawn": (0.0, 0.0, 0.2)})
    assert occ.light_sensors() == (0.0, 0.0)


# ------------------------------------------------------------------ A5 state
def test_set_state_reproduces_trajectory():
    env = TurtleBot3LidarEnv(load_config({**load_config(ROOT / "configs" / "safety_task.yaml").to_dict(),
                                          "lidar": {"num_beams": 72, "noise_std": 0.01}}))
    env.reset(seed=3)
    rng = np.random.default_rng(0)
    for _ in range(10):
        env.step(int(rng.integers(4)))
    st = env.get_state(include_rng=True)
    acts = [int(a) for a in rng.integers(0, 4, 30)]

    def roll():
        out = []
        for a in acts:
            o, r, te, tr, info = env.step(a)
            out.append((o.copy(), r, te, tr, info["safety"]["c_front"]))
            if te or tr:
                break
        return out
    a1 = roll(); env.set_state(st); a2 = roll()
    assert len(a1) == len(a2)
    for x, y in zip(a1, a2):
        assert np.array_equal(x[0], y[0]) and x[1:] == y[1:]


# ------------------------------------------------------------------ A7 safety zone
@pytest.mark.parametrize("v", [0.0, 0.1, 0.2])
def test_safety_zone_geometry(v):
    hf = 0.4 + v * 1.5
    for D, inside in ((hf - 0.05, True), (hf + 0.05, False)):
        env = env_with(safety={"enabled": True}, world={"obstacles": [[D, -0.04, D + 0.2, 0.04]]})
        _, info = env.reset(seed=0, options={"spawn": (0.0, 0.0, 0.0), "velocity": (v, 0.0)})
        s = info["safety"]
        assert s["h_front"] == pytest.approx(hf)
        assert s["zone_width"] == pytest.approx(2 * 0.033 + 0.10)
        assert (s["c_front"] == 0.0) == inside and (s["n_front"] > 0) == inside
    side = env_with(safety={"enabled": True}, world={"obstacles": [[0.1, 0.15, 0.4, 0.3]]})
    _, info = side.reset(seed=0, options={"spawn": (0.0, 0.0, 0.0), "velocity": (0.2, 0.0)})
    assert info["safety"]["c_front"] == 1.0                               # beside, not ahead


def test_rear_zone_optional():
    env = env_with(safety={"enabled": True, "rear": True}, world={"obstacles": [[-0.5, -0.04, -0.3, 0.04]]})
    _, info = env.reset(seed=0, options={"spawn": (0.0, 0.0, 0.0)})
    assert info["safety"]["rear"]["c_rear"] == 0.0 and info["safety"]["c_front"] == 1.0


# ------------------------------------------------------------------ A8 reward
def test_task_reward_speed_term_and_collision_penalty():
    env = TurtleBot3LidarEnv(load_config(ROOT / "configs" / "safety_task.yaml"))
    env.reset(seed=0, options={"spawn": (0.0, 0.0, 0.0), "obstacles": np.zeros((0, 4))})
    _, r, *_ = env.step(F)
    assert r == pytest.approx(0.072 / 0.2)
    env.reset(seed=0, options={"spawn": (1.8, 0.0, 0.0), "velocity": (0.22, 0.0),
                               "obstacles": np.zeros((0, 4))})
    while True:
        _, r, term, _, info = env.step(F)
        if term:
            break
    assert info["collision"] and r == pytest.approx(-10.0)                # v = 0 after block
