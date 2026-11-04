import math

import gymnasium as gym
import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

import simulator
from simulator import ConfigError, InvalidActionError, TurtleBot3LidarEnv, load_config

NO_LIDAR_RULE = {"termination": {"collision_lidar_threshold": None}}


def test_gymnasium_env_checker_passes():
    check_env(TurtleBot3LidarEnv(), skip_render_check=True)


def test_registered_and_make():
    env = gym.make(simulator.ENV_ID)
    obs, info = env.reset(seed=0)
    assert obs.shape == (24,)


def test_spaces_match_brief():
    env = TurtleBot3LidarEnv()
    box = env.observation_space
    assert isinstance(box, gym.spaces.Box)
    assert box.shape == (24,) and box.dtype == np.float32
    assert box.low.min() == 0.0 and box.high.max() == 1.0
    assert isinstance(env.action_space, gym.spaces.Discrete) and env.action_space.n == 5


def test_reset_returns_valid_obs_and_info():
    env = TurtleBot3LidarEnv()
    obs, info = env.reset(seed=1)
    assert env.observation_space.contains(obs)
    assert info["step"] == 0 and not info["collision"]
    assert info["pose"] == (0.0, 0.0, 0.0)           # Gazebo respawn pose
    assert info["ranges_m"].shape == (24,)


def test_observation_consistent_over_many_steps():
    env = TurtleBot3LidarEnv({"spawn": {"mode": "random"},
                              "world": {"obstacles": [[0.4, 0.4, 0.8, 0.8]]}})
    rng = np.random.default_rng(0)
    for ep in range(30):
        obs, _ = env.reset(seed=ep)
        while True:
            assert obs.shape == (24,) and obs.dtype == np.float32
            assert obs.min() >= 0.0 and obs.max() <= 1.0 and np.isfinite(obs).all()
            obs, r, te, tr, info = env.step(int(rng.integers(5)))
            assert env.observation_space.contains(obs)
            if te or tr:
                break


def test_observation_is_normalised_ranges():
    env = TurtleBot3LidarEnv()
    obs, info = env.reset(seed=0)
    assert obs == pytest.approx(info["ranges_m"] / 3.5, abs=1e-6)
    obs, _, _, _, info = env.step(2)
    assert obs == pytest.approx(info["ranges_m"] / 3.5, abs=1e-6)


def test_observation_is_lidar_only_and_goal_hidden():
    """Same geometry + different goals => identical observations."""
    cfg = {"task": {"mode": "goal_navigation"}}
    e1, e2 = TurtleBot3LidarEnv(cfg), TurtleBot3LidarEnv(cfg)
    o1, i1 = e1.reset(seed=0, options={"goal": (1.0, 1.0)})
    o2, i2 = e2.reset(seed=0, options={"goal": (-1.5, 0.5)})
    assert o1.shape == (24,) and np.array_equal(o1, o2)
    assert i1["goal"] != i2["goal"]                     # goal only available in info
    for a in (2, 3, 1):
        o1, *_ = e1.step(a)
        o2, *_ = e2.step(a)
        assert np.array_equal(o1, o2)


@pytest.mark.parametrize("bad", [-1, 5, 2.5, "x", None])
def test_invalid_action_raises_clear_error(bad):
    env = TurtleBot3LidarEnv()
    env.reset(seed=0)
    with pytest.raises(InvalidActionError, match="Action"):
        env.step(bad)


def test_valid_actions_all_work_and_move_correctly():
    for a, (v, w) in enumerate(simulator.SPEC_ACTION_TABLE):
        env = TurtleBot3LidarEnv()
        env.reset(seed=0)
        _, _, _, _, info = env.step(a)
        assert info["linear_velocity"] == v and info["angular_velocity"] == w
        x, y, th = info["pose"]
        assert th == pytest.approx(w * 0.1)
        assert math.hypot(x, y) == pytest.approx(v * 0.1, rel=0.01)


def test_default_turn_direction_and_gazebo_direction():
    spec = TurtleBot3LidarEnv()
    spec.reset(seed=0)
    assert spec.step(1)[4]["pose"][2] < 0                # brief table: action 1 = clockwise (right)
    gz = TurtleBot3LidarEnv(load_config("configs/gazebo_compat.yaml"))
    gz.reset(seed=0)
    assert gz.step(1)[4]["pose"][2] > 0                  # Gazebo mapping: action 1 = left (CCW)
    assert gz.step(1)[4]["linear_velocity"] == 0.2


def test_step_before_reset_and_after_done_raises():
    env = TurtleBot3LidarEnv()
    with pytest.raises(RuntimeError):
        env.step(0)
    env.reset(seed=0)
    for _ in range(10_000):
        _, _, te, tr, _ = env.step(2)
        if te or tr:
            break
    with pytest.raises(RuntimeError, match="reset"):
        env.step(2)
    env.reset()
    env.step(2)                                          # usable again


def test_default_reward_matches_existing_live_code():
    """Gazebo code: reward = 1.0 every step, including the collision step."""
    env = TurtleBot3LidarEnv()
    env.reset(seed=0)
    total, n = 0.0, 0
    while True:
        _, r, te, tr, info = env.step(2)
        assert r == 1.0
        total += r
        n += 1
        if te or tr:
            break
    assert te and info["collision"] and total == n


def test_collision_terminates_not_truncates():
    env = TurtleBot3LidarEnv()
    env.reset(seed=0)
    while True:
        _, _, te, tr, info = env.step(2)
        if te or tr:
            break
    assert te is True and tr is False and info["collision"]


def test_time_limit_truncates_not_terminates():
    env = TurtleBot3LidarEnv({"termination": {"max_episode_steps": 5}})
    env.reset(seed=0)
    for i in range(1, 6):
        _, _, te, tr, info = env.step(2)
        assert not te
        assert tr == (i == 5)
    assert info["step"] == 5 and not info["collision"]


def test_terminated_takes_precedence_over_truncated():
    env = TurtleBot3LidarEnv({"termination": {"max_episode_steps": 3}})
    # 0.015 m per step: ranges ahead are 0.175, 0.160, 0.145 -> crosses the 0.15 m rule on step 3
    env.reset(seed=0, options={"spawn": (1.81, 0.0, 0.0)})
    outs = [env.step(2) for _ in range(3)]
    _, _, te, tr, _ = outs[-1]
    assert te and not tr


def test_no_time_limit_when_null():
    env = TurtleBot3LidarEnv({"termination": {"max_episode_steps": None},
                              "world": {"width": 100.0, "height": 100.0}})
    env.reset(seed=0)
    for _ in range(900):
        _, _, te, tr, _ = env.step(2)
        assert not te and not tr


def test_action_repeat_scales_distance():
    env = TurtleBot3LidarEnv({"motion": {"action_repeat": 4}})
    env.reset(seed=0)
    _, _, _, _, info = env.step(2)
    assert info["pose"][0] == pytest.approx(0.15 * 0.1 * 4)


# ------------------------------------------------------------------ goal mode
GOAL = {"task": {"mode": "goal_navigation"},
        "reward": {"per_step": 0.0, "goal_reward": 100.0, "collision_penalty": -50.0,
                   "terminal_mode": "replace"}}


def test_goal_reached_terminates_with_goal_reward():
    env = TurtleBot3LidarEnv(GOAL)
    _, info = env.reset(seed=0, options={"goal": (0.3, 0.0)})
    assert info["goal"] == (0.3, 0.0)
    for i in range(30):
        _, r, te, tr, info = env.step(2)
        if te:
            break
    assert te and not tr and info["goal_reached"] and not info["collision"]
    assert r == 100.0 and info["goal_distance"] < 0.2


def test_goal_mode_collision_penalty_replace_and_add():
    env = TurtleBot3LidarEnv(GOAL)
    env.reset(seed=0, options={"goal": (-1.5, 0.0)})
    while True:
        _, r, te, tr, info = env.step(2)             # drives away from the goal into the wall
        if te or tr:
            break
    assert info["collision"] and not info["goal_reached"] and r == -50.0

    add = TurtleBot3LidarEnv({**GOAL, "reward": {**GOAL["reward"], "per_step": 1.0,
                                                   "terminal_mode": "add"}})
    add.reset(seed=0, options={"goal": (-1.5, 0.0)})
    while True:
        _, r, te, tr, info = add.step(2)
        if te or tr:
            break
    assert r == pytest.approx(1.0 - 50.0)


def test_goal_disabled_in_obstacle_mode():
    env = TurtleBot3LidarEnv()
    _, info = env.reset(seed=0)
    assert "goal" not in info and "goal_distance" not in info
    _, _, _, _, info = env.step(2)
    assert info["goal_reached"] is False


def test_progress_reward():
    cfg = {"task": {"mode": "goal_navigation"},
           "reward": {"per_step": 0.0, "progress_coef": 10.0}}
    env = TurtleBot3LidarEnv(cfg)
    env.reset(seed=0, options={"goal": (1.5, 0.0)})
    _, r, _, _, _ = env.step(2)                       # moves 0.015 m toward the goal
    assert r == pytest.approx(0.15)
    env.reset(seed=0, options={"goal": (-1.5, 0.0)})
    _, r, _, _, _ = env.step(2)
    assert r == pytest.approx(-0.15)


def test_random_grid_goals_lie_on_gazebo_lattice():
    env = TurtleBot3LidarEnv({"task": {"mode": "goal_navigation"}})
    for s in range(100):
        _, info = env.reset(seed=s)
        for c in info["goal"]:
            assert -2.1 - 1e-9 <= c <= 2.0 + 1e-9
            assert abs(c * 10 - round(c * 10)) < 1e-6    # multiples of 0.1 like randrange(-21,21)/10


# ---------------------------------------------------------------------- config
def test_unknown_config_keys_rejected():
    with pytest.raises(ConfigError):
        TurtleBot3LidarEnv({"lidar": {"num_bems": 24}})
    with pytest.raises(ConfigError):
        TurtleBot3LidarEnv({"nonsense": 1})


def test_invalid_values_rejected():
    for bad in ({"motion": {"dt": 0}}, {"task": {"mode": "x"}},
                {"world": {"obstacles": [[1, 1, 0, 0]]}}, {"lidar": {"max_range": -1}}):
        with pytest.raises(ConfigError):
            TurtleBot3LidarEnv(bad)


def test_all_shipped_yaml_configs_load_and_run():
    from pathlib import Path
    paths = sorted((Path(__file__).parent.parent / "configs").glob("*.yaml"))
    assert len(paths) >= 4
    for p in paths:
        env = TurtleBot3LidarEnv(load_config(p))
        env.reset(seed=0)
        env.step(2)


def test_random_obstacles_only_when_enabled():
    static = TurtleBot3LidarEnv({"world": {"obstacles": [[0.8, 0.8, 1.0, 1.0]]}})
    for s in range(5):
        static.reset(seed=s)
        assert static.unwrapped.world.obstacles.shape == (1, 4)
    rnd = TurtleBot3LidarEnv({"world": {"obstacles": [[0.8, 0.8, 1.0, 1.0]],
                                        "randomize_obstacles": True, "random_obstacle_count": 3}})
    layouts = set()
    for s in range(5):
        rnd.reset(seed=s)
        assert rnd.unwrapped.world.obstacles.shape == (4, 4)
        layouts.add(rnd.unwrapped.world.obstacles.tobytes())
    assert len(layouts) == 5


def test_headless_no_display_needed():
    import os
    import subprocess
    import sys
    code = ("import sys; sys.path.insert(0, '.');"
            "from simulator import TurtleBot3LidarEnv as E; e=E(); e.reset(seed=0); e.step(2);"
            "bad=[m for m in ('rclpy','matplotlib','tkinter','gazebo_msgs') if m in sys.modules];"
            "print('BAD', bad) if bad else print('OK')")
    env = {k: v for k, v in os.environ.items() if k != "DISPLAY"}
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                         cwd=os.path.dirname(os.path.dirname(__file__)), env=env)
    assert out.stdout.strip() == "OK", out.stdout + out.stderr
