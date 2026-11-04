"""The reward helpers in simulator/rewards.py are ports of methods in the user's
dqn_environment.py.  That file imports ROS and cannot be imported here, so the
ORIGINAL method source is pulled out with ``ast`` and executed against a
stand-in ``self``; the ports must agree with it numerically."""
import ast
import math
import textwrap
from pathlib import Path
from types import SimpleNamespace

import numpy
import numpy as np
import pytest

from simulator.lidar import Lidar
from simulator.config import LidarConfig, RewardConfig
from simulator.rewards import (RewardModel, compute_directional_weights,
                               compute_weighted_obstacle_reward, goal_angle)

ORIGINAL = Path(__file__).resolve().parent.parent / "dqn_environment.py"


def _original_methods(*names):
    tree = ast.parse(ORIGINAL.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "RLEnvironment")
    ns = {"numpy": numpy, "math": math}
    for node in cls.body:
        if isinstance(node, ast.FunctionDef) and node.name in names:
            exec(textwrap.dedent(ast.get_source_segment(ORIGINAL.read_text(), node)), ns)
    return [ns[n] for n in names]


@pytest.fixture(scope="module")
def orig():
    assert ORIGINAL.exists(), "original dqn_environment.py must sit next to the package"
    return _original_methods("compute_directional_weights", "compute_weighted_obstacle_reward",
                             "scan_sub_callback", "odom_sub_callback")


def _self():
    s = SimpleNamespace()
    s.compute_directional_weights = lambda *a, **k: ORIG_W(s, *a, **k)
    return s


def test_directional_weights_match_original(orig):
    global ORIG_W
    ORIG_W = orig[0]
    rng = np.random.default_rng(0)
    for _ in range(50):
        ang = rng.uniform(-math.pi / 2, math.pi / 2, rng.integers(1, 13))
        assert np.allclose(compute_directional_weights(ang), orig[0](None, ang))


def test_weighted_obstacle_reward_matches_original(orig):
    global ORIG_W
    ORIG_W = orig[0]
    rng = np.random.default_rng(1)
    ang_all = np.concatenate([np.arange(0, 7) * math.radians(15),
                              np.arange(18, 24) * math.radians(15)])
    n_nonzero = 0
    for _ in range(300):
        s = _self()
        s.front_angles = list(ang_all)
        s.front_ranges = list(rng.uniform(0.05, 1.5, len(ang_all)))
        want = orig[1](s)
        got = compute_weighted_obstacle_reward(s.front_ranges, s.front_angles)
        assert got == pytest.approx(want, abs=1e-12)
        n_nonzero += want != 0.0
    assert n_nonzero > 100                           # the test actually exercised the formula
    s = _self(); s.front_angles = []; s.front_ranges = []
    assert compute_weighted_obstacle_reward([], []) == orig[1](s) == 0.0


def test_front_sector_matches_original_scan_callback(orig):
    """Which beams the original calls 'front' == the RewardModel's mask."""
    L = Lidar(LidarConfig())
    fake_scan = SimpleNamespace(ranges=[1.0] * 24, angle_min=0.0,
                                angle_increment=2 * math.pi / 24)
    s = SimpleNamespace()
    orig[2](s, fake_scan)
    model = RewardModel(RewardConfig(), L.ros_angles)
    assert np.allclose(model._front_angles, s.front_angles)
    assert len(s.front_ranges) == model._front_mask.sum() == 13


def test_goal_angle_matches_original_odom_callback(orig):
    rng = np.random.default_rng(2)
    for _ in range(200):
        rx, ry, gx, gy = rng.uniform(-2, 2, 4)
        th = rng.uniform(-math.pi, math.pi)
        s = SimpleNamespace(goal_pose_x=gx, goal_pose_y=gy)
        s.euler_from_quaternion = lambda q: (0.0, 0.0, th)
        msg = SimpleNamespace(pose=SimpleNamespace(pose=SimpleNamespace(
            position=SimpleNamespace(x=rx, y=ry), orientation=None)))
        orig[3](s, msg)
        assert goal_angle(rx, ry, th, gx, gy) == pytest.approx(s.goal_angle, abs=1e-12)


def test_reward_model_combines_terms_like_the_commented_out_code():
    """reward = yaw_reward + obstacle_reward, terminal values replace it."""
    L = Lidar(LidarConfig())
    cfg = RewardConfig(per_step=0.0, yaw_coef=1.0, obstacle_coef=1.0, goal_reward=100.0,
                       collision_penalty=-50.0, terminal_mode="replace")
    m = RewardModel(cfg, L.ros_angles)
    ranges = np.full(24, 3.5)
    ranges[0] = 0.4                                   # obstacle dead ahead (front sector, <= 0.5)
    ga = 0.5
    want = (1 - 2 * abs(ga) / math.pi) + compute_weighted_obstacle_reward(
        ranges[m._front_mask], m._front_angles)
    assert m.compute(ranges, False, False, ga, 0.0) == pytest.approx(want)
    assert m.compute(ranges, True, False, ga, 0.0) == -50.0
    assert m.compute(ranges, False, True, ga, 0.0) == 100.0
    assert m.compute(ranges, True, True, ga, 0.0) == -50.0          # collision wins
