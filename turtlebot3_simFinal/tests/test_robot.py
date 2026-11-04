import math

import numpy as np
import pytest

from simulator import GAZEBO_ACTION_TABLE, SPEC_ACTION_TABLE, InvalidActionError
from simulator.config import MotionConfig, RobotConfig
from simulator.robot import ActionTable, Robot, arc_pose, wrap_angle


def test_wrap_angle_range():
    for a in np.linspace(-20, 20, 801):
        w = wrap_angle(a)
        assert -math.pi <= w < math.pi
        assert math.isclose(math.cos(w), math.cos(a), abs_tol=1e-12)
        assert math.isclose(math.sin(w), math.sin(a), abs_tol=1e-12)


def test_straight_motion():
    x, y, th = arc_pose(1.0, 2.0, math.pi / 2, v=0.15, omega=0.0, t=2.0)
    assert (x, y, th) == pytest.approx((1.0, 2.3, math.pi / 2), abs=1e-12)


def test_straight_motion_heading_45deg():
    x, y, _ = arc_pose(0.0, 0.0, math.pi / 4, 1.0, 0.0, 1.0)
    assert x == pytest.approx(math.sqrt(0.5)) and y == pytest.approx(math.sqrt(0.5))


def test_pure_arc_matches_closed_form():
    # v=0.15, omega=0.75 -> radius 0.2 m. After a quarter turn (t = pi/2 / omega)
    # starting at origin heading +x the robot is at (r, r) heading +y.
    v, om = 0.15, 0.75
    r = v / om
    t = (math.pi / 2) / om
    x, y, th = arc_pose(0.0, 0.0, 0.0, v, om, t)
    assert (x, y, th) == pytest.approx((r, r, math.pi / 2), abs=1e-12)


def test_full_circle_returns_to_start():
    v, om = 0.15, 1.5
    t = 2 * math.pi / om
    x, y, th = arc_pose(0.3, -0.2, 0.7, v, om, t)
    assert (x, y) == pytest.approx((0.3, -0.2), abs=1e-12)
    assert wrap_angle(th - 0.7) == pytest.approx(0.0, abs=1e-12)


def test_negative_omega_turns_clockwise():
    x, y, th = arc_pose(0.0, 0.0, 0.0, 0.15, -0.75, (math.pi / 2) / 0.75)
    assert th == pytest.approx(-math.pi / 2)
    assert y < 0


def test_arc_consistent_when_split_into_substeps():
    """Exact integration: composing two half steps == one full step."""
    a = arc_pose(0.1, 0.2, 0.3, 0.15, 1.5, 0.1)
    h = arc_pose(0.1, 0.2, 0.3, 0.15, 1.5, 0.05)
    b = arc_pose(*h, 0.15, 1.5, 0.05)
    assert a == pytest.approx(b, abs=1e-12)


def test_euler_formula_from_brief_is_the_small_step_limit():
    dt = 1e-4
    ex = arc_pose(0.0, 0.0, 0.4, 0.15, 0.75, dt)
    eu = (0.15 * math.cos(0.4) * dt, 0.15 * math.sin(0.4) * dt, 0.4 + 0.75 * dt)
    assert ex == pytest.approx(eu, abs=1e-9)


def test_action_table_defaults_match_brief():
    assert SPEC_ACTION_TABLE == ((0.15, -1.5), (0.15, -0.75), (0.15, 0.0), (0.15, 0.75), (0.15, 1.5))
    assert len(ActionTable(SPEC_ACTION_TABLE)) == 5


def test_gazebo_table_matches_existing_code():
    # dqn_environment.py: linear.x = 0.2, angular_vel = [1.5, 0.75, 0.0, -0.75, -1.5]
    assert [w for _, w in GAZEBO_ACTION_TABLE] == [1.5, 0.75, 0.0, -0.75, -1.5]
    assert {v for v, _ in GAZEBO_ACTION_TABLE} == {0.2}


@pytest.mark.parametrize("bad", [-1, 5, 99, 2.0, 2.5, "2", None, True, [2]])
def test_invalid_actions_raise(bad):
    with pytest.raises(InvalidActionError):
        ActionTable(SPEC_ACTION_TABLE).validate(bad)


@pytest.mark.parametrize("ok", [0, 4, np.int64(3), np.int32(1)])
def test_valid_actions(ok):
    assert ActionTable(SPEC_ACTION_TABLE).validate(ok) == int(ok)


def test_wheel_speeds():
    r = Robot(RobotConfig(), MotionConfig())
    r.commit((0, 0, 0), 0.15, 0.0)
    l, rt = r.wheel_speeds()
    assert l == pytest.approx(rt) and l == pytest.approx(0.15 / 0.033)
    r.commit((0, 0, 0), 0.0, 1.0)         # spin in place
    l, rt = r.wheel_speeds()
    assert rt == pytest.approx(-l) and rt == pytest.approx(0.5 * 0.16 / 0.033)


def test_plan_chords_are_short_enough_to_prevent_tunnelling():
    r = Robot(RobotConfig(), MotionConfig())
    r.reset(0, 0, 0)
    _, pts, _ = r.plan(5.0, 0.0, 0.1)
    chords = np.hypot(*np.diff(pts, axis=0).T)
    assert chords.max() <= 0.5 * 0.033 + 1e-12
    assert pts[-1] == pytest.approx([0.5, 0.0])
