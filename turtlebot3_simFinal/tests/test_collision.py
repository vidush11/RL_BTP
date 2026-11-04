"""Wall/obstacle collisions, blocked motion, and no tunnelling.

Most tests set ``collision_lidar_threshold: null`` so that only the GEOMETRIC
disc-vs-geometry test is exercised (otherwise the existing "min range < 0.15 m"
proximity rule ends the episode first).
"""
import math

import numpy as np
import pytest

from simulator import TurtleBot3LidarEnv
from simulator.world import World, segments_to_rects_distance
from simulator.config import WorldConfig

GEOM = {"termination": {"collision_lidar_threshold": None}}


def env_with(obstacles=None, actions=None, **extra):
    cfg = {"termination": {"collision_lidar_threshold": None}}
    if obstacles is not None:
        cfg["world"] = {"obstacles": obstacles}
    if actions is not None:
        cfg["motion"] = {"actions": actions}
    cfg.update(extra)
    return TurtleBot3LidarEnv(cfg)


def test_wall_collision_geometric_blocks_motion():
    env = env_with()
    env.reset(seed=0, options={"spawn": (1.95, 0.0, 0.0)})     # wall at x=2, radius .033
    _, r, te, tr, info = env.step(2)                           # would reach x=1.965 -> 1.998 edge
    # 1.95 + 0.015 = 1.965 ; 1.965 + 0.033 = 1.998 < 2.0  -> still free
    assert not te
    for _ in range(10):
        _, r, te, tr, info = env.step(2)
        if te:
            break
    assert te and not tr
    assert info["collision"] and info["collision_type"] == "geometric"
    x, y, th = info["pose"]
    assert x + 0.033 <= 2.0 + 1e-9                             # never inside the wall
    assert info["linear_velocity"] == 0.0 and info["angular_velocity"] == 0.0


def test_pose_is_unchanged_on_blocked_step():
    env = env_with()
    env.reset(seed=0, options={"spawn": (1.96, 0.0, 0.0)})     # legal: clearance .04 >= radius .033
    before = env.unwrapped.robot.pose
    _, _, te, _, info = env.step(2)                            # 1.96+.015=1.975 -> 2.008 > 2
    assert te and info["collision_type"] == "geometric"
    assert env.unwrapped.robot.pose == before


@pytest.mark.parametrize("heading,spawn", [
    (math.pi, (-1.96, 0.0)), (math.pi / 2, (0.0, 1.96)), (-math.pi / 2, (0.0, -1.96))])
def test_all_four_walls(heading, spawn):
    env = env_with()
    env.reset(seed=0, options={"spawn": (*spawn, heading)})
    _, _, te, _, info = env.step(2)
    assert te and info["collision_type"] == "geometric"


def test_obstacle_collision():
    env = env_with(obstacles=[[0.5, -0.5, 0.7, 0.5]])
    env.reset(seed=0)
    steps = 0
    while True:
        _, _, te, tr, info = env.step(2)
        steps += 1
        if te or tr:
            break
    assert te and info["collision_type"] == "geometric"
    x = info["pose"][0]
    assert x + 0.033 <= 0.5 + 1e-9                              # stopped in front of the box
    assert 0.5 - (x + 0.033) < 0.015 + 1e-9                     # within one step of it


def test_obstacle_side_swipe_detected():
    """Passing the box's corner at less than one radius must collide."""
    env = env_with(obstacles=[[0.5, 0.02, 0.7, 0.5]])           # bottom edge 0.02 above path (<0.033)
    env.reset(seed=0)
    for _ in range(100):
        _, _, te, _, info = env.step(2)
        if te:
            break
    assert te and info["collision_type"] == "geometric"


def test_near_miss_is_not_a_collision():
    env = env_with(obstacles=[[0.5, 0.05, 0.7, 0.5]])           # 0.05 > radius 0.033
    env.reset(seed=0)
    for _ in range(60):                                         # 0.9 m: passes beneath the box
        _, _, te, _, info = env.step(2)
        assert not te
    assert info["pose"][0] > 0.8


@pytest.mark.parametrize("speed", [1.0, 3.0, 20.0, 100.0])
def test_no_tunnelling_through_thin_obstacle(speed):
    """A 5 mm wall must stop a robot moving up to 10 m per step.

    Big world so the arena boundary cannot be what stops the robot; the spawn is
    chosen so the step straddles the wall (start and end points both free)."""
    step = speed * 0.1
    start = 1.0 - 0.5 * step
    env = env_with(obstacles=[[1.0, -1.0, 1.005, 1.0]],
                   actions=[[speed, 0.0]] * 5,
                   world={"width": 40.0, "height": 40.0,
                          "obstacles": [[1.0, -1.0, 1.005, 1.0]]})
    env.reset(seed=0, options={"spawn": (start, 0.0, 0.0)})
    w = env.unwrapped.world
    assert not w.point_hits(start, 0.0, 0.033) and not w.point_hits(start + step, 0.0, 0.033), \
        "test setup: both end points must be free so only a swept check can catch the wall"
    _, _, te, _, info = env.step(2)
    assert te and info["collision_type"] == "geometric", \
        "robot jumped through a thin obstacle (tunnelling)"
    assert info["pose"][0] == pytest.approx(start)              # blocked, not teleported


def test_endpoint_only_check_would_have_tunnelled():
    """Guards the test above: it only means something if the end point is free."""
    w = World(WorldConfig(obstacles=[[1.0, -1.0, 1.005, 1.0]]))
    assert not w.point_hits(0.7, 0.0, 0.033)
    assert not w.point_hits(1.2, 0.0, 0.033)                    # naive check passes (end point free)
    pts = np.array([[0.7, 0.0], [1.2, 0.0]])
    assert w.path_hits(pts, 0.033)                              # swept check catches it


def test_no_tunnelling_on_arc():
    # v=2, omega=5 -> radius 0.4 m, 0.5 rad per step: the arc runs from (0,0) to
    # (0.192, 0.049) and crosses the thin wall at x = 0.14.  Both end points are
    # free (0.14 m and 0.047 m from the wall), so only a swept test can see it.
    wall = [[0.14, -0.5, 0.145, 0.5]]
    env = env_with(obstacles=wall, actions=[[2.0, 5.0]] * 5)
    env.reset(seed=0, options={"spawn": (0.0, 0.0, 0.0)})
    w = env.unwrapped.world
    end = env.unwrapped.robot.plan(2.0, 5.0, 0.1)[0]
    assert not w.point_hits(0.0, 0.0, 0.033) and not w.point_hits(end[0], end[1], 0.033)
    _, _, te, _, info = env.step(2)
    assert te and info["collision_type"] == "geometric"
    assert info["pose"] == (0.0, 0.0, 0.0)


def test_segment_rect_distance_exact_values():
    rect = np.array([[1.0, 1.0, 2.0, 2.0]])
    s = lambda a, b: segments_to_rects_distance(np.array([a], float), np.array([b], float), rect)[0, 0]
    assert s((0, 0), (0.5, 0)) == pytest.approx(np.hypot(0.5, 1.0))   # nearest: (0.5,0)->corner (1,1)
    assert s((0, 1.5), (0.5, 1.5)) == pytest.approx(0.5)              # to left face
    assert s((0, 1.5), (3, 1.5)) == 0.0                               # crosses the box
    assert s((1.5, 1.5), (1.6, 1.6)) == 0.0                           # inside
    assert s((0, 3), (3, 3)) == pytest.approx(1.0)                    # parallel above
    assert s((0, 0), (3, 3)) == 0.0                                   # diagonal through corners
    assert s((3, 0), (3, 3)) == pytest.approx(1.0)                    # vertical right of it
    assert s((1.5, 0.0), (1.5, 0.0)) == pytest.approx(1.0)            # degenerate (point) segment


def test_lidar_proximity_rule_ends_episode_before_geometry():
    """Existing code: min(range) < 0.15 m => collision.  Default config keeps it."""
    env = TurtleBot3LidarEnv()
    env.reset(seed=0, options={"spawn": (1.80, 0.0, 0.0)})              # clearance .2 > .15: legal
    for _ in range(20):
        _, _, te, _, info = env.step(2)
        if te:
            break
    assert te and info["collision_type"] == "lidar"
    assert info["min_range"] < 0.15
    assert info["pose"][0] + 0.033 < 2.0                                # robot never touched the wall


def test_spawn_is_always_collision_free_with_obstacles():
    env = TurtleBot3LidarEnv({"spawn": {"mode": "random"},
                              "world": {"obstacles": [[-1, -1, 1, 1]]}})
    for seed in range(200):
        env.reset(seed=seed)
        x, y, _ = env.unwrapped.robot.pose
        assert env.unwrapped.world.clearance(x, y) >= max(0.033, 0.15)
        assert env.unwrapped.world.clearance(x, y) >= 0.35 - 1e-9


def test_illegal_spawn_is_rejected():
    env = TurtleBot3LidarEnv({"world": {"obstacles": [[-0.1, -0.1, 0.1, 0.1]]}})
    with pytest.raises(ValueError, match="not collision-free"):
        env.reset(seed=0)                                               # default spawn (0,0) is inside


def test_goal_is_collision_free():
    env = TurtleBot3LidarEnv({"task": {"mode": "goal_navigation"},
                              "spawn": {"mode": "random"},
                              "world": {"obstacles": [[-1, -1, 0, 0], [0.5, 0.5, 1.5, 1.5]]}})
    w = env.unwrapped.world
    for seed in range(100):
        _, info = env.reset(seed=seed)
        gx, gy = info["goal"]
        assert w.clearance(gx, gy) >= 0.3 - 1e-9
        assert np.hypot(gx - info["pose"][0], gy - info["pose"][1]) >= 0.5 - 1e-9
