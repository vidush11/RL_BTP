"""LiDAR tests against geometry with analytically known distances."""
import math

import numpy as np
import pytest

from simulator.config import LidarConfig, WorldConfig
from simulator.lidar import Lidar
from simulator.world import World

R = 3.5


def lidar(**kw):
    return Lidar(LidarConfig(**kw))


def world(w=4.0, h=4.0, obstacles=None, center=(0.0, 0.0)):
    return World(WorldConfig(width=w, height=h, obstacles=obstacles or [], center=center))


def test_beam_angles_default_convention():
    L = lidar()
    assert L.num_beams == 24
    assert L.angles == pytest.approx(np.arange(24) * math.radians(15))
    # beam 0 ahead, 6 left (+90 deg CCW), 12 behind, 18 right
    assert L.angles[6] == pytest.approx(math.pi / 2)
    assert L.angles[12] == pytest.approx(math.pi)
    assert L.angles[18] == pytest.approx(3 * math.pi / 2)


def test_centre_of_square_room():
    d = lidar().cast(world(), 0.0, 0.0, 0.0)
    # Axis-aligned beams (0, 6, 12, 18): exactly the half-width.
    assert d[[0, 6, 12, 18]] == pytest.approx([2.0] * 4)
    # 45 deg beams (3, 9, 15, 21): 2 / cos 45 = 2*sqrt 2 = 2.828 (< 3.5)
    assert d[[3, 9, 15, 21]] == pytest.approx([2 * math.sqrt(2)] * 4)
    # general beam: min(2/|cos a|, 2/|sin a|)
    a = lidar().angles
    expected = np.minimum(2 / np.maximum(np.abs(np.cos(a)), 1e-12),
                          2 / np.maximum(np.abs(np.sin(a)), 1e-12))
    assert d == pytest.approx(expected, abs=1e-9)


def test_off_centre_distances():
    d = lidar().cast(world(), 1.0, -0.5, 0.0)
    assert d[0] == pytest.approx(1.0)        # +x wall at 2.0
    assert d[6] == pytest.approx(2.5)        # +y wall at 2.0
    assert d[12] == pytest.approx(3.0)       # -x wall at -2.0
    assert d[18] == pytest.approx(1.5)       # -y wall at -2.0


def test_heading_rotates_the_scan():
    L = lidar()
    d0 = L.cast(world(), 1.0, -0.5, 0.0)
    d90 = L.cast(world(), 1.0, -0.5, math.pi / 2)   # turn left by 6 beams
    assert d90 == pytest.approx(np.roll(d0, -6), abs=1e-9)
    # heading +90: beam 0 now looks along +y -> 2.5
    assert d90[0] == pytest.approx(2.5)


def test_obstacle_ahead():
    w = world(obstacles=[[0.5, -0.2, 0.9, 0.2]])
    d = lidar().cast(w, 0.0, 0.0, 0.0)
    assert d[0] == pytest.approx(0.5)        # front face at x = 0.5
    assert d[12] == pytest.approx(2.0)       # behind: wall
    # beam 1 (15 deg) hits the front face where y = x*tan15 -> x=0.5, y=0.134 < 0.2
    assert d[1] == pytest.approx(0.5 / math.cos(math.radians(15)))
    # beam 2 (30 deg) would reach y=0.2 at x=0.346 (< 0.5): misses the front face,
    # passes above the box -> hits +y wall (2/sin30 = 4 > 3.5 clipped? x-wall: 2/cos30=2.309)
    assert d[2] == pytest.approx(2.0 / math.cos(math.radians(30)))


def test_obstacle_side_face_hit():
    w = world(obstacles=[[0.3, 0.3, 0.6, 0.6]])
    d = lidar().cast(w, 0.0, 0.0, 0.0)
    # beam 3 (45 deg) enters the box corner at (0.3, 0.3): distance = 0.3*sqrt2
    assert d[3] == pytest.approx(0.3 * math.sqrt(2))


def test_ray_grazing_obstacle_edge_hits():
    w = world(obstacles=[[0.5, 0.0, 0.9, 0.4]])      # bottom edge on y = 0
    d = lidar().cast(w, 0.0, 0.0, 0.0)
    assert d[0] == pytest.approx(0.5)                # slab test counts the touch


def test_obstacle_behind_robot_ignored():
    w = world(obstacles=[[-1.0, -0.2, -0.6, 0.2]])
    d = lidar().cast(w, 0.0, 0.0, 0.0)
    assert d[0] == pytest.approx(2.0)                # ahead: wall
    assert d[12] == pytest.approx(0.6)               # behind: obstacle face


def test_nearest_of_several_obstacles():
    w = world(obstacles=[[1.2, -0.1, 1.4, 0.1], [0.4, -0.1, 0.6, 0.1]])
    assert lidar().cast(w, 0.0, 0.0, 0.0)[0] == pytest.approx(0.4)


def test_max_range_clipping_in_large_world():
    w = world(w=20.0, h=20.0)
    d = lidar().cast(w, 0.0, 0.0, 0.0)
    assert np.all(d == R)                              # nothing within 3.5 m
    assert np.all(lidar().normalize(d) == 1.0)


def test_max_range_configurable():
    d = lidar(max_range=1.0).cast(world(), 0.0, 0.0, 0.0)
    assert d.max() == 1.0 and d.min() == 1.0


def test_normalisation_formula_dtype_and_range():
    L = lidar()
    d = np.array([0.0, 1.75, 3.5, 10.0] + [0.5] * 20)
    n = L.normalize(d)
    assert n.dtype == np.float32 and n.shape == (24,)
    assert n[:4] == pytest.approx([0.0, 0.5, 1.0, 1.0])
    assert n.min() >= 0.0 and n.max() <= 1.0


def test_inside_obstacle_reads_zero():
    w = world(obstacles=[[-0.5, -0.5, 0.5, 0.5]])
    assert np.all(lidar().cast(w, 0.0, 0.0, 0.0) == 0.0)


def test_fov_less_than_360():
    L = lidar(num_beams=5, fov_deg=180.0, angle_min_deg=-90.0)
    assert L.angles == pytest.approx(np.radians([-90, -45, 0, 45, 90]))


def test_noise_requires_rng_and_is_clipped_and_seeded():
    L = lidar(noise_std=0.05)
    with pytest.raises(ValueError):
        L.cast(world(), 0, 0, 0)
    a = L.cast(world(), 0, 0, 0, np.random.default_rng(1))
    b = L.cast(world(), 0, 0, 0, np.random.default_rng(1))
    assert np.array_equal(a, b)
    assert a.min() >= 0 and a.max() <= R
    assert not np.allclose(a, L.cast(world(), 0, 0, 0, np.random.default_rng(2)))


def test_offset_world_centre():
    w = world(center=(1.0, 1.0))                       # spans x,y in [-1, 3]
    d = lidar().cast(w, 1.0, 1.0, 0.0)
    assert d[[0, 6, 12, 18]] == pytest.approx([2.0] * 4)
