import sys

import numpy as np
import pytest

from simulator import TurtleBot3LidarEnv, make_vec_env

CFG_RANDOM = {
    "spawn": {"mode": "random"},
    "task": {"mode": "goal_navigation"},
    "lidar": {"noise_std": 0.01},
    "world": {"obstacles": [[0.8, 0.8, 1.1, 1.1]], "randomize_obstacles": True,
              "random_obstacle_count": 2},
}


def trace(env, seed, n=60):
    """Everything random: spawn, goal, obstacles, LiDAR noise, and the action stream."""
    obs, info = env.reset(seed=seed)
    act_rng = np.random.default_rng(seed)
    rec = [obs.copy(), np.array(info["pose"]), np.array(info["goal"]),
           env.unwrapped.world.obstacles.copy()]
    for _ in range(n):
        obs, r, te, tr, info = env.step(int(act_rng.integers(5)))
        rec += [obs.copy(), np.array([r, te, tr, *info["pose"]])]
        if te or tr:
            break
    return rec


def same(a, b):
    return len(a) == len(b) and all(np.array_equal(x, y) for x, y in zip(a, b))


def test_same_seed_same_everything():
    a = trace(TurtleBot3LidarEnv(CFG_RANDOM), 123)
    b = trace(TurtleBot3LidarEnv(CFG_RANDOM), 123)
    assert same(a, b)


def test_reset_seed_123_reproducible_on_same_instance():
    env = TurtleBot3LidarEnv(CFG_RANDOM)
    a = trace(env, 123)
    trace(env, 999)                                   # disturb the stream
    b = trace(env, 123)
    assert same(a, b)


def test_different_seeds_differ():
    a = trace(TurtleBot3LidarEnv(CFG_RANDOM), 1)
    b = trace(TurtleBot3LidarEnv(CFG_RANDOM), 2)
    assert not same(a, b)


def test_seed_from_config_applies_on_first_reset_only():
    cfg = {**CFG_RANDOM, "seed": 7}
    e1, e2 = TurtleBot3LidarEnv(cfg), TurtleBot3LidarEnv(cfg)
    a1, i1 = e1.reset()
    a2, i2 = e2.reset()
    assert np.array_equal(a1, a2) and i1["pose"] == i2["pose"]
    b1, j1 = e1.reset()                               # continues the stream: next reset differs
    assert j1["pose"] != i1["pose"]
    b2, j2 = e2.reset()
    assert j1["pose"] == j2["pose"]                   # ...but identically on both instances


def test_unseeded_resets_continue_deterministic_stream_after_seeding():
    e1, e2 = TurtleBot3LidarEnv(CFG_RANDOM), TurtleBot3LidarEnv(CFG_RANDOM)
    e1.reset(seed=5), e2.reset(seed=5)
    for _ in range(5):
        _, i1 = e1.reset()
        _, i2 = e2.reset()
        assert i1["pose"] == i2["pose"] and i1["goal"] == i2["goal"]


def test_instances_do_not_share_state_or_rng():
    a, b = TurtleBot3LidarEnv(CFG_RANDOM), TurtleBot3LidarEnv(CFG_RANDOM)
    assert a.unwrapped.world is not b.unwrapped.world
    assert a.unwrapped.robot is not b.unwrapped.robot
    # interleaved stepping must equal running each alone
    solo_a = trace(TurtleBot3LidarEnv(CFG_RANDOM), 11)
    solo_b = trace(TurtleBot3LidarEnv(CFG_RANDOM), 22)

    oa, ia = a.reset(seed=11)
    ob, ib = b.reset(seed=22)
    ra, rb = np.random.default_rng(11), np.random.default_rng(22)
    ta, tb = [oa.copy(), np.array(ia["pose"]), np.array(ia["goal"]), a.unwrapped.world.obstacles.copy()], \
             [ob.copy(), np.array(ib["pose"]), np.array(ib["goal"]), b.unwrapped.world.obstacles.copy()]
    doa = dob = False
    for _ in range(60):
        if not doa:
            o, r, te, tr, i = a.step(int(ra.integers(5)))
            ta += [o.copy(), np.array([r, te, tr, *i["pose"]])]
            doa = te or tr
        if not dob:
            o, r, te, tr, i = b.step(int(rb.integers(5)))
            tb += [o.copy(), np.array([r, te, tr, *i["pose"]])]
            dob = te or tr
    assert same(ta, solo_a) and same(tb, solo_b)


def test_module_has_no_shared_mutable_default_config():
    a = TurtleBot3LidarEnv({"world": {"obstacles": [[0.8, 0.8, 1.0, 1.0]]}})
    b = TurtleBot3LidarEnv()
    a.unwrapped.world.set_obstacles([[0.1, 0.1, 0.2, 0.2], [0.3, 0.3, 0.4, 0.4]])
    assert b.unwrapped.world.obstacles.shape == (0, 4)
    assert TurtleBot3LidarEnv().unwrapped.cfg.world.obstacles == []


def test_sync_vector_env_matches_individual_envs():
    n, base = 4, 100
    vec = make_vec_env(n, CFG_RANDOM)
    vobs, _ = vec.reset(seed=base)
    solo = [TurtleBot3LidarEnv(CFG_RANDOM) for _ in range(n)]
    sobs = np.stack([e.reset(seed=base + i)[0] for i, e in enumerate(solo)])
    assert np.array_equal(vobs, sobs)
    assert len({vobs[i].tobytes() for i in range(n)}) == n      # distinct streams per instance
    alive = np.ones(n, bool)
    for t in range(15):
        acts = np.array([(t + i) % 5 for i in range(n)])
        vobs, vr, vte, vtr, _ = vec.step(acts)
        for i, e in enumerate(solo):
            if alive[i]:
                o, r, te, tr, _ = e.step(int(acts[i]))
                assert np.array_equal(vobs[i], o) and vr[i] == r
                alive[i] = not (te or tr)
    vec.close()


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="fork-based async test is Linux-only")
def test_async_vector_env_runs_independent_processes():
    n = 3
    vec = make_vec_env(n, CFG_RANDOM, asynchronous=True)
    obs, _ = vec.reset(seed=50)
    solo = np.stack([TurtleBot3LidarEnv(CFG_RANDOM).reset(seed=50 + i)[0] for i in range(n)])
    assert np.array_equal(obs, solo)
    for _ in range(5):
        obs, *_ = vec.step(np.array([2, 3, 1]))
        assert obs.shape == (n, 24) and obs.dtype == np.float32
    vec.close()
