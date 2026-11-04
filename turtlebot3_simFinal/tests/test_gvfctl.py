"""Tests for gvfctl (GVF controllers vs DQN study)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from simulator import TurtleBot3LidarEnv, load_config
from gvfctl.actions import ActionSpace, SPEC5, SPEED15, mirror_beams
from gvfctl.controllers import (SteerCfg, SpeedCfg, HandCfg, steer_batch, speed_batch,
                                controller_batch, hand_batch, ttc_from_safety)
from gvfctl.dqn import DQN, family
from gvfctl.envs import (action_space, assert_seeds, assert_training_seed, episode_seeds, make_env,
                         PART_ENVS, TEST_BASE, VAL_BASE)
from gvfctl.evaluate import ControllerPolicy, DQNPolicy, HandPolicy, run_episodes, classify_collision
from gvfctl.features import mirror_ranges, rich_features
from gvfctl.gvf import GVFConfig, GVFModel
from gvfctl.nn import MLP
from gvfctl import stats as S


# =================================================================== actions
@pytest.mark.parametrize("table", [SPEC5, SPEED15])
def test_mirror_map_is_symmetry_of_table(table):
    A = ActionSpace(table)
    for a in range(A.n):
        m = A.mirror[a]
        assert A.v[m] == pytest.approx(A.v[a]) and A.w[m] == pytest.approx(-A.w[a])
        assert A.mirror[m] == a                                           # involution
    P = A.persist
    assert np.allclose(P.sum(1), 1.0)
    assert np.allclose(P[np.ix_(A.mirror, A.mirror)], P)                  # policy is mirror-symmetric


def test_speed15_layout_and_persistent_neighbours():
    A = ActionSpace(SPEED15, persist_p=0.1)
    for s in range(3):
        for t in range(5):
            a = s * 5 + t
            assert A.speed_idx[a] == s and A.turn_idx[a] == t and A.grid[s, t] == a
    a = A.grid[1, 2]                     # middle speed, straight: 4 neighbours
    nb = np.flatnonzero(A.persist[a] > 0)
    assert A.persist[a, a] == pytest.approx(0.9)
    assert set(nb) == {a, A.grid[1, 1], A.grid[1, 3], A.grid[0, 2], A.grid[2, 2]}
    assert np.allclose(A.persist[a, [A.grid[1, 1], A.grid[1, 3], A.grid[0, 2], A.grid[2, 2]]], 0.025)


@pytest.mark.parametrize("part", [1, 2])
def test_world_mirror_equivariance(part):
    """Mirroring the world across the robot's axis permutes the beams and the
    mirrored action keeps the trajectories mirrored."""
    A = action_space(part)
    obs = [[0.6, 0.3, 0.9, 0.7], [-0.8, -1.2, -0.4, -0.9], [0.2, -1.0, 0.5, -0.7]]
    mobs = [[x0, -y1, x1, -y0] for x0, y0, x1, y1 in obs]
    act = {"motion": {"actions": [list(a) for a in A.table]}}
    e1 = TurtleBot3LidarEnv(load_config({"world": {"obstacles": obs}, **act}))
    e2 = TurtleBot3LidarEnv(load_config({"world": {"obstacles": mobs}, **act}))
    _, i1 = e1.reset(seed=0, options={"spawn": (-0.5, 0.4, 0.3)})
    _, i2 = e2.reset(seed=0, options={"spawn": (-0.5, -0.4, -0.3)})
    assert np.allclose(mirror_ranges(i1["ranges_m"]), i2["ranges_m"], atol=1e-9)
    rng = np.random.default_rng(0)
    for _ in range(30):
        a = int(rng.integers(A.n))
        _, _, t1, _, i1 = e1.step(a)
        _, _, t2, _, i2 = e2.step(int(A.mirror[a]))
        x1, y1, th1 = i1["pose"]; x2, y2, th2 = i2["pose"]
        assert (x1, y1, th1) == pytest.approx((x2, -y2, -th2), abs=1e-9)
        assert np.allclose(mirror_ranges(i1["ranges_m"]), i2["ranges_m"], atol=1e-9)
        assert t1 == t2
        if t1:
            break


def test_rich_features_mirror_consistent():
    r = np.random.default_rng(1).uniform(0.1, 4, 24)
    f, fm = rich_features(r), rich_features(mirror_ranges(r))
    mb = mirror_beams()
    assert np.allclose(fm[:24], f[:24][mb]) and np.allclose(fm[24:], f[24:][mb])


# ================================================================ controllers
def test_steering_rules_with_stub_predictions():
    A = action_space(1); st = A.straight_turn
    c = SteerCfg(0.6, 0.8, 0.05)
    Q = np.array([[0.1, 0.2, 0.65, 0.2, 0.1]])
    assert steer_batch(Q, np.array([st]), A, c)[0] == st                # straight & safe -> keep
    Q = np.array([[0.1, 0.9, 0.85, 0.2, 0.1]])
    assert steer_batch(Q, np.array([1]), A, c)[0] == st                 # turning & straight >= hi
    Q = np.array([[0.1, 0.7, 0.75, 0.2, 0.1]])
    assert steer_batch(Q, np.array([1]), A, c)[0] == 1                  # keep current safe turn
    Q = np.array([[0.95, 0.5, 0.3, 0.5, 0.93]])
    assert steer_batch(Q, np.array([st]), A, c)[0] == 4 or steer_batch(Q, np.array([st]), A, c)[0] == 0
    # gentlest within delta: 0.94 and 0.92 both within 0.05 of max -> prefer gentle turn index 3
    Q = np.array([[0.95, 0.2, 0.3, 0.92, 0.95]])
    assert steer_batch(Q, np.array([st]), A, c)[0] == 3
    # tie in gentleness: higher Q wins
    Q = np.array([[0.2, 0.90, 0.3, 0.93, 0.2]])
    assert steer_batch(Q, np.array([st]), A, c)[0] == 3


def test_speed_controller_rules_with_stub_predictions():
    A = action_space(2)
    cfg = SpeedCfg(beta_v_lo=0.5, beta_v_hi=0.8, v_target=0.22, k_w=0.0)
    one = lambda q, lvl, vh=0.12, wh=0.0, c=cfg: int(speed_batch(np.array([q]), np.array([lvl]),
                                                                np.array([vh]), np.array([wh]), A, c)[0])
    assert one(0.3, 2) == 1                       # unsafe -> decelerate (one level max)
    assert one(0.1, 1) == 0
    assert one(0.9, 0) == 1                       # safe & below target -> accelerate one level
    assert one(0.9, 1, vh=0.22) == 1              # predicted speed already at target -> hold
    assert one(0.65, 1) == 1                      # hysteresis band -> hold
    capped = SpeedCfg(beta_v_lo=0.5, beta_v_hi=0.8, v_target=0.22, k_w=0.9)
    assert one(0.95, 2, wh=1.5, c=capped) == 1    # upcoming sharp turn caps the speed
    assert one(0.95, 0, c=SpeedCfg(mode="fixed")) == 1
    s = SpeedCfg(mode="safety", beta_v_hi=0.8, v_target=0.22)
    assert one(0.95, 1, vh=0.22, c=s) == 2        # safety-only ignores v_hat (uses v_prev)


def test_controller_batch_part2_uses_chosen_turn_at_current_speed():
    A = action_space(2)
    Qmin = np.full((1, 15), 0.95)
    Vl = np.full((1, 15), 0.12); Va = np.zeros((1, 15))
    a = controller_batch(Qmin, Vl, Va, np.array([A.grid[1, 2]]), A, SteerCfg(), SpeedCfg(k_w=0))
    assert A.turn_idx[a[0]] == A.straight_turn and A.speed_idx[a[0]] == 2


def test_hand_rule_turns_away():
    A = action_space(1)
    r = np.full(24, 3.0); r[[0, 1, 2, 22, 23]] = 0.4; r[3:9] = 0.5        # blocked ahead, left closer
    a = hand_batch(r[None], A, HandCfg(0.5, 0.3))[0]
    assert A.w[a] < 0                                                     # turns right (negative omega)


def test_ttc_from_safety_inverts():
    for g in (0.9, 0.97):
        for T in (1, 10, 50):
            assert ttc_from_safety(1 - g ** T, g) == pytest.approx(T, rel=1e-6)


# ===================================================================== GVF
def _batch(ranges, sp, a, c, term):
    n = len(a)
    return {"s": np.repeat(ranges[None], n, 0).astype(np.float32), "sp": np.repeat(sp[None], n, 0).astype(np.float32),
            "a": np.asarray(a), "c": np.asarray(c, np.float32), "term": np.asarray(term, np.float32)}


def test_velocity_target_matches_expected_sarsa_under_controller():
    A = action_space(2)
    m = GVFModel(GVFConfig(velocity=True, d_min=0.2, headway_tau=0.5, mirror=False), A, seed=0)
    rng = np.random.default_rng(0)
    b = {"s": rng.uniform(0.2, 3, (8, 24)).astype(np.float32), "sp": rng.uniform(0.2, 3, (8, 24)).astype(np.float32),
         "a": rng.integers(0, 15, 8), "c": rng.integers(0, 2, 8).astype(np.float32),
         "term": np.array([0, 0, 1, 0, 0, 0, 1, 0], np.float32)}
    _, a, y, yv = m.targets(b)
    Qn, Vn = m.predict(b["sp"], target=True)
    act = controller_batch(Qn.min(1), Vn[:, 0] * A.v_max, np.clip(Vn[:, 1], 0, None) * A.w_max, a, A, m.steer, m.speed)
    eps, gv = m.cfg.vel_eps, m.g_v
    for i in range(8):
        pi = np.full(15, eps / 15); pi[act[i]] += 1 - eps
        c = np.array([A.v[a[i]] / A.v_max, abs(A.w[a[i]]) / A.w_max, A.w[a[i]] / A.w_max])
        exp = (1 - gv) * c + gv * (1 - b["term"][i]) * (Vn[i] * pi).sum(-1)
        assert np.allclose(yv[i], exp, atol=1e-5)
        for h, g in enumerate(m.cfg.horizons):
            es = (1 - g) * b["c"][i] + g * (1 - b["term"][i]) * (Qn[i, h] * m.P[a[i]]).sum()
            assert y[i, h] == pytest.approx(es, abs=1e-5)


def test_toy_chain_converges_to_analytic_values():
    """Self-loop chain (s' = s), persistence p=0: Q -> c, V_lin -> v(a)/v_max.
    With term=1 every step: Q -> (1-g)c, V -> (1-g_v)c."""
    A = action_space(2)
    cfg = GVFConfig(velocity=True, vel_target="persistent", persist_p=0.0, horizons=(0.9,),
                    vel_H=5, mirror=False, lr=3e-3, polyak=0.05, d_min=0.2, headway_tau=0.0)
    r = np.full(24, 1.5)
    for term, scale in ((0.0, 1.0), (1.0, None)):
        m = GVFModel(cfg, A, seed=0)
        a = np.tile(np.arange(15), 8)
        for _ in range(1500):
            m.update(_batch(r, r, a, np.ones(len(a)), np.full(len(a), term)))
        Q, V = m.predict(r[None])
        gq, gv = 0.9, m.g_v
        exp_q = 1.0 if term == 0 else (1 - gq)
        exp_v = A.v / A.v_max * (1.0 if term == 0 else (1 - gv))
        assert np.allclose(Q[0, 0], exp_q, atol=0.03)
        assert np.allclose(V[0, 0], exp_v, atol=0.03)


def test_symmetrised_prediction_is_mirror_equivariant():
    A = action_space(2)
    m = GVFModel(GVFConfig(velocity=True, d_min=0.2, headway_tau=0.5), A, seed=3)
    r = np.random.default_rng(2).uniform(0.2, 3.5, (5, 24))
    Q1, V1 = m.predict(r); Q2, V2 = m.predict(mirror_ranges(r))
    mi = A.mirror
    assert np.allclose(Q1, Q2[..., mi], atol=1e-6)
    assert np.allclose(V1[:, :2], V2[:, :2][..., mi], atol=1e-6)
    assert np.allclose(V1[:, 2], -V2[:, 2][..., mi], atol=1e-6)


def test_mlp_gradient_check():
    rng = np.random.default_rng(0)
    net = MLP([5, 7, 6, 3], seed=1, dtype=np.float64)
    x, t = rng.normal(size=(4, 5)), rng.normal(size=(4, 3))
    out = net.forward(x, cache=True)
    g = net.backward(out - t)
    W = net.W[1]; num = np.zeros_like(W)
    for i in range(W.shape[0]):
        for j in range(W.shape[1]):
            o = W[i, j]
            W[i, j] = o + 1e-6; lp = 0.5 * np.sum((net.forward(x) - t) ** 2)
            W[i, j] = o - 1e-6; lm = 0.5 * np.sum((net.forward(x) - t) ** 2)
            W[i, j] = o; num[i, j] = (lp - lm) / 2e-6
    assert np.allclose(g[2], num, atol=1e-5)


def test_dqn_terminal_mask_and_bootstrap():
    A = action_space(1)
    r = np.full(24, 1.0, np.float32)
    for term, expected in ((1.0, 1.0), (0.0, 1.0 / (1 - 0.5))):
        d = DQN(family("A", gamma=0.5, lr=3e-3, target_update=50), A, seed=0)
        for _ in range(1500):
            d.update({"s": np.repeat(r[None], 32, 0), "sp": np.repeat(r[None], 32, 0),
                      "a": np.full(32, 2), "r": np.ones(32, np.float32), "term": np.full(32, term, np.float32)})
        assert d.q_values(r)[0, 2] == pytest.approx(expected, abs=0.1)


# ============================================================ seeds / evals
def test_seed_ranges_are_disjoint_and_asserted():
    t = episode_seeds("test", "train", 200); v = episode_seeds("val", "train", 40)
    assert min(t) >= TEST_BASE and max(t) < VAL_BASE and min(v) >= VAL_BASE
    assert not set(t) & set(v)
    with pytest.raises(AssertionError):
        assert_seeds(t, "val")
    with pytest.raises(AssertionError):
        assert_training_seed(TEST_BASE)


@pytest.mark.parametrize("part", [1, 2])
def test_evaluation_is_deterministic(part):
    A = action_space(part)
    m = GVFModel(GVFConfig(velocity=part == 2, d_min=0.25 if part == 1 else 0.2,
                           headway_tau=0.0 if part == 1 else 0.5), A, seed=0)
    seeds = episode_seeds("test", "train", 4)
    for pol in (ControllerPolicy(m, A, SteerCfg()), DQNPolicy(DQN(family("B"), A, 0), A), HandPolicy(A, HandCfg())):
        r1 = run_episodes(pol, "train", part, seeds)
        r2 = run_episodes(pol, "train", part, seeds)
        assert [(x["steps"], x["distance"], x["collided"]) for x in r1] == \
               [(x["steps"], x["distance"], x["collided"]) for x in r2]


def test_gazebo_remap_preserves_physical_turn():
    A = action_space(1)
    env, remap = make_env("gazebo", 1)
    for a in range(A.n):
        assert env.actions.table[remap[a]][1] == pytest.approx(A.w[a])
    assert "gazebo" not in PART_ENVS[2]


def test_collision_classifier():
    scans = np.full((30, 24), 2.0); scans[-1, 0] = 0.1
    assert classify_collision(scans, np.zeros(29), np.full(29, 0.15), None) == "head_on"
    scans2 = np.full((30, 24), 2.0); scans2[-1, 6] = 0.1
    assert classify_collision(scans2, np.full(29, 1.5), np.full(29, 0.15), None) == "side_clip_turning"
    scans3 = np.full((30, 24), 0.5); scans3[-1, 0] = 0.1
    assert classify_collision(scans3, np.zeros(29), np.full(29, 0.15), None) == "dead_end"


# ==================================================================== stats
def test_student_t_quantiles_and_ci():
    assert S.t_ppf(0.975, 9) == pytest.approx(2.2622, abs=1e-3)
    assert S.t_ppf(0.975, 2) == pytest.approx(4.3027, abs=1e-3)
    m, h, n = S.mean_ci([1, 2, 3, 4, 5])
    assert m == 3 and h == pytest.approx(2.7764 * np.std([1, 2, 3, 4, 5], ddof=1) / np.sqrt(5), rel=1e-3)


def test_welch_mannwhitney_bootstrap_cohen():
    a = np.array([5.1, 4.9, 5.3, 5.0, 5.2]); b = np.array([4.0, 4.2, 3.9, 4.1, 4.3])
    w = S.welch(a, b)
    assert w["t"] > 0 and w["p"] < 0.001
    mw = S.mann_whitney(a, b)
    assert mw["U"] == 25 and mw["p"] < 0.05
    bs = S.paired_bootstrap(a, b, n=2000)
    assert bs["ci_lo"] > 0 and bs["p"] < 0.01
    assert S.cohen_d(a, b) > 3
    same = S.welch(a, a)
    assert same["p"] == pytest.approx(1.0)


def test_kaplan_meier():
    xs, ys = S.kaplan_meier([5, 10, 10, 20], [True, True, False, False])
    assert ys == pytest.approx([1.0, 0.75, 0.5])
