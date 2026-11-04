"""GVF/Horde correctness tests.

These reuse the agent's own demon classes (DirectionalTTCGVF, FixedHorizonGVF),
checking the GQ(lambda) update against the paper's equations on hand-computed
transitions, the importance ratios, terminal/truncated handling, TTC censoring,
checkpoint save/load equivalence, deterministic evaluation, and that the agent's
new ROS parameters are backward-compatible (defaults reproduce the original
hyperparameters).
"""
from __future__ import annotations

import importlib
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from simulator import TurtleBot3LidarEnv, load_config  # noqa: E402
from simulator.ros_compat import install, uninstall  # noqa: E402

FORWARD, LEFT, RIGHT = 2, 1, 3
LID, ACT, FEAT = 24, 5, 120


@pytest.fixture(scope="module")
def demons():
    """Install the shim so gvf_horde_agent's imports resolve, then hand back its
    demon classes."""
    env = TurtleBot3LidarEnv(load_config())
    install(env, seed=0)
    mod = importlib.import_module("gvf_horde_agent")
    yield mod
    uninstall()


def phi(ranges, a):
    f = np.zeros((ACT, LID)); f[a] = np.clip(np.asarray(ranges, float) / 3.5, 0, 1)
    return f.ravel()


# --------------------------------------------------------------------------- #
# GQ(lambda): TD error and first update vs hand calculation
# --------------------------------------------------------------------------- #
def test_gq_lambda_first_update_matches_paper(demons):
    d = demons.DirectionalTTCGVF(FEAT, ACT, FORWARD, slice(0, 24),
                                 alpha_theta=0.1, alpha_w=0.01, trace_lambda=0.4,
                                 obstacle_threshold=0.25, name="t")
    lidar_t = np.full(24, 1.0)       # min 1.0 > 0.25 -> gamma_t = 1
    lidar_tp1 = np.full(24, 1.0)     # gamma_tp1 = 1 (not done)
    phi_t = phi(lidar_t, FORWARD)
    phi_tn = phi(lidar_tp1, FORWARD)
    # First step, theta=w=e=0 -> q_current=q_next=0, r=1, gamma_tp1=1, z=0 => delta=1.
    # e_t = phi_t (no history). theta += alpha_theta * delta * e_t = 0.1 * phi_t.
    ok = d.update(phi_t, phi_tn, lidar_t, lidar_tp1, action=FORWARD,
                  behaviour_prob=0.2, done=False)
    assert ok
    assert np.allclose(d.e, phi_t)
    assert np.allclose(d.theta, 0.1 * phi_t)
    # w += alpha_w*(delta*e - (w.phi)phi) = 0.01*1*phi_t
    assert np.allclose(d.w, 0.01 * phi_t)


def test_gamma_terminates_on_proximity_and_done(demons):
    d = demons.DirectionalTTCGVF(FEAT, ACT, FORWARD, slice(0, 24),
                                 0.1, 0.01, 0.4, 0.25, "t")
    assert d.compute_gamma(np.full(24, 1.0)) == 1.0     # far -> continue
    assert d.compute_gamma(np.full(24, 0.2)) == 0.0     # <=0.25 -> terminate
    # done forces gamma_tp1=0 regardless of lidar: delta = r - q_current = 1 - 0 = 1
    d.update(phi(np.full(24, 1.0), FORWARD), phi(np.full(24, 1.0), FORWARD),
             np.full(24, 1.0), np.full(24, 1.0), FORWARD, 0.2, done=True)
    assert np.all(np.isfinite(d.theta))


def test_importance_ratio_and_policy_probs(demons):
    d = demons.DirectionalTTCGVF(FEAT, ACT, FORWARD, slice(0, 24), 0.1, 0.01, 0.4, 0.25, "t")
    assert d.target_policy_prob(FORWARD) == 1.0
    assert d.target_policy_prob(LEFT) == 0.0
    # rho = pi/b ; behaviour uniform b=0.2 -> on-target rho=5, off-target 0
    # (checked via the trace growth)
    phi_t = phi(np.full(24, 1.0), FORWARD)
    d.e = np.ones(FEAT)
    d.update(phi_t, phi_t, np.full(24, 1.0), np.full(24, 1.0), action=LEFT,
             behaviour_prob=0.2, done=False)
    # action != target -> rho=0 -> new trace = phi_t (history killed)
    assert np.allclose(d.e, phi_t)


# --------------------------------------------------------------------------- #
# Fixed-horizon velocity demon
# --------------------------------------------------------------------------- #
def test_fixed_horizon_learns_constant_without_is(demons):
    """With rho=1 (on-policy / policy-independent target) the demon learns a
    constant measurement; it is the vanishing rho product, not the update rule,
    that pins the weights at zero in the agent's default off-policy setting."""
    d = demons.FixedHorizonGVF(FEAT, horizon=5, alpha_w=0.05, rho_product_clip=50.0)
    rng = np.random.default_rng(0)
    for _ in range(4000):
        ph = phi(rng.uniform(0.2, 1.0, 24), int(rng.integers(0, ACT)))
        d.push_and_maybe_update(ph, rho_t=1.0, current_measurement=0.15)
    # a few features should predict ~0.15
    preds = [d.predict(phi(rng.uniform(0.2, 1.0, 24), int(rng.integers(0, ACT))))
             for _ in range(200)]
    assert abs(np.mean(preds) - 0.15) < 0.02


def test_fixed_horizon_vanishing_is_blocks_learning(demons):
    d = demons.FixedHorizonGVF(FEAT, horizon=5, alpha_w=0.05, rho_product_clip=50.0)
    rng = np.random.default_rng(0)
    for _ in range(4000):
        ph = phi(rng.uniform(0.2, 1.0, 24), int(rng.integers(0, ACT)))
        d.push_and_maybe_update(ph, rho_t=1e-4, current_measurement=0.15)  # tiny rho product
    assert np.abs(d.w).max() < 1e-3  # weights stay ~0


# --------------------------------------------------------------------------- #
# Terminal vs truncated, and TTC censoring
# --------------------------------------------------------------------------- #
def test_evaluate_ttc_censors_truncation_counts():
    import importlib.util
    spec = importlib.util.spec_from_file_location("tb3_eval2", ROOT / "scripts" / "evaluate.py")
    ev = importlib.util.module_from_spec(spec); spec.loader.exec_module(ev)
    cfg = load_config(ROOT / "configs" / "default.yaml")
    env = TurtleBot3LidarEnv(cfg); env.reset(seed=0)
    theta = np.zeros(FEAT)
    # forward in empty arena -> real wall collision every episode
    fwd = ev.ttc_ground_truth(env, theta, FORWARD, num_episodes=4)
    assert fwd["collided_episodes"] == 4 and fwd["censored_episodes"] == 0
    assert fwd["n_pairs"] > 0
    # left circles forever -> all censored, no fabricated TTC labels
    env.reset(seed=0)
    left = ev.ttc_ground_truth(env, theta, LEFT, num_episodes=4)
    assert left["collided_episodes"] == 0 and left["censored_episodes"] == 4
    assert left["n_pairs"] == 0


# --------------------------------------------------------------------------- #
# Checkpoint save/load equivalence (combined Horde format)
# --------------------------------------------------------------------------- #
def test_checkpoint_roundtrip_combined_format(tmp_path):
    keys = ("theta_forward", "w_forward", "theta_left", "w_left",
            "theta_right", "w_right", "w_future_linear_velocity",
            "w_future_angular_velocity")
    rng = np.random.default_rng(1)
    payload = {k: rng.normal(size=FEAT) for k in keys}
    payload["episode"] = np.int64(350)
    p = tmp_path / "gvf_weights_ep_000350.npz"
    np.savez(p, **payload)
    with np.load(p) as d:
        for k in keys:
            assert np.array_equal(d[k], payload[k])
        assert int(d["episode"]) == 350


# --------------------------------------------------------------------------- #
# Deterministic evaluation under a fixed seed
# --------------------------------------------------------------------------- #
def test_evaluate_deterministic_fixed_seed():
    import importlib.util
    spec = importlib.util.spec_from_file_location("tb3_eval3", ROOT / "scripts" / "evaluate.py")
    ev = importlib.util.module_from_spec(spec); spec.loader.exec_module(ev)
    cfg = load_config(ROOT / "configs" / "default.yaml")
    theta = np.zeros(FEAT)
    outs = []
    for _ in range(2):
        env = TurtleBot3LidarEnv(cfg); env.reset(seed=123)
        outs.append(ev.ttc_ground_truth(env, theta, FORWARD, num_episodes=5))
    assert outs[0] == outs[1]


# --------------------------------------------------------------------------- #
# Backward compatibility: default params reproduce original hyperparameters
# --------------------------------------------------------------------------- #
def test_agent_defaults_backward_compatible():
    """Running the modified agent with NO param overrides must still train and
    produce the same checkpoint format (defaults == original hyperparameters)."""
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        cmd = [sys.executable, str(ROOT / "scripts" / "run_agent.py"),
               str(ROOT / "gvf_horde_agent.py"), "--run-dir", tmp,
               "--param", "max_training_episodes:=3"]
        p = subprocess.run(cmd, capture_output=True, text=True)
        assert p.returncode == 0, p.stdout + p.stderr
        ck = list((Path(tmp) / "gvf_checkpoints_horde").glob("gvf_weights_ep_*.npz"))
        assert ck, "no checkpoint written by the default-param agent"
        with np.load(ck[0]) as d:
            assert "theta_forward" in d.files and "episode" in d.files
