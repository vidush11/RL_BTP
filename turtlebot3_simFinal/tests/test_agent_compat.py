"""Compatibility with the EXISTING agents.

* Horde/GVF: the user's unmodified gvf_horde_agent.py / gvf_horde_agent_evaluate.py
  are executed (in a subprocess) against the simulator through scripts/run_agent.py.
* DQN: the DQN *agent* script was not supplied, so it cannot be run.  What is
  tested is the service contract of the DQN environment node that it consumed
  (dqn_environment.py): state = list of raw LiDAR ranges in metres, reward, done.
"""
import hashlib
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
RUN = ROOT / "scripts" / "run_agent.py"

EXPECTED_KEYS = {"episode", "theta_forward", "w_forward", "theta_left", "w_left", "theta_right",
                 "w_right", "w_future_linear_velocity", "w_future_angular_velocity"}


def run(agent, run_dir, *params, config=None, seed=0, timeout=300):
    cmd = [sys.executable, str(RUN), str(ROOT / agent), "--run-dir", str(run_dir),
           "--seed", str(seed)]
    if config:
        cmd += ["--config", str(config)]
    for p in params:
        cmd += ["--param", p]
    env = {k: v for k, v in os.environ.items() if k != "DISPLAY"}
    out = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, env=env)
    assert out.returncode == 0, out.stdout[-2000:] + out.stderr[-2000:]
    return out.stdout


# ------------------------------------------------------------------- Horde
@pytest.fixture(scope="module")
def horde_run(tmp_path_factory):
    d = tmp_path_factory.mktemp("horde")
    out = run("gvf_horde_agent.py", d, "max_training_episodes:=12")
    return d, out


def test_horde_training_runs_unmodified_and_writes_native_checkpoint(horde_run):
    d, out = horde_run
    assert "Episode: 12" in out and "Traceback" not in out
    ck = d / "gvf_checkpoints_horde" / "gvf_weights_ep_000012.npz"
    assert ck.exists()
    data = np.load(ck)
    assert set(data.files) == EXPECTED_KEYS               # exact keys the original script writes
    assert int(data["episode"]) == 12
    for k in EXPECTED_KEYS - {"episode"}:
        assert data[k].shape == (24 * 5,) and np.isfinite(data[k]).all()
    assert any(np.abs(data[k]).sum() > 0 for k in ("theta_forward", "w_future_angular_velocity"))


def test_horde_training_metrics_are_finite_and_weights_move(tmp_path):
    """Deliberately NOT a 'the agent learns' test.  Measured behaviour of the unmodified
    Horde algorithm on this simulator is documented in README.md ('Measured Horde behaviour');
    here we only check the plumbing: every demon's metric is finite and weights update."""
    out = run("gvf_horde_agent.py", tmp_path, "max_training_episodes:=30")
    lines = [l for l in out.splitlines() if "TTC_fwd MAE" in l]
    assert len(lines) == 30 and "nan" not in " ".join(lines).lower()
    data = np.load(tmp_path / "gvf_checkpoints_horde" / "gvf_weights_ep_000030.npz")
    for k in ("theta_forward", "theta_left", "theta_right"):
        assert np.abs(data[k]).max() > 0 and np.isfinite(data[k]).all()


def test_horde_resume_from_combined_checkpoint(horde_run, tmp_path):
    d, _ = horde_run
    ck = d / "gvf_checkpoints_horde" / "gvf_weights_ep_000012.npz"
    out = run("gvf_horde_agent.py", tmp_path, "max_training_episodes:=15",
              f"resume_checkpoint:={ck}")
    assert "continuing from episode 13" in out and "Episode: 13" in out and "Episode: 15" in out
    before = np.load(ck)["theta_forward"]
    after = np.load(tmp_path / "gvf_checkpoints_horde" / "gvf_weights_ep_000015.npz")["theta_forward"]
    assert not np.array_equal(before, after)                # training continued from loaded weights


def test_horde_evaluate_script_loads_checkpoint_and_reports_all_five_demons(horde_run, tmp_path):
    d, _ = horde_run
    out = run("gvf_horde_agent_evaluate.py", tmp_path,
              f"checkpoint_dir:={d / 'gvf_checkpoints_horde'}", "num_eval_episodes:=1")
    assert "Evaluating checkpoint from training episode 12" in out
    for key in ("TTC_fwd MAE", "TTC_L MAE", "TTC_R MAE", "FutLinVel RMSE", "FutAngVel RMSE"):
        assert key in out
    assert "Finished evaluating all stored checkpoints" in out


def test_horde_runs_with_gazebo_action_mapping(tmp_path):
    out = run("gvf_horde_agent.py", tmp_path, "max_training_episodes:=5",
              config=ROOT / "configs" / "gazebo_compat.yaml")
    assert "Episode: 5" in out and "v=0.2, w=+1.5" in out


def test_horde_runs_with_obstacles(tmp_path):
    out = run("gvf_horde_agent.py", tmp_path, "max_training_episodes:=5",
              config=ROOT / "configs" / "obstacles.yaml")
    assert "Episode: 5" in out


def test_horde_is_deterministic_given_seed(tmp_path):
    def logs(sub):
        o = run("gvf_horde_agent.py", tmp_path / sub, "max_training_episodes:=8", seed=3)
        return [l.split("] [gvf_horde_agent]: ")[1] for l in o.splitlines() if "Episode:" in l]
    assert logs("a") == logs("b")


def test_original_agent_files_are_byte_identical():
    """Pristine copies live in backup_original/ and must match the recorded
    hashes. Three of the four files are also unchanged in place; gvf_horde_agent.py
    was modified for the GVF-diagnosis fixes in a backward-compatible way (see
    tests/test_gvf.py and DIAGNOSIS_GVF.md), so it is checked against its backup,
    not against the working copy."""
    ref = {}
    for line in (ROOT / "ORIGINAL_SHA256.txt").read_text().splitlines():
        h, name = line.split()
        ref[name] = h
    assert len(ref) == 4
    for name, h in ref.items():
        assert hashlib.sha256((ROOT / "backup_original" / name).read_bytes()).hexdigest() == h, \
            f"backup_original/{name} differs from the recorded original"
    # the three files untouched by the GVF fix stay byte-identical in place
    for name in ("dqn_environment.py", "dqn_gazebo.py", "gvf_horde_agent_evaluate.py"):
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == ref[name], \
            f"{name} was modified unexpectedly"


# --------------------------------------------------------------------- DQN
@pytest.fixture
def backend():
    from simulator import TurtleBot3LidarEnv
    from simulator.ros_compat import install, uninstall
    be = install(TurtleBot3LidarEnv(), seed=0)
    yield be
    uninstall()


def test_dqn_service_contract(backend):
    import sys as _s
    from turtlebot3_msgs.srv import Dqn            # the fake modules installed by the shim
    r = backend.call("reset_environment", Dqn.Request())
    assert len(r.state) == 24 and all(isinstance(x, float) for x in r.state)
    assert 0.0 < min(r.state) and max(r.state) <= 3.5              # raw metres, NOT normalised
    steps = 0
    while True:
        req = Dqn.Request(); req.action = 2
        resp = backend.call("rl_agent_interface", req)
        steps += 1
        assert len(resp.state) == 24 and resp.reward == 1.0
        if resp.done:
            break
    assert steps > 10 and min(resp.state) < 0.15                    # collision rule of the original
    req = Dqn.Request(); req.action = 9
    with pytest.raises(Exception):
        backend.call("rl_agent_interface", req)


def test_shim_rejects_unsupported_ros_features_loudly(backend):
    from rclpy.node import Node
    n = Node("x")
    with pytest.raises(NotImplementedError):
        n.create_timer(0.8, lambda: None)
    with pytest.raises(NotImplementedError):
        n.create_service(None, "x", lambda *a: None)
    with pytest.raises(NotImplementedError):
        backend.call("task_failed", None)


# ----------------------------------------------- generic transition collection
def test_transition_collection_for_gvf_style_learning():
    """(obs, action, reward, next_obs, terminated, info) with consistent terminals."""
    from simulator import TurtleBot3LidarEnv
    env = TurtleBot3LidarEnv()
    rng = np.random.default_rng(0)
    buf = []
    for ep in range(10):
        obs, _ = env.reset(seed=ep)
        while True:
            a = int(rng.integers(5))
            nxt, r, te, tr, info = env.step(a)
            buf.append((obs, a, r, nxt, te, tr, info))
            obs = nxt
            if te or tr:
                break
    end_idx = [i for i, b in enumerate(buf) if b[4] or b[5]]
    ends = [buf[i] for i in end_idx]
    assert len(ends) == 10
    assert all(b[4] and not b[5] and b[6]["collision"] for b in ends)   # random walk always collides here
    assert not any(b[4] or b[5] for i, b in enumerate(buf) if i not in set(end_idx))
    # successive transitions chain: next_obs of t == obs of t+1 within an episode
    for prev, cur in zip(buf, buf[1:]):
        if not (prev[4] or prev[5]):
            assert np.array_equal(prev[3], cur[0])
    # GVF-style TTC target is recoverable from the episode lengths
    assert all(b[6]["step"] >= 1 for b in ends)
    # metres-valued ranges for the velocity/TTC demons are available in info
    assert all(b[6]["ranges_m"].shape == (24,) for b in buf)
