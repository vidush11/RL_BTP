"""Tests for the HPC-support utilities (simulator.hpc) and the wrapper scripts.

These never touch a cluster; they exercise the crash-safe writes, checkpoint
discovery/resume selection, resource detection, and metadata/summary writers,
plus a guard that the four original agent files remain byte-identical.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from simulator import hpc  # noqa: E402


# --------------------------------------------------------------------------- #
# Atomic writes
# --------------------------------------------------------------------------- #
def test_atomic_savez_roundtrip(tmp_path):
    p = hpc.atomic_savez(tmp_path / "ck.npz", theta=np.arange(5.0), episode=np.int64(50))
    assert p.exists() and p.suffix == ".npz"
    with np.load(p) as d:
        assert np.array_equal(d["theta"], np.arange(5.0))
        assert int(d["episode"]) == 50


def test_atomic_savez_leaves_no_tmp(tmp_path):
    hpc.atomic_savez(tmp_path / "ck.npz", x=np.zeros(3))
    leftovers = [q for q in tmp_path.iterdir() if q.name != "ck.npz"]
    assert leftovers == [], f"temp files left behind: {leftovers}"


def test_atomic_savez_replaces_existing(tmp_path):
    hpc.atomic_savez(tmp_path / "ck.npz", x=np.ones(2))
    hpc.atomic_savez(tmp_path / "ck.npz", x=np.full(2, 7.0))
    with np.load(tmp_path / "ck.npz") as d:
        assert np.array_equal(d["x"], np.full(2, 7.0))


def test_write_json_atomic(tmp_path):
    p = hpc.write_json_atomic(tmp_path / "m.json", {"a": 1, "b": [1, 2]})
    assert json.loads(p.read_text()) == {"a": 1, "b": [1, 2]}
    assert list(tmp_path.iterdir()) == [p]


def test_write_json_atomic_numpy_types(tmp_path):
    p = hpc.write_json_atomic(tmp_path / "m.json",
                              {"i": np.int64(3), "f": np.float64(1.5),
                               "arr": np.arange(3)})
    got = json.loads(p.read_text())
    assert got == {"i": 3, "f": 1.5, "arr": [0, 1, 2]}


# --------------------------------------------------------------------------- #
# Checkpoint discovery / resume
# --------------------------------------------------------------------------- #
def test_episode_of():
    assert hpc.episode_of("gvf_weights_ep_002000.npz") == 2000
    assert hpc.episode_of("gvf_weights_ep_000050.npz") == 50
    assert hpc.episode_of("no_episode_here.npz") is None


def _mk_ckpt(d: Path, ep: int):
    return hpc.atomic_savez(d / f"gvf_weights_ep_{ep:06d}.npz",
                            episode=np.int64(ep), w=np.zeros(3))


def test_scan_and_latest_sorted(tmp_path):
    for ep in (50, 150, 100):
        _mk_ckpt(tmp_path, ep)
    found = hpc.scan_checkpoints(tmp_path)
    assert [ep for ep, _ in found] == [50, 100, 150]
    assert hpc.latest_checkpoint(tmp_path)[0] == 150


def test_scan_skips_corrupt_latest(tmp_path):
    _mk_ckpt(tmp_path, 50)
    _mk_ckpt(tmp_path, 100)
    # Corrupt the newest file (truncate the zip) -> resume must fall back to 50.
    bad = tmp_path / "gvf_weights_ep_000100.npz"
    bad.write_bytes(b"PK\x03\x04 not a real npz")
    assert hpc.latest_checkpoint(tmp_path)[0] == 50
    assert [ep for ep, _ in hpc.scan_checkpoints(tmp_path)] == [50]


def test_scan_requires_key(tmp_path):
    # A file without the required 'episode' key is rejected.
    hpc.atomic_savez(tmp_path / "gvf_weights_ep_000070.npz", w=np.zeros(3))
    assert hpc.scan_checkpoints(tmp_path, require_key="episode") == []
    assert len(hpc.scan_checkpoints(tmp_path, require_key=None)) == 1


def test_scan_empty_dir(tmp_path):
    assert hpc.scan_checkpoints(tmp_path) == []
    assert hpc.latest_checkpoint(tmp_path) is None
    assert hpc.scan_checkpoints(tmp_path / "does-not-exist") == []


# --------------------------------------------------------------------------- #
# Resource detection + threads
# --------------------------------------------------------------------------- #
def test_detect_resources_host(monkeypatch):
    for k in list(os.environ):
        if k.startswith("SLURM_"):
            monkeypatch.delenv(k, raising=False)
    res = hpc.detect_resources()
    assert res.scheduler == "none"
    assert res.cpus >= 1
    assert res.job_id is None


def test_detect_resources_slurm(monkeypatch):
    monkeypatch.setenv("SLURM_JOB_ID", "12345")
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "4")
    monkeypatch.setenv("SLURM_MEM_PER_NODE", "8192")
    monkeypatch.setenv("SLURM_ARRAY_TASK_ID", "3")
    res = hpc.detect_resources()
    assert res.scheduler == "slurm"
    assert res.cpus == 4
    assert res.mem_mb == 8192
    assert res.array_task_id == 3
    assert res.job_id == "12345"


def test_configure_threads(monkeypatch):
    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        monkeypatch.delenv(v, raising=False)
    out = hpc.configure_threads(2)
    assert out["OMP_NUM_THREADS"] == "2"
    assert os.environ["OPENBLAS_NUM_THREADS"] == "2"


# --------------------------------------------------------------------------- #
# Metadata + summary
# --------------------------------------------------------------------------- #
def test_write_run_metadata(tmp_path):
    p = hpc.write_run_metadata(tmp_path, config={"seed": None, "lidar": {"num_beams": 24}},
                               seed=7, agent="gvf_horde_agent")
    meta = json.loads(p.read_text())
    assert meta["seed"] == 7 and meta["agent"] == "gvf_horde_agent"
    assert "numpy" in meta["versions"] and "resources" in meta


def test_write_run_summary(tmp_path):
    ck = tmp_path / "gvf_checkpoints_horde"
    ck.mkdir()
    _mk_ckpt(ck, 50)
    _mk_ckpt(ck, 100)
    p = hpc.write_run_summary(tmp_path, status="completed", checkpoint_dir=ck)
    s = json.loads(p.read_text())
    assert s["status"] == "completed"
    assert s["num_checkpoints"] == 2
    assert s["latest_episode"] == 100
    assert s["artifacts_ok"] is True


# --------------------------------------------------------------------------- #
# Original agent files: provenance.
#   * backup_original/ keeps the four pristine files and MUST match ORIGINAL_SHA256.
#   * The three files never touched by the GVF fix stay byte-identical in place.
#   * gvf_horde_agent.py was modified (GVF diagnosis fixes) in a BACKWARD-
#     COMPATIBLE way -- see test_agent_defaults_backward_compatible below.
# --------------------------------------------------------------------------- #
def _original_hashes():
    expected = {}
    for line in (ROOT / "ORIGINAL_SHA256.txt").read_text().splitlines():
        if line.strip():
            h, name = line.split()
            expected[name] = h
    return expected


def test_backup_matches_original_hashes():
    for name, want in _original_hashes().items():
        got = hashlib.sha256((ROOT / "backup_original" / name).read_bytes()).hexdigest()
        assert got == want, f"backup_original/{name} differs from the recorded original"


def test_untouched_originals_unchanged():
    for name in ("dqn_environment.py", "dqn_gazebo.py", "gvf_horde_agent_evaluate.py"):
        got = hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
        assert got == _original_hashes()[name], f"{name} must stay byte-identical"


# --------------------------------------------------------------------------- #
# train.py overwrite safeguard (integration, fast)
# --------------------------------------------------------------------------- #
def test_train_overwrite_safeguard(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    # Pre-seed a metadata file describing a DIFFERENT experiment.
    (run_dir / "run_metadata.json").write_text(json.dumps(
        {"agent": "other_agent", "seed": 99, "config_hash": "deadbeef"}))
    cmd = [sys.executable, str(ROOT / "scripts" / "train.py"),
           "--agent", "gvf_horde_agent.py", "--seed", "0",
           "--run-dir", str(run_dir), "--episodes", "1"]
    p = subprocess.run(cmd, capture_output=True, text=True)
    assert p.returncode == 2, p.stdout + p.stderr
    assert "REFUSING" in (p.stdout + p.stderr)


# --------------------------------------------------------------------------- #
# evaluate.py ground-truth helpers
# --------------------------------------------------------------------------- #
def _load_evaluate_module():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "tb3_evaluate", ROOT / "scripts" / "evaluate.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_feature_vector_places_lidar_in_action_block():
    ev = _load_evaluate_module()
    ranges = np.full(24, 1.75)  # half of max_range -> 0.5 after /3.5
    phi = ev.feature_vector(ranges, action=2)
    phi2d = phi.reshape(ev.ACTION_SIZE, ev.LIDAR_SIZE)
    assert np.allclose(phi2d[2], 0.5)
    assert np.all(phi2d[[0, 1, 3, 4]] == 0.0)


def test_rho_product_diagnostic_matches_readme():
    ev = _load_evaluate_module()
    d = ev.rho_product_diagnostic(horizon=20, trials=200_000, seed=0)
    # README: the 19-step product exceeds 1e-3 only ~0.03% of the time.
    assert 0.0 <= d["frac_gt_1e-3"] < 0.002
    assert d["median"] < 1e-6  # heavily suppressed


def test_ttc_ground_truth_censors_truncations():
    """Deterministic LEFT in the empty default arena circles forever -> every
    episode truncates -> all censored, MAE is nan (not a bogus countdown)."""
    import math
    ev = _load_evaluate_module()
    from simulator import load_config, TurtleBot3LidarEnv
    cfg = load_config(ROOT / "configs" / "default.yaml")
    env = TurtleBot3LidarEnv(cfg)
    env.reset(seed=0)
    theta = np.zeros(ev.ACTION_SIZE * ev.LIDAR_SIZE)
    res = ev.ttc_ground_truth(env, theta, ev.LEFT_ACTION, num_episodes=3)
    assert res["collided_episodes"] == 0
    assert res["censored_episodes"] == 3
    assert math.isnan(res["mae"])
