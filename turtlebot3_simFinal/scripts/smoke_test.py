#!/usr/bin/env python3
"""Minimal end-to-end smoke test (no GUI, no ROS, no Gazebo).

    python scripts/smoke_test.py                 # environment only (~1 s)
    python scripts/smoke_test.py --horde         # + run your REAL Horde script: train, save, resume, evaluate

Demonstrates: create env -> reset -> random actions -> several episodes ->
vector env -> (optionally) save and reload a checkpoint with the existing agent code.
"""
import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import gymnasium as gym  # noqa: E402

import simulator  # noqa: E402


def env_smoke():
    env = gym.make(simulator.ENV_ID)                       # same env for DQN and Horde
    obs, info = env.reset(seed=0)
    print(f"reset: obs shape {obs.shape} dtype {obs.dtype} range [{obs.min():.3f}, {obs.max():.3f}]")
    assert obs.shape == (24,) and obs.dtype == np.float32

    lengths, returns = [], []
    for ep in range(5):
        obs, info = env.reset(seed=ep)
        ret, n = 0.0, 0
        while True:
            obs, r, terminated, truncated, info = env.step(env.action_space.sample())
            ret += r
            n += 1
            if terminated or truncated:
                break
        lengths.append(n)
        returns.append(ret)
        print(f"episode {ep}: {n:4d} steps, return {ret:7.1f}, terminated={terminated} "
              f"truncated={truncated}, collision={info['collision_type']}")
    vec = simulator.make_vec_env(4)
    vobs, _ = vec.reset(seed=0)
    for _ in range(20):
        vobs, *_ = vec.step(vec.action_space.sample())
    print(f"vector env: obs batch {vobs.shape}")
    vec.close()
    print("environment smoke test: OK")


def horde_smoke():
    run = [sys.executable, str(ROOT / "scripts" / "run_agent.py")]
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        print("\n--- Horde: train 10 episodes (your unmodified gvf_horde_agent.py) ---")
        subprocess.run(run + [str(ROOT / "gvf_horde_agent.py"), "--run-dir", str(tmp / "a"),
                              "--param", "max_training_episodes:=10"], check=True)
        ck = tmp / "a" / "gvf_checkpoints_horde" / "gvf_weights_ep_000010.npz"
        assert ck.exists(), "checkpoint was not written"
        print("\n--- Horde: reload that checkpoint and continue to episode 13 ---")
        subprocess.run(run + [str(ROOT / "gvf_horde_agent.py"), "--run-dir", str(tmp / "b"),
                              "--param", "max_training_episodes:=13",
                              "--param", f"resume_checkpoint:={ck}"], check=True)
        print("\n--- Horde: evaluate the stored checkpoint ---")
        subprocess.run(run + [str(ROOT / "gvf_horde_agent_evaluate.py"), "--run-dir", str(tmp / "c"),
                              "--param", f"checkpoint_dir:={ck.parent}",
                              "--param", "num_eval_episodes:=1"], check=True)
    print("\nHorde smoke test: OK")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--horde", action="store_true")
    args = ap.parse_args()
    env_smoke()
    if args.horde:
        horde_smoke()
