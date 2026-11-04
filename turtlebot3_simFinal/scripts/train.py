#!/usr/bin/env python3
"""Resource-aware, resumable training wrapper for the HPC workflow.

This is the entry point the SLURM scripts call.  It wraps ``run_agent.py``
(which runs your UNMODIFIED agent) and adds everything a long cluster run needs:

  * reads the scheduler's CPU allocation and pins the numeric-library thread
    pools (no oversubscription when one task == one core);
  * **auto-resume**: finds the newest VALID checkpoint already in the run dir
    and passes it as ``resume_checkpoint`` so a requeued/restarted job continues
    instead of starting over (``--no-resume`` to force a fresh start);
  * **overwrite safeguard**: refuses to write into a run dir that already holds a
    DIFFERENT experiment (different agent/seed/config) unless ``--force``;
  * snapshots the resolved config and writes ``run_metadata.json`` before and
    ``run_summary.json`` after (even if the run is interrupted), so every run is
    reproducible and its outputs are discoverable.

Example (local):
    python scripts/train.py --agent gvf_horde_agent.py --seed 0 \
        --run-dir runs/horde_s0 --episodes 2000

The four original agent files are never modified; this only orchestrates them.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from simulator import hpc, load_config  # noqa: E402

DEFAULT_CHECKPOINT_SUBDIR = "gvf_checkpoints_horde"
EPISODE_PARAM = "max_training_episodes"


def _config_hash(cfg_dict: dict) -> str:
    blob = json.dumps(cfg_dict, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:12]


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--agent", default="gvf_horde_agent.py",
                    help="agent script to run (default: gvf_horde_agent.py)")
    ap.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--run-dir", default=None,
                    help="output directory (default: runs/<agent>_seed<seed>)")
    ap.add_argument("--episodes", type=int, default=None,
                    help=f"maps to the agent's '{EPISODE_PARAM}' parameter")
    ap.add_argument("--checkpoint-subdir", default=DEFAULT_CHECKPOINT_SUBDIR,
                    help="where the agent writes checkpoints, relative to run-dir")
    ap.add_argument("--no-resume", action="store_true",
                    help="start fresh even if checkpoints exist in the run dir")
    ap.add_argument("--force", action="store_true",
                    help="allow writing into a run dir that holds a different experiment")
    ap.add_argument("--threads", type=int, default=None,
                    help="numeric-library threads (default: from the SLURM allocation)")
    ap.add_argument("--realtime", action="store_true",
                    help="pass through to run_agent.py (keep time.sleep pacing)")
    ap.add_argument("--param", action="append", default=[], metavar="NAME:=VALUE",
                    help="extra ROS-style parameter(s), passed straight to the agent")
    return ap.parse_args()


def main() -> int:
    args = _parse_args()

    agent_path = Path(args.agent)
    if not agent_path.is_absolute():
        agent_path = (ROOT / args.agent) if (ROOT / args.agent).exists() else agent_path
    agent_name = agent_path.stem

    config_path = Path(args.config).expanduser().resolve()
    cfg = load_config(config_path)
    cfg_dict = cfg.to_dict()
    cfg_hash = _config_hash(cfg_dict)

    run_dir = Path(args.run_dir or ROOT / "runs" / f"{agent_name}_seed{args.seed}").expanduser().resolve()
    ckpt_dir = run_dir / args.checkpoint_subdir

    # --- resource awareness ------------------------------------------------
    res = hpc.detect_resources()
    threads = args.threads if args.threads is not None else 1  # 1 core per task by design
    thread_env = hpc.configure_threads(threads)
    print(f"[train] resources: scheduler={res.scheduler} cpus={res.cpus} "
          f"mem_mb={res.mem_mb} time_left_s={res.time_limit_s} "
          f"job={res.job_id} array={res.array_task_id}")
    print(f"[train] threads pinned to {threads}: {','.join(thread_env)}")

    # --- overwrite safeguard ----------------------------------------------
    run_dir.mkdir(parents=True, exist_ok=True)
    meta_path = run_dir / "run_metadata.json"
    if meta_path.exists():
        try:
            prev = json.loads(meta_path.read_text())
        except Exception:
            prev = {}
        mismatch = []
        if prev.get("agent") not in (None, agent_name):
            mismatch.append(f"agent {prev.get('agent')} != {agent_name}")
        if prev.get("seed") not in (None, args.seed):
            mismatch.append(f"seed {prev.get('seed')} != {args.seed}")
        if prev.get("config_hash") not in (None, cfg_hash):
            mismatch.append(f"config {prev.get('config_hash')} != {cfg_hash}")
        if mismatch and not args.force:
            print(f"[train] REFUSING to reuse {run_dir}: it holds a different "
                  f"experiment ({'; '.join(mismatch)}). Use --force or a new --run-dir.",
                  file=sys.stderr)
            return 2

    # --- auto-resume -------------------------------------------------------
    resume_from = None
    user_gave_resume = any(p.startswith("resume_checkpoint:=") for p in args.param)
    if not args.no_resume and not user_gave_resume:
        latest = hpc.latest_checkpoint(ckpt_dir)
        if latest:
            resume_from = latest[1]
            print(f"[train] auto-resume from {resume_from} (episode {latest[0]})")
        else:
            print(f"[train] no valid checkpoint in {ckpt_dir}; starting fresh")

    # --- config snapshot + metadata ---------------------------------------
    try:
        import yaml
        (run_dir / "config_snapshot.yaml").write_text(yaml.safe_dump(cfg_dict, sort_keys=False))
    except Exception as e:
        print(f"[train] warning: could not snapshot config: {e}")
    hpc.write_run_metadata(
        run_dir, config=cfg_dict, seed=args.seed, agent=agent_name,
        extra={"config_hash": cfg_hash, "config_path": str(config_path),
               "resume_checkpoint": str(resume_from) if resume_from else None,
               "episodes_requested": args.episodes,
               "threads": threads})

    # --- build the run_agent.py command -----------------------------------
    cmd = [sys.executable, str(ROOT / "scripts" / "run_agent.py"), str(agent_path),
           "--config", str(config_path), "--seed", str(args.seed),
           "--run-dir", str(run_dir)]
    if args.realtime:
        cmd.append("--realtime")
    if args.episodes is not None:
        cmd += ["--param", f"{EPISODE_PARAM}:={args.episodes}"]
    if resume_from is not None:
        cmd += ["--param", f"resume_checkpoint:={resume_from}"]
    for p in args.param:
        cmd += ["--param", p]

    child_env = dict(os.environ)
    child_env.update(thread_env)
    child_env["MPLBACKEND"] = "Agg"

    print(f"[train] exec: {' '.join(cmd)}")
    t0 = time.perf_counter()
    status = "unknown"
    rc = 1
    try:
        rc = subprocess.run(cmd, env=child_env).returncode
        status = "completed" if rc == 0 else f"agent_exit_{rc}"
    except KeyboardInterrupt:
        status = "interrupted"
    finally:
        elapsed = time.perf_counter() - t0
        summary = hpc.write_run_summary(
            run_dir, status=status, checkpoint_dir=ckpt_dir,
            extra={"elapsed_s": round(elapsed, 1), "agent": agent_name,
                   "seed": args.seed, "return_code": rc})
        data = json.loads(Path(summary).read_text())
        print(f"\n[train] status={status} in {elapsed:.1f}s")
        print(f"[train] checkpoints: {data['num_checkpoints']} "
              f"(latest episode {data['latest_episode']}) in {ckpt_dir}")
        print(f"[train] run summary: {summary}")
        print(f"[train] output size: {data['output_bytes'] / 1e6:.2f} MB")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
