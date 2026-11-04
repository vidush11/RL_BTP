#!/usr/bin/env python3
"""Run an existing ROS 2 agent script UNMODIFIED, headless, against the simulator.

    python scripts/run_agent.py gvf_horde_agent.py \
        --config configs/default.yaml --seed 0 --run-dir runs/horde_s0 \
        --param max_training_episodes:=2000

    python scripts/run_agent.py gvf_horde_agent_evaluate.py \
        --run-dir runs/horde_eval --param checkpoint_dir:=runs/horde_s0/gvf_checkpoints_horde

What it does around your script (your file is never edited):
  * puts fake rclpy/message modules in place that answer the environment
    services from the simulator (see simulator/ros_compat/shim.py);
  * ``chdir`` into --run-dir, so relative output paths such as your hard-coded
    'gvf_checkpoints_horde' land inside the run directory;
  * forces the Agg matplotlib backend and turns off per-episode redraws (the
    figure is saved to the run directory when the script ends);
  * removes ``time.sleep`` throttling (they exist to pace Gazebo);
    use --realtime to keep them;
  * seeds Python's ``random`` and NumPy (your agents draw from the global RNG).
``--param name:=value`` is the same syntax as ``ros2 run ... -p name:=value``.
"""

import argparse
import importlib.util
import os
import random
import sys
import time as _time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


class _FastTime:
    """Proxy for the ``time`` module whose ``sleep`` does nothing."""

    def __getattr__(self, name):
        return getattr(_time, name)

    @staticmethod
    def sleep(_seconds):
        return None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("agent", help="path to the agent script (e.g. gvf_horde_agent.py)")
    ap.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--run-dir", default=None,
                    help="working/output directory (default: runs/<agent>_seed<N>)")
    ap.add_argument("--param", action="append", default=[], metavar="NAME:=VALUE",
                    help="ROS-style parameter override; repeatable")
    ap.add_argument("--realtime", action="store_true", help="keep time.sleep() calls")
    ap.add_argument("--live-plots", action="store_true",
                    help="keep matplotlib interactive/redrawing (needs a display)")
    args = ap.parse_args()

    agent_path = Path(args.agent).expanduser().resolve()
    if not agent_path.exists():
        cand = ROOT / args.agent
        if cand.exists():
            agent_path = cand.resolve()
        else:
            sys.exit(f"Agent script not found: {args.agent}")
    config_path = Path(args.config).expanduser().resolve()
    run_dir = Path(args.run_dir or ROOT / "runs" / f"{agent_path.stem}_seed{args.seed}").expanduser().resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    overrides = {}
    for item in args.param:
        if ":=" not in item:
            sys.exit(f"--param expects NAME:=VALUE, got '{item}'")
        k, v = item.split(":=", 1)
        overrides[k] = v

    if not args.live_plots:
        os.environ["MPLBACKEND"] = "Agg"
    import matplotlib
    if not args.live_plots:
        matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    if not args.live_plots:
        for fn in ("pause", "draw", "tight_layout", "show"):
            setattr(plt, fn, lambda *a, **k: None)

    import numpy as np
    random.seed(args.seed)
    np.random.seed(args.seed)

    from simulator import TurtleBot3LidarEnv, load_config
    from simulator.ros_compat import install

    cfg = load_config(config_path)
    env = TurtleBot3LidarEnv(cfg)
    backend = install(env, seed=args.seed, param_overrides=overrides)

    os.chdir(run_dir)
    print(f"[run_agent] agent={agent_path.name} config={config_path.name} seed={args.seed}")
    print(f"[run_agent] run dir: {run_dir}")
    table = ", ".join(f"{i}:(v={v:g}, w={w:+g})" for i, (v, w) in enumerate(env.actions.table))
    print(f"[run_agent] action table: {table}")
    print("[run_agent] note: positive angular velocity = counter-clockwise (left, REP-103)")

    spec = importlib.util.spec_from_file_location("agent_under_test", agent_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["agent_under_test"] = module
    spec.loader.exec_module(module)
    if not args.realtime and hasattr(module, "time"):
        module.time = _FastTime()

    sys.argv = [str(agent_path)]
    t0 = _time.perf_counter()
    try:
        module.main()
    finally:
        elapsed = _time.perf_counter() - t0
        for num in plt.get_fignums():
            fig = plt.figure(num)
            try:
                fig.tight_layout()
            except Exception:
                pass
            out = run_dir / f"{agent_path.stem}_figure{num}.png"
            fig.savefig(out, dpi=120)
            print(f"[run_agent] saved figure: {out}")
        print(f"[run_agent] done in {elapsed:.1f}s: {backend.episodes} episodes, "
              f"{backend.total_steps} env steps ({backend.total_steps / max(elapsed, 1e-9):.0f} steps/s)")
        print(f"[run_agent] episode endings: {backend.terminations} terminated "
              f"(collision/goal), {backend.truncations} truncated (step limit)")
        if backend.truncations:
            print("[run_agent] WARNING: truncated episodes did not end in a collision. Your agents "
                  "see them as 'done' and (e.g. the TTC targets np.arange(steps, 0, -1)) treat them "
                  "as collisions, so their TTC error is not meaningful. A deterministic always-left/"
                  "right policy circles forever in an empty arena; see README 'Known limitations'.")


if __name__ == "__main__":
    main()
