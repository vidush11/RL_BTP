#!/usr/bin/env python3
"""Omni cluster DISCOVERY + environment VALIDATION, in one script.

Run this on the IIIT Delhi Omni cluster (login node first, then inside a short
interactive/batch job).  It answers the questions the usage policy leaves open
and checks that the simulator actually works in the allocated environment.  It
changes nothing on the cluster, needs no root, and prints a human summary plus
a machine-readable JSON (``--json out.json``).

    # on the login node (discovery only; the sim checks still run if deps exist)
    python scripts/validate_omni.py --json omni_login.json

    # inside a job (see hpc/submit_validation style); adds the SLURM allocation
    srun -A <GROUP> -p short -q short --ntasks=1 --cpus-per-task=2 --mem=4G \
         --pty python scripts/validate_omni.py --json omni_node.json

What DISCOVERY collects (Phase 1 of the brief):
  * Python version/path; whether ``module`` and ``conda`` exist; ``module avail``.
  * SLURM presence and, if available, ``sinfo`` partitions and this job's
    allocation (cpus, mem, time remaining, scratch dir).
  * Candidate home/scratch paths and their free space / quota (``quota``, ``df``).
  * Whether the compute node can reach the internet (so we know if pip needs an
    offline wheelhouse) -- a short, non-fatal TCP probe to PyPI.
  * Installed versions of numpy / gymnasium / PyYAML / matplotlib / pytest.

What VALIDATION checks (Phase 5A of the brief):
  * simulator imports, config loading, a LiDAR cast, a geometric collision,
    the Gymnasium reset/step contract, the ROS-shim service contract, output
    write access, and a short headless simulation.

Exit status: 0 if all *validation* checks that could run passed; 1 otherwise.
Discovery items never fail the run -- an unknown cluster setting is reported,
not treated as an error (that is the whole point of running this).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

OK, WARN, FAIL, INFO = "ok", "warn", "fail", "info"


class Report:
    def __init__(self) -> None:
        self.items: list[dict] = []

    def add(self, section: str, name: str, status: str, detail="", data=None):
        self.items.append({"section": section, "name": name, "status": status,
                            "detail": str(detail), "data": data})

    def any_fail(self) -> bool:
        return any(i["status"] == FAIL for i in self.items)

    def print(self) -> None:
        glyph = {OK: "PASS", WARN: "WARN", FAIL: "FAIL", INFO: "INFO"}
        section = None
        for i in self.items:
            if i["section"] != section:
                section = i["section"]
                print(f"\n=== {section} ===")
            line = f"  [{glyph[i['status']]}] {i['name']}"
            if i["detail"]:
                line += f": {i['detail']}"
            print(line)


def _run(cmd: list[str], timeout=15) -> tuple[int, str]:
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return p.returncode, (p.stdout or "") + (p.stderr or "")
    except FileNotFoundError:
        return 127, "not found"
    except Exception as e:  # pragma: no cover - environment dependent
        return 1, str(e)


# --------------------------------------------------------------------------- #
# Discovery
# --------------------------------------------------------------------------- #
def discover(rep: Report) -> None:
    sec = "discovery: python & package managers"
    rep.add(sec, "python", INFO, f"{sys.version.split()[0]} at {sys.executable}")
    for tool in ("module", "conda", "mamba", "virtualenv"):
        path = shutil.which(tool)
        rep.add(sec, tool, INFO if path else WARN, path or "not on PATH")
    # `module` is usually a shell function, not an executable -> probe via bash.
    rc, out = _run(["bash", "-lc", "module avail 2>&1 | head -n 40"])
    if rc == 0 and out.strip() and "not found" not in out.lower():
        rep.add(sec, "module avail", INFO, "see data", data=out.strip().splitlines())
    else:
        rep.add(sec, "module avail", WARN, "no environment modules detected")

    sec = "discovery: scheduler"
    rc, _ = _run(["bash", "-lc", "command -v sbatch"])
    rep.add(sec, "sbatch", OK if rc == 0 else WARN,
            "SLURM present" if rc == 0 else "sbatch not found")
    from simulator import hpc  # local import so discovery still runs w/o deps
    res = hpc.detect_resources()
    rep.add(sec, "allocation", INFO if res.scheduler == "slurm" else WARN,
            (f"cpus={res.cpus} mem_mb={res.mem_mb} time_left_s={res.time_limit_s} "
             f"job={res.job_id} array={res.array_task_id}")
            if res.scheduler == "slurm" else
            f"not inside a SLURM job (host cpus={res.cpus}); run again under srun",
            data=res.to_dict())
    rc, out = _run(["bash", "-lc",
                    "sinfo -h -o '%P %l %c %m' 2>/dev/null | head -n 20"])
    if rc == 0 and out.strip():
        rep.add(sec, "sinfo partitions", INFO, "partition time cpus mem (see data)",
                data=out.strip().splitlines())
    else:
        rep.add(sec, "sinfo partitions", WARN,
                "sinfo unavailable (run on a login node); policy lists short/medium/long/cai")

    sec = "discovery: storage"
    for label, path in (("HOME", os.environ.get("HOME")),
                        ("SCRATCH", os.environ.get("SCRATCH")),
                        ("SLURM_TMPDIR", os.environ.get("SLURM_TMPDIR")),
                        ("TMPDIR", os.environ.get("TMPDIR")),
                        ("/scratch", "/scratch"),
                        ("CWD", os.getcwd())):
        if not path:
            rep.add(sec, label, WARN, "unset")
            continue
        exists = Path(path).exists()
        free = None
        if exists:
            try:
                du = shutil.disk_usage(path)
                free = f"{du.free / 1e9:.1f} GB free of {du.total / 1e9:.1f} GB"
            except OSError:
                free = "df failed"
        rep.add(sec, label, INFO if exists else WARN,
                f"{path}" + (f" ({free})" if free else " (does not exist)"))
    rc, out = _run(["bash", "-lc", "quota -s 2>/dev/null | head -n 10"])
    if rc == 0 and out.strip():
        rep.add(sec, "quota", INFO, "see data", data=out.strip().splitlines())
    else:
        rep.add(sec, "quota", WARN, "quota command gave nothing; group limit is 5 TB per policy")

    sec = "discovery: internet from this node"
    reachable = _probe_internet()
    rep.add(sec, "pypi.org:443", INFO if reachable else WARN,
            "reachable -> pip can download directly" if reachable else
            "NOT reachable -> build an offline wheelhouse on the login node "
            "(see hpc/environment_setup.sh --download)")

    sec = "discovery: installed packages"
    for name in ("numpy", "gymnasium", "yaml", "matplotlib", "pytest"):
        try:
            mod = __import__(name)
            rep.add(sec, name, OK, getattr(mod, "__version__", "unknown"))
        except Exception:
            rep.add(sec, name, WARN, "not importable in this interpreter")


def _probe_internet(host="pypi.org", port=443, timeout=4.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except Exception:
        return False


# --------------------------------------------------------------------------- #
# Validation (the simulator must actually work here)
# --------------------------------------------------------------------------- #
def validate(rep: Report, out_dir: Path) -> None:
    sec = "validation: imports"
    try:
        import numpy as np  # noqa
        import simulator
        from simulator import (TurtleBot3LidarEnv, load_config, make_vec_env,  # noqa
                               ENV_ID)
        rep.add(sec, "import simulator", OK, f"env id {ENV_ID}")
    except Exception as e:
        rep.add(sec, "import simulator", FAIL, repr(e))
        return  # nothing else can run

    import numpy as np
    from simulator import TurtleBot3LidarEnv, load_config, make_vec_env

    sec = "validation: config"
    try:
        cfg = load_config(ROOT / "configs" / "default.yaml")
        rep.add(sec, "load default.yaml", OK,
                f"{cfg.lidar.num_beams} beams, dt={cfg.motion.dt}")
    except Exception as e:
        rep.add(sec, "load default.yaml", FAIL, repr(e))
        return

    sec = "validation: lidar & collision"
    try:
        env = TurtleBot3LidarEnv(cfg)
        obs, info = env.reset(seed=0)
        ranges = info["ranges_m"]
        ok = (obs.shape == (cfg.lidar.num_beams,) and obs.dtype == np.float32
              and float(obs.min()) >= 0.0 and float(obs.max()) <= 1.0
              and np.all(np.isfinite(ranges)))
        rep.add(sec, "lidar cast", OK if ok else FAIL,
                f"obs {obs.shape} {obs.dtype} range [{obs.min():.3f},{obs.max():.3f}]")
    except Exception as e:
        rep.add(sec, "lidar cast", FAIL, repr(e))
        return
    try:
        # Drive straight into the +x wall; a geometric or lidar collision must fire.
        env2 = TurtleBot3LidarEnv(cfg)
        env2.reset(seed=0, options={"spawn": (1.8, 0.0, 0.0)})
        hit = False
        for _ in range(200):
            _o, _r, term, trunc, info = env2.step(2)  # action 2 = straight
            if term and info["collision"]:
                hit = True
                break
            if term or trunc:
                break
        rep.add(sec, "collision fires at wall", OK if hit else WARN,
                f"collision_type={info.get('collision_type')}")
    except Exception as e:
        rep.add(sec, "collision fires at wall", FAIL, repr(e))

    sec = "validation: gymnasium API"
    try:
        env3 = TurtleBot3LidarEnv(cfg)
        o1, _ = env3.reset(seed=123)
        o2, _ = env3.reset(seed=123)
        determ = np.array_equal(o1, o2)
        o, r, term, trunc, info = env3.step(env3.action_space.sample())
        contract = (env3.observation_space.contains(o)
                    and isinstance(r, float)
                    and isinstance(term, bool) and isinstance(trunc, bool))
        rep.add(sec, "reset/step contract", OK if contract else FAIL,
                "obs in space, scalar reward, bool term/trunc")
        rep.add(sec, "reset(seed) reproducible", OK if determ else FAIL,
                "same seed -> identical first observation")
    except Exception as e:
        rep.add(sec, "reset/step contract", FAIL, repr(e))

    sec = "validation: ROS shim contract"
    try:
        from simulator.ros_compat import install, uninstall
        env4 = TurtleBot3LidarEnv(cfg)
        backend = install(env4, seed=0)
        from turtlebot3_msgs.srv import Dqn
        reset_resp = backend.call("reset_environment", Dqn.Request())
        req = Dqn.Request(); req.action = 2
        step_resp = backend.call("rl_agent_interface", req)
        ok = (len(reset_resp.state) == cfg.lidar.num_beams
              and hasattr(step_resp, "reward") and hasattr(step_resp, "done"))
        # Unsupported service must raise, not silently no-op.
        raised = False
        try:
            backend.call("nonexistent_service", None)
        except NotImplementedError:
            raised = True
        uninstall()
        rep.add(sec, "3 services answer", OK if ok else FAIL,
                f"reset state len={len(reset_resp.state)}")
        rep.add(sec, "unknown service raises", OK if raised else FAIL,
                "NotImplementedError on unsupported API")
    except Exception as e:
        rep.add(sec, "ROS shim contract", FAIL, repr(e))

    sec = "validation: output write access"
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        from simulator import hpc
        probe = out_dir / "_write_probe.npz"
        hpc.atomic_savez(probe, x=np.arange(3))
        loaded = np.load(probe)["x"]
        probe.unlink()
        rep.add(sec, f"write+atomic save to {out_dir}", OK,
                f"round-tripped {loaded.tolist()}")
    except Exception as e:
        rep.add(sec, "output write access", FAIL, f"{out_dir}: {e!r}")

    sec = "validation: short headless simulation"
    try:
        t0 = time.perf_counter()
        env5 = TurtleBot3LidarEnv(cfg)
        env5.reset(seed=0)
        steps = 2000
        done_count = 0
        for _ in range(steps):
            _o, _r, term, trunc, _i = env5.step(env5.action_space.sample())
            if term or trunc:
                done_count += 1
                env5.reset()
        dt = time.perf_counter() - t0
        sps = steps / dt if dt > 0 else float("inf")
        rep.add(sec, "2000 steps headless", OK,
                f"{sps:,.0f} steps/s, {done_count} episodes, no display needed")
    except Exception as e:
        rep.add(sec, "headless simulation", FAIL, repr(e))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json", default=None, help="write the full report as JSON here")
    ap.add_argument("--out-dir", default=None,
                    help="directory to probe for write access (default: runs/_validate)")
    ap.add_argument("--skip-discovery", action="store_true")
    ap.add_argument("--skip-validation", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out_dir).expanduser() if args.out_dir else ROOT / "runs" / "_validate"
    rep = Report()

    print(f"turtlebot3_sim Omni validator  (host={socket.gethostname()}, "
          f"time={time.strftime('%Y-%m-%d %H:%M:%S')})")
    if not args.skip_discovery:
        try:
            discover(rep)
        except Exception as e:
            rep.add("discovery", "fatal", FAIL, repr(e))
    if not args.skip_validation:
        validate(rep, out_dir)

    rep.print()
    n_fail = sum(1 for i in rep.items if i["status"] == FAIL)
    n_warn = sum(1 for i in rep.items if i["status"] == WARN)
    print(f"\nSummary: {n_fail} fail, {n_warn} warn, "
          f"{sum(1 for i in rep.items if i['status']==OK)} pass "
          f"({len(rep.items)} checks).")

    if args.json:
        Path(args.json).write_text(json.dumps(
            {"host": socket.gethostname(), "time": time.time(), "items": rep.items},
            indent=2))
        print(f"Wrote JSON report: {args.json}")

    if rep.any_fail():
        print("\nRESULT: validation FAILED -- fix the FAIL items before submitting training.")
        return 1
    print("\nRESULT: validation OK. Address any WARN items (usually cluster-config "
          "discovery) then submit a short training job.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
