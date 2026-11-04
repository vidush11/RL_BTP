"""HPC-support utilities for headless, resumable, resource-aware runs.

Nothing here changes simulation semantics.  These helpers are used by the
``scripts/`` entry points (``validate_omni.py``, ``train.py``, ``evaluate.py``)
and by the SLURM job scripts in ``hpc/``.  They are deliberately dependency-free
(only the Python standard library + NumPy, both already required by the
simulator) so they run on a bare login node and inside a minimal venv.

Design notes
------------
* **Atomic writes.**  ``atomic_savez`` / ``write_json_atomic`` write to a temp
  file in the *same directory* and ``os.replace`` it into place, so a job killed
  by the scheduler's wall-time limit can never leave a half-written checkpoint
  that a later resume would choke on.  The original agent's own
  ``save_checkpoint`` is left byte-identical; these helpers are available to the
  wrapper scripts and to any future native agent.
* **Resume = newest VALID checkpoint.**  ``scan_checkpoints`` opens every
  candidate ``.npz`` and discards any that fail to load or lack the ``episode``
  key, so a corrupt latest file does not block resume -- the run falls back to
  the newest intact one.
* **Resource awareness.**  ``detect_resources`` reads the scheduler's own
  environment (``SLURM_*``) and falls back to the host.  ``configure_threads``
  pins the numerical-library thread pools so one process per allocated core does
  not oversubscribe the node (the real control is the ``export`` in the sbatch,
  before Python starts; this is the in-process backstop + a single source of
  truth for the value).
"""

from __future__ import annotations

import json
import os
import platform
import re
import sys
import tempfile
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np

# Default glob for the Horde agent's combined checkpoints.
CHECKPOINT_GLOB = "gvf_weights_ep_*.npz"
_EP_RE = re.compile(r"ep_(\d+)")


# --------------------------------------------------------------------------- #
# Atomic file writes
# --------------------------------------------------------------------------- #
def atomic_savez(path: os.PathLike | str, **arrays: Any) -> Path:
    """``np.savez`` that is crash-safe: write a temp file in the same directory,
    flush+fsync, then atomically ``os.replace`` it onto ``path``.

    Returns the final path.  NumPy appends ``.npz`` to ``np.savez`` targets, so
    we write the temp file *with* that suffix and rename the result.
    """
    path = Path(path)
    if path.suffix != ".npz":
        path = path.with_suffix(".npz")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=path.stem + ".", suffix=".npz.tmp",
                                    dir=str(path.parent))
    os.close(fd)
    tmp = Path(tmp_name)
    # np.savez would add ".npz" to a name without it; our name already ends in
    # ".npz.tmp", so pass a file object to control the exact bytes/path.
    try:
        with open(tmp, "wb") as fh:
            np.savez(fh, **arrays)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()
    return path


def write_json_atomic(path: os.PathLike | str, obj: Any) -> Path:
    """Serialise ``obj`` to JSON atomically (temp file + ``os.replace``)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=path.stem + ".", suffix=".json.tmp",
                                    dir=str(path.parent))
    try:
        with os.fdopen(fd, "w") as fh:
            json.dump(obj, fh, indent=2, sort_keys=True, default=_json_default)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)
    return path


def _json_default(o: Any):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    return str(o)


# --------------------------------------------------------------------------- #
# Checkpoint discovery / resume
# --------------------------------------------------------------------------- #
def episode_of(path: os.PathLike | str) -> Optional[int]:
    """Episode number encoded in a checkpoint filename (``..._ep_002000.npz``),
    or ``None`` if the name does not match."""
    m = _EP_RE.search(Path(path).name)
    return int(m.group(1)) if m else None


def _loadable(path: Path, require_key: Optional[str]) -> bool:
    try:
        with np.load(path, allow_pickle=False) as data:
            if require_key is not None and require_key not in data.files:
                return False
            # Touch one array so a truncated zip is actually detected.
            first = require_key if require_key is not None else (
                data.files[0] if data.files else None)
            if first is not None:
                _ = data[first]
        return True
    except Exception:
        return False


def scan_checkpoints(directory: os.PathLike | str,
                     glob: str = CHECKPOINT_GLOB,
                     require_key: Optional[str] = "episode",
                     validate: bool = True) -> List[Tuple[int, Path]]:
    """Return ``[(episode, path), ...]`` for valid checkpoints, sorted by
    episode ascending.

    ``episode`` is taken from the filename; a file that neither matches the name
    pattern's episode group nor can be opened is skipped.  With ``validate`` the
    ``.npz`` is actually opened and required to carry ``require_key`` -- this is
    what makes a half-written latest file fall back to the newest intact one.
    """
    directory = Path(directory)
    if not directory.is_dir():
        return []
    found: List[Tuple[int, Path]] = []
    for p in sorted(directory.glob(glob)):
        ep = episode_of(p)
        if ep is None:
            continue
        if validate and not _loadable(p, require_key):
            continue
        found.append((ep, p))
    found.sort(key=lambda t: t[0])
    return found


def latest_checkpoint(directory: os.PathLike | str,
                      glob: str = CHECKPOINT_GLOB,
                      require_key: Optional[str] = "episode") -> Optional[Tuple[int, Path]]:
    """Newest valid checkpoint as ``(episode, path)`` or ``None``."""
    found = scan_checkpoints(directory, glob, require_key)
    return found[-1] if found else None


# --------------------------------------------------------------------------- #
# Resource detection + thread pinning
# --------------------------------------------------------------------------- #
@dataclass
class Resources:
    cpus: int                       # cores this process should use
    mem_mb: Optional[int]           # memory allocated (MB), if known
    time_limit_s: Optional[int]     # wall-time limit (seconds), if known
    scheduler: str                  # "slurm" | "none"
    job_id: Optional[str]
    array_task_id: Optional[int]
    scratch: Optional[str]          # node-local/scratch dir, if advertised
    source: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def _slurm_mem_mb() -> Optional[int]:
    # SLURM_MEM_PER_NODE is in MB; SLURM_MEM_PER_CPU * cpus otherwise.
    node = os.environ.get("SLURM_MEM_PER_NODE")
    if node and node.isdigit():
        return int(node)
    per_cpu = os.environ.get("SLURM_MEM_PER_CPU")
    cpus = os.environ.get("SLURM_CPUS_PER_TASK") or os.environ.get("SLURM_CPUS_ON_NODE")
    if per_cpu and per_cpu.isdigit() and cpus and cpus.isdigit():
        return int(per_cpu) * int(cpus)
    return None


def _slurm_time_limit_s() -> Optional[int]:
    """Parse ``SLURM_JOB_END_TIME`` (epoch secs) relative to now, else None."""
    end = os.environ.get("SLURM_JOB_END_TIME")
    if end and end.lstrip("-").isdigit():
        remaining = int(end) - int(time.time())
        return remaining if remaining > 0 else 0
    return None


def detect_resources(default_cpus: int = 1) -> Resources:
    """Discover the CPU/memory/time budget from the scheduler, falling back to
    the host.  Never raises; unknown values are ``None``."""
    env = os.environ
    is_slurm = "SLURM_JOB_ID" in env
    cpus_raw = (env.get("SLURM_CPUS_PER_TASK")
                or env.get("SLURM_CPUS_ON_NODE"))
    if cpus_raw and cpus_raw.isdigit():
        cpus = int(cpus_raw)
    else:
        cpus = os.cpu_count() or default_cpus
    scratch = (env.get("SLURM_TMPDIR") or env.get("TMPDIR")
               or env.get("SCRATCH") or None)
    array_id = env.get("SLURM_ARRAY_TASK_ID")
    return Resources(
        cpus=max(1, cpus),
        mem_mb=_slurm_mem_mb(),
        time_limit_s=_slurm_time_limit_s() if is_slurm else None,
        scheduler="slurm" if is_slurm else "none",
        job_id=env.get("SLURM_JOB_ID"),
        array_task_id=int(array_id) if array_id and array_id.lstrip("-").isdigit() else None,
        scratch=scratch,
        source={k: v for k, v in env.items() if k.startswith("SLURM_")},
    )


_THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")


def configure_threads(n: int = 1) -> Dict[str, str]:
    """Pin numerical-library thread pools to ``n`` by setting the standard env
    vars.  Returns the values set.

    NOTE: to take full effect these must be set *before* NumPy/OpenBLAS import,
    which is why the sbatch exports them.  Calling this after import is a
    harmless no-op for already-initialised pools but keeps a single source of
    truth and covers libraries that read the env lazily.
    """
    n = max(1, int(n))
    out = {}
    for var in _THREAD_VARS:
        os.environ[var] = str(n)
        out[var] = str(n)
    return out


# --------------------------------------------------------------------------- #
# Run metadata + summary
# --------------------------------------------------------------------------- #
def environment_versions() -> Dict[str, str]:
    """Versions of the libraries that matter for reproducibility."""
    out = {"python": sys.version.split()[0], "platform": platform.platform()}
    for name in ("numpy", "gymnasium", "yaml", "matplotlib"):
        try:
            mod = __import__(name)
            out[name] = getattr(mod, "__version__", "unknown")
        except Exception:
            out[name] = "not-installed"
    return out


def write_run_metadata(run_dir: os.PathLike | str, *,
                       config: Mapping[str, Any],
                       seed: int,
                       agent: Optional[str] = None,
                       extra: Optional[Mapping[str, Any]] = None) -> Path:
    """Record everything needed to reproduce a run: resolved config, seed,
    library versions, and the scheduler allocation.  Written atomically to
    ``run_dir/run_metadata.json``."""
    run_dir = Path(run_dir)
    meta = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "seed": int(seed),
        "agent": agent,
        "config": dict(config),
        "versions": environment_versions(),
        "resources": detect_resources().to_dict(),
        "argv": list(sys.argv),
    }
    if extra:
        meta.update(dict(extra))
    return write_json_atomic(run_dir / "run_metadata.json", meta)


def _dir_size_bytes(path: Path) -> int:
    total = 0
    if not path.exists():
        return 0
    for p in path.rglob("*"):
        if p.is_file():
            try:
                total += p.stat().st_size
            except OSError:
                pass
    return total


def write_run_summary(run_dir: os.PathLike | str, *,
                      status: str,
                      checkpoint_dir: Optional[os.PathLike | str] = None,
                      extra: Optional[Mapping[str, Any]] = None) -> Path:
    """Write ``run_dir/run_summary.json`` reporting output paths, the newest
    checkpoint, on-disk size, and whether key artifacts were saved.  Safe to
    call in a ``finally`` so it records even an interrupted run."""
    run_dir = Path(run_dir)
    ck_dir = Path(checkpoint_dir) if checkpoint_dir else run_dir / "gvf_checkpoints_horde"
    latest = latest_checkpoint(ck_dir)
    ckpts = scan_checkpoints(ck_dir)
    summary = {
        "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "status": status,
        "run_dir": str(run_dir),
        "checkpoint_dir": str(ck_dir),
        "num_checkpoints": len(ckpts),
        "latest_checkpoint": str(latest[1]) if latest else None,
        "latest_episode": latest[0] if latest else None,
        "output_bytes": _dir_size_bytes(run_dir),
        "artifacts_ok": bool(latest) if checkpoint_dir is not False else True,
    }
    if extra:
        summary.update(dict(extra))
    return write_json_atomic(run_dir / "run_summary.json", summary)


__all__ = [
    "CHECKPOINT_GLOB", "atomic_savez", "write_json_atomic", "episode_of",
    "scan_checkpoints", "latest_checkpoint", "Resources", "detect_resources",
    "configure_threads", "environment_versions", "write_run_metadata",
    "write_run_summary",
]
