#!/bin/bash
# User-space environment setup for turtlebot3_sim on IIIT Delhi Omni (no root).
#
# Usage:
#   bash hpc/environment_setup.sh                 # create venv + install from PyPI
#   bash hpc/environment_setup.sh --download WH    # (login node) build offline wheelhouse WH
#   bash hpc/environment_setup.sh --offline  WH    # (compute node) install from wheelhouse WH
#
# Why the two offline steps: Omni's usage policy does not state whether COMPUTE
# nodes have internet.  Run `python scripts/validate_omni.py` first -- if its
# "pypi.org:443" check says NOT reachable, build the wheelhouse on the login
# node (--download) and install from it inside the job (--offline).
#
# Nothing here needs sudo.  It does NOT run `pip install` as root and does not
# touch system Python.  Set VENV to choose where the venv lives (default below).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # project root
VENV="${VENV:-$HOME/venvs/tb3sim}"
REQ="$HERE/requirements.txt"
MODE="online"
WHEELHOUSE=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --download) MODE="download"; WHEELHOUSE="${2:?--download needs a directory}"; shift 2;;
    --offline)  MODE="offline";  WHEELHOUSE="${2:?--offline needs a directory}";  shift 2;;
    -h|--help)  grep '^#' "$0" | sed 's/^# \{0,1\}//'; exit 0;;
    *) echo "unknown arg: $1" >&2; exit 2;;
  esac
done

# --- pick a Python. Prefer a module if the cluster provides one. -------------
# CHANGE ME if `module avail` (see validate_omni) lists a specific Python, e.g.:
#   module load python/3.11
PYBIN="${PYBIN:-python3}"
echo "[setup] python: $("$PYBIN" --version 2>&1) at $(command -v "$PYBIN")"
PYVER="$("$PYBIN" -c 'import sys;print("%d.%d"%sys.version_info[:2])')"
case "$PYVER" in
  3.9|3.10|3.11|3.12|3.13) : ;;
  *) echo "[setup] WARNING: Python $PYVER is untested; requirements verified on 3.10-3.13." >&2;;
esac

if [[ "$MODE" == "download" ]]; then
  mkdir -p "$WHEELHOUSE"
  echo "[setup] downloading wheels for $REQ into $WHEELHOUSE (run this on a node WITH internet)"
  "$PYBIN" -m pip download -r "$REQ" -d "$WHEELHOUSE"
  echo "[setup] done. Copy '$WHEELHOUSE' to the cluster and use --offline there."
  exit 0
fi

echo "[setup] creating venv at $VENV"
"$PYBIN" -m venv "$VENV"
# shellcheck disable=SC1091
source "$VENV/bin/activate"
python -m pip install --upgrade pip >/dev/null

if [[ "$MODE" == "offline" ]]; then
  echo "[setup] installing from wheelhouse $WHEELHOUSE (no internet needed)"
  python -m pip install --no-index --find-links "$WHEELHOUSE" -r "$REQ"
else
  echo "[setup] installing from PyPI"
  python -m pip install -r "$REQ"
fi

echo "[setup] verifying imports"
python -c "import numpy,gymnasium,yaml,matplotlib; print('[setup] OK:',numpy.__version__,gymnasium.__version__)"
echo "[setup] activate later with:  source $VENV/bin/activate"
