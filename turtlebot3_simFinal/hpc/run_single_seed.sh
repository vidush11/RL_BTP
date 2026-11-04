#!/bin/bash
# Train (and optionally evaluate) ONE seed, right here, no scheduler.
# Use it for a quick local/interactive check or inside `srun --pty bash`.
#
#   bash hpc/run_single_seed.sh 0                         # seed 0, default episodes
#   SEED=2 EPISODES=200 bash hpc/run_single_seed.sh       # seed via env
#   EVAL=1 bash hpc/run_single_seed.sh 0                  # train then evaluate
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$HERE"

SEED="${1:-${SEED:-0}}"
EPISODES="${EPISODES:-500}"
CONFIG="${CONFIG:-configs/default.yaml}"
AGENT="${AGENT:-gvf_horde_agent.py}"
RUN_ROOT="${RUN_ROOT:-runs}"
RUN_DIR="${RUN_ROOT}/$(basename "${AGENT%.py}")_seed${SEED}"

export MPLBACKEND=Agg
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

[[ -n "${VENV:-}" && -f "${VENV}/bin/activate" ]] && source "${VENV}/bin/activate"

echo "[single] training seed ${SEED} -> ${RUN_DIR}"
python scripts/train.py --agent "$AGENT" --config "$CONFIG" --seed "$SEED" \
    --run-dir "$RUN_DIR" --episodes "$EPISODES"

if [[ "${EVAL:-0}" == "1" ]]; then
  echo "[single] evaluating ${RUN_DIR}/gvf_checkpoints_horde"
  python scripts/evaluate.py --checkpoint-dir "${RUN_DIR}/gvf_checkpoints_horde" \
      --out-dir "${RUN_DIR}/eval" --config "$CONFIG" --num-eval-episodes "${EVAL_EPISODES:-20}"
fi
echo "[single] done. See ${RUN_DIR}/run_summary.json"
