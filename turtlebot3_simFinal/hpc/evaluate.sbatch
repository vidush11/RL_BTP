#!/bin/bash
# SLURM batch script: evaluate saved checkpoints with independent ground truth.
# Submit through hpc/submit_evaluation.sh. Reads checkpoints READ-ONLY and
# writes metrics to a separate eval/ dir, so it never disturbs training output.
# CPU-only: no --gres.
#SBATCH --job-name=tb3-eval
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=01:00:00
#SBATCH --output=%x_%A_%a.out
#SBATCH --open-mode=append
#SBATCH --array=0

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
export MPLBACKEND=Agg
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
       NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
source "${VENV:-$HOME/venvs/tb3sim}/bin/activate"

SEED="${SLURM_ARRAY_TASK_ID:-0}"
CONFIG="${CONFIG:-configs/default.yaml}"
AGENT="${AGENT:-gvf_horde_agent.py}"
RUN_ROOT="${RUN_ROOT:-runs}"
EVAL_EPISODES="${EVAL_EPISODES:-20}"
EVAL_SEED="${EVAL_SEED:-1000}"
RUN_DIR="${RUN_ROOT}/$(basename "${AGENT%.py}")_seed${SEED}"
CKPT_DIR="${RUN_DIR}/gvf_checkpoints_horde"

echo "[evaluate.sbatch] seed=${SEED} checkpoints=${CKPT_DIR}"
python scripts/evaluate.py \
    --checkpoint-dir "${CKPT_DIR}" \
    --out-dir "${RUN_DIR}/eval" \
    --config "${CONFIG}" \
    --seed "${EVAL_SEED}" \
    --num-eval-episodes "${EVAL_EPISODES}"
echo "[evaluate.sbatch] wrote ${RUN_DIR}/eval/metrics.json"
