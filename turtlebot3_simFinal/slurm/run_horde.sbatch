#!/bin/bash
# COMPATIBILITY WRAPPER. The maintained Omni workflow now lives in hpc/ (see
# hpc/README.md): hpc/submit_training.sh / hpc/train.sbatch, which fill in your
# account + queue and give you auto-resume, run metadata and a run summary.
#
# This file is kept because the top-level README referenced it. It submits one
# seed (or an array) by delegating to scripts/train.py. CPU-only: NO GPU is
# requested (Omni forbids idle GPUs).
#
#   sbatch --account=<group> --partition=short --qos=short slurm/run_horde.sbatch
#   sbatch --account=<group> --partition=short --qos=short --array=0-4%3 slurm/run_horde.sbatch
#SBATCH --job-name=horde-sim
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --open-mode=append
#SBATCH --requeue
#SBATCH --array=0

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
mkdir -p logs

export MPLBACKEND=Agg
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

# CHANGE ME if Omni needs `module load python/...` first.
source "${VENV:-$HOME/venvs/tb3sim}/bin/activate"

SEED=${SLURM_ARRAY_TASK_ID:-0}
EPISODES=${EPISODES:-5000}
CONFIG=${CONFIG:-configs/default.yaml}
RUN_DIR=${RUN_DIR:-runs/horde_seed${SEED}}

# train.py auto-resumes from the newest valid checkpoint and writes
# run_metadata.json / run_summary.json into RUN_DIR.
python scripts/train.py --agent gvf_horde_agent.py \
    --config "$CONFIG" --seed "$SEED" --run-dir "$RUN_DIR" --episodes "$EPISODES"

# Evaluate this seed's checkpoints with independent ground truth.
python scripts/evaluate.py \
    --checkpoint-dir "$RUN_DIR/gvf_checkpoints_horde" \
    --out-dir "${RUN_DIR}/eval" --config "$CONFIG" --num-eval-episodes 20
