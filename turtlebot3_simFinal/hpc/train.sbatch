#!/bin/bash
# SLURM batch script for turtlebot3_sim Horde training on IIIT Delhi Omni.
# Submit it through hpc/submit_training.sh (which fills in your account and the
# queue) -- do not edit account/partition here; pass them on the sbatch CLI.
#
# IMPORTANT (Omni policy): this is a CPU-ONLY 2-D simulator. Do NOT add a
# `--gres=gpu:...` line. Omni's policy prohibits "intentionally keeping GPUs
# idle", so requesting a GPU you will not use is a policy violation. Whether the
# queues accept a job with no GPU is not documented -- confirm with a short
# `scripts/validate_omni.py` job first.
#
# Resource defaults below are deliberately small (the sim needs ~1 core, <1 GB).
# submit_training.sh overrides --account/--partition/--qos/--time/--array.
#SBATCH --job-name=tb3-train
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=02:00:00
#SBATCH --output=%x_%A_%a.out
#SBATCH --open-mode=append
#SBATCH --requeue
#SBATCH --array=0

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

# One process per core: pin BLAS/OpenMP BEFORE Python imports NumPy.
export MPLBACKEND=Agg
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
       NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1

# Activate the user-space venv (see hpc/environment_setup.sh).
# CHANGE ME if Omni needs a `module load python/...` first.
source "${VENV:-$HOME/venvs/tb3sim}/bin/activate"

SEED="${SLURM_ARRAY_TASK_ID:-0}"
EPISODES="${EPISODES:-5000}"
CONFIG="${CONFIG:-configs/default.yaml}"
AGENT="${AGENT:-gvf_horde_agent.py}"
# Persistent run root (keep code + checkpoints in $HOME; the 5 TB group quota
# lives there). Override RUN_ROOT to use scratch for high-volume temp output.
RUN_ROOT="${RUN_ROOT:-runs}"
RUN_DIR="${RUN_ROOT}/$(basename "${AGENT%.py}")_seed${SEED}"

echo "[train.sbatch] host=$(hostname) job=${SLURM_JOB_ID:-?} array=${SLURM_ARRAY_TASK_ID:-?} seed=${SEED}"
echo "[train.sbatch] agent=${AGENT} episodes=${EPISODES} config=${CONFIG} run_dir=${RUN_DIR}"

# train.py auto-resumes from the newest valid checkpoint, so a requeued job
# (wall-time hit, preemption) continues instead of restarting.
python scripts/train.py \
    --agent "${AGENT}" \
    --config "${CONFIG}" \
    --seed "${SEED}" \
    --run-dir "${RUN_DIR}" \
    --episodes "${EPISODES}"

echo "[train.sbatch] finished seed ${SEED}"
