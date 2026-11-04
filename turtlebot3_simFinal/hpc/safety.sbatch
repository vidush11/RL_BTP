#!/bin/bash
# Safety task (paper GVF controller + given RL_BTP GVFs vs DQN), ONE job =
#   STAGE=tune : validation tuning (both methods, equal budget), uses all cores
#   STAGE=run  : METHOD (gvf|dqn) for seed SLURM_ARRAY_TASK_ID, 30,000 episodes
# Runs resume from checkpoint.pkl, so --requeue / a resubmit continues the run.
# Submit via hpc/submit_safety.sh. CPU-only: NO --gres. Headless.
#SBATCH --job-name=tb3-safety
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=6G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --open-mode=append
#SBATCH --requeue
#SBATCH --array=0

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
export MPLBACKEND=Agg
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
       NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
source "${VENV:-$HOME/venvs/tb3sim}/bin/activate"

STAGE="${STAGE:-run}"
RUN_ROOT="${RUN_ROOT:-runs_safety}"
SEED="${SLURM_ARRAY_TASK_ID:-0}"
EPISODES="${EPISODES:-30000}"

if [ "$STAGE" = "tune" ]; then
  python -m safety_exp.tune --out "${RUN_ROOT}/tuning" --workers "${SLURM_CPUS_PER_TASK:-1}"
  exit 0
fi

METHOD="${METHOD:?set METHOD=gvf or METHOD=dqn}"
TUNED="${TUNED:-${RUN_ROOT}/tuning/tuning.json}"
EXTRA=()
if [ "$METHOD" = "gvf" ] && [ "$SEED" = "0" ] && [ -z "${TAG:-}" ]; then EXTRA+=(--sweep-gammas); fi
for s in ${SIM:-}; do EXTRA+=(--sim "$s"); done        # e.g. SIM="lidar.num_beams=24"
echo "[safety.sbatch] method ${METHOD} seed ${SEED} episodes ${EPISODES} tuned ${TUNED} ${EXTRA[*]+${EXTRA[*]}}"
python -m safety_exp.run --method "$METHOD" --seed "$SEED" --episodes "$EPISODES" \
    --tuned "$TUNED" --out "${RUN_ROOT}/${METHOD}${TAG:-}_s${SEED}" ${EXTRA[@]+"${EXTRA[@]}"}
echo "[safety.sbatch] finished ${METHOD} seed ${SEED}"
