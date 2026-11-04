#!/bin/bash
# Train Agents A, B, C for ONE seed (= SLURM_ARRAY_TASK_ID) on Omni.
# Submit via hpc/submit_dqn.sh, which pretrains the shared GVF checkpoint first.
# CPU-only (NumPy DQN): NO --gres. Headless, no Gazebo/display.
#SBATCH --job-name=tb3-dqn
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=06:00:00
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

SEED="${SLURM_ARRAY_TASK_ID:-0}"
CONFIG="${CONFIG:-configs/dqn/experiment.yaml}"
GVF_CKPT="${GVF_CKPT:-runs_dqn/gvf_pretrained.npz}"
RUN_ROOT="${RUN_ROOT:-runs_dqn}"

for MODE in A B C; do
  EXTRA=""
  [[ "$MODE" != "A" ]] && EXTRA="--gvf-checkpoint ${GVF_CKPT}"
  echo "[dqn.sbatch] mode ${MODE} seed ${SEED}"
  python scripts/train_dqn.py --mode "$MODE" --seed "$SEED" \
      --config "$CONFIG" --out "${RUN_ROOT}/${MODE}_s${SEED}" ${EXTRA}
done
echo "[dqn.sbatch] finished seed ${SEED} (A,B,C)"
