#!/bin/bash
# Experiment D (LiDAR-only DQN, GVF-shaped reward) for ONE seed = SLURM_ARRAY_TASK_ID.
# Runs every variant in $VARIANTS (default "D D0") so D and its lambda=0 control
# share seeds. Submit via hpc/submit_expD.sh. CPU-only: NO --gres. Headless.
#SBATCH --job-name=tb3-expD
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
CONFIG="${CONFIG:-configs/dqn/experiment_d.yaml}"
RUN_ROOT="${RUN_ROOT:-runs_expD}"
VARIANTS="${VARIANTS:-D D0}"

for V in $VARIANTS; do
  case "$V" in
    D)     SET=() ;;
    D0)    SET=(--set reward.lam=0 --set experiment_id=D0_lidar_only_terminal_reward) ;;
    Dpred) SET=(--set reward.prediction_only=true --set experiment_id=Dpred_prediction_only_reward) ;;
    Dabs)  SET=(--set reward.shaping=absolute --set experiment_id=Dabs_absolute_gvf_reward) ;;
    *) echo "unknown variant $V" >&2; exit 2 ;;
  esac
  echo "[expD.sbatch] variant ${V} seed ${SEED}"
  python scripts/train_expD.py --config "$CONFIG" --seed "$SEED" \
      --out "${RUN_ROOT}/${V}_s${SEED}" "${SET[@]}"
done
echo "[expD.sbatch] finished seed ${SEED}"
