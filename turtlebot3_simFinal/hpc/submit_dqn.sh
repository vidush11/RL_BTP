#!/bin/bash
# Pretrain the shared GVF checkpoint, then submit the A/B/C DQN comparison as a
# per-seed SLURM array on Omni. CPU-only (no GPU).
#
#   OMNI_ACCOUNT=my_group bash hpc/submit_dqn.sh                 # seeds 0-4
#   OMNI_ACCOUNT=my_group SEEDS=0-9 STEPS=200000 bash hpc/submit_dqn.sh
#
# Pretraining runs here (login node is fine; it is short and CPU-only) so every
# seed/agent shares the EXACT same GVF checkpoint. Each array task trains A, B, C
# for one seed. Respects the queue's concurrent-job cap.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$HERE"
: "${OMNI_ACCOUNT:?set OMNI_ACCOUNT to your Omni group/account name}"
PARTITION="${PARTITION:-short}"; QOS="${QOS:-$PARTITION}"
SEEDS="${SEEDS:-0-4}"; VENV="${VENV:-$HOME/venvs/tb3sim}"
CONFIG="${CONFIG:-configs/dqn/experiment.yaml}"
RUN_ROOT="${RUN_ROOT:-runs_dqn}"
GVF_CKPT="${GVF_CKPT:-$RUN_ROOT/gvf_pretrained.npz}"
PRETRAIN_EPISODES="${PRETRAIN_EPISODES:-2000}"
case "$PARTITION" in short) CC=3; T=06:00:00;; medium) CC=2; T=1-00:00:00;; *) CC=1; T=1-00:00:00;; esac
CONCURRENCY="${CONCURRENCY:-$CC}"; TIME="${TIME:-$T}"
mkdir -p logs "$RUN_ROOT"

if [[ ! -f "$GVF_CKPT" ]]; then
  echo "[submit_dqn] pretraining shared GVF checkpoint -> $GVF_CKPT"
  source "$VENV/bin/activate"
  MPLBACKEND=Agg OMP_NUM_THREADS=1 python scripts/pretrain_gvf.py \
      --config "$CONFIG" --episodes "$PRETRAIN_EPISODES" --seed 0 --out "$GVF_CKPT"
fi

echo "[submit_dqn] account=$OMNI_ACCOUNT partition=$PARTITION array=${SEEDS}%${CONCURRENCY}"
exec sbatch \
  --account="$OMNI_ACCOUNT" --partition="$PARTITION" --qos="$QOS" \
  --time="$TIME" --array="${SEEDS}%${CONCURRENCY}" --output="logs/%x_%A_%a.out" \
  --export=ALL,VENV="$VENV",CONFIG="$CONFIG",GVF_CKPT="$GVF_CKPT",RUN_ROOT="$RUN_ROOT" \
  hpc/dqn.sbatch
