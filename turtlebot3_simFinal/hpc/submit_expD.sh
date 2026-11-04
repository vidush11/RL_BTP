#!/bin/bash
# Submit Experiment D as a per-seed SLURM array on Omni (CPU-only).
#   OMNI_ACCOUNT=my_group bash hpc/submit_expD.sh                          # D + D0, seeds 0-4
#   OMNI_ACCOUNT=my_group VARIANTS="D D0 Dpred" SEEDS=0-9 bash hpc/submit_expD.sh
#   OMNI_ACCOUNT=my_group CONFIG=configs/dqn/experiment_d_fixed.yaml RUN_ROOT=runs_expD_fixed \
#       bash hpc/submit_expD.sh                                            # fixed scenario
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$HERE"
: "${OMNI_ACCOUNT:?set OMNI_ACCOUNT to your Omni group/account name}"
PARTITION="${PARTITION:-short}"; QOS="${QOS:-$PARTITION}"
SEEDS="${SEEDS:-0-4}"; VENV="${VENV:-$HOME/venvs/tb3sim}"
CONFIG="${CONFIG:-configs/dqn/experiment_d.yaml}"
RUN_ROOT="${RUN_ROOT:-runs_expD}"; VARIANTS="${VARIANTS:-D D0}"
case "$PARTITION" in short) CC=3; T=06:00:00;; medium) CC=2; T=1-00:00:00;; *) CC=1; T=1-00:00:00;; esac
CONCURRENCY="${CONCURRENCY:-$CC}"; TIME="${TIME:-$T}"
mkdir -p logs "$RUN_ROOT"
echo "[submit_expD] account=$OMNI_ACCOUNT partition=$PARTITION array=${SEEDS}%${CONCURRENCY} variants='$VARIANTS'"
exec sbatch \
  --account="$OMNI_ACCOUNT" --partition="$PARTITION" --qos="$QOS" \
  --time="$TIME" --array="${SEEDS}%${CONCURRENCY}" --output="logs/%x_%A_%a.out" \
  --export=ALL,VENV="$VENV",CONFIG="$CONFIG",RUN_ROOT="$RUN_ROOT",VARIANTS="$VARIANTS" \
  hpc/expD.sbatch
