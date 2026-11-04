#!/bin/bash
# Submit checkpoint evaluation for each trained seed (independent ground truth).
#   OMNI_ACCOUNT=my_group bash hpc/submit_evaluation.sh                 # seeds 0-4
#   OMNI_ACCOUNT=my_group SEEDS=0-9 EVAL_EPISODES=50 bash hpc/submit_evaluation.sh
# Env vars mirror submit_training.sh (OMNI_ACCOUNT required). CPU-only, no GPU.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$HERE"

: "${OMNI_ACCOUNT:?set OMNI_ACCOUNT to your Omni group/account name}"
PARTITION="${PARTITION:-short}"
QOS="${QOS:-$PARTITION}"
SEEDS="${SEEDS:-0-4}"
CONFIG="${CONFIG:-configs/default.yaml}"
AGENT="${AGENT:-gvf_horde_agent.py}"
RUN_ROOT="${RUN_ROOT:-runs}"
VENV="${VENV:-$HOME/venvs/tb3sim}"
EVAL_EPISODES="${EVAL_EPISODES:-20}"
EVAL_SEED="${EVAL_SEED:-1000}"
TIME="${TIME:-01:00:00}"
case "$PARTITION" in short) CC=3;; medium) CC=2;; *) CC=1;; esac
CONCURRENCY="${CONCURRENCY:-$CC}"

mkdir -p logs
echo "[submit-eval] account=$OMNI_ACCOUNT partition=$PARTITION array=${SEEDS}%${CONCURRENCY}"
exec sbatch \
  --account="$OMNI_ACCOUNT" \
  --partition="$PARTITION" --qos="$QOS" \
  --time="$TIME" \
  --array="${SEEDS}%${CONCURRENCY}" \
  --output="logs/%x_%A_%a.out" \
  --export=ALL,VENV="$VENV",CONFIG="$CONFIG",AGENT="$AGENT",RUN_ROOT="$RUN_ROOT",EVAL_EPISODES="$EVAL_EPISODES",EVAL_SEED="$EVAL_SEED" \
  hpc/evaluate.sbatch
