#!/bin/bash
# Submit a multi-seed Horde training array on Omni.
#
#   OMNI_ACCOUNT=my_group bash hpc/submit_training.sh            # seeds 0-4, short queue
#   OMNI_ACCOUNT=my_group SEEDS=0-9 PARTITION=medium EPISODES=20000 \
#       bash hpc/submit_training.sh
#
# Env vars:
#   OMNI_ACCOUNT  (REQUIRED) your SLURM account / group name (`--account`).
#   PARTITION     queue name; partition and QoS share a name on Omni.
#                 short (<=6h, 3 jobs/grp) | medium (<=1d, 2) | long (<=3d, 1) | cai.
#   SEEDS         array spec, e.g. "0-4" (default) or "0-9".
#   CONCURRENCY   max simultaneous array tasks; MUST respect the queue's
#                 "max jobs per group" (short=3, medium=2, long/cai=1).
#   EPISODES, CONFIG, AGENT, RUN_ROOT, VENV, TIME  -> see hpc/train.sbatch.
#
# This CPU-only workload requests NO GPU on purpose (Omni forbids idle GPUs).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$HERE"

: "${OMNI_ACCOUNT:?set OMNI_ACCOUNT to your Omni group/account name (see usage policy / sacctmgr)}"
PARTITION="${PARTITION:-short}"
QOS="${QOS:-$PARTITION}"             # Omni: QoS name == partition name
SEEDS="${SEEDS:-0-4}"
EPISODES="${EPISODES:-5000}"
CONFIG="${CONFIG:-configs/default.yaml}"
AGENT="${AGENT:-gvf_horde_agent.py}"
RUN_ROOT="${RUN_ROOT:-runs}"
VENV="${VENV:-$HOME/venvs/tb3sim}"

# Default per-queue wall time (max allowed); override with TIME.
case "$PARTITION" in
  short)  DEF_TIME="06:00:00"; DEF_CC=3;;
  medium) DEF_TIME="1-00:00:00"; DEF_CC=2;;
  long)   DEF_TIME="3-00:00:00"; DEF_CC=1;;
  cai)    DEF_TIME="3-00:00:00"; DEF_CC=1;;
  *)      DEF_TIME="06:00:00"; DEF_CC=1;;
esac
TIME="${TIME:-$DEF_TIME}"
CONCURRENCY="${CONCURRENCY:-$DEF_CC}"

mkdir -p "$RUN_ROOT" logs
ARRAY="${SEEDS}%${CONCURRENCY}"      # throttle to the group's concurrent-job cap

echo "[submit] account=$OMNI_ACCOUNT partition=$PARTITION qos=$QOS array=$ARRAY time=$TIME"
echo "[submit] agent=$AGENT episodes=$EPISODES config=$CONFIG run_root=$RUN_ROOT"
echo "[submit] (CPU-only: no --gres requested)"

exec sbatch \
  --account="$OMNI_ACCOUNT" \
  --partition="$PARTITION" --qos="$QOS" \
  --time="$TIME" \
  --array="$ARRAY" \
  --output="logs/%x_%A_%a.out" \
  --export=ALL,VENV="$VENV",EPISODES="$EPISODES",CONFIG="$CONFIG",AGENT="$AGENT",RUN_ROOT="$RUN_ROOT" \
  hpc/train.sbatch
