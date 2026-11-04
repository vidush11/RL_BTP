#!/bin/bash
# Submit the safety-task experiment on Omni (CPU-only). Order matters: tune first.
#   OMNI_ACCOUNT=my_group STAGE=tune bash hpc/submit_safety.sh                # ~1-2 h, 10 cores
#   OMNI_ACCOUNT=my_group METHOD=gvf bash hpc/submit_safety.sh                # seeds 0-2, 30k episodes
#   OMNI_ACCOUNT=my_group METHOD=dqn bash hpc/submit_safety.sh
#   OMNI_ACCOUNT=my_group METHOD=gvf SEEDS=0 SIM="lidar.num_beams=24" TAG=_b24 \
#       bash hpc/submit_safety.sh                                           # beam ablation
# Then, on any node:  python -m safety_exp.analyze --runs runs_safety
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$HERE"
: "${OMNI_ACCOUNT:?set OMNI_ACCOUNT to your Omni group/account name}"
STAGE="${STAGE:-run}"; VENV="${VENV:-$HOME/venvs/tb3sim}"; RUN_ROOT="${RUN_ROOT:-runs_safety}"
if [ "$STAGE" = "tune" ]; then
  PARTITION="${PARTITION:-short}"; CPUS=10; SEEDS=0; CONCURRENCY=1; TIME="${TIME:-06:00:00}"
else
  # A GVF seed is ~4 h on a 2023 laptop core (dominated by the unedited RL_BTP Horde
  # update), so `medium` (1 day) is the safe default; resume makes `short` usable too.
  PARTITION="${PARTITION:-medium}"; CPUS=1; SEEDS="${SEEDS:-0-2}"
  case "$PARTITION" in short) CC=3; T=06:00:00;; medium) CC=2; T=1-00:00:00;; *) CC=1; T=1-00:00:00;; esac
  CONCURRENCY="${CONCURRENCY:-$CC}"; TIME="${TIME:-$T}"
fi
QOS="${QOS:-$PARTITION}"
mkdir -p logs "$RUN_ROOT"
echo "[submit_safety] stage=$STAGE method=${METHOD:-} account=$OMNI_ACCOUNT partition=$PARTITION array=${SEEDS}%${CONCURRENCY}"
exec sbatch \
  --account="$OMNI_ACCOUNT" --partition="$PARTITION" --qos="$QOS" --cpus-per-task="$CPUS" \
  --time="$TIME" --array="${SEEDS}%${CONCURRENCY}" --output="logs/%x_%A_%a.out" \
  --export=ALL,VENV="$VENV",STAGE="$STAGE",METHOD="${METHOD:-}",RUN_ROOT="$RUN_ROOT",SIM="${SIM:-}",TAG="${TAG:-}" \
  hpc/safety.sbatch
