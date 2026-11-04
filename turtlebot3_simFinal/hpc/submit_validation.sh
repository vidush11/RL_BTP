#!/bin/bash
# Submit a tiny validation+discovery job (runs scripts/validate_omni.py on a
# COMPUTE node). This is step 1 on the cluster: it tells you the queue
# allocation, storage paths, whether the node has internet, and whether the
# simulator works there. It also tests whether a CPU-only job (no GPU) is
# accepted by the chosen queue.
#   OMNI_ACCOUNT=my_group bash hpc/submit_validation.sh
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$HERE"
: "${OMNI_ACCOUNT:?set OMNI_ACCOUNT to your Omni group/account name}"
PARTITION="${PARTITION:-short}"
QOS="${QOS:-$PARTITION}"
VENV="${VENV:-$HOME/venvs/tb3sim}"
mkdir -p logs
exec sbatch \
  --account="$OMNI_ACCOUNT" \
  --partition="$PARTITION" --qos="$QOS" \
  --job-name=tb3-validate --ntasks=1 --cpus-per-task=2 --mem=4G --time=00:15:00 \
  --output="logs/%x_%j.out" \
  --export=ALL,VENV="$VENV" \
  --wrap='set -euo pipefail; export MPLBACKEND=Agg OMP_NUM_THREADS=1;
          source "${VENV}/bin/activate";
          python scripts/validate_omni.py --json "logs/omni_validate_${SLURM_JOB_ID}.json"'
