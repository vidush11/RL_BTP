#!/bin/bash
# Full GVF-controller vs DQN study (gvfctl). One command, resumable.
#
#   bash scripts/run_full_study.sh                      # new results/<timestamp>/
#   bash scripts/run_full_study.sh results/20261026_1200  # resume an existing run
#   PHASES="2" bash scripts/run_full_study.sh results/X   # only phase 2 (+ report)
#
# Phase 1: DQN tuning grid + headway-tau selection (VALIDATION seeds) + cached
#          Monte-Carlo safety targets. Phase 2: main comparison (10 seeds),
#          ablations (5 seeds), 1M-step strong DQN (3 seeds). Then the report.
# Uses a SLURM array if `sbatch` exists (set SLURM_ARGS, e.g. "--account=G -p short -q short";
# NO_SLURM=1 forces local), otherwise one process per job via xargs -P $(nproc).
# A job whose results/<ts>/jobs/<id>.json exists is skipped.
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT="${1:-results/$(date +%Y%m%d_%H%M%S)}"
BUDGET="${BUDGET:-300000}"
PHASES="${PHASES:-1 2}"
NPROC="${NPROC:-$(getconf _NPROCESSORS_ONLN 2>/dev/null || nproc)}"
PY="${PYTHON:-python}"
mkdir -p "$ROOT/logs" "$ROOT/jobs"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 MPLBACKEND=Agg
export GVF_ROOT="$ROOT" GVF_BUDGET="$BUDGET" GVF_PY="$PY"
echo "[study] root=$ROOT budget=$BUDGET workers=$NPROC phases='$PHASES'"

run_phase() {
  local phase=$1 list="$ROOT/jobs_phase$1.txt"
  $PY -m gvfctl.study list --phase "$phase" > "$list"
  local n; n=$(wc -l < "$list" | tr -d ' ')
  echo "[study] phase $phase: $n jobs ($(ls "$ROOT/jobs" | wc -l | tr -d ' ') results already present)"
  if command -v sbatch >/dev/null 2>&1 && [ -z "${NO_SLURM:-}" ]; then
    sbatch --wait --array=1-"$n"%"${SLURM_CONCURRENCY:-3}" ${SLURM_ARGS:-} --ntasks=1 --cpus-per-task=1 \
      --mem=4G --time="${SLURM_TIME:-06:00:00}" --output="$ROOT/logs/slurm_%A_%a.out" \
      --export=ALL --wrap="J=\$(sed -n \"\${SLURM_ARRAY_TASK_ID}p\" $list); $PY -m gvfctl.study run \$J --root $ROOT --budget $BUDGET"
  else
    xargs -P "$NPROC" -I{} sh -c '$GVF_PY -m gvfctl.study run {} --root "$GVF_ROOT" --budget "$GVF_BUDGET" > "$GVF_ROOT/logs/{}.log" 2>&1 || echo "[study] FAILED {} (see $GVF_ROOT/logs/{}.log)"' < "$list"
  fi
  local missing; missing=$(while read -r j; do [ -f "$ROOT/jobs/$j.json" ] || echo "$j"; done < "$list")
  if [ -n "$missing" ]; then echo "[study] phase $phase incomplete; missing:"; echo "$missing" | head -20; fi
}
for ph in $PHASES; do run_phase "$ph"; done
$PY -m gvfctl.report --root "$ROOT"
echo "[study] report: $ROOT/REPORT.md"
