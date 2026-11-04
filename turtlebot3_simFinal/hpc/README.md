# Running turtlebot3_sim on IIIT Delhi Omni

Everything here is CPU-only: a 2-D LiDAR simulator in NumPy. **Do not request a
GPU.** Omni's usage policy prohibits "intentionally keeping GPUs idle", so a
`--gres=gpu:...` line for this workload would violate it. None of these scripts
request one.

> Omni facts used below come from <https://it.iiitd.edu.in/static/hpc/usage_policy.html>.
> Items the policy does not document (storage paths, Python modules, compute-node
> internet, whether CPU-only jobs are accepted) are discovered by
> `scripts/validate_omni.py` — run it first.

## Queues (partition name == QoS name)

| Queue  | Max time | Max jobs/group | Max cores | Max RAM | Use for |
|--------|----------|----------------|-----------|---------|---------|
| short  | 6 h      | 3              | 10        | 64 GB   | validation, all Horde training (fast), eval |
| medium | 1 day    | 2              | 16        | 256 GB  | very long sweeps |
| long   | 3 days   | 1              | 20        | 512 GB  | — (overkill here) |
| cai    | 3 days   | 1              | 32        | 512 GB  | CAI faculty only |

"Max jobs/group" is the concurrency cap, so the training array throttles to it
(`SEEDS%CONCURRENCY`, default 3 on `short`). Horde training is cheap
(5000 episodes ≈ minutes), so `short` is the right default.

## One-time setup

```bash
# 0. transfer the project from your Mac (see repo README §Quick start)
rsync -av --exclude runs --exclude __pycache__ ./ <user>@<omni-login>:~/tb3sim/

# 1. create the user-space venv (no root). If `module avail` shows a Python,
#    `module load` it first and set PYBIN.
cd ~/tb3sim
bash hpc/environment_setup.sh
#    If validate_omni later reports no internet on compute nodes, build a
#    wheelhouse on the login node and install offline inside jobs:
#      bash hpc/environment_setup.sh --download ~/tb3_wheels      # login node
#      # (then in the job / interactively on a node:)
#      bash hpc/environment_setup.sh --offline  ~/tb3_wheels
```

## Workflow

```bash
export OMNI_ACCOUNT=<your_group>          # REQUIRED for every submit (sacctmgr show user $USER)

# 1. VALIDATE + DISCOVER (15-min job): queue allocation, storage, internet, sim works,
#    and whether a no-GPU job is accepted here.
bash hpc/submit_validation.sh
#    read logs/tb3-validate_<jobid>.out and logs/omni_validate_<jobid>.json

# 2. TRAIN a 5-seed array (throttled to the queue's job cap)
bash hpc/submit_training.sh                                   # seeds 0-4, short
SEEDS=0-9 PARTITION=medium EPISODES=20000 bash hpc/submit_training.sh

# 3. EVALUATE every seed's checkpoints with independent ground truth
bash hpc/submit_evaluation.sh

# single seed, no scheduler (inside `srun --pty bash`, or on your Mac):
EVAL=1 EPISODES=500 bash hpc/run_single_seed.sh 0
```

## Monitor / cancel / resume

```bash
squeue --me                      # your queued/running jobs
sacct -j <jobid> --format=JobID,State,Elapsed,MaxRSS,ReqMem   # usage after it runs
scancel <jobid>                  # cancel one job  (scancel <jobid>_<n> for one array task)
scancel -u $USER                 # cancel all your jobs
```

**Resume is automatic.** `scripts/train.py` finds the newest *valid* checkpoint
in the run dir and continues from it, so a requeued job (wall-time hit or
preemption; the scripts set `--requeue`) picks up at the last checkpoint
(every 50 episodes) instead of restarting. Re-submitting the same array is also
safe — finished seeds resume at their last checkpoint and run to the target
episode count. To force a fresh start, pass `--no-resume` (edit the sbatch) or
use a new `RUN_ROOT`.

## Storage

- Keep **code + checkpoints** under `$HOME` (persistent; the 5 TB group quota
  lives there). That is the default `RUN_ROOT=runs`.
- Use scratch only for high-volume *temporary* output, and only if
  `validate_omni` found a scratch path — set `RUN_ROOT=$SCRATCH/tb3_runs`. Scratch
  is **not** backed up; copy important checkpoints back to `$HOME` afterwards
  (`rsync -a $SCRATCH/tb3_runs/ ~/tb3sim/runs/`).
- Output is small: one `.npz` (~70 KB) every 50 episodes + logs + a JSON summary,
  so a 5000-episode run is well under a few MB. The simulator never writes
  per-step files.
```
```

## Retrieve results to your Mac

```bash
rsync -av <user>@<omni-login>:~/tb3sim/runs/ ./runs/
# checkpoints are the unchanged .npz files; load them with your resume/eval scripts anywhere.
```
