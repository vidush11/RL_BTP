# Omni upgrade — deliverables, commands, and limitations

This summarises the HPC upgrade. The simulator and the four original agent files
were already in the repo; this pass added the cluster workflow, resumable
training, honest evaluation, and tests. **The four original files
(`dqn_environment.py`, `dqn_gazebo.py`, `gvf_horde_agent.py`,
`gvf_horde_agent_evaluate.py`) are byte-identical** — `pytest` enforces it
(`tests/test_hpc.py::test_original_files_unchanged`) and `sha256sum -c
ORIGINAL_SHA256.txt` passes.

## What was added

| Area | File(s) | What it does |
|---|---|---|
| Cluster discovery + validation | `scripts/validate_omni.py`, `hpc/submit_validation.sh` | On Omni: finds the queue allocation, storage paths, Python/modules, compute-node internet, and whether a CPU-only job is accepted; validates imports, config, LiDAR, collision, Gymnasium API, ROS shim, write access, and a headless sim. Writes a JSON report. |
| HPC utilities | `simulator/hpc.py` | Atomic `.npz`/JSON writes (crash-safe), newest-**valid**-checkpoint scan (skips a half-written latest), SLURM resource detection, thread pinning, run metadata + run summary. |
| Resumable training | `scripts/train.py`, `hpc/train.sbatch`, `hpc/submit_training.sh` | Resource-aware wrapper around your unmodified agent: auto-resume, overwrite safeguard, config snapshot, `run_metadata.json` + `run_summary.json`. Multi-seed via SLURM array throttled to the queue's job cap. |
| Independent evaluation | `scripts/evaluate.py`, `hpc/evaluate.sbatch`, `hpc/submit_evaluation.sh` | Read-only checkpoint eval with TRUE ground truth: TTC against real steps-to-collision (truncations censored, not counted as collisions), future-velocity vs the velocity measured `horizon` steps later, baselines, and the importance-sampling suppression diagnostic. |
| Local/interactive | `hpc/run_single_seed.sh` | Train (and optionally eval) one seed with no scheduler. |
| Setup | `hpc/environment_setup.sh` | User-space venv; online or offline (wheelhouse) install; no root. |
| Docs/tests | `hpc/README.md`, `tests/test_hpc.py`, README updates | Cluster runbook; 20 new tests; refreshed README §2/§5/§6/§7/§8. |

## Exact commands

### On your Mac
```bash
cd turtlebot3_sim
python3 -m venv ~/venvs/tb3sim && source ~/venvs/tb3sim/bin/activate
pip install -r requirements.txt
python -m pytest                       # expect 135 passed, 1 skipped
python scripts/validate_omni.py        # expect 0 fail (cluster items show WARN off-cluster)
# quick end-to-end (train 500 ep + evaluate one seed, no scheduler):
EVAL=1 EPISODES=500 VENV=~/venvs/tb3sim bash hpc/run_single_seed.sh 0
```

### On Omni
```bash
# transfer
rsync -av --exclude runs --exclude __pycache__ ./ <user>@<omni-login>:~/tb3sim/
ssh <user>@<omni-login>
cd ~/tb3sim
bash hpc/environment_setup.sh          # add --download/--offline if compute nodes have no internet

export OMNI_ACCOUNT=<your_group>       # from `sacctmgr show user $USER` / your PI
bash hpc/submit_validation.sh          # 15-min job: discovery + validation; read logs/omni_validate_*.json
bash hpc/submit_training.sh            # seeds 0-4 on `short`, throttled to 3 concurrent
bash hpc/submit_evaluation.sh          # independent ground-truth metrics per seed
squeue --me                            # monitor;  scancel <jobid> to cancel
rsync -av <user>@<omni-login>:~/tb3sim/runs/ ./runs/   # pull results back
```
Resume is automatic: re-submitting, or a requeued job after a wall-time hit,
continues from the newest valid checkpoint (every 50 episodes).

## Results actually obtained (local; NOT on Omni)

* **Tests:** `135 passed, 1 skipped` (the skip is a Linux-only fork test). The
  116→136 jump is the new `tests/test_hpc.py`.
* **Benchmark (dev Mac, Apple Silicon, 10 cores):** ~51,500 steps/s single env
  empty, ~29,800 with 4 obstacles; LiDAR cast 7.6/16.7 µs. See README §7 for the
  2-core sandbox comparison.
* **Phase-7 reproduction (via `scripts/evaluate.py`):** forward-TTC MAE ≈ 50
  (zero-predictor 62.5, mean-predictor 31); TTC_left/right = `nan` over 0
  collided / N censored in the empty arena; velocity-demon rho-product exceeds
  1e-3 only ≈ 0.035 % of the time; future-velocity RMSE ≈ mean-predictor
  baseline. These match the README's prior observations with correct targets.

## Limitations and unverified assumptions

1. **Not run on Omni.** I have no cluster account. The SLURM scripts follow the
   documented policy and pass `bash -n` syntax checks, but no job has executed on
   Omni. `validate_omni.py` is the first thing to run there.
2. **CPU-only job acceptance is unconfirmed.** Every policy example requests a
   GPU; whether `short`/`medium` accept a job with no `--gres` is not documented.
   The validation job tests exactly this. If a queue rejects CPU-only jobs, ask
   the helpdesk which partition serves CPU work.
3. **Account/group name** must be supplied (`OMNI_ACCOUNT`); it is not knowable
   from here.
4. **Storage paths / compute-node internet / available Python modules** are
   discovered at runtime, not assumed. Defaults keep code + checkpoints in
   `$HOME`; scratch is opt-in via `RUN_ROOT`.
5. **DQN agent** was not supplied, so only the DQN *service contract* is tested
   (`tests/test_agent_compat.py`). `train.py --agent <your_dqn>.py` should work if
   it uses the same ROS surface as the Horde scripts.
6. **Learning algorithm unchanged.** Per the brief, the GVF demons' slow/absent
   learning is diagnosed and measured honestly, not "fixed" by editing the
   algorithm; old checkpoints stay loadable.
7. **No Gazebo comparison.** The simulator is still not validated against the
   original Gazebo world (its obstacle geometry was never supplied).
8. **Graceful wall-time handling** relies on frequent checkpoints (every 50
   episodes) + auto-resume, not on signal trapping inside the unmodified agent.
   Resume granularity is therefore one checkpoint interval.
