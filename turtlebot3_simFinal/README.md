# Gazebo-free TurtleBot3 Burger simulator (DQN + Horde/GVF)

A headless 2-D simulator in plain Python + NumPy that gives your DQN and Horde/GVF
agents the same LiDAR-only environment without Gazebo, ROS or ROS 2.
Your agent/algorithm files are **byte-identical** to what you supplied
(`sha256sum -c ORIGINAL_SHA256.txt`); the simulator is a separate package and your
scripts reach it through a small ROS stand-in (`simulator/ros_compat`).

## 1. What your code does (inspected, not assumed)

| Item | Existing behaviour (file) | In the simulator |
|---|---|---|
| Env API | ROS 2 services `make_environment`, `reset_environment`, `rl_agent_interface` (`dqn_environment.py`) | same three services via the shim, plus a Gymnasium env |
| Observation | list of 24 raw ranges in **metres** (Inf→3.5, NaN→0); Horde normalises by 3.5 itself | Gymnasium obs = normalised `Box(0,1,(24,),float32)`; shim returns metres (`info["ranges_m"]`) so unmodified agents see what they saw |
| LiDAR | 360°, beam 0 = straight ahead, CCW, 24 beams assumed by Horde | same convention, 3.5 m max |
| Reward (live) | constant `1.0` per step | default `reward.per_step = 1.0` (identical) |
| Termination (live) | `min(range) < 0.15 m`; goal and 800-step timeout are commented out | same rule kept (`collision_lidar_threshold: 0.15`); `max_episode_steps: 800` is enabled and reported as **truncated** |
| Actions | `linear.x = 0.2`, `angular = [+1.5, +0.75, 0, −0.75, −1.5]` | **see conflict below** |
| Spawn | respawn at (0,0), heading 0 | default `spawn.mode: fixed` at (0,0,0) |
| Checkpoints | one `.npz` per 50 episodes, keys `theta_*`, `w_*`, `w_future_*`, `episode` | untouched; written by your script |
| Gazebo/ROS deps | `dqn_gazebo.py` (goal spawn, reset), `rclpy`, `/odom`, `/get_action`, `result`, matplotlib window | replaced by the simulator + shim (`/odom` is fed from the robot state; publishers are no-ops) |

**Action-table conflict (needs your decision).** The brief says `v = 0.15`, `ω = −1.5…+1.5` for actions 0…4.
Your code uses `v = 0.2` and the *opposite* order. Positive ω is a left turn and Horde calls action 1 "left",
so with the brief's table **action 1 turns right** and `TTC_left`/`TTC_right` swap meaning. The default
follows the brief; `configs/gazebo_compat.yaml` reproduces your Gazebo mapping.
Use that config for Gazebo-trained Horde checkpoints.

## 2. Layout

```
simulator/          config.py robot.py world.py lidar.py rewards.py environment.py vector.py
                    hpc.py  (atomic ckpt save, resume scan, resource detection, run metadata/summary)
simulator/ros_compat/   shim.py  (fake rclpy / message modules answered by the simulator)
configs/            default.yaml  gazebo_compat.yaml  obstacles.yaml  goal_nav.yaml
scripts/            run_agent.py (run your agent scripts unmodified)  smoke_test.py
                    validate_omni.py  (cluster discovery + env validation)
                    train.py          (resource-aware, auto-resuming training wrapper)
                    evaluate.py        (checkpoint eval with INDEPENDENT ground truth)
tests/              136 tests (pytest); tests/test_hpc.py covers the HPC utilities
benchmarks/         benchmark_environment.py
hpc/                environment_setup.sh  submit_validation.sh  submit_training.sh
                    submit_evaluation.sh  run_single_seed.sh  train.sbatch  evaluate.sbatch  README.md
slurm/              run_horde.sbatch (compatibility wrapper -> scripts/train.py; see hpc/)
dqn_environment.py dqn_gazebo.py gvf_horde_agent.py gvf_horde_agent_evaluate.py   <- your files, unchanged
ORIGINAL_SHA256.txt requirements.txt pytest.ini
```

## 3. Model (what is and is not simulated)

* **Kinematics:** unicycle, exact constant-(v, ω) arc integration (no Euler drift), `dt = 0.1 s`,
  heading wrapped to [−π, π). The brief's Euler formulas are the small-dt limit (tested).
  Optional `motion.action_repeat` runs several sub-steps per agent step.
* **World:** 4 m × 4 m rectangle + axis-aligned rectangular obstacles from YAML. Random obstacles
  only if `world.randomize_obstacles: true`. Spawn/goal are always collision-free.
* **Collision:** the robot is a disc (r = 0.033 m). Its swept path is tested exactly against every
  nearby obstacle (segment-vs-rectangle distance on chords ≤ r/2) and walls, so **no speed can
  tunnel through a thin obstacle** (tested up to 10 m per step against a 5 mm wall; broad-phase
  checked against the exact test on 20 000 random paths, 0 disagreements). On contact the robot
  stays where it was, velocity is zeroed, the episode ends. Collisions are reported as `geometric`
  (disc touched geometry) or `lidar` (your `min range < 0.15 m` rule fired first, which is what
  happens with default settings because 0.15 > 0.033).
* **LiDAR:** vectorised ray–slab casting against walls and obstacles; `info["ranges_m"]` in metres,
  observation `min(d, 3.5) / 3.5`. Optional Gaussian noise (off by default, seeded).
* **Observation is LiDAR-only.** Pose, heading and goal are only in `info`.
* **Rewards/termination:** see the docstring of `simulator/rewards.py`. `obstacle_avoidance`
  (default) = your live code. `goal_navigation` (hidden goal) uses the terms from the *commented-out*
  code (yaw reward, weighted obstacle term, +100 goal, −50 collision) — ports of your functions
  checked numerically against them, but **not** claimed to match any earlier experiment.
  `terminated` = collision or goal; `truncated` = step limit; terminated wins ties.
* **RNG:** one `numpy.random.Generator` per env instance, no global state; `reset(seed=s)` reproduces
  spawn, goal, random obstacles and LiDAR noise.

## 4. Install

```bash
# Linux or macOS (no root needed)
python3 -m venv ~/venvs/tb3sim && source ~/venvs/tb3sim/bin/activate
pip install -r requirements.txt
```

## 5. Run

```bash
python scripts/smoke_test.py                  # environment only
python scripts/smoke_test.py --horde          # + your Horde script: train, save, resume, evaluate
python -m pytest                              # 136 tests
python benchmarks/benchmark_environment.py
python scripts/validate_omni.py               # discovery + validation (run this on Omni too)

# Resource-aware training wrapper (auto-resume + run metadata/summary; wraps run_agent.py):
python scripts/train.py --agent gvf_horde_agent.py --seed 0 --run-dir runs/horde_s0 --episodes 5000
# re-running the SAME command resumes from the newest valid checkpoint automatically.

# Independent ground-truth evaluation (read-only; writes runs/horde_s0/eval/metrics.json):
python scripts/evaluate.py --checkpoint-dir runs/horde_s0/gvf_checkpoints_horde --num-eval-episodes 20

# Horde training / evaluation (your scripts, unmodified; --param is ros2's  -p name:=value)
python scripts/run_agent.py gvf_horde_agent.py --seed 0 --run-dir runs/horde_s0 \
       --param max_training_episodes:=5000
python scripts/run_agent.py gvf_horde_agent_evaluate.py --run-dir runs/horde_s0_eval \
       --param checkpoint_dir:=runs/horde_s0/gvf_checkpoints_horde --param num_eval_episodes:=10
# resume: --param resume_checkpoint:=runs/horde_s0/gvf_checkpoints_horde/gvf_weights_ep_002000.npz

# Gazebo action mapping / obstacles / goal task
python scripts/run_agent.py gvf_horde_agent.py --config configs/gazebo_compat.yaml ...
```

`run_agent.py` puts the shim in place, `chdir`s into `--run-dir` (so your hard-coded
`gvf_checkpoints_horde` lands there), forces matplotlib's Agg backend (the figure is saved as a PNG at
the end), removes `time.sleep` pacing (`--realtime` keeps it) and seeds `random`/NumPy.

**DQN.** Your DQN *agent* file was not supplied, so it has not been run. What is tested is the
service contract of the DQN environment node it consumed (`tests/test_agent_compat.py::test_dqn_service_contract`).
If the agent uses only the same ROS surface as the Horde scripts, `python scripts/run_agent.py <your_dqn_agent>.py --config ...`
should work; anything outside that surface (`create_timer`, `create_service`, other topics) raises
`NotImplementedError` or a warning instead of silently doing nothing. Send the file and I will test it.
For a native (non-ROS) integration use the Gymnasium env directly:
`gym.make("TurtleBot3Lidar-v0")`, 24-float32 observation, `Discrete(5)`.

## 6. Running on Omni (IIIT Delhi HPC)

The scheduler, queues and limits below come from the official usage policy
(<https://it.iiitd.edu.in/static/hpc/usage_policy.html>); items it leaves open
(storage paths, Python modules, compute-node internet, whether a CPU-only job is
accepted) are discovered on the cluster by `scripts/validate_omni.py`. I have no
Omni account, so **nothing here has been executed on Omni** — it has been built
to the documented policy and verified locally. Full instructions: `hpc/README.md`.

* **Scheduler:** SLURM. Queues (partition name == QoS name): `short` (≤6 h, 3
  jobs/group, 10 cores, 64 GB), `medium` (≤1 d, 2 jobs, 16 cores), `long` (≤3 d,
  1 job), `cai` (faculty). `--account=<group>` is mandatory.
* **No GPU.** This is a CPU-only 2-D simulator. The scripts request **no**
  `--gres=gpu`; Omni forbids keeping GPUs idle, so do not add one.

```bash
# 1. transfer + set up (login node; no root)
rsync -av --exclude runs --exclude __pycache__ ./ <user>@<omni-login>:~/tb3sim/
cd ~/tb3sim && bash hpc/environment_setup.sh        # --download/--offline if no node internet

export OMNI_ACCOUNT=<your_group>
# 2. validate + discover on a compute node (15-min job)
bash hpc/submit_validation.sh
# 3. train 5 seeds (array throttled to the queue's job cap)
bash hpc/submit_training.sh
# 4. evaluate every seed with independent ground truth
bash hpc/submit_evaluation.sh
# monitor / cancel: squeue --me ; scancel <jobid>
```

Resume is automatic (newest valid checkpoint); a requeued job continues instead
of restarting. Bring results back with
`rsync -av <user>@<omni-login>:~/tb3sim/runs/ ./runs/`; the `.npz` checkpoints are
unchanged, so they load with your existing scripts anywhere.

## 7. Benchmark (hardware-dependent — run `benchmarks/benchmark_environment.py` yourself)

Two machines, same workload. **Neither is Omni** (no measurement on the cluster
has been made — do not extrapolate these as Omni numbers).

| | 2-core sandbox (Py 3.13, NumPy 2.5.3) | dev Mac, Apple Silicon 10-core (Py 3.13, NumPy 2.5.4) |
|---|---|---|
| single env, empty arena | ~14,800 steps/s | ~51,500 steps/s (316 ep/s) |
| single env, 4 obstacles | ~8,850 steps/s | ~29,800 steps/s (280 ep/s) |
| LiDAR cast, 24 beams | ~23 µs / ~48 µs (empty / 4 obs) | ~7.6 µs / ~16.7 µs |
| `SyncVectorEnv`, 1/2/4 envs | ~6.0k / 6.5k / 7.1k steps/s | ~21k / 23k / 24k steps/s |
| `AsyncVectorEnv`, 2/4 envs | ~3.6k / 4.2k steps/s | ~15k / 27k steps/s |

Each step costs well under 0.1 ms, so cross-process pipes cost more than the work: **in-process
multi-env speedups do not appear here** (async is slower than sync). Parallelism is best obtained
by running independent processes (SLURM array tasks, one seed each).
Your Horde training through the shim ran at ~1,000–2,200 agent steps/s in my runs (the agent's own Python work dominates).

## 8. Measured Horde behaviour on this simulator (unmodified algorithm — worth knowing before long runs)

I ran your Horde script for 1,500 episodes and evaluated all 30 checkpoints with your evaluation script
(default config). These are observations, not simulator validation:

* **Forward TTC barely learns.** On-policy MAE stays ≈ 50 from checkpoint 50 to 1500 (a zero predictor scores ≈ 62).
  The features *can* represent it: a least-squares fit on your exact features reaches MAE 4.6 on ground-truth
  forward TTC from random poses. The trained θ_forward has max |weight| 4.7 where ≈ 225 is needed.
  So it is slow TD learning at the given step sizes, not a missing signal.
* **Per-step distance matters.** With the brief's 0.15 m/s × 0.1 s the robot moves 1.5 cm per step (forward TTC ≈ 123 steps).
  With `motion.action_repeat: 5` (TTC ≈ 25 steps) the on-policy forward MAE is ≈ 5.5 against a ≈ 13 zero baseline — better relative error — but
  across random poses the learned weights were still no better than predicting the mean after 1,500 episodes.
  I have **not** established which per-step distance corresponds to your Gazebo runs.
* **Velocity demons do not learn.** Your 19-step importance-weight product has mean 1 but, after the clip at 50, exceeds 1e-3 only
  0.03 % of the time under uniform-random behaviour, so the weights stay ≈ 0 (max |w| ≈ 0.0015); linear-velocity RMSE sits at 0.150.
  This is arithmetic about the estimator and independent of the simulator.
* **TTC_left / TTC_right evaluation is degenerate in an empty arena.** A constant-turn policy circles forever (radius 0.1–0.2 m),
  hits the 800-step limit (*truncated*), and your TTC target `arange(steps, 0, −1)` then treats it as a collision (MAE ≈ 380).
  `run_agent.py` prints a warning when this happens. Fixing it means changing the evaluation design (e.g. spawn near walls), which I left alone.

**`scripts/evaluate.py` reproduces all of the above with honest ground truth** and
is the recommended way to measure checkpoints. It drives the Gymnasium env
directly, and for the TTC demons it uses the TRUE remaining-steps-to-collision,
**excluding** step-limit truncations as *censored* instead of scoring them as a
bogus countdown. On the empty default arena it therefore reports TTC_forward MAE
≈ 50 (vs a zero-predictor 62.5 and a mean-predictor 31 — i.e. the demon is worse
than predicting the mean), and TTC_left/right as `nan` over 0 collided / N
censored episodes rather than ≈ 380. Its `rho_product_diagnostic` reproduces the
velocity-demon suppression (the 19-step importance-weight product exceeds 1e-3
only ≈ 0.035 % of the time under uniform behavior), and the future-velocity RMSE
matches the mean-predictor baseline — the velocity demons learn nothing. Per the
brief, the learning algorithm itself is left unchanged and old checkpoints stay
loadable; `evaluate.py` only measures them correctly.

## 9. Differences from Gazebo / limitations

* Kinematic only: instant velocity changes, no wheel slip, inertia, motor lag, contact physics or sensor latency.
* Time: one agent step = exactly `dt × action_repeat` of simulated time. In Gazebo a step lasts as long as the service round trip,
  and `calculate_state` returns the scan *before* the commanded motion has taken effect; neither is modelled.
* LiDAR: ideal geometric rays; no beam divergence, no dropouts, no Inf/NaN readings (no-hit = `max_range`); noise optional.
* Collisions: disc footprint r = 0.033 m (from the brief) vs. the real Burger body; the 0.15 m proximity rule usually fires first.
* World: the supplied files do not contain the Gazebo `turtlebot3_dqn_world` obstacle geometry, so `configs/obstacles.yaml` is an
  illustrative layout, **not** a copy of it. Goals follow the same 0.1 m lattice as `dqn_gazebo.py`.
* `/odom` velocities are the commanded values; position odometry is ground truth.
* **No comparison against Gazebo has been performed, so nothing here is validated against it.**
* Not tested: the DQN agent, Omni itself, macOS (only Linux was available), any ROS-installed machine.

## 10. API changes

None to your files. New public API: `TurtleBot3LidarEnv`, `SimConfig`/`load_config`, `make_vec_env`, `ros_compat.install`.

## 11. Experiment D — LiDAR-only DQN with a GVF-shaped reward

Separate from the A/B/C study (`RESEARCH_DQN_GVF.md`), which it does not change.
The DQN sees **only the 24 LiDAR values**; the three directional-TTC GVFs learn
concurrently from the same interaction stream and enter **only the reward**:
`r = r_task + lam * (gamma*Phi(s') - Phi(s))` (potential-based safety shaping;
terminal rewards +100 goal / -50 collision). Collision ends the episode
immediately and takes precedence over goal success. Full design, GVF semantics,
temporal-consistency rules, limitations and measured results:
`DESIGN_EXPERIMENT_D.md`.

```bash
# train (auto-resumes from <out>/agent.pkl)
python scripts/train_expD.py --seed 0 --out runs_expD/D_s0
# lambda = 0 control (identical except no GVF shaping) -- the key comparison for D
python scripts/train_expD.py --seed 0 --out runs_expD/D0_s0 \
    --set reward.lam=0 --set experiment_id=D0_lidar_only_terminal_reward
# ablations: --set reward.prediction_only=true   |   --set reward.shaping=absolute
# fixed, LiDAR-identifiable scenario:
python scripts/train_expD.py --config configs/dqn/experiment_d_fixed.yaml --seed 0 --out runs_expD_fixed/D_s0

# plots (mean +/- std over seeds)
python scripts/plot_expD.py --group "D=runs_expD/D_s*" --group "D0=runs_expD/D0_s*" --out runs_expD/plots

# task outcomes for A, B, C, D on identical eval scenarios (returns are NOT compared)
python scripts/compare_agents.py --abc-dir runs_dqn/compare --d-dir runs_expD \
    --eval-seed 20000 --eval-episodes 100 --out runs_expD/compare_ABCD.json

# tests
python -m pytest -q tests/test_expD.py

# Omni (CPU-only array; D and D0 share seeds)
OMNI_ACCOUNT=<group> SEEDS=0-4 bash hpc/submit_expD.sh
```

## 12. GVF predictive controller vs DQN (`gvfctl/`)

A complete, paired comparison of a GVF-based predictive controller against DQN
on LiDAR-only navigation, in two parts:
* Part 1: safety GVF + steering, 5 actions at 0.15 m/s.
* Part 2: + a speed-aware (headway) safety cumulant, future linear and angular
  velocity GVFs, and a speed controller, over 15 actions.

Baselines:
* A: standard DQN.
* B: strong DQN (double-Q, mirror augmentation, rich features).
* C: a hand-written reactive rule.
* D: the original linear Horde demons from `gvf_horde_agent.py`.

It also runs ablations, OOD layouts, LiDAR noise, Gazebo action-table
transfer, run-until-collision survival analysis, Monte-Carlo prediction
quality, and statistics.

Pure NumPy, new files only. `simulator/` and the original agents are untouched.
Design, fairness notes and deviations are in `gvfctl/SETUP.md`; results go to
`results/<timestamp>/REPORT.md`.

```bash
# everything (phase 1: tuning + MC targets; phase 2: main/ablations/1M DQN; then report). Resumable.
bash scripts/run_full_study.sh                       # -> results/<timestamp>/
bash scripts/run_full_study.sh results/<timestamp>   # resume
BUDGET=100000 bash scripts/run_full_study.sh         # smaller budget per run

# single job / report only
python -m gvfctl.study list --phase 2
python -m gvfctl.study run main__p1__gvf__s0 --root results/X
python -m gvfctl.report --root results/X

# tests
python -m pytest -q tests/test_gvfctl.py
```
On a SLURM cluster, `run_full_study.sh` submits arrays automatically if
`sbatch` exists. Set `SLURM_ARGS="--account=<group> -p short -q short"`, or
`NO_SLURM=1` to run locally.

## 13. Paper reproduction: perception as prediction (GVFs) vs DQN (`safety_exp/`)

Graves, Rezaee & Scheideman (IROS 2019) on this simulator. One shared task
(`configs/safety_task.yaml`): drive at `v_target` = 0.2 m/s among 3 random obstacles
without colliding; inertial dynamics, actions FORWARD/LEFT/RIGHT/STOP, 72-beam LiDAR
+ v + omega, 3 stacked frames. Three controllers are compared:

* **paper GVF controller** — action-conditioned GVF networks (Algorithm 1, one per
  gamma), trained off-policy from a behaviour policy, frozen, then driving with the
  Alg. 3 + Alg. 2 rule controller (`agents/gvf/`);
* **given GVFs** — the unedited RL_BTP Horde demons (`third_party/rl_btp/`), learned
  from the same data through `agents/horde/adapter.py`, driving their own predictive
  controller (`agents/horde/controller.py`);
* **DQN** on the task reward (`agents/dqn/`).

All new simulator features are opt-in; `tests/test_golden_regression.py` proves the
defaults are bit-identical. Every choice is in [DECISIONS.md](DECISIONS.md); results
in [REPORT.md](REPORT.md).

```bash
# 1. validation tuning (equal budget for both methods; validation seeds only)
python -m safety_exp.tune --out runs_safety/tuning --workers 10
# 2. 30,000 training episodes per method per seed (resumable)
python -m safety_exp.run --method gvf --seed 0 --tuned runs_safety/tuning/tuning.json --out runs_safety/gvf_s0 --sweep-gammas
python -m safety_exp.run --method dqn --seed 0 --tuned runs_safety/tuning/tuning.json --out runs_safety/dqn_s0
# optional 24-beam ablation
python -m safety_exp.run --method gvf --seed 0 --tuned runs_safety/tuning/tuning.json --sim lidar.num_beams=24 --out runs_safety/gvf_b24_s0
# 3. figures + tables (learning curves, final test/hard tables, Fig. 4/5 analogues,
#    prediction quality of both GVF families vs Monte-Carlo returns, gamma/beta table)
python -m safety_exp.analyze --runs runs_safety
# tests
pytest -q tests/test_safety_sim.py tests/test_horde_adapter.py tests/test_safety_agents.py tests/test_golden_regression.py
# Omni (CPU-only)
OMNI_ACCOUNT=<group> STAGE=tune bash hpc/submit_safety.sh
OMNI_ACCOUNT=<group> METHOD=gvf bash hpc/submit_safety.sh   # then METHOD=dqn
```
