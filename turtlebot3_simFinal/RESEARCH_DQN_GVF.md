# GVF-enhanced state representations for DQN navigation

A fair, reproducible comparison of three agents on the existing TurtleBot3
simulator: **A** LiDAR-only DQN, **B** frozen-GVF DQN, **C** adaptive-GVF DQN.
Implements the "perception as prediction" idea (Graves, Rezaee & Scheideman):
use policy-based GVF *safety* predictions (directional time-to-collision) as
compact predictive features for control.

Status: the implementation is complete, tested (16 new tests, full suite green),
and smoke-validated end-to-end. The full 5-seed comparison has **not** been run
to convergence — per the brief, that long run is launched separately (on Omni),
and no claim is made that any agent outperforms another without it.

## 1. Architecture audit (existing code)

* **Env** `simulator/environment.py` — Gymnasium `TurtleBot3LidarEnv`. Obs =
  `Box(0,1,(24,))` normalised LiDAR only; pose/goal live in `info`
  (`goal_distance`, `goal_angle`, `ranges_m`, `linear/angular_velocity`).
  Actions `Discrete(5)` `[v,ω]`. **`terminated`** (collision/goal) and
  **`truncated`** (800-step limit) are distinct — DQN bootstraps on truncation
  but not on termination. Reward: `obstacle_avoidance` = +1/step;
  `goal_navigation` = yaw+obstacle shaping, +100 goal, −50 collision.
* **GVFs** `gvf_horde_agent.py` — `DirectionalTTCGVF` (GQ(λ), verified against
  the Horde paper). Each directional demon predicts the scalar `θ·φ(s,a_d)`
  under a deterministic target policy (forward/left/right) ≈ steps-to-collision.
  Policy-conditioned (not action-value). The velocity demons are **degenerate**
  (linear speed constant, angular noise-like — proven in RESULTS_GVF_IMPROVED.md)
  and are **excluded** from the DQN features.
* **Output dims / meaning**: 3 predictive features, each a directional TTC
  (steps). Left/right are only well-posed with obstacles (`obstacles_train.yaml`).
* **Reused**: the demon classes (via `rl/gvf_predictor.py`), `simulator/hpc.py`
  (atomic ckpt, resources, metadata), the `hpc/` SLURM patterns, and the
  pure-NumPy ethos (the DQN is NumPy, so no torch and Omni stays offline-OK).
* **Preserved failure modes**: terminal≠truncation, TTC censoring, no fabricated
  TTC. The four original agent files remain byte-identical (`backup_original/`).

## 2. Architecture & key design choices

* **Common everything** (`rl/experiment.py`): one env, one DQN, one reward path.
  Only the input dimension differs (A: 27 = 24 LiDAR + 3 goal; B/C: 30 = +3 GVF).
  Identical hidden layers `[64,64]`, Adam, γ, lr, batch, target-update, ε-schedule.
* **Fair task** (`rl/encoder.py`): goal-navigation with a hidden goal, so
  goal-relative features (dist, sin/cos angle) are added to the base observation
  of **all three** agents — the baseline is never handicapped (brief §8).
* **DQN** (`rl/dqn.py`): NumPy two-layer MLP + Adam, Huber TD loss, target
  network. Target `y = r + γ(1−terminated)·maxₐ Q_target(s',a)` — the terminal
  mask is `terminated`, not `done`, so truncated episodes still bootstrap.
  Backprop is gradient-checked (`tests/test_dqn.py::test_mlp_gradient_check`).
* **GVF features as safety perception** (`rl/gvf_predictor.py`): 3 directional
  TTCs, z-scored with **train-only** stats, clipped to ±5; non-finite
  predictions are clipped and counted, never silently replaced.

## 3. Replay-buffer strategy for adaptive predictions (the stale-feature fix)

The replay buffer (`rl/replay.py`) stores **raw base observations** (LiDAR [+
goal]), never augmented features. For every minibatch the DQN rebuilds the GVF
features from the raw `s` and `s'` using the **current** predictor (Agent C) or
the frozen one (Agent B) — same convention for `s` and `s'`
(`rl/agent.py::_features`). So Agent C never trains on features produced by an
outdated GVF. Agent C's GVF demons learn from the **live transition stream**
(their own GQ(λ) objective, with ε-greedy behaviour probability for ρ), never
from the DQN TD error. Tests
`test_agent_recomputes_features_from_raw`, `test_frozen_predictor_never_changes`,
and `test_adaptive_predictor_changes_from_same_init` lock this down.

## 4. Files changed / created

New (all additive; nothing existing was modified except adding
`configs/dqn/experiment.yaml` and `tests/test_dqn.py`):

```
rl/__init__.py  dqn.py  replay.py  encoder.py  gvf_predictor.py  agent.py  metrics.py  experiment.py
scripts/pretrain_gvf.py  train_dqn.py  evaluate -> (reuse) run_comparison.py  plot_comparison.py
configs/dqn/experiment.yaml
hpc/dqn.sbatch  hpc/submit_dqn.sh
tests/test_dqn.py       (16 tests)
RESEARCH_DQN_GVF.md
```

## 5. Tests (actually run)

`python -m pytest -q` → **161 passed, 1 skipped** (the 16 `test_dqn.py` tests plus
the pre-existing suite; 1 Linux-only skip). Coverage: MLP gradient check; Huber
clip; DQN terminal-vs-truncation mask; weights change on update; replay stores
raw obs with separate terminal/truncated flags; encoder dims per agent; GVF
feature shape/finiteness and non-finite handling; **B frozen / C adaptive** from
identical init; stale-feature avoidance; checkpoint save/load equivalence (Q
values identical after reload; replay restored); shuffled-feature ablation;
end-to-end short run + eval for A, B, and C (in_dim 27/30/30).

## 6. Smoke comparison (ran; NOT the full experiment)

GVF pretrain 800 episodes (56,772 interactions, reported separately), then A/B/C
at **20k DQN steps × 2 seeds** (ε still 0.37–0.68 → heavily exploring, far from
converged). Cross-seed mean±std (eval = 50 greedy episodes, unseen seed 10000):

| mode | success rate | collision rate | reward | timeout |
|---|---|---|---|---|
| A LiDAR-only | 0.18 ± 0.04 | 0.82 ± 0.04 | 97 ± 9 | 0.00 |
| B frozen-GVF | 0.17 ± 0.07 | 0.83 ± 0.07 | 137 ± 9 | 0.00 |
| C adaptive-GVF | 0.12 ± 0.12 | 0.85 ± 0.15 | 126 ± 46 | 0.03 |

**These differences are within noise** (2 seeds, unconverged); they only show the
pipeline runs and produces comparable metrics. **No agent is claimed to
outperform another.** Plot: `runs_dqn/compare/plots/learning_curves.png`.

## 7. Exact commands

```bash
source ~/venvs/tb3sim/bin/activate && cd turtlebot3_sim
python -m pytest -q tests/test_dqn.py                      # acceptance tests

# 1. pretrain the shared GVF checkpoint (B and C use the SAME one)
python scripts/pretrain_gvf.py --config configs/dqn/experiment.yaml \
    --episodes 2000 --seed 0 --out runs_dqn/gvf_pretrained.npz

# 2. train one agent  (A has no --gvf-checkpoint)
python scripts/train_dqn.py --mode A --seed 0 --out runs_dqn/A_s0 --config configs/dqn/experiment.yaml
python scripts/train_dqn.py --mode B --seed 0 --out runs_dqn/B_s0 --config configs/dqn/experiment.yaml \
    --gvf-checkpoint runs_dqn/gvf_pretrained.npz
python scripts/train_dqn.py --mode C --seed 0 --out runs_dqn/C_s0 --config configs/dqn/experiment.yaml \
    --gvf-checkpoint runs_dqn/gvf_pretrained.npz

# 3. full comparison across seeds + plot
python scripts/run_comparison.py --config configs/dqn/experiment.yaml --seeds 0 1 2 3 4 \
    --gvf-checkpoint runs_dqn/gvf_pretrained.npz --out runs_dqn/compare
python scripts/plot_comparison.py --dir runs_dqn/compare

# on Omni (CPU-only array; pretrains the shared GVF first):
OMNI_ACCOUNT=<group> SEEDS=0-4 bash hpc/submit_dqn.sh
```

Resume is automatic (`<out>/agent.pkl` restores the Q-net, target net, optimizer
state, replay buffer, epsilon step and GVF weights). Each agent/seed writes to
its own directory; nothing overwrites the GVF baselines in `runs_improved/`.

## 8. Results schema

`runs_dqn/<MODE>_s<seed>/`: `run_metadata.json` (config, seed, versions),
`history.json` (per-eval-interval: env_step, epsilon, success/collision/timeout
rate, reward, length, mean loss, gvf_updates, gvf_nonfinite), `summary.json`,
`agent.pkl` (resumable state). `run_comparison.py` writes
`compare/comparison.json` (per-seed + cross-seed mean/std); `plot_comparison.py`
writes the learning-curve PNG.

## 9. What this experiment can and cannot establish

**Can** (once the full 5-seed run completes): whether 3 directional-TTC safety
features change DQN sample efficiency / collision rate / success rate vs raw
LiDAR; whether *continuing* to learn those GVFs during DQN (C) differs from
freezing them (B), from an identical pretrained init and identical DQN budget;
and the extra wall-clock cost of C (reported separately from env interactions).

**Cannot**: prove GVFs help in general (one simulator, one task, kinematic sim,
tiny turning radius); the velocity GVFs are excluded as degenerate; goal-nav
success depends on the shaped reward; and the smoke numbers above are not
evidence of any ordering. Adaptive (C) improving sample efficiency would **not**
imply wall-clock efficiency — C does extra GVF work per step.

## 10. Limitations & next experiment

1. Run the **full** comparison: ≥5 seeds, `total_env_steps` ≥ 150k, on Omni
   (`hpc/submit_dqn.sh`). Then compare success/collision learning curves with
   cross-seed CIs and the "interactions-to-threshold" metric.
2. Add the **randomized-feature ablation** as a 4th condition (B with shuffled
   GVF features) to test whether the *information* in the predictions matters,
   not just the extra inputs. The unit-level check exists; wiring it as a run
   condition is a one-line mode addition.
3. Report GVF prediction error on held-out transitions alongside DQN metrics
   (reuse `scripts/evaluate.py`) so feature quality and task gain are separable.
4. Sweep the GVF feature set (forward-only vs forward+left+right) to see which
   safety predictions carry DQN-relevant signal.
