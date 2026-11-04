# Experiment D — LiDAR-only DQN with a GVF-shaped reward

Experiment id: `D_lidar_only_gvf_reward`. Independent of A/B/C
(`RESEARCH_DQN_GVF.md`); nothing in those agents changed.

**Question.** Can a DQN that observes only the LiDAR scan learn better
goal-directed navigation when its reward is informed by GVF predictions that
learn concurrently from the same interaction stream?

**Headline caveat (read first).** In the main scenario the goal is resampled
every episode and is *not* observable from LiDAR, and the existing GVFs predict
only *safety* (time-to-proximity), not goal progress. So D's shaping can only
make the robot safer; it cannot tell it where the goal is. Goal success in the
main scenario is therefore expected to stay near chance for every LiDAR-only
variant. The fixed scenario (§9) is included so goal-directed learning is
possible in principle.

## 1. Variants (same code, same seeds; CLI `--set` overrides)

| id | reward | purpose |
|---|---|---|
| **D** `D_lidar_only_gvf_reward` | `r_task + 1.0 * r_GVF` (potential shaping) | main experiment |
| **D0** `D0_lidar_only_terminal_reward` | `r_task` only (`reward.lam=0`) | **control**: isolates the effect of GVF shaping; identical in every other respect |
| Dpred | `r_GVF` only (`reward.prediction_only=true`) | ablation: prediction-only reward; collision still terminates |
| Dabs | `r_task + Phi(s')` (`reward.shaping=absolute`) | ablation: non-potential reward, expected to invite safe-circling reward hacking |
| D/D0 fixed | as D/D0 on `configs/fixed_goal_nav.yaml` | identifiable scenario (§9) |

D vs D0 is the clean test of the research question. D vs A/B/C compares task
outcomes only. Those agents have goal features in their state and a different
(env) reward, so the comparison is confounded and returns are never compared.

## 2. Episodic termination and collision detection

* **Authoritative collision.** `world.path_hits` sweeps the robot's disc along
  the *whole* motion arc (chord polyline) and tests it against all four walls
  and every rectangular obstacle. Collisions *between* observation times are
  caught (tested at 1 m/step against a 5 mm wall). On contact the robot is held
  at its last free pose, velocity is zeroed, and `terminated=True`.
* **Additional rule.** The episode also ends when any LiDAR ray reads < 0.15 m
  (the existing task's proximity rule). `info["collision_type"]` records
  `geometric` or `lidar`. In practice the LiDAR rule fires first.
* **Precedence.** `goal_reached = (not collision) and dist < 0.20 m`. Collision
  always wins (tested with the goal 4 cm ahead and the wall 16 cm ahead). The
  reward module applies the same precedence.
* **Goal success** → `terminated=True`, recorded separately (`success`).
* **Step limit (800)** → `truncated=True`, only if not already terminated. It
  is never treated as a collision or a success.
* **No stepping after termination.** The env raises `RuntimeError` until
  `reset()`. The training loop stores the final transition in replay and writes
  the episode record *before* resetting, then resets the env and the GVF traces.
* **Limitations (stated, not hidden).** The geometric disc radius is 0.033 m,
  smaller than the real Burger body (≈0.09–0.10 m half-width). The 0.15 m LiDAR
  rule (measured from the robot centre) is larger than that half-width, so for
  geometry a ray can see, the episode ends no later than true body contact.
  Thin geometry *between* the 24 rays is caught only by the 0.033 m disc. The
  simulator reports the collision *type*, not which object was hit.

## 3. Observation contract: LiDAR only

`s_t^DQN` = the env's 24-dim normalised LiDAR vector (`Box(0,1,(24,),float32)`).
It is used for action selection, the online and target Q-networks
(`W[0].shape[0] == 24`), the replay buffer (stores exactly that vector for `s`
and `s'`), evaluation, and checkpoint reload. No goal features, GVF outputs,
TTCs, velocities, or collision labels are appended anywhere. The GVF predictor
is a separate object used only by the reward module. Tests:
`test_dqn_input_is_lidar_only`, `test_replay_never_contains_gvf_predictions`,
`test_checkpoint_and_eval_keep_same_observation_contract`.

**Partial observability.** With a random goal per episode the goal direction is
unidentifiable from LiDAR; the reward still depends on the goal (via +100 on
success). This is the existing observation design, not a bug, and it is not
patched by adding goal information.

## 4. GVF semantics (the three demons used)

All three are the existing `DirectionalTTCGVF` demons (GQ(λ), Horde paper
equations, verified in `DIAGNOSIS_GVF.md`).

| | forward | left | right |
|---|---|---|---|
| question | expected steps until the full 24-ray scan reads ≤ 0.25 m, if the robot followed the probe policy from s | same | same |
| target policy | always action 2 | always action 1 | always action 3 |
| cumulant | 1 per step | 1 | 1 |
| continuation γ(s) | 1, or 0 once the scan is ≤ 0.25 m or the episode truly terminates | same | same |
| on/off-policy | off-policy: behaviour = the DQN's ε-greedy policy, ρ = π(a\|s)/μ(a\|s) with μ = ε/5 + (1−ε)·[a = greedy] | same | same |
| range | steps ≥ 0. Learned values can be negative early, so they are clipped to 0 in Φ | same | same |
| training | concurrent, from zero (or `gvf.init_checkpoint`), α_θ = 0.1, λ = 0.4 | same | same |
| evaluation | forward-TTC MAE against independent ground truth at each eval; censored episodes excluded | — | — |

* **Calibration.** These are *not* probabilities and *not* the outcome of the
  DQN's chosen action. In this obstacle arena the forward demon was previously
  measured at about the mean-predictor baseline (`RESULTS_GVF_IMPROVED.md`).
  The safety signal is therefore weak and is treated as a heuristic potential.
* **Coverage.** A demon is updated only on steps where the chosen action equals
  its target action (ρ = 0 otherwise). As the DQN becomes greedy, a demon whose
  action the policy rarely chooses learns slowly.
* **Termination flag.** The GVF update receives `terminated`, not
  `terminated or truncated`. A time-limit cut does not end the predicted
  future. (The original Horde script passed `done` for both; this is a
  deliberate, documented difference that applies only to D.)
* **Excluded.** The velocity demons are excluded (linear speed is constant;
  future angular velocity is state-independent noise, proven earlier). There is
  **no goal-progress GVF**, so none is used and none was invented.

## 5. Reward

    r_t = r_task,t + lam * r_GVF,t                                (rl/prediction_reward.py)

    r_task: collision -50 (precedence) | goal +100 | otherwise step_reward (0)
            [+ timeout_penalty (0) on truncation]
    r_GVF : gamma * Phi(s_{t+1}) - Phi(s_t),   Phi(s_{t+1}) := 0 if terminated
    Phi(s) = mean_d tanh( clip(TTC_d(s), 0, 400) / 50 )   in [0, 1)

* **Potential-based** (Ng, Harada & Russell 1999), with γ equal to the DQN's γ.
  Over an episode the shaping telescopes to γ^T Φ(s_T) − Φ(s_0), so the agent
  cannot farm shaping reward by circling in a safe region. Staying stationary
  is impossible anyway, because every action has v = 0.15 m/s.
* **Censoring.** Never-colliding futures (unbounded TTC) saturate tanh at Φ ≈ 1,
  meaning "safe". The reward is bounded: |r_GVF| ≤ 1 + γ.
* **Non-finite predictions.** These contribute 0 to Φ and are counted
  (`reward_nonfinite`, `gvf_nonfinite` in `history.json`), not hidden.
* **Weights.** `reward.*` in `configs/dqn/experiment_d.yaml`. `aggregate` can be
  `mean|max|min|forward`.
* **Caveat.** The optimal-policy-invariance guarantee of potential shaping
  holds only for a *fixed* Φ. Here Φ changes as the GVFs learn, so it holds
  approximately.

## 6. Temporal consistency

Order per transition (`rl/experiment_d.py::process_transition`):

1. `env.step(a)`
2. Φ(s_t) and Φ(s_{t+1}) computed with the **same, pre-update** GVF weights
3. reward
4. `replay.add(s, a, r, s', terminated, truncated)`
5. GVF update
6. DQN update

Only s_t, s_{t+1}, and this step's outcome flags are used. The stored reward is
**never recomputed**, so replay holds rewards produced by older GVF weights (a
non-stationary reward, logged via `phi_mean`, `ttc_pred_mean`, and
`gvf_abs_td_mean` per episode). Tested by
`test_reward_uses_pre_update_gvf_and_is_stored_unchanged`.

## 7. DQN

This reuses `rl/dqn.py` and `rl/agent.py` (the Agent A class, with goal
features off), so D shares A/B/C's network, optimizer, and hyperparameters:

* ε-greedy exploration, linear decay 1.0 → 0.05 over 30k steps
* two-layer [64, 64] MLP, Adam (lr 1e-3), Huber loss, gradient clip 10
* target network synced every 500 updates
* TD target `y = r + γ(1 − terminated)·max_a Q_target(s', a)`: no bootstrap
  after a true termination; truncation bootstraps

Checkpoints (`agent.pkl`) restore the Q and target networks, the Adam state,
the replay buffer, RNGs, the GVF weights, and the step and episode counters. A
resumed run starts a fresh episode: the partial episode at checkpoint time is
dropped and the episode log is trimmed to the checkpoint step.

## 8. Logging

* **`episodes.jsonl`** (one record per episode): experiment id, seed, episode,
  env_step, length, return_total, return_task, gvf_reward_sum (unweighted and
  weighted), success, collision, collision_type, timeout, final_goal_distance,
  phi_mean, ttc_pred_mean[3], gvf_abs_td_mean[3] (GVF loss), dqn_loss_mean,
  epsilon, elapsed_s.
* **`history.json`** (greedy eval every `eval_interval` on fixed eval seeds):
  success/collision/timeout rates, length, task and shaped return, final goal
  distance, forward-TTC GVF MAE vs ground truth (+ mean baseline, collided and
  censored counts), max |θ|, non-finite counters, wall time.
* **Plots:** `scripts/plot_expD.py`.

## 9. Measured results (executed locally; not on Omni)

**Protocol.** 60,000 DQN environment steps per run (the A/B/C budget). Greedy
evaluations every 10k steps: 50 episodes on fixed eval seeds (10000+). Paired
training seeds across variants. The four-way comparison re-evaluates every
final checkpoint on 100 *new* episodes (seed 20000). The GVF learns from the
same interactions as the DQN, so it adds **no** environment interactions;
pretraining interactions for frozen variants are counted separately (below).
8 runs ran in parallel on a 10-core Mac, so wall times are contended:
A ≈ 24 s, D/D0 ≈ 35 s, B/C ≈ 140 s per run (B/C recompute GVF features per
minibatch).

### 9a. Main scenario (random goal each episode; 3 seeds)

Task outcomes on identical eval scenarios (`scripts/compare_agents.py`,
100 episodes per checkpoint, mean ± std over 3 seeds):

| agent | state | reward | success | collision | timeout | length |
|---|---|---|---|---|---|---|
| A | LiDAR + goal | env | 0.053 ± 0.041 | 0.927 ± 0.066 | 0.020 | 98 |
| B | LiDAR + goal + frozen GVF | env | 0.077 ± 0.082 | 0.923 ± 0.082 | 0.000 | 101 |
| C | LiDAR + goal + adaptive GVF | env | 0.170 ± 0.116 | 0.830 ± 0.116 | 0.000 | 88 |
| **D** | **LiDAR only** | terminal + GVF shaping | 0.023 ± 0.021 | 0.977 ± 0.021 | 0.000 | 69 |
| **D0** | **LiDAR only** | terminal only | 0.033 ± 0.012 | 0.927 ± 0.068 | 0.040 | 105 |
| Dfwd | LiDAR only | terminal + forward-GVF shaping | 0.033 ± 0.026 | 0.963 ± 0.031 | 0.003 | 82 |

Training (ε-greedy) success stays at about 1–3 % for D, D0, and Dfwd through
all four quarters of training. **All LiDAR-only variants are at chance, as
expected: the goal is not observable.** D shows no measurable benefit over its
control D0. A/B/C are not a fair reference for D: they observe the goal, and
3 seeds are too few to rank them.

### 9b. Fixed, LiDAR-identifiable scenario (5 seeds)

| variant | training success by quarter | last quarter | greedy success, mean over evals ≥ 20k | greedy success at final ckpt | fwd-GVF MAE (mean-baseline 31) |
|---|---|---|---|---|---|
| **D0** control | .155 .144 .179 .191 | **0.19 ± 0.19** | **0.45** | 4/5 seeds | 45 |
| **D** (Φ from 3 GVFs, concurrent) | .136 .109 .028 .054 | 0.05 ± 0.06 | 0.05 | 0/5 | 354 |
| Dfwd (Φ from forward GVF, concurrent) | .183 .089 .115 .101 | 0.10 ± 0.12 | 0.10 | 0/5 | 279 |
| Dfrozen (Φ from forward GVF, pretrained and frozen; *ablation*) | .177 .174 .140 .084 | 0.08 ± 0.07 | 0.05 | 0/5 | 56 |

* **Uncertainty.** The control learns the route; every shaped variant declines
  over training. One-sided permutation tests, D0 > variant:
  * D: p = 0.09 (last-quarter success), p = 0.06 (greedy evals)
  * Dfwd: p = 0.21, p = 0.11
  * Dfrozen: p = 0.16, p = 0.06
  * Final greedy success, 4/5 vs 0/5: Fisher one-sided p = 0.024 for each
    variant.
  * Pooling all 15 shaped runs against D0 gives p = 0.013 (pooling is post
    hoc).
* **The fixed scenario is deterministic.** All 50 greedy evaluation episodes
  replay a single trajectory, so per-checkpoint greedy success is 0 or 1 and
  flips between checkpoints. Use the training-success and "mean over evals"
  columns for judgement.
* **Loss dynamics.** The DQN loss rises over training for D and Dfwd, stays
  flat for D0, and falls for Dfrozen. This is consistent with a non-stationary
  reward for the concurrent variants.

### 9c. Why the shaping does not help: GVF quality in obstacle fields

Forward-TTC GVF trained on 1500 episodes of uniform-random behaviour in the
fixed layout (independent of the DQN), scored on held-out forward-probe
rollouts from random spawns:

| α_θ | MAE vs collision-time ground truth (baseline 34) | MAE vs the GVF's *own* question, "any ray ≤ 0.25 m" (baseline 32) | max \|θ\| |
|---|---|---|---|
| 0.005 | 46 | 39 | 9 |
| 0.02 | 54 | 47 | 12 |
| 0.05 | 60 | 53 | 15 |
| 0.1 | 71 | 63 | 17 |

The best linear fit on the same 24 LiDAR features (held-out) reaches MAE 21
against a mean baseline of 35. In the empty arena it is exact (0.0).

* **Representation is adequate.** The linear features can beat the baseline.
* **TD learning falls short.** Off-policy GQ(λ) TD learning ends up worse than
  a constant at every step size, and worse as α grows. This happens with or
  without the DQN. (The α = 0.1 recommended in `RESULTS_GVF_IMPROVED.md` was
  validated in the empty arena only.)
* **Left/right add instability.** In open space the turning probe never meets
  geometry, so the left/right demons are ill-posed and grow without bound
  (|θ| up to 2×10⁶ in the D0 runs, where they do not affect the reward).
* **Effect on D.** Φ is inaccurate in every D variant, and in the concurrent
  ones also non-stationary.

Other checks: no non-finite rewards occurred; across 22,313 logged training
episodes no episode was marked both collision and success; every collision was
ended by the 0.15 m LiDAR rule (the geometric check never fired first).

Pretraining interactions (separate from the 60k DQN budget): Dfrozen 140,862;
B/C 56,772.

## 10. Answer to the research question (what this does and does not show)

* **No evidence that GVF reward shaping helps** a LiDAR-only DQN here. In the
  main scenario every LiDAR-only variant is at chance (the goal is
  unobservable), and D ≈ D0.
* **In the identifiable scenario the shaped variants did worse than the
  unshaped control** in every comparison. The evidence is suggestive, not
  conclusive (n = 5; p ≈ 0.02–0.2 depending on the metric). It holds for
  concurrent, forward-only, and frozen GVFs.
* **Likely cause:** the safety predictions are inaccurate in obstacle fields
  (§9c), so Φ is a misleading heuristic. With function approximation and
  finite data, potential shaping acts like a biased Q-initialisation and can
  steer learning away from a goal that lies close to geometry. Concurrent
  learning adds non-stationarity on top.
* **What this does not show:** that *accurate* GVF predictions would fail as a
  reward signal, or that safety shaping cannot help in other tasks. It tests
  these particular GVFs, as they can currently be learned.

## 11. Limitations and next experiments

1. **More seeds** (≥ 10) per arm in the fixed scenario, on Omni
   (`hpc/submit_expD.sh`). Each run takes about 35 s of CPU.
2. **Fix GVF learning before re-testing shaping.**
   * Try on-policy or lower-variance off-policy corrections (e.g. tree-backup
     or Retrace-style truncation of ρ).
   * Use a well-posed left/right question (finite horizon, or obstacle-rich
     training).
   * Gate shaping on validated accuracy: only shape with a demon whose held-out
     error beats the mean baseline.
3. **Goal-progress GVF.** None exists. The research question needs one, and it
   would have to be implemented and validated explicitly.
4. **Collision footprint.** Replace the 0.033 m disc with the real Burger body
   so that the geometric check, not the proximity rule, is authoritative.
5. **Main-scenario observability.** Any LiDAR-only, random-goal comparison is
   limited to safety behaviour, whatever the reward.

## 12. Files

New: `rl/prediction_reward.py`, `rl/experiment_d.py`, `scripts/train_expD.py`,
`scripts/compare_agents.py`, `scripts/plot_expD.py`,
`configs/dqn/experiment_d.yaml`, `configs/dqn/experiment_d_fixed.yaml`,
`configs/dqn/pretrain_fixed.yaml`, `configs/fixed_goal_nav.yaml`,
`hpc/expD.sbatch`, `hpc/submit_expD.sh`, `tests/test_expD.py` (23 tests),
`DESIGN_EXPERIMENT_D.md`, `results_expD/`.

Modified (additive, A/B/C behaviour unchanged):
* `rl/gvf_predictor.py`: `update()` now also returns the pre-update TD errors.
* `rl/experiment.py`: `make_env` resolves config paths from any working
  directory.
* `README.md`: §11.

The four original agent files are unchanged (`backup_original/` hashes OK).

