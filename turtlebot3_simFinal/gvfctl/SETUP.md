## Setup

**Task.** LiDAR-only obstacle avoidance on the headless TurtleBot3 simulator.
Observation: 24 LiDAR ranges. An episode ends with a collision (`terminated`;
LiDAR < 0.15 m or swept-disc contact) or after 800 steps (`truncated`,
bootstrapped by every learner). There is no goal: "success" means surviving
800 steps without a collision.

**Action spaces.**
* Part 1: the simulator's 5 actions at v = 0.15 m/s, ω ∈ {−1.5, −0.75, 0, 0.75, 1.5} rad/s.
* Part 2: `configs/speed_actions.yaml`, 3 speeds {0.05, 0.12, 0.22} m/s × the same
  5 turn rates (index = speed·5 + turn).
* Episode-start "previous action": straight at the middle speed.

**Environments** (no simulator edits; layouts are config files):

| env | layout |
|---|---|
| `train` | `obstacles_train.yaml`: 10 random boxes, random spawn |
| `dense` | 18 boxes |
| `empty` | random spawn, walls only |
| `corridor` | zig-zag corridors, ~0.9 m wide |
| `clutter` | 28 small boxes |
| `noise001` / `noise003` | `train` with LiDAR noise σ = 0.01 / 0.03 m |
| `gazebo` | `gazebo_compat.yaml` exactly (Part 1 only) |
| `gazebo_train` | training layout with the Gazebo action table (Part 1 only) |

**Seeds** (disjoint ranges, asserted in code):

| use | seeds |
|---|---|
| training, main | 0–9 |
| training, ablations | 0–4 |
| training, 1M-step DQN | 0–2 |
| training, tuning runs | 100–101 |
| test episodes | 1,000,000+ (final evaluation only; identical for every method, so paired) |
| validation episodes | 2,000,000+ (all tuning and threshold selection) |
| learning-curve episodes | 3,000,000+ |
| prediction-quality states | 4,000,000+ |

**Methods.**

* **GVF controller** (`gvfctl/gvf.py`, `gvfctl/controllers.py`)
  * Network: MLP 48→128→128→heads.
  * Safety heads: sigmoid outputs, BCE loss, γ ∈ {0.90, 0.97}, persistent
    target policy (stay 0.9, otherwise move to a neighbouring turn rate or
    speed). The update has no importance sampling and no max.
  * Cumulant: c = 1 if min range > 0.25 m (Part 1), or > 0.2 m + v·τ
    (Part 2, τ chosen on validation).
  * Velocity heads (Part 2, same trunk): normalised v, |ω| and signed ω, with
    H = 20. Expected-SARSA bootstrap under the ε-softened controller
    (ε = 0.05), evaluated with target-network predictions.
  * Training: Adam (lr 1e-3), replay 100k, batch 64, Polyak 0.005, mirror
    augmentation and symmetrised prediction.
  * Behaviour: ~30-step segments that are either a smooth random walk or the
    controller; ε decays 1.0 → 0.1 over the first 30% of steps.
  * Steering and speed rules as specified. The speed rule follows the paper's
    Alg. 2, with Alg. 3's hysteresis, a turn-rate cap and a limit of one speed
    level per step.
  * "48 rich features" = [min(d, 3.5)/3.5, exp(−d/0.3)] per beam. The brief
    referred to an existing `gvfctl/` that does not exist in this repository,
    so this is a fresh implementation of the brief.
* **DQN A**: raw 24 → 128 → 128 → n. Huber loss, hard target network, ε
  1 → 0.05 over 40% of steps. Done-mask on collision only.
* **DQN B**: A + double-Q + mirror augmentation + the 48 rich features.
* **DQN rewards:**
  * Part 1: +1 per step.
  * Part 2: v/v_max, optionally − λ|ω|/ω_max; λ ∈ {0, 0.1} is part of the
    tuning grid.
* **Hand rule (C)**: turn away from the closer side when the front sector
  (±30°) is closer than d_turn, and turn hard below d_hard. Part 2 also sets
  speed from front clearance. Thresholds are tuned on validation. It is
  deterministic, so n = 1.
* **Original demons (D)**: the unmodified `DirectionalTTCGVF` /
  `FixedHorizonGVF` classes from `gvf_horde_agent.py`, trained with uniform
  random behaviour as in the original script.
  * Variants: `default` (α_θ 0.005, H-step IS product) and `fixed` (α_θ 0.1,
    velocity IS off).
  * Their TTC is converted to safety as Q = 1 − γ^TTC, then fed to the same
    controller with its own validation-tuned thresholds.
  * They provide only 3 directional demons and no speed-conditioned safety.

**Shared protocol.** Every method gets the same:
* env, 300k-step budget, update-to-data ratio (1 update/step, batch 64), and
  replay size;
* learning-curve evaluation: greedy, 40 curve-seed episodes every 20k steps,
  plus at 50k and 100k;
* final evaluation: 200 test episodes per env, cap 800;
* run-until-collision evaluation: 100 test episodes per env, 20,000-step
  safety cap (censored).

## Fairness notes and deviations (all deliberate, all documented)

1. **DQN gets more tuning than the GVF learner.**
   * Grid: γ ∈ {0.95, 0.99} × lr ∈ {3e-4, 1e-3} × target update ∈ {500, 2000}
     × reward scale ∈ {1, 0.1} (× turn penalty ∈ {0, 0.1} in Part 2).
   * Each family (A, B) is tuned separately on validation seeds. Compute
     limits the tuning runs to 100k steps and 2 tuning seeds.
   * The GVF learning hyperparameters were not tuned; only controller
     thresholds were, on validation. This favours DQN.
2. **Threshold tuning is per trained agent, on validation seeds only:**
   * Part 1: 42 steering configs.
   * Part 2: 42 steering configs, then 16 speed configs.
   * Ties go to the documented defaults.
   * Learning curves use the default thresholds, because tuned thresholds
     exist only after training.
3. **Velocity-GVF target policy.** It is the controller with the
   training-time (default) thresholds. The deployed controller uses the
   validation-tuned ones, so v̂ predicts speed under a slightly different
   controller. The Monte-Carlo ground truth uses the training-time controller,
   i.e. the GVF's own question.
4. **Decision-time ablations.** The speed-controller and velocity-GVF
   ablations (fixed / safety-only / lin / ang / both) reuse the same trained
   networks. Ablations that need training (H, separate network, persistent
   velocity target, Part 1 variants) are separate 5-seed runs.
5. **Pareto sweep.** For the GVF, v_target is swept at decision time. For
   DQN, a decision-time speed bias Q + b·span(Q)·speed_level is swept. DQN was
   not retrained per point.
6. **Gazebo.** `gazebo_compat.yaml` has an empty arena with a fixed spawn, so
   every episode there is one trajectory. `gazebo_train` is added as the
   informative transfer test. Agents act in physical (v, ω) and are remapped by
   ω. Part 2 is excluded because the Gazebo table has 5 actions.
7. **Determinism.** The determinism check (evaluate twice and assert) runs on
   10 test episodes per agent, and a unit test covers it. A full double
   evaluation of every agent would double the evaluation compute.
8. **TTC readout.** Reading TTC from a discounted safety value is
   ill-conditioned far from obstacles: Q ≈ 1, and the readout saturates
   beyond ~1/(1−γ) steps. The last-40-steps error is the meaningful number.
9. **Statistics.**
   * Unit = training seed (n = 10 main, 5 ablations, 3 for 1M steps).
   * 95% CIs use Student-t; the paired bootstrap uses 10,000 resamples.
   * Mann-Whitney uses the normal approximation. Welch's t is also reported.
   * The hand rule is compared with a one-sample t-test.
10. **Provenance.** This is not a git repository, so each job records the
    sha256 of `gvfctl/*.py` instead of a git sha.
11. **Hardware.** All runs used one 10-core Apple M4 laptop, one process per
    job, up to 10 in parallel. Wall-clock numbers are contended.
