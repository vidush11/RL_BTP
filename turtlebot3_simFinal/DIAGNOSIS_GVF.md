# GVF / Horde diagnosis and fixes

Scope: diagnose why the five Horde demons predict poorly on the headless
TurtleBot3 sim, then apply the smallest set of evidence-backed fixes. The
GQ(λ) update in `gvf_horde_agent.py` was checked line-by-line against the Horde
paper (Sutton et al. 2011, `horde1.pdf`) and **the learning rule is correct** —
δ, the θ/w updates, and the trace `e_t = φ(S_t,A_t) + γ(S_t)λ ρ_t e_{t-1}` with
`ρ_t = π/b` all match. The failures are in step size, off-policy variance, and
problem well-posedness, not in the algorithm.

All numbers below are reproduced locally (CPU, no cluster) from the provided
`gvf_debug_bundle` checkpoints and from controlled single-variable experiments
(`scripts/gvf_experiment.py`, which reuses the agent's own demon classes).

## 1. Baseline (reproduced)

Checkpoint `gvf_weights_ep_001000.npz`, eval seed 42, 20 episodes, `default.yaml`:

| demon | metric | learned | baseline | notes |
|---|---|---|---|---|
| TTC forward | MAE | 49.69 | 31.0 (mean) | worse than predicting the mean |
| TTC left | MAE | n/a | n/a | 20/20 episodes censored (never collide) |
| TTC right | MAE | n/a | n/a | 20/20 episodes censored |
| future linear vel | RMSE | 0.1497 | 0.0 (mean) | target is constant 0.15 → baseline is perfect |
| future angular vel | RMSE | 0.2517 | 0.2517 (mean) | no improvement over the mean |

Weight magnitudes at ep 1000: `θ_forward` max|·|=5.46 (ℓ2=26); velocity weights
max|·|≈2e-4 (essentially zero). Importance-weight product median 1.6e-17,
99.86 % ≤ 1e-6.

## 2. Root causes (proven)

**(A) Forward TTC — step size ~40× too small. [proven]**
The features represent forward steps-to-collision almost perfectly: a
least-squares fit on the exact `φ(s, forward)` features reaches **MAE 0.00**
(max|θ|≈34). So this is optimization, not representation. Raising `alpha_theta`
(controlled, single-variable, 300 episodes — *fewer* than the 1000-ep baseline):

| alpha_theta | forward MAE | max|θ| |
|---|---|---|
| 0.005 (original) | 50.6 | 3.2 |
| 0.05 | 27.4 | 25.4 |
| 0.1 | 19.9 | 37.4 |
| 0.2 | 9.2 | 51.9 |
| 0.4 | diverges (inf) | — |

Stable and strongly improving up to ~0.2; diverges at 0.4.

**(B) Future velocity — the H-step importance-sampling product collapses, and
for linear velocity it is unnecessary. [proven]**
The velocity demons multiply ~19 per-step ratios. Under uniform behaviour
(b=0.2) and an ε-greedy-forward target, that product is ≈0 almost always
(median 1.6e-17), so `w += α·ρ_product·δ·φ` never moves → weights stay at 0 →
prediction 0. Crucially, **commanded linear speed is a constant 0.15 m/s for
every action** (verified: std = 0.0), so future linear velocity is
*policy-independent* and the off-policy correction is not just high-variance but
unnecessary. Setting ρ=1 for the velocity demons (on-policy, valid because the
target is policy-independent):

| velocity IS | linear-vel RMSE | max|w| | mean prediction (target 0.15) |
|---|---|---|---|
| on (original) | 0.150 | 0.0000 | 0.000 |
| off (ρ=1) | 0.0034 | 0.014 | 0.1525 |

**(C) Left/right TTC — ill-posed in the open arena. [proven]**
The deterministic hard-turn target policies drive a ~0.2 m-radius circle that
never reaches a wall, so γ (=0 only when the full scan sees ≤0.25 m) never
terminates. The forward-left/right TTC return (Σ of r=1 with γ=1 forever) has
**no finite fixed point**: every evaluation episode is censored, and the weights
grow without bound (θ_left ℓ2 reached 9.9e9 at alpha_theta=0.2). This is why
left/right are both un-evaluable *and* unstable — one cause, not two.

**(D) Evaluation target offset. [minor, proven]**
The GVF's γ terminates at full-scan ≤0.25 m, but the evaluator (and the agent's
own per-episode metric) scores against steps-to-termination (0.15 m / geometric).
Mean gap ≈ 6.8 steps — a systematic offset, small next to (A).

## 3. Changes applied (minimal, backward-compatible)

| # | file | change | why |
|---|---|---|---|
| 1 | `gvf_horde_agent.py` (hyperparam block ~ln 269; `rho_vel` ~ln 562) | hyperparameters (`alpha_theta`, `alpha_w`, `trace_lambda`, `obstacle_threshold`, `alpha_w_horizon`, `rho_product_clip`, `epsilon_target_velocity`) are now ROS params with **identical defaults**; new `velocity_importance_sampling` (default **true**) switches ρ=1 for the velocity demons | lets fixes (A),(B) be applied/tuned via `-p name:=value` without changing default behaviour |
| 2 | `scripts/gvf_experiment.py` (new) | controlled single-demon harness reusing the agent's demon classes | evidence for (A),(B) without touching the agent |
| 3 | `configs/eval_ttc_near_wall.yaml` (new) | spawn beside a wall so turn policies collide | demonstration remedy for (C) |
| 4 | `tests/test_gvf.py` (new) | GQ(λ) update vs hand calc, ρ/policy probs, fixed-horizon learning, terminal/truncated, TTC censoring counts, checkpoint round-trip, deterministic eval, backward-compat smoke | lock in the fixes |

The four original files are preserved byte-identical in `backup_original/`; the
three not touched by this work remain byte-identical in place. Checkpoint format
and the `.npz` keys are unchanged, so old checkpoints still load.

**Not done deliberately:** the learning algorithm was not replaced, baselines
were not redefined, and no TTC labels were fabricated for censored episodes.

## 4. Before / after (identical eval: seed 42, 20 episodes, `default.yaml`)

| demon | metric | original (ep 1000) | fixed (ep **250**) | command |
|---|---|---|---|---|
| TTC forward | MAE | 49.69 | **5.06** | `-p alpha_theta:=0.2` |
| future linear vel | RMSE | 0.1497 | **0.006** | `-p velocity_importance_sampling:=false` |
| future angular vel | RMSE | 0.2517 | ~0.25 (unchanged; see limits) | — |
| TTC left/right | MAE | censored | censored in open arena; measurable (~4-step TTC) with `eval_ttc_near_wall.yaml` | — |

The fixed run used **250** episodes vs the baseline's 1000, so the gain is from
the fix, not more training.

## 5. Reproduce

```bash
source ~/venvs/tb3sim/bin/activate       # or your venv
cd turtlebot3_sim

# tests
python -m pytest -q                                   # 145 passed, 1 skipped

# controlled evidence (fast, no cluster)
python scripts/gvf_experiment.py forward_ttc --episodes 300 --alpha-theta 0.005   # ~MAE 50
python scripts/gvf_experiment.py forward_ttc --episodes 300 --alpha-theta 0.2     # ~MAE 9
python scripts/gvf_experiment.py linear_velocity --episodes 300                   # RMSE 0.15
python scripts/gvf_experiment.py linear_velocity --episodes 300 --no-importance-sampling  # RMSE ~0.003

# train the fixed agent (new run dir; never overwrites the baseline) + evaluate
python scripts/train.py --agent gvf_horde_agent.py --seed 0 --run-dir runs/horde_fixed_s0 \
    --episodes 1000 --param alpha_theta:=0.1 --param velocity_importance_sampling:=false
python scripts/evaluate.py --checkpoint-dir runs/horde_fixed_s0/gvf_checkpoints_horde \
    --only-latest --seed 42 --num-eval-episodes 20
```

On Omni, add the same `--param` flags to `EPISODES=... hpc/submit_training.sh`
via the agent args, or submit with a config that sets them.

## 6. Remaining limitations and next experiments

1. **Angular-velocity demon is near-unlearnable here.** Under ε-greedy-forward
   the true future ω is 0 ≈ 96 % of the time with rare ±0.75/±1.5 spikes; the
   mean (~0) is already near-optimal (MAE 0.05). Turning off IS lets it learn
   the conditional mean but cannot beat the irreducible noise. Next: evaluate
   under a target policy that actually commands turns, or predict |ω|.
2. **Linear-velocity metric is degenerate** (zero-variance target → mean
   baseline RMSE 0). The fix makes the demon *represent* the constant
   (RMSE→0.003); no estimator can "beat" a perfect baseline. A meaningful
   velocity task needs action-dependent linear speed.
3. **Left/right TTC** is only identifiable where the turning circle meets
   geometry. Principled fix (not yet applied): add a finite max-horizon γ cutoff
   so the hard-turn TTC value is bounded, then retrain; or train/eval in a
   cluttered arena. Per-demon learning rates would also help (forward tolerates
   higher α than the unstable turn demons).
4. **alpha_theta sweet spot** shown for forward (0.1–0.2 stable, 0.4 diverges);
   multi-seed confirmation and a short 1000-ep × 3-seed cluster run are the next
   step, which the `hpc/` workflow already supports.
5. **Evaluation offset (D)** (~6.8 steps) is not corrected in code; it is small
   relative to the forward MAE and documented here.
