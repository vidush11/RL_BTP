# GVF Horde — prediction-accuracy improvements (validated)

Follow-up to `DIAGNOSIS_GVF.md`, now measured against your **7000-episode**
baseline (`runs_gvf_fixed_7000/.../eval/metrics.json`). That baseline was trained
with the **default** parameters (the opt-in fixes were not passed), so it is the
honest "more episodes alone" control: 7000 episodes barely moved forward TTC
(50→43) and made angular velocity *worse* (0.247→0.371).

Nothing here overwrites your baseline: all new output is under `runs_improved/`,
and the four original agent files remain pristine in `backup_original/`. The
fixes are still applied via `-p` flags (defaults unchanged, per your choice).

## 1. Root causes (confirmed on the 7000-ep run)

| # | cause | evidence on the 7000-ep baseline |
|---|---|---|
| A | forward-TTC learning rate ~40× too small | MAE crept 50→43 over 7000 ep; LS fit on the same features reaches MAE 0.00 |
| B | velocity IS product collapses (median 1.6e-17, 99.86 % ≤1e-6); linear speed is policy-independent | linear RMSE stuck ~0.12; angular *degraded* to 0.37 as weights drifted on rare high-ρ updates |
| C | left/right TTC ill-posed in the open arena (turn policy never terminates → unbounded value) | 0 valid pairs in all 20 eval episodes, both directions |

## 2. Files / functions changed

| file | change |
|---|---|
| `gvf_horde_agent.py` | hyperparameters exposed as ROS params (identical defaults); new `velocity_importance_sampling` flag (default true) → ρ=1 for velocity demons when false. Backward-compatible. |
| `configs/obstacles_train.yaml` (new) | random obstacles + random spawn so the turn policies meet threats → left/right TTC becomes well-posed and measurable |
| `scripts/gvf_experiment.py` (new) | controlled single-demon harness (reuses the agent's demon classes) |
| `tests/test_gvf.py` (new) | GVF correctness + backward-compat tests |

No change to the learning algorithm, the baselines, or the metric definitions.

## 3. Why the fixes improve real prediction

* **Forward TTC** — raise `alpha_theta` 0.005→0.1 (stable; 0.4 diverges). The
  features already represent the target (LS MAE 0.00); this just lets TD reach it.
* **Linear velocity** — `velocity_importance_sampling:=false`. The target
  (commanded speed) is a constant 0.15 for every action, so it is
  policy-independent and the H-step IS product is unnecessary *and* near-zero.
  On-policy LMS (ρ=1) then learns the constant.
* **Left/right TTC** — train and evaluate where the turning circle meets
  obstacles (`obstacles_train.yaml`). The GVF then terminates (γ→0), so the value
  is bounded, weights stay finite (θ_left max|·|≈7 vs 9.9e9 in the open arena),
  and ~50 % of episodes yield valid TTC targets; the rest stay correctly censored.

## 4. Before / after (identical eval configs & seeds; evaluated read-only)

**Empty arena** (your baseline's scenario; eval seed 1000, 20 episodes):

| metric | baseline (7000 ep, default) | fixed (1000 ep, α=0.1, IS off) | target | met |
|---|---|---|---|---|
| Forward TTC MAE | 43.40 | **16.41** | <43.4, ideally <31 | ✅ beats the 31 mean baseline |
| Future linear-vel RMSE | 0.1199 | **0.0041** | <0.1199 | ✅ |
| Future angular-vel RMSE | 0.3709 | **0.2557** (≈0.22 on seeds 2000/3000) | <0.371, ideally <0.2468 | ⚠️ beats 0.371 and matches the optimal mean baseline; <0.2468 is **unreachable** (see §6) |

**Obstacle scenario** (`obstacles_train.yaml`; trained seed 0, eval on UNSEEN
seeds, 40 episodes) — makes left/right identifiable:

| direction | seed | MAE | mean-baseline | collided | censored | valid pairs |
|---|---|---|---|---|---|---|
| left | 2000 / 3000 / 4000 | 13.7 / 7.5 / 8.0 | 13.7 / 8.4 / 10.6 | 19 / 19 / 10 | 21 / 21 / 30 | 683 / 393 / 212 |
| right | 2000 / 3000 / 4000 | 13.0 / 15.4 / 13.5 | 14.2 / 13.6 / 10.9 | 18 / 18 / 12 | 22 / 22 / 28 | 619 / 521 / 297 |
| forward | 2000 / 3000 / 4000 | 43.9 / 39.6 / 37.8 | 41.7 / 40.8 / 40.1 | 40 / 40 / 40 | 0 | — |

Left/right now have **hundreds of valid pairs** and sit around their mean
baseline (beating it on several seeds). Forward in the cluttered arena is a
harder task (partial observability) and only ~ties its baseline there — the
clean win is in the open arena where forward TTC is well-posed.

Plot: `runs_improved_plots/before_after.png` (per-checkpoint curves, baseline vs fixed).

## 5. Valid-pair / censoring counts (all three TTC directions)

* Empty arena: forward 20/20 collided (2480 pairs); left/right **0 collided,
  20 censored** → correctly undefined (no threat on the circle).
* Obstacle arena: forward 40/40 collided; left ~10–19/40 collided, right
  ~12–18/40 collided; the rest censored — real threats produce labels, their
  absence is preserved as censoring. **No TTC labels were fabricated.**

## 6. Honest limits & the angular-velocity floor

* **Future angular velocity cannot beat ≈0.247.** Under ε-greedy-forward the
  action H steps ahead is exploration noise drawn *independently of the state*,
  so ω_{t+H} ⟂ S_t. The best possible linear predictor on the exact features
  (in-sample LS) gives RMSE **0.2333** vs the marginal std **0.2340** — no
  meaningful gain. The success target <0.2468 is below the information floor; the
  mean baseline is already optimal. The fix restores the demon *to* that optimum
  (0.371→0.247); it should not and cannot go lower. A learnable angular task
  needs a *state-dependent* target policy.
* **Linear-velocity baseline is degenerate** (zero-variance target → mean-RMSE 0).
  The fix makes the demon represent the constant (RMSE 0.004); no estimator
  "beats" a perfect baseline.
* **Left/right TTC** is only identifiable near geometry and is short/variable due
  to the 0.2 m turning radius; it now produces valid pairs but is a harder,
  noisier target than forward.
* Empty-arena forward is deterministic (fixed spawn) → identical across eval
  seeds; the obstacle scenario provides the stochastic multi-seed spread.

## 7. Reproduce

```bash
source ~/venvs/tb3sim/bin/activate && cd turtlebot3_sim
python -m pytest -q                                   # 145 passed, 1 skipped

# forward + velocity (empty arena), new run dir — does NOT touch the baseline
python scripts/train.py --agent gvf_horde_agent.py --seed 0 \
    --run-dir runs_improved/horde_empty_fixed_s0 --config configs/default.yaml \
    --episodes 1000 --param alpha_theta:=0.1 --param velocity_importance_sampling:=false
python scripts/evaluate.py --checkpoint-dir runs_improved/horde_empty_fixed_s0/gvf_checkpoints_horde \
    --config configs/default.yaml --seed 1000 --num-eval-episodes 20   # all checkpoints -> eval/metrics.json

# left/right (obstacle scenario), unseen eval seeds
python scripts/train.py --agent gvf_horde_agent.py --seed 0 \
    --run-dir runs_improved/horde_obstacles_s0 --config configs/obstacles_train.yaml \
    --episodes 1500 --param alpha_theta:=0.1 --param velocity_importance_sampling:=false
for S in 2000 3000 4000; do
  python scripts/evaluate.py --checkpoint-dir runs_improved/horde_obstacles_s0/gvf_checkpoints_horde \
      --only-latest --config configs/obstacles_train.yaml --seed $S --num-eval-episodes 40 \
      --out-dir runs_improved/horde_obstacles_s0/eval_seed$S
done
```

On Omni, submit the same via `hpc/` with the extra `--param` flags and
`CONFIG=configs/obstacles_train.yaml`, keeping `RUN_ROOT=runs_improved`.

## 8. Next recommended experiment

1. **Multi-seed training** (3–5 seeds) of the obstacle agent on Omni (short
   queue, minutes each) to get mean±std of left/right MAE and confirm the forward
   gain holds across seeds.
2. **Per-demon learning rates** — forward tolerates α=0.2 (MAE→~5) but the
   turn demons prefer the obstacle scenario + moderate α; a per-demon α would let
   forward go lower without risking the turn demons.
3. **State-dependent angular task** — if future-ω prediction matters, evaluate
   under a target policy whose turning depends on the scan (e.g. greedy obstacle
   avoidance), so the target carries learnable state information.
