# Gate Decomposition v6 Findings

**Status:** research candidate; not production.
**Date:** 2026-08-07
**Production baseline:** `phase_g_v4_timing_correct`

## Architecture

Three independent XGBoost classifiers predict the persisted v4 gate labels:

- Gate 1: `CAR > +3%` (`pass_g1`)
- Gate 2: event volume ratio `> 2x` baseline (`pass_g2`)
- Gate 3: market-adjusted MaxDD `> -1.5%` (`pass_g3`)

The persisted identity was verified:

```text
pead_pass == pass_g1 AND pass_g2 AND pass_g3
```

The v4 matrix contains 16,789 rows and 23 point-in-time features. Gate base
rates are 30.25%, 36.83%, and 40.43%; combined PEAD rate is 10.68%.

## Validation protocol

`54_gate_decomposition_v6_nested.py` uses four outer walk-forward folds. For each
outer fold:

1. Each gate's HP is selected on the preceding sweep window only.
2. The ensemble rule and threshold are selected on that same sweep window only.
3. Each selected gate model is refit on train + sweep.
4. The untouched outer test window is evaluated once.

The HP grid is the same 60-combination grid used in the v4/v5 comparison:

- gamma: 1, 3, 5, 8, 12
- min_child_weight: 20, 50, 100, 200
- max_depth: 2, 3, 4

Candidate ensemble rules were product, minimum, and hard conjunction, with
thresholds from 0.02 through 0.50.

## Nested OOS result

| Metric | v4 single classifier | Nested gate decomposition |
|---|---:|---:|
| Executed trades | 99 | 168 |
| Win rate | 57.6% | 62.5% |
| Average trade | +2.78% | +2.37% |
| Compounded NAV | +89.7% | +149.2% |
| Minimum fold NAV | +1.94% | +6.87% |
| Executed PEAD precision | 24.3% | 26.2% |

Raw model-pick precision across the outer test rows was approximately 16.2%
for the decomposition versus approximately 24.3% for v4 at its fixed 0.20
threshold. This difference reflects that the decomposition's selected rules
favor a much larger candidate set; executable slot selection produces a higher
precision subset.

Outer-fold selected rules:

- Fold 1: minimum probability, threshold 0.30
- Fold 2: minimum probability, threshold 0.25
- Fold 3: product probability, threshold 0.05
- Fold 4: product probability, threshold 0.05

Outer-fold nested raw candidate counts and precision:

- Fold 1: 155 candidates, 23.9% precision
- Fold 2: 443 candidates, 11.5% precision
- Fold 3: 311 candidates, 19.6% precision
- Fold 4: 344 candidates, 15.7% precision

## Interpretation

This is materially more promising than grades-historical v5. The decomposition
increases executed-trade NAV and win rate under the nested protocol, while also
slightly improving executed PEAD precision. However:

- It uses more candidate events and a different ensemble threshold in each fold.
- The raw candidate precision is not better than v4.
- Average trade return is lower than v4.
- The result is based on four outer folds and requires bootstrap uncertainty
  analysis.
- Gate labels inherit the current v4 verification definitions and should not be
  changed during comparison.

Therefore the decomposition is **not promoted** yet. The next validation step is
trade-level and weekly-NAV bootstrap confidence intervals, plus fold-level
comparison against v4 using the exact selected rules. No production model or
live inference path was changed.

## Bootstrap validation

The first v6 validation run used 10,000 resamples with seed `20260807` and
reconstructed the v4 trade stream directly from the persisted v4 matrix and
fixed v4 HP (`gamma=3`, `min_child_weight=100`, `max_depth=2`).

| Metric | v6 | v4 |
|---|---:|---:|
| Trades | 168 | 99 |
| Win rate | 62.5% | 57.6% |
| Average trade | +2.37% | +2.78% |
| Weekly-NAV point estimate | +149.2% | +89.7% |

Trade-level bootstrap 95% intervals:

- v6 average trade: **+0.61% to +4.13%**
- v4 average trade: **+0.51% to +5.19%**
- v6 win rate: **55.4% to 69.6%**
- v4 win rate: **47.5% to 67.7%**
- v6 minus v4 average trade: **-3.36 to +2.42 percentage points**
- Probability bootstrap v6 average trade exceeds v4: **38.8%**

Weekly-NAV bootstrap intervals were wide:

- v6: point estimate +149.2%; 95% CI **+14.6% to +437.6%**
- v4: point estimate +89.7%; 95% CI **+9.7% to +241.9%**

Fold 2 is the main weakness: v6 produced 48 trades at +0.73% average return
and +6.9% fold NAV versus v4's 30 trades at +2.95% and +23.6% NAV. Folds 1,
3, and 4 were positive for v6.

The bootstrap supports positive expectancy for v6, but does **not** establish
that v6 has higher expectancy than v4. The difference interval includes zero,
and the v6 average-trade point estimate is lower. The higher NAV is therefore
not sufficient for promotion by itself. Further validation should focus on
trade-overlap dependence, fold-level robustness, and whether the v6 selection
rule can be simplified without relying on fold-specific rule changes.

## Stability validation

The next validation used the fold-legitimate gate HPs selected inside each
outer fold, then compared a fixed family of ensemble policies on the untouched
outer test rows:

- product >= 0.05
- minimum >= 0.25
- minimum >= 0.30
- hard conjunction >= 0.25

The adaptive nested policy remained at 168 trades, +2.37% average trade, and
+149.2% NAV. The strongest fixed policy in this pre-registered family was
`minimum(p1,p2,p3) >= 0.30`:

| Metric | Fixed minimum >= 0.30 |
|---|---:|
| Trades | 158 |
| Win rate | 63.3% |
| Average trade | +3.82% |
| Compounded NAV | +314.7% |
| Minimum fold NAV | +24.7% |

Trade bootstrap results for the fixed policy:

- average trade 95% CI: **+1.90% to +5.81%**
- win rate 95% CI: **55.7% to 70.9%**
- weekly NAV IID bootstrap 95% CI: **+100.7% to +811.2%**
- weekly NAV four-week block bootstrap 95% CI: **+93.1% to +857.8%**

Compared with v4 on common calendar weeks, the fixed policy's weekly-return
difference was +1.00 percentage points:

- IID paired-week 95% CI: **+0.20 to +1.85 pp**
- four-week block paired 95% CI: **+0.18 to +1.91 pp**
- bootstrap probability fixed v6 exceeds v4: approximately **99.3% IID** and
  **99.3% block bootstrap**

There are 50 common event entries between v4 and adaptive v6; their returns
are identical because the same event/entry/exit is being compared. The fixed
policy comparison adds events rather than claiming a different fill price for
common trades.

Caution: although the fixed policy family was defined in the validation
script, selecting the best member after inspecting this result creates a
multiple-policy selection consideration. The `minimum >= 0.30` result is
therefore **strong research evidence, not yet a promotion-grade final claim**.
It should be confirmed on a future untouched rolling holdout or with a
pre-committed deployment threshold before replacing v4.

## Frozen-policy final holdout

The fixed policy was frozen before evaluating the final historical holdout:

```text
score = min(p1, p2, p3)
threshold = 0.30
```

Gate HPs were the fold-4 HPs selected using only data through 2025-12-31's
preceding sweep window. Models were refit through 2025-12-31 and evaluated
once on the untouched 2026 H1 window.

| Metric | Frozen V6 | V4 |
|---|---:|---:|
| Executed trades | 47 | 27 |
| Win rate | 63.8% | 59.3% |
| Average trade | +3.88% | +1.85% |
| Compounded NAV | +53.4% | +12.1% |
| Raw candidate precision | 14.4% | 24.7% |

Frozen V6 2026 H1 trade bootstrap:

- average trade 95% CI: **+0.19% to +7.93%**
- win rate 95% CI: **51.1% to 76.6%**
- four-week block NAV 95% CI: **-5.9% to +175.1%**

V4 four-week block NAV 95% CI was **-12.4% to +47.7%**. The holdout is
encouraging: V6 outperformed v4 on all executable portfolio statistics in the
single final window, and its average-trade CI remained above zero. However,
the block-NAV interval is wide and the holdout contains only 47 executed
trades. The lower raw candidate precision shows that the edge comes from
ranking/portfolio selection rather than a higher hit rate across all
candidates.

This is sufficient to justify shadow paper trading, not live promotion. V6
should generate hypothetical orders alongside v4 while v4 continues to submit
paper orders. The final holdout is now consumed and must not be used for any
further threshold or HP tuning.

## Artifacts

Research scripts:

- `04_backtest/53_gate_decomposition_v6.py` — fixed-HP and descriptive sweep
- `04_backtest/54_gate_decomposition_v6_nested.py` — valid nested walk-forward
- `04_backtest/55_validate_gate_decomposition_v6.py` — trade/week bootstrap
- `04_backtest/56_validate_gate_decomposition_v6_stability.py` — fixed-policy and dependence tests
- `04_backtest/57_validate_v6_final_holdout.py` — frozen 2026 H1 final holdout

Results:

- `04_backtest/archive/experiments/gate_decomposition_v6/results.json`
- `04_backtest/archive/experiments/gate_decomposition_v6/nested_results.json`
- `04_backtest/archive/experiments/gate_decomposition_v6/validation.json`
- `04_backtest/archive/experiments/gate_decomposition_v6/stability_validation.json`
- `04_backtest/archive/experiments/gate_decomposition_v6/final_holdout.json`
- `04_backtest/archive/experiments/gate_decomposition_v6/policy.json`
