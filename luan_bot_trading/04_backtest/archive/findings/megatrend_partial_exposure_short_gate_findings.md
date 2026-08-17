# RC-4 Short-List Gate — Partial Theme Exposure

**Script:** `04_backtest/84_megatrend_partial_exposure_short_gate.py`
**Date:** 2026-08-17
**Status:** Short gate completed; no operational promotion.

## Fixed shortlist

```text
A: price-only,       cap 50%, step 10%
B: price-only,       cap 50%, step 15%
C: price + capex,    cap 50%, step 10%
D: price + capex,    cap 70%, step 10%
```

All configurations use a 10% theme floor. Costs tested: 0, 50, and 100 bps
per one-way turnover. The test also reports fixed blocks and a 12-month moving-
block bootstrap with 2,000 resamples.

## Results

| Config | Cost | Total | Annualized | Max DD | 2022 | Bootstrap P(total > 0) |
|---|---:|---:|---:|---:|---:|---:|
| A | 0 bp | +1,385% | +24.1% | -46.3% | -37.9% | 98.9% |
| A | 50 bp | +1,342% | +23.8% | -46.4% | -38.1% | 98.9% |
| A | 100 bp | +1,299% | +23.5% | -46.5% | -38.2% | 98.9% |
| B | 0 bp | +1,472% | +24.7% | -46.3% | -37.9% | 99.0% |
| B | 50 bp | +1,419% | +24.3% | -46.4% | -38.0% | 98.9% |
| B | 100 bp | +1,369% | +24.0% | -46.4% | -38.2% | 98.9% |
| C | 0 bp | +1,413% | +24.3% | -45.4% | -38.9% | 99.5% |
| C | 50 bp | +1,385% | +24.1% | -45.4% | -39.0% | 99.3% |
| C | 100 bp | +1,358% | +23.9% | -45.5% | -39.0% | 99.2% |
| D | 0 bp | +1,701% | +26.0% | -43.2% | -36.7% | 99.8% |
| D | 50 bp | +1,655% | +25.8% | -43.3% | -36.8% | 99.8% |
| D | 100 bp | +1,610% | +25.5% | -43.4% | -36.9% | 99.7% |

All four configurations were positive in each fixed calendar block at 50 bps
and had block maximum drawdown better than -50%:

```text
positive all blocks:           A/B/C/D = TRUE
block maxDD > -50%:            A/B/C/D = TRUE
```

## Interpretation

### 1. The shortlist is robust to moderate implementation friction

The relative ordering and broad behavior survive 0–100 bps turnover costs. The
partial-rotation concept is not being driven by a single zero-cost assumption.

### 2. D has the strongest gross profile, but is not selected

The price-plus-capex / 70% cap configuration has the highest gross return and
lowest maximum drawdown in this sample. That is evidence to carry into the
next test, not permission to select D after observing the results. It remains
more concentrated than A–C and relies on the capex prior.

### 3. The 2022 problem is reduced, not solved

All configurations remain invested and lose approximately 37–39% in 2022.
That is better than the equal-theme Cycle-1 result (-45.2%) but still a severe
loss. The proposed approach is therefore a participation/rotation framework,
not crash protection.

### 4. Bootstrap result must not be overstated

The high bootstrap probability of positive absolute return reflects the long
2014–2026 growth sample and is not a test of superiority versus SPY, downside
utility, or future generalization. The bootstrap is supportive of positive
historical expectancy only; it is not a promotion criterion by itself.

## Gate status

```text
Short-list stability:      PASSED descriptively
Cost sensitivity:          PASSED descriptively through 100 bps
2022 whipsaw reduction:    PARTIAL — improved, still severe
Benchmark superiority:     NOT tested as a formal gate here
Forward robustness:        NOT established
Operational promotion:     NO
```

## Next and final validation requirements

Before changing the manual watcher or core allocation, the next cycle must:

- compare the four configurations against SPY and static 60/40 on return, drawdown,
and recovery capture;
- use rolling walk-forward chronology rather than only fixed full-sample
  targets;
- include proxy substitution and delayed/missing capex stress;
- report exposure and theme turnover in actual portfolio terms;
- test whether D's apparent advantage survives excluding the AI mega-trend
  period or applying a leave-one-theme-out stress;
- use block bootstrap for excess return and drawdown, not only absolute return.

Until then, the operational component remains the manual monthly panel report.
