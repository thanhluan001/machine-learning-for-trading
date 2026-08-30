# RC-4 Partial Exposure Cycle 2 — Robustness Gate

**Script:** `04_backtest/archive/megatrend_wave_2026/83_megatrend_partial_exposure_cycle2_robustness.py`
**Date:** 2026-08-17
**Status:** Informative; promotion gate not cleared.

## Purpose

Cycle 2 tests whether Cycle 1's encouraging result is stable across practical
weight constraints rather than being a single-parameter outcome.

The pre-registered grid is:

```text
Theme floors:       5%, 10%, 15%
Theme caps:        50%, 70%, 90%
Monthly steps:      5%, 10%, 15%
Rotation:           price-only, price + point-in-time capex
Costs:              0, 25, 50, 100 bps per one-way turnover
```

Fixed reporting blocks:

```text
2014-02 to 2019-12
2020-01 to 2022-12
2023-01 to 2026-07
```

A separate fixed recession overlay uses the Cycle-1 configuration:
price+capex, floor 10%, cap 70%, step 10%, absolute exposure 50% after the
Sahm-style trigger. No recession parameters were swept.

## Main results at 50 bps

The 2020–2022 block is the relevant test of the user's hypothesis: remain
invested while the market decides, but rotate gradually.

### Price-only rotation

```text
cap  step   2020-22 total   2022 return   block maxDD
50%  10%        +66.0%        -38.1%        -46.4%
50%  15%        +68.9%        -38.0%        -46.4%
70%  10%        +47.0%        -39.5%        -48.3%
70%  15%        +56.0%        -37.0%        -46.7%
90%  10%        +47.0%        -39.5%        -48.3%
90%  15%        +56.0%        -37.0%        -46.7%
```

### Price-plus-capex rotation

```text
cap  step   2020-22 total   2022 return   block maxDD
50%   5%        +65.7%        -39.0%        -45.4%
50%  10%        +67.5%        -39.0%        -45.4%
50%  15%        +67.5%        -39.0%        -45.4%
70%   5%        +55.1%        -37.8%        -44.3%
70%  10%        +60.5%        -36.8%        -43.3%
70%  15%        +61.5%        -37.2%        -43.7%
90%  10%        +59.0%        -36.9%        -43.4%
90%  15%        +60.0%        -37.3%        -43.8%
```

Floor values had little effect in the cap-50% cases because the target map
and feasible simplex already kept weights above the floor. The cap and step
are the more consequential controls in this grid.

## Findings

### 1. The partial-rotation idea is not a one-parameter accident

Across the fixed grid, the price-plus-capex variant remains less negative in
2022 than Cycle 1 equal-theme exposure (-45.2%), generally in the
approximately -37% to -39% range at 50 bps. The price-only variant is usually
approximately -37% to -40%.

This supports the narrower hypothesis:

```text
bounded theme rotation can reduce damage during an undecided regime
without forcing a full exit.
```

It does not establish acceptable absolute risk. A -37% year remains a large
loss.

### 2. More concentration is not clearly better

The 50% theme cap is more stable in the 2020–2022 block than the 70–90% caps.
Allowing a high cap does not reliably improve 2022 and exposes the portfolio to
more single-theme concentration. This is consistent with the Phase-2 lesson:
concentration needs to earn its place out of sample.

### 3. Very slow movement can be too inert

The 5% monthly step generally gives back some performance relative to 10–15%
steps. The 10–15% region is more responsive without becoming a binary switch.
The difference is not large enough to select a final step yet.

### 4. Price-plus-capex remains promising but unproven

The capex blend generally improves the 2020–2022 block's drawdown/return
tradeoff relative to price-only at the 70% cap. However, the capex natural
experiment still rules out using capex as a death-timing signal. Its only
permitted role here is a bounded, slow sponsorship prior.

### 5. Recession overlay is not cleared

At 50 bps, the fixed price+capex recession overlay produces:

```text
2020–2022 total: approximately +2.6%
2020 return:    approximately +42.6%
2022 return:    approximately -36.8%
mean exposure:  approximately 84.7%
```

It still misses too much of the rapid 2020 recovery. This is a warning that a
macro recession flag should not automatically halve exposure without a more
careful recovery protocol. It remains research-only.

## Gate status

```text
Cycle 1 premise:       supported directionally
Robustness:            encouraging but incomplete
Theme cap:             50% currently looks safer than 70–90%
Monthly step:          10–15% currently more practical than 5%
Capex role:            bounded prior only
Recession overlay:     not approved
Operational promotion: no
```

No single configuration is promoted because the strict stability bar requires
positive/acceptable behavior across every fixed block, cost assumption, and
special regime. The grid is descriptive, not a license to choose the best cell
after the fact.

## Next gate

The next research step should not expand the parameter grid. It should test a
small pre-registered shortlist:

```text
A. price-only, cap 50%, step 10%
B. price-only, cap 50%, step 15%
C. price+capex, cap 50%, step 10%
D. price+capex, cap 70%, step 10%
```

For those four only:

- walk-forward training/decision chronology;
- bootstrap confidence intervals;
- turnover/cost sensitivity through 100 bps;
- explicit 2020 recovery and 2022 decision logs;
- proxy substitution and missing-history stress;
- gradual absolute exposure ladder tested separately from theme rotation.

Until that gate clears, the manual monthly watcher remains unchanged.
