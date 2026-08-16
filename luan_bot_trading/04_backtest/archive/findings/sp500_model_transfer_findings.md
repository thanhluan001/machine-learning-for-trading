# S&P 500 Model Transfer Test: Frozen V6 (S&P 400-trained) Applied to S&P 500

**Script:** `04_backtest/archive/edge_search_2026/67_sp500_model_transfer.py` (data in `01_data/db_sp500.h5`)
**Date:** 2026-08-13
**Question:** Does the sp400-trained V6 model perform better on the S&P 500 than on its home universe?

## Method

- Features: the 23 DEPLOY_FEATURES recomputed from scratch for all 503 sp500
  constituents (FMP earnings full history + report times, FMP grades, Tiingo
  prices, SPY benchmark, GICS→SPDR sector ETFs, FRED macros) using the exact
  production formulas (Sunday-safe windows ending T-1).
- Model: frozen `phase_g_v6_gate_decomposition` gates, min-gate ≥ 0.33, XLF
  excluded — identical to live policy.
- Timing contract: BMO entry Close[T-1]/exit Close[T+5]; AMC entry Close[T]/exit
  Close[T+5] (same as live).
- Execution: 63's slot simulator — weekly top-4 slate, force-refresh mh=4,
  10% stop on force-sells, weekly 1/4-slot NAV compounding.
- Window: 2023-07-01 → 2026-06-30, plus a 2026-H1 sub-window (pure holdout for
  both sides).

## Results

### Full 3-year window (2023-07 → 2026-06)

| | trades | win% | avgW% | avgL% | maxDD% | NAV% |
|---|---|---|---|---|---|---|
| S&P 400 (home) | 264 | 61.4 | 11.6 | −8.8 | −15.5 | **+821** |
| S&P 500 (transfer) | 328 | 55.5 | 11.9 | −8.5 | −20.7 | +663 |

### 2026 H1 only (pure holdout for both)

| | trades | win% | avgW% | avgL% | maxDD% | NAV% |
|---|---|---|---|---|---|---|
| S&P 400 (home) | 46 | 71.7 | 11.5 | −10.2 | −6.2 | +78.5 |
| S&P 500 (transfer) | 55 | 61.8 | 16.6 | −8.5 | −16.0 | +135.0 |

## Caveats

1. **In-sample asymmetry favors sp400** for the 3y window (V6 dev folds cover
   sp400 2023-2025). 2026-H1 is clean for both.
2. **Survivorship favors sp500** in both windows (current members applied
   backward — 2026's survivors include exactly the names that rallied).
3. sp500 SUE/grades come straight from FMP; sp400 matrix earnings lineage is
   EODHD-era + FMP — some feature distribution shift is baked in.

## Findings

1. **The model transfers.** A sp400-trained V6 run on sp500 produces +663%
   NAV over 3 years — strongly positive. The model learned general PEAD
   structure, not sp400 noise. This is a meaningful robustness result.
2. **Home universe still wins over 3 years** — higher win rate (61.4 vs 55.5),
   higher NAV, shallower drawdown — even with sp500's survivorship tailwind.
   Directionally consistent with script 66's finding (PEAD events rarer in
   sp500: 8.5% vs 10.7%).
3. **Threshold-passer rates are similar** (sp500 891/6,482 = 13.7%; sp400
   721 ≈ 15.6%) — the model fires at comparable rates on both universes, but
   pick QUALITY is lower on sp500 (win rate gap).
4. **The 2026-H1 holdout wrinkle:** sp500 transfer actually beat home
   (+135% vs +78.5% NAV, avg win 16.6% vs 11.5%), at the cost of a much deeper
   drawdown (−16% vs −6.2%) and a much lower win rate. With n=55 trades, one
   strong half, and maximal survivorship bias (2026 survivors = 2026 winners),
   this is NOT decisive for switching — but it goes on the watch list. If it
   repeats in a future clean half, a mixed-universe variant deserves a
   separately approved research cycle.

## Decision

**Stay with the S&P 400.** No policy change. Record kept for the robustness
evidence and the 2026-H1 anomaly flag.
