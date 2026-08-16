# S&P 500 vs S&P 400 PEAD Frequency (2015–2026)

**Script:** `04_backtest/archive/edge_search_2026/66_sp500_pead_comparison.py` (data cached in `01_data/db_sp500.h5`)
**Date:** 2026-08-13
**Question:** Are PEAD events as common in the S&P 500 as the S&P 400, and by how much has large-cap PEAD eroded?

## Method

Identical gate definition to the production pipeline (`_pead_target_retrain.py`):
- g1: car_10d > +3% (log CAR vs benchmark, T+1..T+11)
- g2: (Vol_T..T+2)/3 > 2× vma20
- g3: max relative drawdown T+1..T+11 > −1.5%
- PEAD event = all three

S&P 500: 503 current constituents (Wikipedia), earnings dates from FMP
`/stable/earnings`, prices from Tiingo (2014-06+), benchmark SPY.
S&P 400: persisted `train_matrix_v4_timing_correct` (gates vs IJH, historical
membership intervals). Both sides filtered to report_date ≥ 2015-01-01.

## Results

| | events | PEAD rate | g1 | g2 | g3 |
|---|---|---|---|---|---|
| S&P 500 | 22,216 | **8.49%** | 28.2% | 30.7% | 42.5% |
| S&P 400 | 16,789 | **10.68%** | 30.3% | 36.8% | 40.4% |

Per earnings event, the S&P 400 produces **~1.26× more PEAD events**. The
main differentiator is **g2 (volume confirmation)**: 36.8% vs 30.7% — mid-cap
post-earnings volume expansion is materially stronger. g1 (CAR) is close
(30.3 vs 28.2); g3 similar.

### Erosion (early vs recent)

| | 2015–17 | 2020–22 (trough) | 2023–25 |
|---|---|---|---|
| S&P 500 PEAD rate | 9.54% | 5.2–6.8% | 9.06% |
| S&P 400 PEAD rate | 10.90% | 7.2–9.4% | 12.28% |

- S&P 500: modest net erosion 2015-17 → 2023-25 (9.5 → 9.1%), with a deep
  trough in 2020–2022 (5–7%) and partial recovery. Never catches sp400.
- S&P 400: **no net erosion** — 2023–25 is its *best* period (12.3%), and
  2025–26 runs 10.5–11.5% vs sp500's 8.0–8.6%. The gap **widened** in recent
  years.

## Caveats

1. **Survivorship (sp500 side only):** current constituents only — removed
   members (underperformers) excluded, which if anything *flatters* the
   sp500 PEAD rate. The true gap is likely wider than measured.
2. Benchmarks differ (SPY vs IJH) — each index measured against its own
   benchmark, consistent with the production definition.
3. sp400 uses historical membership intervals; sp500 uses current membership
   applied backward (no add-date gating).
4. Raw event counts per year are similar in absolute terms (sp500 ~1,900/yr
   × 8.5% ≈ 160 PEAD/yr vs sp400 ~1,500/yr × 11% ≈ 165/yr); the advantage is
   in rate and 4-slot opportunity cost, plus sp400's better model fit.

## Conclusion

**Universe choice confirmed.** S&P 500 PEAD is not gone (8.5% of events) but
it is ~20% rarer, its volume confirmation is weaker, and it showed genuine
erosion in 2020–22. The S&P 400 has the higher, *more recently improving*
PEAD rate. No action item — this is evidence supporting the current universe.
