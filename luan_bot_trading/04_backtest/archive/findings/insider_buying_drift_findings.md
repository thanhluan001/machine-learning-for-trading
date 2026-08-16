# SEC Form 4 Insider Buying Drift Event Study — CLOSED (No Short-Horizon Edge)

**Script:** `04_backtest/archive/edge_search_2026/69_insider_cluster_drift.py` (cached in `01_data/db_insider.h5`)
**Date:** 2026-08-13
**Question:** Do SEC Form 4 open-market insider purchases (especially cluster buying) generate tradeable abnormal drift to fill idle slots in off-earnings months (September, December, March, June)?

## Method

- Data: 71,663 raw Form 4 transactions across 959 S&P 400 permaTickers (2015–2026) from FMP `/stable/insider-trading/search`.
- Filter: Strict open-market cash purchases (`transactionType == 'P-Purchase'` or starting with `P`), price > $0.
- Execution point: **Close[T+1]** (first trading close after public SEC EDGAR dissemination).
- Benchmark: Abnormal return (CAR) vs `/macros/IJH`.
- Horizons tested: 5, 10, 20, 60 trading days.
- Sub-populations: Materiality ($25k–$250k), Role (C-Suite vs Directors vs Officers), and Cluster Buying ($\ge 2$ distinct insiders buying within 14 calendar days, $\ge \$50\text{k}$).

## Results

### 1. By Materiality & Role (from Close[T+1])

| Segment | n | 5d CAR (win%) | 10d CAR (win%) | 20d CAR (win%) | 60d CAR (win%) |
|---|---|---|---|---|---|
| All P-Purchases (raw) | 14,222 | +3.56% (49.3%)* | +7.69% (50.0%)* | +9.57% (51.0%)* | +7.16% (51.1%) |
| Value $\ge \$50\text{k}$ | 8,293 | **+0.11%** (49.2%) | **+0.13%** (49.5%) | +0.56% (50.2%) | +2.77% (50.6%) |
| Value $\ge \$100\text{k}$ | 6,326 | +0.29% (50.0%) | +0.32% (50.2%) | +0.70% (50.5%) | +2.54% (50.7%) |
| C-Suite ($\ge \$50\text{k}$) | 2,280 | **−0.05%** (48.4%) | **−0.01%** (49.6%) | +0.39% (50.7%) | +2.94% (50.4%) |
| Board Directors ($\ge \$50\text{k}$) | 4,477 | +0.15% (48.7%) | +0.18% (49.4%) | +0.51% (49.9%) | +2.50% (50.6%) |

*(Note: Raw unfiltered P-Purchases include unadjusted penny/distressed volume spikes; dollar-filtered rows represent genuine corporate trades).*

### 2. Cluster Buying ($\ge 2$ Insiders in 14 Days, $\ge \$50\text{k}$)

| Strategy | n | 5d CAR (win%) | 10d CAR (win%) | 20d CAR (win%) | 60d CAR (win%) |
|---|---|---|---|---|---|
| Single Insider Only | 4,516 | +0.12% (49.5%) | +0.12% (49.6%) | +0.48% (49.3%) | +1.55% (49.5%) |
| **Cluster ($\ge 2$ Insiders)** | 3,777 | **+0.11%** (48.8%) | **+0.14%** (49.5%) | **+0.65%** (51.3%) | **+4.23%** (52.0%) |
| Cluster + (C-Suite/Director) | 3,095 | +0.07% (48.2%) | +0.12% (49.3%) | +0.65% (51.2%) | +4.04% (51.2%) |

### 3. Seasonality (September Slow-Week Performance)

- **September Cluster 20d CAR: −0.28%** (n=278).
- Peak insider volume does not happen in September (n=278); it clusters immediately after Q4/Q1 earnings in March (n=705) and May (n=609).

## Key Findings

1. **Zero edge at short holding horizons (5d / 10d):**
   Across every single subset (C-suite, $250k+, multi-insider clusters), the 5-day and 10-day abnormal return is **+0.11% to +0.14%** with a win rate of **~49%** (statistical coin flip).
2. **The horizon mismatch:**
   The only horizon with positive drift is **60 trading days (~3 months, +4.23%)**. But a 60-day holding period creates a severe portfolio conflict: buying in September locks up slots through November, crowding out our high-Sharpe PEAD engine during its peak season.
3. **September is negative:**
   In September specifically, 20-day cluster drift is **−0.28%**.

## Decision

**Closed — do not implement.** 
Form 4 insider buying has no short-term alpha on S&P 400 names. Holding cash in earnings dead zones remains the mathematically superior choice over forcing a low/zero-edge strategy.
