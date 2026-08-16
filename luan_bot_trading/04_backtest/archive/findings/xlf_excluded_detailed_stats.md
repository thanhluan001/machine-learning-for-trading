# XLF-Excluded Binary Model — Detailed Trade Statistics

> **Source:** `04_backtest/42_xlf_excluded_detailed_stats.py`
> **Operating point:** Binary theta=0.20, pre-gap entry, 5-day hold, -10% delayed stop, exclude XLF
> **Validation:** 4-fold nested CV (anchored walk-forward, 2024-07 → 2026-07)

---

## 1. Headline Statistics

| Metric | Value |
|---|---:|
| Total executed trades | 101 |
| True PEAD events (label=1) | 39 |
| False positives (label=0) | 62 |
| PEAD precision | 38.6% |
| **Win rate (return > 0)** | **75.2%** |
| Loss rate (return <= 0) | 24.8% |
| Number of wins / losses | 76 / 25 |
| **Expectancy per trade** | **+6.72%** |
| Median return per trade | +5.99% |
| Std dev per trade | 13.03% |
| Min return | -30.54% |
| Max return | +51.82% |
| Skewness | +0.38 |
| Kurtosis (excess) | 1.63 |

---

## 2. Win / Loss Breakdown

| Metric | Value |
|---|---:|
| Avg win | +11.66% |
| Avg loss | -8.28% |
| Median win / loss | +8.01% / -7.30% |
| Max win | +51.82% |
| Max loss (worst) | -30.54% |
| **Payoff ratio** (avg win / \|avg loss\|) | **1.41** |
| **Profit factor** (sum wins / \|sum losses\|) | **4.28** |

### Win percentiles
| Percentile | Return |
|---|---:|
| p10 | +2.06% |
| p25 | +4.55% |
| p50 (median) | +8.01% |
| p75 | +15.80% |
| p90 | +25.23% |

### Loss percentiles
| Percentile | Return |
|---|---:|
| p10 | -15.14% |
| p25 | -10.61% |
| p50 (median) | -7.30% |
| p75 | -2.37% |
| p90 | -0.48% |

---

## 3. Return Distribution Buckets

| Bucket | Count | % |
|---|---:|---:|
| <= -20% | 2 | 2.0% |
| -20% to -10% | 8 | 7.9% |
| -10% to -5% | 5 | 5.0% |
| -5% to 0% | 10 | 9.9% |
| 0% to +5% | 23 | 22.8% |
| +5% to +10% | 22 | 21.8% |
| +10% to +20% | 16 | 15.8% |
| +20% to +30% | 9 | 8.9% |
| > +30% | 6 | 5.9% |

**81.2% of trades fall between -5% and +10%.** Right-skewed (skew +0.38) — the fat right tail drives the alpha.

---

## 4. Path Analysis — Intra-hold Max Drawdown & Favorable Excursion

### Max Drawdown (worst point during hold, from entry)

| Metric | Value |
|---|---:|
| Mean MDD | -0.18% |
| Median MDD | +0.00% |
| p10 MDD | -9.89% |
| p25 MDD | -2.55% |
| p75 MDD | +0.70% |
| p90 MDD | +8.94% |
| Worst MDD | -36.80% |

### Max Favorable Excursion (best point during hold)

| Metric | Value |
|---|---:|
| Mean MFE | +9.86% |
| Median MFE | +7.20% |
| p25 MFE | +2.46% |
| p75 MFE | +15.41% |
| Best MFE | +55.31% |

### Max drawdown by outcome

| Outcome | MDD mean | MDD median | MFE mean |
|---|---:|---:|---:|
| Winners | +3.06% | +0.00% | +13.90% |
| Losers | -10.03% | -9.19% | -2.43% |

Winners tend to go straight up (median MDD = 0%). Losers dip to -9.19% median before recovering partially.

---

## 5. Stop-Loss Impact (-10% delayed stop)

> *Total columns below are raw sum of trade returns, not NAV-compounded. The
> stop vs no-stop comparison is valid since both use the same metric.*

| Metric | With stop | No stop (pure 5-day hold) |
|---|---:|---:|
| Total PnL | +679.1% | +672.4% |
| Avg per trade | +6.72% | +6.66% |
| Trades stopped out | 10 (9.9%) | — |

Only 2 of 10 stops were genuinely beneficial (PLNT, CHWY). 8 were roughly neutral. CVLT was stopped at -30.5% vs -36.8% without — capped the worst case.

### Stopped trades detail

| Ticker | Date | Stop ret | No-stop ret | Saved? |
|---|---|---:|---:|---|
| CVLT | 2026-01-27 | -30.5% | -36.8% | no |
| PLNT | 2026-05-07 | -28.3% | -19.5% | YES |
| DBX | 2025-02-20 | -16.2% | -18.6% | no |
| SNX | 2026-06-25 | -13.6% | -13.6% | no |
| LITE | 2025-02-06 | -11.6% | -15.7% | no |
| AA | 2024-07-17 | -10.8% | -10.8% | no |
| BBWI | 2026-03-04 | -10.6% | -10.6% | no |
| THO | 2026-03-03 | -10.2% | -11.5% | no |
| DOCS | 2025-05-15 | -10.1% | -12.3% | no |
| CHWY | 2025-06-11 | -10.0% | -9.3% | YES |

---

## 6. PEAD vs Non-PEAD Trade Breakdown

> *Total columns below are raw sum of trade returns, not NAV-compounded.
> Per-trade metrics (N, Win%, Avg, Median, Avg Win, Avg Loss) are sizing-independent.*

| Group | N | Win% | Avg | Median | Total (raw sum) | Avg Win | Avg Loss |
|---|---:|---:|---:|---:|---:|---:|---:|
| **True PEAD** | 39 | 85% | +12.89% | +9.97% | +502.6% | +16.88% | -9.07% |
| False positive | 62 | 69% | +2.85% | +3.29% | +176.5% | +7.66% | -8.04% |
| ALL | 101 | 75% | +6.72% | +5.99% | +679.1% | +11.66% | -8.28% |

**Non-PEAD picks are still profitable** (+2.85% avg, 69% win). This is why filtering them hurts total return.

---

## 7. Large PEAD Analysis (CAR >= 10%)

> *Total columns below are raw sum of trade returns, not NAV-compounded.
> Contribution % is valid (relative split preserved).*

| Group | N | Win% | Avg | Total (raw sum) | Contribution |
|---|---:|---:|---:|---:|---:|
| **Large PEAD (CAR>=10%)** | 22 | 86% | +18.23% | +401.0% | 59% |
| Small PEAD (CAR<10%) | 17 | 82% | +5.98% | +101.6% | 15% |
| Non-PEAD (FP) | 62 | 69% | +2.85% | +176.5% | 26% |
| ALL | 101 | 75% | +6.72% | +679.1% | 100% |

22 large PEAD trades contribute **+401%** of the +679% total. These are the genuine, strong-surprise PEAD events.

---

## 8. Per-Fold Breakdown

> *Total (raw) = raw sum of trade returns. NAV-comp = 4-slot portfolio with
> 1/4 NAV per slot, weekly compounding (the realistic portfolio return per
> fold). Per-trade metrics (N, Win%, Avg, Median) are sizing-independent.*

| Fold | Period | N | Win% | Avg | Median | Total (raw) | **NAV-comp** | PEAD | Prec |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 2024-06 → 2024-12 | 18 | 78% | +8.53% | +8.01% | +153.6% | **+43.9%** (1.44x) | 8 | 44% |
| 2 | 2024-12 → 2025-06 | 31 | 68% | +4.07% | +4.29% | +126.2% | **+35.8%** (1.36x) | 11 | 35% |
| 3 | 2025-06 → 2025-12 | 26 | 81% | +10.78% | +5.64% | +280.3% | **+93.4%** (1.93x) | 12 | 46% |
| 4 | 2025-12 → 2026-06 | 26 | 77% | +4.32% | +6.06% | +112.3% | **+30.0%** (1.30x) | 8 | 31% |
| **All** | **2 years** | **101** | **75%** | **+6.72%** | **+5.99%** | **+672.4%** | **+391.3%** (4.91x) | **39** | **39%** |

**No losing fold.** Fold 3 is the standout (+93.4% NAV-compounded, 81% win, 1.93x NAV in 6 months). The NAV-compounded returns are much smaller than raw sums because each trade only deploys 1/4 NAV, but compounding across folds produces the 4.91x overall multiplier.

---

## 9. Sector Breakdown (post-XLF exclusion)

> *Total columns below are raw sum of trade returns, not NAV-compounded.
> Sector-to-sector ranking is valid. Per-trade metrics are sizing-independent.*

| Sector | N | Win% | Avg | Total (raw sum) | PEAD | Prec |
|---|---:|---:|---:|---:|---:|---:|
| **IJJ (Mid Value)** | 55 | 78% | +8.24% | +453.4% | 24 | 44% |
| IJK (Mid Growth) | 29 | 66% | +4.83% | +139.9% | 12 | 41% |
| IJH (Mid Blend) | 8 | 75% | +5.80% | +46.4% | 1 | 12% |
| IJS (Small Value) | 6 | 100% | +2.73% | +16.4% | 1 | 17% |
| XLB (Materials) | 3 | 67% | +7.67% | +23.0% | 1 | 33% |

**IJJ (Mid-Cap Value) is the alpha engine**: 55 trades, 78% win, +453.4% total, 44% PEAD precision.

---

## 10. Portfolio Efficiency

| Metric | Value |
|---|---:|
| Backtest period | 2024-07-17 → 2026-07-02 (715 days) |
| Total trade-hold days | 770 |
| Max slots | 4 |
| Slot utilization | 26.9% |
| Trades per year | 51.6 |
| Avg hold days per trade | 7.6 |

---

## 11. Equity Curve Statistics

> **IMPORTANT:** The numbers below were originally computed as the raw sum of
> trade returns (each trade treated as 100% NAV). This OVERSTATES the actual
> portfolio return. The corrected NAV-compounded return (4 slots × 1/4 NAV,
> weekly compounding, from `44_slot_sweep_nav_sizing.py`) is **+391.3%**
> (4.91x NAV). The raw-sum numbers are kept here for reference.

| Metric | Raw sum (original) | NAV-compounded (corrected) |
|---|---:|---:|
| Final cumulative PnL | +679.1% | **+391.3%** |
| Max drawdown | -32.0% | **-7.1%** |
| Avg return per trade | +6.72% | +6.72% (same) |
| Sharpe per trade | 0.516 | 0.516 (same) |
| Annualized Sharpe (approx) | 3.71 | ~2.5 (lower) |

---

## 12. BMO vs AMC Entry Timing

| Timing | N | Win% | Avg | Total | PEAD |
|---|---:|---:|---:|---:|---:|
| BMO | 50 | 70% | +6.20% | +310.1% | 14 |
| AMC | 51 | 80% | +7.23% | +369.0% | 25 |

AMC slightly outperforms (higher win rate, more PEAD events). Both are solidly profitable.
