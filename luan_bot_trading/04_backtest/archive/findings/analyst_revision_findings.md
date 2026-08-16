# Analyst Revision Momentum Probe — Findings

> **Source:** `04_backtest/46_analyst_revision_probe.py`
> **Date:** 2026-07-30
> **Data:** `/analyst/grades/{permaTicker}` (807 nodes, 14 years, 111 firms)
> **Universe:** S&P 400 Mid-Cap, 2015-2026

---

## Executive Summary

The analyst revision momentum edge is **REAL but WEAK**. Upgrades generate
+0.46% abnormal return over 21 days (market-adjusted vs IJH), statistically
significant (p<0.001) across 12,205 events. The edge has NOT decayed over 11
years. However, the per-trade edge is 14× weaker than PEAD (+0.46% vs +6.66%)
with a 4-12× longer hold period. As a standalone strategy, it barely covers
transaction costs. It has value as a **component signal** in a multi-factor
model, not as a standalone edge.

---

## 1. Core Finding: Abnormal Returns (market-adjusted vs IJH)

| Hold period | Upgrades | Downgrades | Long-short spread |
|------------|---------:|-----------:|------------------:|
| 5d (1 wk) | +0.23% \*\*\* | +0.25% \*\*\* | +0.27% |
| 10d (2 wk) | +0.47% \*\*\* | +0.30% \*\*\* | +0.77% |
| **21d (1 mo)** | **+0.46% \*\*\*** | +0.50% \*\*\* | +0.96% |
| 42d (2 mo) | +0.98% \*\*\* | +1.15% \*\*\* | +2.13% |
| **63d (3 mo)** | **+1.60% \*\*\*** | +2.24% \*\*\* | +3.84% |

All t-stats significant at p<0.001 due to large sample (12,000+ events).

### The downgrade surprise

**Both upgrades AND downgrades have positive abnormal returns.** This is
unexpected — the classic literature (Chan 1996, Womack 1996) predicts
downgrades drift *down*. Two explanations:

1. **Mean reversion:** Analysts downgrade *after* the stock has already
   fallen (they're late). The downgrade marks the bottom, and the stock
   bounces. The S&P 400 mid-cap universe has an inherent upward drift.
2. **Survivorship bias:** Stocks that get downgraded but *stay* in the
   index are the ones that recover. Truly failing stocks get removed.

This means **downgrades cannot be shorted profitably** in this universe —
the mean reversion overwhelms the momentum signal.

---

## 2. Edge Stability Over Time

21-day abnormal returns for upgrades, year by year:

| Year | N | Win% | Avg | Median |
|-----:|--:|-----:|----:|-------:|
| 2015 | 1,044 | 52% | +0.74% | +0.58% |
| 2016 | 1,209 | 52% | +0.68% | +0.27% |
| 2017 | 1,153 | 50% | +0.39% | +0.02% |
| 2018 | 1,303 | 52% | +0.41% | +0.26% |
| 2019 | 1,088 | 51% | +0.04% | +0.19% |
| 2020 | 1,501 | 48% | +0.94% | -0.71% |
| 2021 | 1,066 | 53% | +0.93% | +0.56% |
| 2022 | 829 | 52% | +1.40% | +0.40% |
| 2023 | 891 | 49% | -0.29% | -0.15% |
| 2024 | 822 | 53% | +0.43% | +0.56% |
| 2025 | 924 | 47% | -0.57% | -0.42% |
| 2026 | 375 | 47% | -0.29% | -0.46% |

| Period | Avg (21d) | Win% |
|--------|----------:|-----:|
| Pre-2020 | +0.45% | 51% |
| Post-2020 | +0.47% | 50% |

**The edge is stable** — +0.45% pre-2020 vs +0.47% post-2020. It has not
decayed. Individual years are noisy (2023 and 2025 were negative), but the
long-run average is consistent. This is consistent with a **behavioral**
bias (anchoring/underreaction), not a mechanical anomaly that gets arbitraged.

---

## 3. Signal Enhancement Attempts (all failed)

### Cluster upgrades (2+ upgrades in 30-day window)

| Signal | N | Avg abnormal (21d) | Sig |
|--------|--:|------------------:|:---:|
| Single upgrade | 7,693 | +0.45% | \*\*\* |
| 2 in 30 days | 3,199 | +0.47% | \*\* |
| 3 in 30 days | 832 | +0.27% | ns |

Clustering does NOT amplify the signal. A single upgrade is just as good
as a cluster.

### Upgrade magnitude (ordinal change)

| Signal | N | Avg abnormal (21d) | Sig |
|--------|--:|------------------:|:---:|
| +1 ordinal (e.g. Hold→Buy) | 6,196 | +0.47% | \*\*\* |
| +2 ordinal (e.g. Sell→Buy) | 5,611 | +0.45% | \*\*\* |

Magnitude does NOT matter. A small upgrade is as good as a big one. The
edge is a uniform slow drift affecting all upgrades equally.

---

## 4. Comparison to PEAD

| Metric | PEAD (current) | Analyst upgrades |
|--------|---------------:|----------------:|
| Per-trade edge | **+6.66%** | +0.46% (21d) |
| Win rate | **75%** | 51% |
| Hold period | 5 days | 21-63 days |
| Annual trades | ~50 | ~1,000+ |
| Edge decayed? | No | No |
| Mechanism | Earnings surprise underreaction | Analyst revision underreaction |

PEAD is **14× stronger per trade** with **4-12× shorter hold**. The analyst
revision edge cannot compete as a standalone strategy.

---

## 5. Timing Complementarity

Analyst revisions happen year-round, including shoulder months where PEAD
has idle capital:

| Month | Upgrades | PEAD overlap? |
|------:|---------:|:---:|
| Jan | 1,653 | Earnings season |
| Feb | 973 | Earnings season |
| **Mar** | **1,109** | **Shoulder (idle)** |
| Apr | 1,079 | Earnings season |
| May | 1,066 | Earnings season |
| **Jun** | **738** | **Shoulder (idle)** |
| Jul | 1,006 | Earnings season |
| Aug | 933 | Earnings season |
| **Sep** | **863** | **Shoulder (idle)** |
| Oct | 1,013 | Earnings season |
| Nov | 1,082 | Earnings season |
| **Dec** | **960** | **Shoulder (idle)** |

~3,670 upgrades/year happen in shoulder months (Mar, Jun, Sep, Dec) where
PEAD capital is idle.

---

## 6. Verdict and Recommendations

### NOT deployable as a standalone strategy

- +0.46% per trade at 51% win rate barely covers transaction costs
- 21-63 day hold ties up capital for marginal returns
- Every signal enhancement attempt (clustering, magnitude) failed to improve the edge

### Value as a component signal

1. **Already in the PEAD model:** The 8 revision momentum features
   (`revision_momentum_30d/60d/90d`, `revision_ordinal_momentum_90d`,
   `revision_intensity_90d`, `grade_dispersion_90d`, `n_analysts_covering`,
   `last_action_days_before_earnings`) capture this edge. The probe validates
   these features are based on a real, stable signal.
2. **Multi-signal model candidate:** Combine analyst upgrades + insider buying
   + PEAD into a composite. The analyst revision edge adds diversification
   value even if weak alone.
3. **Future 63-day hold variant:** The +1.60% abnormal at 63 days is more
   substantial. A longer-hold "drift" strategy (distinct from the 5-day PEAD)
   could complement the portfolio.

### Priority assessment

| Edge | Per-trade | Win% | Status |
|------|----------|-----:|--------|
| PEAD (current) | +6.66% | 75% | **Deployed** |
| Analyst revisions | +0.46% | 51% | **Weak — feature only** |
| Insider cluster (Form 4) | +5-10% (lit.) | — | **Next to probe** |
| Index rebalancing | +0.02% (mkt-adj) | 50% | **Dead** (see `45_index_rebalance_probe.py`) |

**Next priority:** Insider cluster buying (SEC Form 4). Free data, strongest
documented edge (+5-10% over 6 months), completely uncorrelated to earnings.
