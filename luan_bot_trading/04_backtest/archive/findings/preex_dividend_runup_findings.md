# Pre-Ex-Divididend Run-Up Event Study — CLOSED (tail is dispersion, not alpha)

**Script:** `04_backtest/archive/edge_search_2026/70_preex_dividend_runup.py` (cached in `01_data/db_div.h5`)
**Date:** 2026-08-15
**Question:** What fraction of ex-dividend events have a "big" run-up (CAR > +1% vs IJH
pre-ex), and is the tail predictable ex-ante (PEAD-style gated-event framing)?

## Method

- 20,015 ex-dividend events, 925 S&P 400 tickers, 2015–2026 (Tiingo `divCash` field,
  fetched fresh into `db_div.h5`; production db.h5 untouched).
- Entry Close[T−N], exit Close[T−1] (never hold through ex-date), N ∈ {3, 5, 10}.
- CAR vs IJH. Tail thresholds +1% / +2%. Yield-quartile conditioning (ex-ante observable).

## Results

```text
window     mean     P(>1%)   P(>2%)
T-3      -0.02%    31.5%    18.4%
T-5      -0.05%    35.4%    23.9%
T-10     +0.00%    39.5%    31.3%

Yield conditioning (T-5):
Q1 smallest   +0.15%  (50.5% win)
Q4 largest    -0.29%  (46.7% win)   ← anti-predictive
```

- Mean run-up ≈ 0 in every window, every year (2015–2026), stable.
- P(>1%) = 35% LOOKS like a fat tail, but a zero-mean 5-day mid-cap return with ~3%
  std produces exactly this by chance — the tail is return dispersion, not a
  harvestable effect.
- The ex-ante conditioning variable (dividend yield) is ANTI-predictive: large
  dividends drift DOWN pre-ex (−0.29%), consistent with capture desks shorting
  into the ex-date. No predictor separates the tail from noise.

## Comparison to PEAD (why one lives and this dies)

PEAD: 10.7% gated base rate, but mean +2.78%/trade when model-selected (win 60%+).
Run-up: 35% exceed 1%, but mean ≈ 0 and no selection variable works — an oracle
is required, which means no edge.

## Follow-up: user-specified filter stack (yield + trend + liquidity)

Requested filters (all ex-ante observable), window T-5:
- F1: annualized TTM dividend yield > 2.5%
- F2: + stock AND IJH above their 50-day SMA at entry
- F3: + ADV20 dollar volume >= $50M (F3s: >= $100M)

```text
layer                          n       mean     win%    P(>1%)
all events                 20,014   -0.05%    48.6%    35.4%
F1  yield > 2.5% ann.       9,611   -0.18%    47.4%    34.8%   <- harmful
F2  + both > SMA50          4,069   -0.13%    46.3%    33.8%   <- no rescue
F3  + ADV20 >= $50M         1,103   -0.10%    47.5%    36.2%
F3s + ADV20 >= $100M          318   -0.22%    45.0%    34.9%   <- worst
```

- Yield filter is ANTI-predictive (confirms capture-desk shorting zone).
- Trend filter: above-SMA vs below-SMA events equally dead (-0.13% vs -0.22%) —
  the effect is absent, not regime-dependent.
- Liquidity filter makes it slightly worse: the most liquid high-yielders are
  the most efficiently priced.
- Year-by-year at F3: 8 of 12 years negative, means -1.16%..+0.44%, no persistence.
- Supply at F3: Sep ~91 events (~4-5/week), median ADV $76M, median yield 3.75%.

**Verdict unchanged and strengthened: CLOSED.** The classic dividend-capture
filter trio selects the most-arbitraged corner of the universe.

## Decision

**Closed — do not pursue.** The 35% tail frequency exceeds the user's 10–20%
"interesting" bar, but the tail is statistical dispersion, not alpha; and the
only ex-ante conditioning variable points the wrong way (short-side, wrong book).

## Pattern now established across slow-week candidates

| candidate | verdict | script |
|---|---|---|
| Analyst upgrades | DEAD (Day-0 pricing, zero drift) | 68 |
| Index additions | DEAD (front-run to 0 vs IJH) | 45 |
| Insider cluster buying | DEAD (5–10d CAR ≈ +0.1%, win 49%) | 69 |
| Pre-ex dividend run-up | DEAD (zero mean, noise tail, anti-predictive yield) | 70 |

Every scheduled/known-in-advance event in S&P 400 is arbitraged to zero. The
earnings PEAD anomaly survives precisely because underreaction takes days.
Idle-slot cash through dead zones remains the September policy.
