# Analyst Upgrade Drift Event Study — CLOSED (no tradeable edge)

**Script:** `04_backtest/archive/edge_search_2026/68_analyst_upgrade_drift.py`
**Date:** 2026-08-13
**Question:** Can analyst upgrades fill idle slots in earnings dead zones (Sep/Dec/Mar/Jun)?

## Method

All 31,318 upgrade/downgrade events (FMP `/stable/grades`, 807 sp400
permaTickers, 2012–2026). Execution-honest: drift measured from **Close[T+1]**
(next close after the event) vs IJH, horizons 5/10 trading days. Upgrades split
by earnings proximity (±3 td), new rating ordinal, ordinal delta, year.

## Results

| segment | n | rel 5d | win 5d |
|---|---|---|---|
| all upgrades | 15,063 | **+0.09%** | 49.7% |
| upgrades FAR from earnings | 13,682 | +0.09% | 49.5% |
| FAR upgrades to Buy/Strong Buy (ord 4–5) | 11,178 | ~+0.1% | ~49% |
| FAR upgrades, delta=+2 or more | 8,300 | +0.05% | 49.2% |
| downgrades (reference) | 16,255 | +0.18% | 49.3% |

- No cell (ordinal, delta, earnings-proximity, year) shows a tradeable edge.
  Recent years (2023–2026) are slightly NEGATIVE.
- Announcement-DAY move (untradeable): upgrades to ord 4–5 average **+3.27%**
  (median +1.75%) on day 0; downgrades −2.47%. **The reaction is immediate
  and complete** — by the next close, zero drift remains.

## Conclusion

**The upgrade edge exists but is entirely in the announcement-day move**,
captured by faster participants. Post-announcement drift in mid-caps is
statistically zero at our horizons. This is consistent with (a) V5
grades-historical rejection and (b) revision-momentum features being weak
model inputs.

## Decision

**Closed — do not pursue.** No slow-week filler edge from analyst grades.
Idle slots in earnings dead zones are the correct behavior; the strategy's
edge is earnings PEAD, and holding cash in September costs nothing vs. the
expected value of forced trades (~0% edge minus costs/slippage).

Slow-week idea inventory after this: exhausted (sp500 universe — rejected
script 67; analyst upgrades — closed script 68; MOC routing — operational,
not an edge).
