# Megatrend Watcher Phase 1 — Dead-Trends Exit Test (Kill Test A): PASSED

**Script:** `04_backtest/71_megatrend_phase1_dead_trends.py` (data: `01_data/db_megatrend.h5`, 26 Tiingo series)
**Date:** 2026-08-16
**Design ref:** Design.md §18 RC-4. Role: overlay/warning indicator for the core ~90% book (real estate + index + blue chips), NOT the PEAD sleeve.

## Test design

The floor machinery (deliberately trivial — this is the Kill test C bar):
state machine per asset, entry on close > N-day MA, exit after M consecutive
closes below. Variants: MA {150, 200, 250} × confirm {1, 3, 5}, daily and
monthly cadence. Universe deliberately DOMINATED BY DEAD megatrends (a
backtest on winners only is a eulogy written in advance):

```text
dead_2021 : ARKK, KWEB, XBI, TAN, ICLN, COIN, MSTR, ZM, PTON, SHOP, PYPL, XYZ, SPAK
cycles    : GDX, XLE, EEM, XLU, URA, LIT, IBB, VNQ   (died AND revived)
live_2026 : SMH, NVDA, TSM
reference : SPY, QQQ
```

Pass criteria fixed before running: A1 dead-class mean giveback-from-peak at
exit ≥ −35% AND rule maxDD < B&H maxDD for every dead trend; A2 live-class
capture ≥ 70%; A3 any single variant meets both.

## Results

### Dead trends (the reason this test exists)

| proxy | B&H | rule | B&H maxDD | rule maxDD | giveback@peak |
|---|---|---|---|---|---|
| ARKK | +340% | +44% | −81.0% | −68.1% | −29.9% |
| COIN | **−55%** | **+11%** | −90.9% | −60.6% | −38.7% |
| MSTR | +653% | **+2157%** | −89.3% | −69.8% | −28.8% |
| PTON | −78% | −53% | −98.3% | −82.7% | −40.3% |
| SHOP | +5909% | +2381% | −84.8% | −50.9% | −33.4% |
| SPAK | −43% | −3% | −62.1% | −4.0% | already out |
| (13 total) | | | | mean: | **−28.6%** |

The rule exits dead megatrends ~29% below their peaks on average, versus
buy-and-hold drawdowns of −62% to −98%. MSTR is the showcase: avoiding the
2022 −93% crash turned +653% B&H into +2157%.

### Live trends (2026 AI complex)

| proxy | B&H | rule | capture | B&H maxDD | rule maxDD |
|---|---|---|---|---|---|
| SMH | +3074% | +2286% | 92% | −45.3% | −28.7% |
| NVDA | +60187% | +48243% | 97% | −66.3% | −46.9% |
| TSM | +3239% | +1346% | 76% | −56.5% | −30.4% |

88% mean capture at the headline variant — being "late" costs little because
diffusion takes years (the motivating NVDA/Pelosi thesis, now measured).

### Variant sweep (corrected evaluation)

```text
variant              dead GB   13/13 DD-wins   live cap   cycles cap
MA150/1cf daily      −26.7%        13/13          69%         47%
MA200/1cf daily      −29.1%        13/13          83%         58%   ← passes all
MA200/3cf daily      −28.6%        12/13          88%         55%
MA250/5cf daily      −30.9%        13/13          83%         38%
MA200/3cf MONTHLY    −29.0%        13/13          80%         64%   ← passes all; ops-realistic
```

## Verdict

**PASSED (A3).** The trivial floor already: exits dead megatrends at ~−29%
mean giveback (vs −62..−98% B&H), keeps 80–83% of live-trend upside with
drawdowns cut 17–26pp — and the MONTHLY-cadence variant passes everything
while requiring only month-end decisions (best cycles capture too, 64%).
ICLN was the lone DD failure at 3cf confirm (+1.8pp worse than B&H — choppy
cleantech whipsaw); 1cf and monthly variants fix it.

## Honest caveats

1. **Evaluation sign bug caught and fixed mid-run**: `dd < bhdd` initially
   counted drawdown-worse cases as wins (drawdowns negative). Table results
   were always correct; only the pass/fail rollup was inverted. Corrected
   before interpreting.
2. **Ranging assets whipsaw** (GDX capture 11%, VNQ 1%, XLU 28%): expected —
   a per-asset MA is a trend detector, and those names aren't megatrends.
   Phase 2's cluster-ranking layer (only trend assets enter the watchlist)
   addresses this; Phase 1 tests exit machinery on trend assets specifically.
3. **Index reference**: SPY/QQQ capture 66–72% — the rule gives up return on
   the broad market (no persistent trend). Confirms this is an overlay for
   trending clusters, not a whole-portfolio replacement — matching its role
   as carve-out governor for the core book.
4. No transaction costs modeled (turnover is months-scale; whipsaw is the
   enemy). Survivorship: universe chosen ex-post BY lifecycle class — but
   that is the test's design (exit quality), not cherry-picking winners.

## Next phases (if pursued)

- Phase 2: cluster construction (correlation-based, not GICS), cluster-level
  composite scoring (12m momentum + breadth), watchlist ranking — must beat
  the Phase-1 floor (Kill test C: beat/match MA200 variants after costs).
- Phase 3: regime table (2008/2011/2015/2018Q4/2020/2022/2023 chop) on the
  composite; AI-bubble scenario: current SMH/NVDA/TSM state as live case.
- News APIs (FMP /stable/news, Tiingo /news/) noted as candidate sentiment
  features — NOT Phase 1 inputs (doctrine: narratives arrive last).
