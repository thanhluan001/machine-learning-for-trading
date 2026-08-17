# Megatrend Watcher Phase 3 — Broad-Universe Regime Table: FAILED as strategy; SALVAGED as warning indicator

**Script:** `04_backtest/73_megatrend_phase3_broad_regime.py` (diagnostics inline)
**Date:** 2026-08-16 | **Design ref:** Design.md §18 RC-4 Phase 3.

## Test

The Phase-2 floor (equal-weight, monthly, member exits below 10m mean) on the
BROAD deployment universe — 26 category-chosen assets (11 GICS sectors, broad/
intl/bond/gold, standard theme-ETF menu), expanding universe from 2006 —
versus SPY B&H and static 60/40, across the full regime table 2006–2026.
Category selection, not winner selection.

## Headline (2006-06 → 2026-08)

```text
                       total    ann%   maxDD
FLOOR-27 (broad)       +245%   +6.3%  -33.4%
FLOOR 60/40 AGG        +164%   +4.9%  -26.7%
SPY buy-and-hold       +784%  +10.8%  -50.8%
static 60/40           +375%   +7.7%    —
```

## Regime table (FLOOR − SPY, pp)

```text
2008: +34   (the design case: -11% vs -46%)
2011:  -4   2015: -9   2018:  -8
2020:   0   2022: -20  2023: -12   2024: -14
13 of 20 years trail SPY; worst underperformance -20pp (2022)
```

Kill criteria: P1 pass (DD −33.4% vs −50.8%), P2/P3/P4 all fail.

## The 2022 diagnosis (the decisive finding)

Month-by-month anatomy shows the basket empties correctly at every break
(Jan 20→8, Feb 8→9, Jul 0, Oct 0) — **exit machinery works**. The −40% year
comes from **re-entry whipsaw**: 1-month momentum bounces re-fill the basket
(Apr: 16 assets, Aug: 1, Nov: 15) into fresh legs down. 2022 was a sequence of
bear rallies — the exact whipsaw regime the Phase-1 caveat predicted for
"ranging assets," now measured at portfolio scale.

Dual-confirm fix (price > MA10m AND MA10m rising): 2008 improves (−7%) but
2022 WORSENS (−43%) — the whipsaw is regime-structural, not a parameter
problem. No in-family fix converges; the honest read is that a long-only
equal-weight trend basket cannot both ride trends and dodge bear rallies
without short-term information we deliberately exclude (doctrine: narratives/
speed are not our game).

## Verdict

**As a STRATEGY (carve-out vehicle replacing/augmenting core holdings): FAILED.**
It converts SPY's −51% crash risk into −33% with 2008 insurance, but gives up
5.4pp/year across 20 years — and the insurance is regime-dependent (2022 was
worse than holding SPY). Equal-weight across all above-MA assets holds ~17
names: too diversified to be a *megatrend* vehicle (the top-cluster
concentration that Phase 2 rejected was the megatrend exposure; diversification
killed it).

**As a WARNING INDICATOR (the original deployment plan): VALIDATED.**
The breadth signal — *fraction of universe above 10m mean* — is the useful
output: at 2022's breaks it collapsed to 0–5 of 26 before each major leg down;
today it reads 20/26 (see script [6], monthly refresh). For the core book's
"when to de-risk the carve-out / when is the megatrend fading" question, this
is the dashboard — with no capital at risk on its accuracy as a *return*
strategy.

**RC-4 final status: warning-indicator role adopted; strategy role closed.**
The three-phase arc (A pass → C fail → regime fail) is the expected shape of
an honest funnel: exit machinery validated, selection layers rejected,
deployment reality tested. Deployed component = monthly breadth report on the
26-asset universe (one page, month-end cadence, no capital).

## Lesson

The Phase-2 survivorship lesson had a mirror here: on the hand-picked trend
universe the floor's diversification looked like robustness; on the real
universe that same diversification is revealed as *dilution* of the megatrend
exposure — and the concentration that "failed" Phase 2 was the only part that
actually expressed the thesis. Universe choice doesn't just flatter results;
it can invert conclusions about strategy structure itself.
