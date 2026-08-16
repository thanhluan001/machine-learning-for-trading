# Megatrend Watcher Phase 2 — Cluster Machinery vs Floor (Kill Test C): FAILED, floor retained

**Script:** `04_backtest/72_megatrend_phase2_clusters.py` (+ diagnostics in this doc)
**Date:** 2026-08-16
**Design ref:** Design.md §18 RC-4 Phase 2.

## Test design

Monthly cadence (Phase-1 validated). Machinery: SPY-beta-residualized 36m
correlation → avg-linkage clustering (ρ≥0.35) → cluster 12m momentum score →
breadth gate (≥60% members above 10m mean) → hold top cluster equal-weight,
members below their 10m mean dropped. Compared head-to-head against the
Phase-1 floor (MA10m equal-weight basket over all assets), SPY-MA, SPY B&H.

Pass criterion (fixed before running): cluster beats floor-all on return AND
max drawdown.

## Headline results (2018-08 → 2026-08, same 26-asset universe)

```text
                                   total   ann%   maxDD   in-mkt
A. CLUSTER machinery               +251%  +14.8%  -17.4%    77%
B. FLOOR: MA10m all 26 assets      +458%  +20.2%  -29.7%   100%
C. FLOOR: MA10m SPY                 +86%   +7.3%  -23.0%    80%
D. SPY buy-and-hold                +212%  +14.1%  -23.9%   100%

2022 (crash year):   A +1%   B -8%   C -24%   D -20%
```

Cluster wins risk (DD −17.4 vs −29.7; 2022 +1%), loses return. Holdings
timeline economically sensible (clean energy 2019–21, GDX 2020, URA/XLE 2022,
AI 2023–26) — the semantic layer works.

## The diagnostics that decided it

**1. Top-2 clusters (the diagnosed parallel-trend fix): WORSE.** +224%,
DD −23.4%, 2022 −16% — the second cluster dilutes into weaker trends.

**2. Survivorship-stress (the decisive test): strip the seven hand-picked
mega-winners (MSTR, SHOP, NVDA, SMH, TSM, XYZ, PYPL):**

```text
floor_all  ex mega-winners:   +192%   DD -21.6%    ← robust structure
cluster    ex mega-winners:    -11%   DD -49.3%    ← COLLAPSES
```

The cluster strategy's +251% was carried almost entirely by concentration
luck in the one cluster (AI complex) that kept trending. Without those names,
top-cluster concentration rides dying trends (2019–21 it would have
concentrated in the pandemic cluster) into −49% drawdowns, while the
diversified floor's cross-trend rotation (clean energy dies → energy/GDX
rises) is exactly what survives.

## Verdict

**KILL TEST C: FAILED — cluster-selection layer REJECTED for execution.**
The kill test did its job: the headline comparison flattered the machinery,
and the stress test caught a second-order survivorship effect (concentration
luck in a hand-picked universe, distinct from the classic winner-picking bias
— here even a "losing" comparison arm was contaminated).

**What survives, per doctrine:**
- **The floor IS the strategy**: equal-weight trend basket with MA10m
  member exits (Phase-1 validated exit machinery + this test's robustness).
  Trivial, robust, and the bar every future elaboration must beat.
- **Cluster labels as descriptive semantics** (which trend is active) —
  useful reporting for the core-book warning role, not a selector.
- Phase 3 redirect: regime table + BROAD deployment universe (GICS sectors +
  major theme ETFs, where survivorship-flattery is structurally weaker) for
  the FLOOR structure — the open question is whether it beats SPY B&H with
  lower DD on the real universe.

## Lesson recorded

The near-miss pattern worth remembering: sophisticated layers can look better
than simple ones *because of* universe selection, not selection logic. The
floor's win here is diversification across trends; the machinery's loss is
concentration in one. When a complex strategy beats a simple one, stress-test
the universe before believing the complexity earned it.
