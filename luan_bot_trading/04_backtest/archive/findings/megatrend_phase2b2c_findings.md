# Megatrend Phase 2b/2c — Gradual Pivot & Capex Tilt: BOTH REJECTED (procyclical at trend death)

**Scripts:** `75_megatrend_phase2b_pivot.py`, `76_megatrend_phase2c_capex.py`
**Date:** 2026-08-16 | **Design ref:** RC-4 (post-Phase-3 refinement path, now closed)

## The hypothesis (user)

Phase 2 failed with BINARY cluster selection. Proposed alternative: start
equal-weight, then GRADUALLY pivot toward demonstrated winners (slow sell of
losers, slow buy of winners) — megatrend persistence means the pivot needn't
be early, only eventually right. Tilt variable candidates: price momentum
(2b) and CAPEX FLOWS (2c: "no one spends on Bitcoin but massive money flows
to AI" — capital-cycle confirmation).

## Phase 2b: momentum pivot — REJECTED

Monthly, eligibility = floor (close > MA10m), weight ∝ clip(1 + k·z(12m
momentum), 0.5, 2.0), k ∈ {0.25, 0.5, 1.0}. Kill criteria fixed in advance:
beat floor on return AND DD (full universe), no collapse ex-winners.

```text
FULL universe:        total    maxDD          STRESS (ex 7 winners):  total   maxDD
FLOOR  k=0            +390%   -29.7%          FLOOR                   +174%  -21.6%
PIVOT  k=1.0          +449%   -33.0%   ←K1 F  PIVOT k=1.0             +141%  -26.8%  ←K2 F
```

Year table shows the mechanism: the tilt adds return only while winners keep
winning (2024: +35% vs +22%) and subtracts at every trend death (2018, 2021,
2022 all worse). **Maximal trailing momentum = maximal weight immediately
before death** — gradualness does not fix being fattest at the top. On the
stress universe the tilt is worse on BOTH dimensions (it concentrates into
the pandemic cluster exactly as top-1 selection did, just slower).

## Phase 2c: capex confirmation — scale signal CONFIRMED, tilt signal REJECTED

Bellwether panel, FMP `/stable/cash-flow-statement`, TTM aggregates:

```text
Theme TTM capex:    2018    2022    2024    2026H1
AI/hyperscale        69B    160B    242B    574B   (accelerating)
clean_energy           8B     13B     13B     26B
crypto                 0B      1B     24B      1B

User's scale hypothesis: CONFIRMED — AI outspends crypto 174x-401x.
```

**The natural experiment (decision rule pre-registered):** clean-energy
capex vs ICLN price through the trade's death (ICLN peaked 2020-12-31,
−58% by end-2024):

```text
            2021Q4   2022Q4   2023Q2   2023Q4   2024Q4   2025Q2
capex vs peak  -19%     +4%    +12%    +13%     +3%     -10%
ICLN vs peak   -24%    -28%    -33%    -43%    -58%    -51%
```

**Capex ROSE for three years into a −58% price decline**, peaking 2023Q4 —
three years AFTER the ETF peaked — and only rolled over in 2024-25, after
the damage. Capex LAGS price at trend death; it has zero death-timing
information. Per the pre-registered rule: a capex tilt inherits 2b's
procyclicality in worse form (capex peaks with the theme — the classic
capital-cycle OVERSUPPLY failure mode, the 2000s-fiber scenario, exactly as
flagged in the framing: high capex = max competition building into the top).

## Verdict

**Both refinements REJECTED.** The pivot path is closed at every layer:
binary selection (P2), gradual momentum (2b), fundamental capex confirmation
(2c) — all fail the same way: every signal that confirms a trend is maximal
at its end. RC-4's surviving deployment remains exactly as Phase 3 left it:
**the manual monthly panelized breadth warning indicator** (`05c_megatrend_watcher/monthly_panel_report.py`),
zero capital.

## Salvage: capex as indicator CONTEXT (not tilt)

The one legitimate surviving use: capex aggregates as narrative-independent
SCALE evidence in the monthly warning report — "the AI trend is real: $574B
TTM, accelerating" (vs clean-energy's decade of ~$13B) belongs in the
breadth report's context block, the same role cluster labels hold
(descriptive semantics, never selection). Optional enhancement to Script 74;
data already cached (`db_capex.h5`, 18 series).

## The general law this adds to the doctrine

For long-only trend participation, ANY confirmation signal (price momentum,
capital flows, analyst breadth) is procyclical at trend death — the
confirmation IS the crowding. Exit machinery + equal weight + information
(the breadth reading) is the frontier of what monthly-cadence long-only can
extract; every attempt to convert information into concentration has now
failed three independent ways.
