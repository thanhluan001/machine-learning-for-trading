# RC-9 Undecided-State Detector — Findings

> **Source:** `04_backtest/86_rc9_undecided_state_detector.py` (backfill) +
> `05c_megatrend_watcher/monthly_panel_report.py` §[13] (operational, advisory)
> **Date:** 2026-08-23
> **Data:** `01_data/db_megatrend.h5` (/mt series; themes start 2014-01, COIN 2021-04)
> **Role:** advisory section [13] + `rc9_state` log field. No auto-allocation.

---

## Executive summary

The 3-metric state map (P leadership persistence, B theme breadth, C bloc
correlation) was built with pre-registered thresholds (P≥0.50, B≥50%,
C≥60th pct) and backfilled monthly over 2014-2026. Episode validation:
**2 clean passes, 1 partial, 1 premise failure** — the 2020-21
"everything-bid" era read CONCENTRATED(bloc), not UNDECIDED, because
leadership persistence stayed HIGH: the pandemic cohort was decided by
mid-2020 and led for a year. The premise conflated "many trends bid"
(breadth) with "winner unknown" (persistence); the map disentangled them.
Promoted to the month-end panel as advisory with that caveat recorded.

## The metrics (monthly, prior-month data only)

- **P** = Spearman autocorr of theme 3m relative returns vs SPY, lag 1m.
  High = same leaders keep leading. Median in-sample ≈ 0.7 in trend regimes.
- **B** = fraction of 10 theme proxies above own MA10 with positive 6m
  relative momentum vs SPY.
- **C-pct** = avg pairwise 60d correlation of theme daily returns as an
  expanding percentile of own history (≥36m; themes valid per-window
  because COIN launches 2021-04).

## Episode validation (pre-registered)

| Episode | Expected | Actual | Verdict |
|---|---|---|---|
| 2020-02..2021-06 | UNDECIDED-heavy | CONCENTRATED(bloc) 11m, UNDECIDED 3m | **FAIL (premise)** — P stayed 0.77-0.94; pandemic winners were decided and persistent; the "money everywhere" phase was ONE liquidity factor carrying all trends, which C (80-100th pct) correctly flagged as bloc |
| 2021 H2 winner emergence | DIFFERENTIATING | DIFFERENTIATING 5 of 6m (P collapsed 0.94 → −0.49 in Jul-Nov) | **PASS** — called the everything-trade breakup in real time |
| 2023-2025 AI era | CONCENTRATED-heavy | CONCENTRATED(+bloc) 18 of 36m, DIFF 12, DISP 5, UND 1 | **PARTIAL** — 2024-08..10 DISPERSAL (yen-carry unwind window) defensible; 2025 chop flips states |
| 2018 Q4 | DISPERSAL appears | DISPERSAL Oct+Nov (+Feb, Apr) | **PASS** |

Extra (not pre-registered, noted honestly): DISPERSAL also fired
2022-09..11 and 2024-08..10 — both real stress windows. The state with
the worst consequences for the old megatrend book (DISPERSAL) is the one
the map catches most reliably.

## What UNDECIDED actually looks like

Genuinely rare: 2020-03/04/06 (3m), 2025-11, 2026-04/05. The current
2026 reading is instructive: April-May UNDECIDED (P 0.43-0.49, B 50-60%,
C 91-92nd pct — breadth with no persistence = rotation churning), then
July-August snapped to CONCENTRATED(bloc) (P 0.94 → 0.54, B collapsed
20%, C 75th) — leadership reasserted (AI), breadth narrowed. The map's
current verdict: a winner is again being paid; the fractional-rotation
posture of April-May no longer matches the tape.

## Honest limitations

1. Theme data starts 2014 → only ~2 full cycles; the 36m percentile for C
   starts 2017. 2015-16 unvalidated.
2. The DIFFERENTIATING label is the noisiest state (remainder bucket);
   its operational meaning is only "not the other three" — treat as
   transition/watch, not action.
3. P's Spearman on 6 themes has coarse granularity (rank steps of 1/6);
   monthly flips of 0.3+ are common in chop (2025).
4. Advisory only. The 2020-21 misread is exactly the failure mode the
   doctrine warns about: a classifier can describe a state correctly
   while the human premise about that state is wrong. Postures remain
   manual decisions.

## Promotion

Added as section [13] of the month-end panel + `rc9_state` log field
(commit on 2026-08-23). Next scheduled reading: 2026-08-31 month-end run.
Backfill artifact: `archive/experiments/rc9_state_backfill.csv`.
