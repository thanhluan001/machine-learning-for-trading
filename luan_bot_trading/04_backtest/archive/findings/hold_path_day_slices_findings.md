# Hold-Path Day-Slice Decomposition — Findings

> **Source:** ad-hoc event scan (user question), 10,422 SP400 events,
> 258-ticker random sample (seed 42, consistent with RC-2/RC-10), 2015-2026.
> **Companion to:** archive/phase_g_v3_v4_era/52_hold_period_comparison.py
> (the gated-pick hold-period study that set T+5).
> **Date:** 2026-08-26

---

## Question (user, from live paper trading)

"Most of the profit happened during T+1→T+3. T+4 and T+5 are almost
always losses. Can you test that?"

## Setup

Per-day abnormal (log) return slices vs IJH around each realized earnings
event: `gap` (entry close → first post-print close d1), then d1→d2, d2→d3,
d3→d4, d4→d5. Cohorts: all events / favorable print (gap>0, n=5,368) /
strong beat (gap>+2%, n=3,723) / path-conditional (in-profit at d3).

## Results

### 1. The profit is the gap, not the drift

| slice | favorable cohort | strong beat |
|---|---:|---:|
| gap (entry→d1) | **+5.46%** | **+7.45%** |
| d1→d2 | −0.03% | +0.02% |
| d2→d3 | −0.07% | −0.09% |
| d3→d4 | +0.03% | +0.07% |
| d4→d5 | **−0.07% (t=−2.2)** | −0.09% (t=−2.1) |
| cum entry→d3 | +5.40% | +7.44% |
| cum entry→d5 | +5.35% | +7.38% |

Virtually 100% of the mean edge is in the print reaction (the overnight
gap + day-1 close we capture by entering pre-print). Every post-d1 daily
slice is within ±0.1% — days 2, 3, 5 bleed a whisper (t≈−2), day 4 ticks
up slightly.

### 2. The user's claim, path-conditioned

Among favorable-print positions **in profit at d3** (n=5,279):

- d3→d4: **+0.27%, win 55%, t=+7.7** — winners carry one more day
- d4→d5: −0.06%, win 49%, t=−1.6
- d3→d5 combined: +0.28%, win 53%, t=+5.1

So T+4 is actually the BEST post-gap day for winning positions; only T+5
is mildly negative. "Almost always loss" is not supported — win rates are
49-55%, not 20-30%. With 8 live trades, the perception is small-sample
selection (BILL bled late; DBX bled late; the winners' late days were
forgettable rather than lossy).

### 3. Decision comparison: T+3 exit vs T+5 exit (favorable cohort)

Holding d3→d5 adds **−0.05% mean / +0.01% median, 50% win** — statistically
indistinguishable from zero and below round-trip cost differences. The
gated-pick version of this question (script 52) validated T+5 with the
hold-period sweep; this scan confirms its mechanism at daily granularity:
nothing tradeable lives in the intra-hold path.

## Verdict

No change to the frozen T+5 exit. The real structure is: **gap ≈ everything,
intra-hold days ≈ noise (±0.1%), T+4 mildly positive for winners, T+5
mildly negative for everyone (−0.06..−0.09%, t≈−2)** — the last is the
kernel of truth in the user's observation, but it is an order of magnitude
too small to pay for changing a frozen exit rule (would need the script-63
walk-forward + bootstrap bar, and the expected uplift is ≈ 0.05%).

---

## Addendum (2026-08-26): the GATED head-to-head — T+4 exit NOT promoted

`88_t4_vs_t5_exit_test.py`: identical V6 gates (θ=0.33), folds, slate,
slots, force-refresh (mh=4) and stops; ONLY the natural exit moves one
trading day earlier. Paired per-trade comparison on common trades:

| window | n | paired diff (T4−T5) | bootstrap 95% CI | verdict |
|---|---:|---:|---|---|
| DEV 1-3 | 122 | +0.046% | [−0.33%, +0.44%] | includes 0 |
| fold 1 (2024 H2) | 32 | +0.298% | [−0.45%, +1.10%] | includes 0 |
| fold 2 (2025 H1) | 49 | +0.375% | [−0.25%, +1.02%] | includes 0 |
| fold 3 (2025 H2) | 41 | **−0.543%** | [−1.13%, +0.07%] | includes 0, sign FLIPPED |
| fold 4 holdout (2026 H1) | 62 | +0.026% | [−0.41%, +0.48%] | includes 0 |

The ungated −0.07%/trade last-day slice does NOT transfer to gated picks:
DEV diff +0.046% (sign-flipped from naive), CI includes zero in every
window, fold 3 is negative, and the paired diff wins only 36% of trades
(median 0.000% — most trades have identical or flat last days).
**T+5 stays, frozen. Paper-phase optimization candidate tested at the
full bar and rejected.**
