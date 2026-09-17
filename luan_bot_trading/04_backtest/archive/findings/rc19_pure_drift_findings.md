# RC-19 Closure — Pure-Drift Label (First Positive Expectancy, Uncertifiable at This Power)

**Status:** CLOSED 2026-09-16. G1/G2 FAIL as registered. No promotion.
**Spec:** `rc19_pure_drift_pre_registration.md` (0915d54) — frozen before run.

## Result

```text
                     trades  win%   avg trade   NAV(79w)   weekly-mean CI(95%)
v6n   (3-gate trio)    129   44.2   −1.581%    −42.8%     [−1.684, +0.122]
rc19  (pure drift)     158   53.8   +0.718%    +21.4%     [−0.685, +1.400]   ← PRIMARY
rc19_cost (>cost)      190   54.7   +1.066%    +51.7%     (declared non-gating)
rc19_pos   (>0)        190   53.7   +0.037%     −5.8%     (declared non-gating)

G1 FAIL: avg +0.718% > 0 but CI includes 0
G2 FAIL: paired vs v6n +0.846pp/wk, CI [−0.494, +2.231], prob_pos 53.3% (60 wks)
```

## What this established (the honest positives)

1. **The label redesign fixed the sign.** Dropping volume/drawdown from the
   target moved the book from −1.58% to **+0.718% avg trade, +21.4% NAV,
   53.8% win rate** — the first positive expectancy of the entire honest
   program (RC-16 reset onward), reproduced on the same 16,587-key matrix,
   same folds, same simulator that produced every failure before it.
   The composite 3-gate label was indeed the defect the probe identified.
2. **Trade count scales as designed** (129 → 158) without degrading quality.

## Cause of death (named, as registered)

1. **Power.** 79 weeks; CI half-width ±1.04pp around a +0.36pp weekly mean.
   The registration anticipated exactly this: "a fail certifies
   uncertifiable, not zero."
2. **Decoupling persists at the score level.** Even trained directly on
   car_10d>+3%, spearman(score, car_10d) = −0.039 among selected rows;
   decile means non-monotone (top deciles −1.3% to −2.1%). The expectancy
   does NOT come from the score ranking CAR — it emerges from the
   threshold (0.33) plus the weekly top-pick selection layer. The 24
   features still do not "see" which specific events drift; they see
   enough to pass/fail a 36% base rate and the weekly cut does the rest.
   Any future claim must respect this: this is not a proven alpha model;
   it is a construction whose edge, if real, lives partly outside the
   score's ranking.

## Discipline notes (no archaeology)

- rc19_cost (+1.07%/trade) is the strongest arm — it was DECLARED
  non-gating before the run and stays non-gating. Promoting it now
  because it won would be post-hoc selection. It is memorialized as a
  motivating diagnostic only.
- No threshold/HP/hold sweeps. Gates executed as registered.

## Door forward (requires new registration; user decision)

**RC-20 candidate — power extension by universe replication:** the exact
frozen RC-19 spec re-run on the combined v7c matrix (33,598 rows, SP400+
SP600, ~2× sample, more paired weeks) — no design choices, no new
parameters, gates unchanged. If +0.7–1.0pp avg trade replicates there
with tighter CIs, certification becomes possible; if it doesn't, the
point estimate was noise. The feature ledger (forward OOS logging)
remains available as a complementary path.

Artifacts: `archive/experiments/rc19/gauntlet.json`, `executed.h5`.
