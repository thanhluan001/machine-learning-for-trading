# RC-17 Closure — No-Drift Baseline

**Status:** CLOSED 2026-09-15. G1/G2/G3 all FAIL. Per the pre-committed rule,
the 23-feature PEAD construction at this universe/label definition is
retired from the research queue.
**Registration:** `rc17_nodrift_pre_registration.md` (2026-09-15, committed
before any result). Frozen HPs, frozen thresholds, identical folds and
bootstrap as RC-16 R3 (10k week-block resamples, seed 20260807).

## Result

| arm | exec | win% | avg trade | week-block 95% CI | verdict |
|---|---:|---:|---:|---|---|
| v6n (gates, 22 feats) | 129 | 44.2 | −1.581% | [−1.684, +0.122] | G1 FAIL |
| v4n (single, 22 feats) | 69 | 47.8 | −0.419% | [−1.266, +0.999] | G2 FAIL |
| v7n (combined, 22+1 feats) | 131 | 47.3 | −0.723% | [−1.383, +0.777] | G3 FAIL |

G4 (paired vs RC-16 c-companions, reporting only): removing the
honest-but-noisy drift feature changes nothing — v6n−v6c +0.146pp
CI [−0.60, +0.90]; v4n−v4c −0.366pp; v7n−v7c −0.527pp (all CIs include 0).

## Reading (what RC-16 + RC-17 jointly establish)

1. The historical V4/V6/V7 backtest edges were carried by the
   `car_drift_historical_q1` look-ahead (RC-16: 91.2% of the damage).
2. With the feature honest, the gate trio still tilts on its noise and
   selects badly (RC-16: v6c significantly negative).
3. With the feature REMOVED ENTIRELY, the remaining 22 features carry no
   honest edge either — all three families' point estimates are mildly
   negative and none approaches significance (RC-17).

The two-run arc is the cleanest possible demonstration of the audit's
thesis: the edge lived in information that did not exist at decision time,
and once honesty is enforced at every joint (features, labels, folds,
macros), this construction does not have one. Nothing here was tuned,
re-thresholded, or shopped; both programs ran exactly as registered.

## Consequences (executing the pre-committed rules)

- **PEAD research line PAUSES.** No re-tuning of the old feature set, no
  threshold archaeology, no BMO/AMC rule adoption (stays a registered
  future hypothesis). Revival requires NEW features or NEW labels under a
  new registration.
- **Nightly paper book continues as process data only** (frozen V6
  semantics, corrected contracts since 2026-09-14/15). Its ledger is a
  chronology of process — no promotion reviews exist or will exist.
- The corrected matrices (v6c/v7c), retrains (*_rc16), and gauntlet
  artifacts remain archived for any future audit or revival effort.
- Unaffected programs: megatrend watcher (RC-4/RC-9, no ML), RC-15
  lifecycle (invariants, no ML), tax reporting, data infrastructure.
- `basic_pead_project/` skeleton: carries a frozen V6 whose edge did not
  survive audit — README caveat to be added (pipeline educational, edge
  historical artifact).

## What is NOT concluded

- "No edge at this construction" ≠ "PEAD does not exist in SP400/SP600."
  The label definition (3-gate 10-day outcome), the universe, the vendor
  vintages, and the 23-feature set are one construction among many.
- Forward ledgers under honest contracts remain the only clean test; if a
  future construction is registered, it starts from zero credibility and
  must earn it the same way this one lost it.
