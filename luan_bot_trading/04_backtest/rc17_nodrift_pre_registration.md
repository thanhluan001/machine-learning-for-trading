# RC-17 Pre-Registration — No-Drift Baseline: Do the Remaining 22 Features Carry Anything?

**Status:** REGISTERED 2026-09-15, before any result is computed.
**Trigger:** RC-16 closure (G1-G3 FAIL; 91.2% of the V6 damage attributed to
the `car_drift_historical_q1` leak). RC-16 deliberately did not test the
null: a model family with the drift feature REMOVED entirely. This
registration pre-specifies exactly that experiment and nothing else.

## Motivation

RC-16's corrected arms kept the drift feature in honest form (45-session,
maturity-masked) — and the gates still assigned it weight, tilting selections
on noise (v6c significantly negative). The open question is whether the
remaining 22 features carry an honest edge at this construction. This is the
"is there anything left" question, answered properly or not at all.

## Arms (one run each, no sweeps)

All on the RC-16 corrected matrices, identical folds (label-mature splits,
`bt.DEFAULT_FOLDS`), identical simulator (matrix `pregap_return`), identical
bootstrap (10k week-block resamples, seed 20260807), identical row-key
restriction for the V6/V4 family (the 16,587 common keys).

```text
v6n   v6c matrix, V6 gate trio (frozen HPs: g1 (8,20,3), g2 (12,50,3),
      g3 (1,50,3), 300 trees, lr .05), min-gate 0.33,
      FEATURES = 22 (DEPLOY_FEATURES minus car_drift_historical_q1)
v4n   v6c matrix, single pead_pass classifier (V4 uniform HP), theta 0.20,
      same 22 features
v7n   v7c matrix, V7 uniform-HP gate trio, min-gate 0.33,
      22 features + is_sp400, ADV>=10M + XLF exclusion as in RC-16
```

The min-gate 0.33 and theta 0.20 stay FIXED even though the probability
scale shifts with a changed feature set — recalibration would be tuning.

## Gates (pre-specified)

```text
G1 (V6 family)   v6n mean trade > 0 AND week-block bootstrap CI excludes 0
G2 (V4 family)   v4n passes the same test
G3 (V7 family)   v7n passes the same test
G4 (reporting)   paired week-block diffs vs the RC-16 c-companions
                 (v6n-v6c, v4n-v4c, v7n-v7c): did removing the honest-but-
                 noisy drift feature help or hurt? REPORTING ONLY.
```

Power honesty: ~60-65 weekly observations per arm; the test detects edges of
roughly the magnitude the old contaminated backtests showed (+1.5%/week-block
mean on the weekly scale). "Positive but insignificant" is a FAIL, not a
signal to keep digging.

## Decision rules (pre-committed)

```text
Any gate passes    -> that arm goes to SHADOW (paper, alongside the frozen
                      V6 process book) with a pre-registered promotion review
                      at >= 8 fresh OOS weeks.
All gates fail     -> the 23-feature PEAD construction at this universe/
                      label definition has no honest edge. The PEAD research
                      line PAUSES: no re-tuning of the old feature set, no
                      threshold archaeology, no BMO/AMC rule adoption (that
                      remains a separate future hypothesis). Forward ledgers
                      continue for process data only. Any revival requires
                      NEW features or labels under a new registration.
Closure memo       names the outcome regardless, per doctrine.
```

## Limitations (inherited)

- Corrected matrices still train on current vendor vintages (no point-in-time
  vendor archive); all numbers carry this label.
- Historical folds remain diagnostic — consumed windows stay consumed.
- Single-seed, single-threshold by design: this is a hypothesis test, not an
  optimization.
