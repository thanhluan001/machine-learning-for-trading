# RC-23 Pre-Registration — Momentum-Conditioned High-Confidence Beat (Excess Returns)

**Registered:** 2026-09-16, before any RC-23 computation. Unamendable gate.

## Motivation (all on record)

The beat model (OOS AUC 0.673) has zero EXCESS expectancy at p≥0.8
(+0.022%, t=0.07 — the raw edge was market beta; see
`market_adjustment_findings.md`). Exactly one conditioning survived the
market adjustment in the exploratory analysis: strong pre-event momentum.

```text
exploratory estimate: p≥0.8 & rel_ret_20d high tercile
  n=302, excess +0.969%, week-block t=1.36, CI [−0.43, +2.34]
```

## Frozen rule

```text
1. BEAT MODEL (identical to the exploratory run — no changes)
   XGBoost: 22 deploy features (DEPLOY_FEATURES minus car_drift),
   objective binary:logistic, n_estimators=300, lr=0.05, max_depth=3,
   min_child_weight=20, gamma=8, reg_lambda=1.0, subsample=0.7,
   colsample_bytree=0.7, seed=42
   label: beat  = sue_score>0 (SP400, v6c join)
                | actual>estimate (SP600, earnings_full)
   folds: DEFAULT_FOLDS on label_end — train ≤ sve, test in (sve, tse]
   score: p_beat = predicted P(beat)

2. SELECTION (frozen constants)
   p_beat ≥ 0.80  AND  rel_ret_20d ≥ 0.0369
   (0.0369 = 67th percentile of rel_ret_20d within the OOS p≥0.8 bucket,
   computed once 2026-09-16; XLF excluded as always)

3. TRADE AND RETURN MEASUREMENT
   enter at matrix entry_date close, exit at exit_date (5-session hold)
   EXCESS return = stock raw return − own benchmark return over the
   IDENTICAL window (IJH for SP400, IJR for SP600), recomputed from
   prices. pregap_return (raw, with long-side stop) is NOT the metric.
```

## Gate (unamendable)

```text
G1  mean EXCESS return per trade > 0 AND week-block bootstrap CI
    (10,000 draws, seed 20260807) excludes 0

PASS -> candidate for a forward shadow book (parameters were derived from
        these same OOS data, so PASS alone is NOT deployable evidence;
        forward confirmation required)
FAIL -> candidate closed; no conditioning variants, no threshold search
```

## Declared non-gating diagnostics

Raw-return mean; unconditioned p≥0.8 baseline; beat rate of picks;
per-universe split; n.

## Power (stated before the run)

Expected n ≈ 302; per-trade excess SD ≈ 13–15% → SE ≈ 0.8pp; detection
floor for t=2 is ≈ 1.6pp/trade. The exploratory estimate (+0.97%) is
below the floor: an inconclusive FAIL is the expected outcome unless the
true effect is materially larger than estimated. This is registered as
an honest test, not an expected pass.

Execution: `rc23_momentum_beat.py`, artifacts to `archive/experiments/rc23/`.
