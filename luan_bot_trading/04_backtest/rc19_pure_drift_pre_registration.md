# RC-19 Pre-Registration — Pure-Drift Label Construction

**Registered:** 2026-09-16, before any RC-19 computation. Unamendable gates.

## Motivation (all on record before this doc)

1. Threshold probe (`_rc18_threshold_probe.py`, 14881e0): label-return
   decoupling — min-gate score predicts the composite 3-gate label
   (PEAD rate 6.2%→15.2% monotone across deciles) but not returns
   (top-decile mean CAR −1.50%; spearman(score, car_10d) ≈ −0.02).
2. RC-18 P3 (355fdfd): two leak-tested CI-positive features moved avg
   trade +1.21pp yet could not clear zero — because the label they serve
   conflates drift (g1) with post-print path conditions (g2 volume,
   g3 drawdown) that are outcomes, not payments.
3. RC-17 (06d4fcf): v6n baseline −1.581% avg trade — reproduced again
   inside RC-18 P3.

## Hypothesis

The 3-gate composite label is predictable but is not the quantity the
book is paid on. A single classifier on the pure drift target
(car_10d > +3%), fed by the 22 retired features plus the two RC-18
survivors, will outperform v6n and be non-negative — with the explicit
caveat that ~65 weekly observations can only certify edges of roughly
+1.5pp avg trade or larger. A fail certifies "uncertifiable," not "zero."

## Frozen specification

```text
matrix          /features/train_matrix_v6c restricted to the 16,587
                common keys (identical to RC-16/17/18 — baseline continuity)
label (PRIMARY) pass_g1 = car_10d > +0.03  — the protocol's registered
                drift definition since inception; NOT a new threshold,
                deliberately the old g1 to avoid archaeology
model           ONE classifier (V4 architecture), score = predicted
                probability; selection threshold 0.33 (V6-era convention)
hyperparams     frozen V6 g1 HPs: gamma=8, min_child_weight=20,
                max_depth=3, n_estimators=300, lr=0.05, subsample=0.7,
                colsample=0.7, seed=42
features        22 retired + F1_sb_h3 + A2 = 24 (both truncation-tested:
                120/120, 240/240)
folds           DEFAULT_FOLDS partitioned on label_end (maturity-honest)
simulator       pregap_return, N_SLOTS=4, select_weekly, EXCLUDE_SECTORS
                — identical to RC-16 R3 / RC-17 / RC-18 P3
baseline        v6n (22 features, gate trio, min-gate 0.33), re-run in the
                same script for exact pairing
bootstrap       week-block 10,000 draws, seed 20260807

SECONDARY (declared non-gating diagnostics — reported, never gated):
  labels car_10d > +0.001 (cost) and car_10d > 0: same 24 features,
  same HPs; stats only
  decoupling check: spearman(score, car_10d) + decile table for the
  primary arm (did the redesign actually fix decoupling?)
```

## Gates (unamendable)

```text
G1  rc19 avg trade > 0 AND week-block CI excludes 0
G2  rc19 beats v6n: avg trade greater AND paired weekly diff CI excludes 0
    (56+ paired weeks)

PASS  -> user decides on shadow paper book; promotion review after
         >= 8 fresh OOS weeks (same rule as RC-18's amendment)
FAIL  -> RC-19 closes; closure memo names the cause; NO threshold/HP
         archaeology, no soft landing (the power caveat above is an
         expectation registered in advance, not an escape hatch)
```

## What this program deliberately does NOT do

- No sweep of label threshold, score threshold, HPs, or hold length.
- No BMO/AMC changes, no slot changes, no exclusions changes.
- No peeking at interim arms; all arms run once, verdict computed from
  the primary arm only.

Execution: `rc19_pure_drift.py`, artifacts to `archive/experiments/rc19/`.
