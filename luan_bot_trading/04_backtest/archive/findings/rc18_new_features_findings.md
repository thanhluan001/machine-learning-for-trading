# RC-18 Closure — New Feature Construction (Firewall-Protected)

**Status:** CLOSED 2026-09-16. P3 gates G1/G2 FAIL. Cause of death named below.
**Registrations:** `rc18_new_features_pre_registration.md` (families A-E),
Amendment 1 (family F, learned PEAD-behavior similarity), Amendment 2
(post-hoc, user-approved re-specification of P2 condition 2). All committed
before the corresponding computations. Firewall (P0) passed throughout:
truncation-equivalence 300/300 + 120/120 + 240/240; negative control
detected every real injected leak.

## What was found (real, at the diagnostic level)

```text
F1_sb_h3  SVD-behavior-similarity read-across, 3-day peer partial:
          +0.64pp car_10d quintile spread, CI [+0.10, +1.19], 86% coverage,
          orthogonalized vs 6 momentum/sector/revision confounders.
A2        same-ticker follow-through personality (extend-vs-fade):
          +0.56pp, CI [+0.005, +1.11], 96% coverage, orthogonalized.
```

Both are the first CI-positive, leak-tested, orthogonalized feature-level
associations found in this program since the RC-16 reset. Notably they are
the honest descendants of the two constructs RC-16 killed: cross-sectional
read-across (the peer version of car_drift) and same-ticker drift
personality (the statistic version of car_drift). The *ideas* survived;
only the stolen measurements died.

## P3 result (the decisive test)

```text
                       trades  win%   avg trade   NAV     raw precision
v6n   (22 features)      129   44.2   −1.581%    −42.8%     16.3%
v6c18 (22 + F1_sb_h3     142   53.5   −0.372%    −17.7%     23.2%
       + A2)

G1  FAIL  avg trade −0.372% < 0; week-block CI [−1.23, +0.82] includes 0
G2  FAIL  paired weekly diff +0.585pp, CI [−0.50, +2.40] includes 0
          (56 paired weeks; point estimate favors the new features)
```

(v6n baseline reproduced RC-17's numbers exactly — clean replication.)

## Cause of death

Not the features — the construction and the power.

1. **The base remains negative.** The 22-feature construction sits at
   −1.58% avg trade; two features with ~+0.6pp diagnostic spreads lifted
   it +1.21pp to −0.37% but cannot carry a negative base above zero. The
   deeper defect stands from the threshold probe: the 3-gate label is
   predictable but does not pay (top score decile: 15.2% PEAD rate,
   −1.5% mean CAR). Until the LABEL is redesigned, feature additions are
   re-arranging furniture in a negative-expectancy room.
2. **Power.** ~60-65 weekly observations can only certify large edges. A
   +1.2pp avg-trade improvement — large by any practical standard — is
   statistically indistinguishable from zero at this sample. Same wall
   RC-14 hit: the weekly granularity of a 4-slot book is the binding
   constraint on what can ever be proven.

## What survives, and the honest door forward

- The two features, their specs, and the firewall machinery are archived
  and reusable; any future construction starts from them, not from zero.
- The label-redesign hypothesis (predict CAR>cost directly, or drop the
   volume/drawdown gates from the LABEL while keeping them as features)
   remains unregistered and untested — it is the deepest open question,
   and the threshold probe motivated it before RC-18 began.
- Optional, doctrine-compatible next step: a nightly FEATURE LEDGER —
  log F1_sb_h3 / A2 values at decision time (no trading, no promotion),
  accumulating clean forward OOS evidence at zero cost so any future
  program starts with the sample size this one lacked.

Per the amendment: no shadow, no promotion review. Live books unchanged
(frozen V6 = process data only).
