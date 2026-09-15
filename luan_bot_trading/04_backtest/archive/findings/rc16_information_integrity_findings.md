# RC-16 Closure — Information-Integrity Revalidation

**Status:** CLOSED 2026-09-15. All three gates FAIL. Cause of death named below.
**Registration:** `rc16_information_integrity_pre_registration.md` (2026-09-14,
committed before any corrected result). No thresholds tuned, no variant
shopping; frozen HPs and thresholds throughout (V6 trio HPs, V4 uniform HP,
min-gate 0.33, V4 theta 0.20, 10k resamples seed 20260807).

## The verdict

| arm | exec | win% | avg trade | week-block 95% CI |
|---|---:|---:|---:|---|
| base_v6 (frozen mechanics, old contracts, common keys) | 132 | 63.6 | **+3.593%** | [+0.797, +3.033] |
| f1_only (pub-aware macros) | 129 | 63.6 | +4.100% | [+1.002, +3.426] |
| **f2_only (45-session drift + maturity mask)** | 122 | 47.5 | **−1.432%** | [−1.490, +0.169] |
| f3_only (label-mature folds) | 127 | 62.2 | +4.080% | [+1.053, +3.471] |
| f4_only (unlabelled-not-negative) | 130 | 63.8 | +3.619% | [+0.788, +3.075] |
| **v6c (all fixes)** | 124 | 44.4 | **−1.914%** | **[−1.744, −0.067]** |
| v4c (V4 classifier, corrected data) | 74 | 50.0 | +0.481% | [−1.214, +1.691] |
| v4_base (old data, context) | 93 | 54.8 | +2.243% | — |
| v7c (combined universe, corrected) | 131 | 55.0 | +0.537% | [−0.844, +1.404] |

```text
G1  FAIL  v6c mean trade negative AND CI excludes 0 on the NEGATIVE side.
G2  FAIL  v6c (−1.914%) does not beat v4c (+0.481%); paired weekly diff
          −1.46pp, CI [−2.957, +0.019].
G3  FAIL  v7c CI includes 0; v7c-vs-v6c paired diff +1.07pp,
          CI [−0.153, +2.404] — v7c is *less damaged* than v6c but not
          positive-significant.
```

## Cause of death (G4 one-factor attribution)

```text
total delta (base -> v6c):  −5.51pp avg trade
  F2  car_drift maturity mask + 45-session window:  −5.03pp  (91.2%)
  F1  macro publication joins:                       +0.51pp
  F3  label-mature fold splits:                      +0.49pp
  F4  unlabelled-not-negative:                       +0.03pp
```

The historical V6 edge was carried by ONE feature: `car_drift_historical_q1`.
Its old 60-session window silently used bars that did not exist at the
feature cutoff (27.8% of rows chronically; every filled-by-future row is a
leak). With an honest window the feature does not merely lose informativeness
— the gate trio's selections underperform (win rate 63.6% → 47.5%). The
audit's variant-E sign-flip instability was this same mechanism seen from
outside: the "edge" lived in the leak, and any correction to the leak removes
it.

The other three fixes are mildly POSITIVE — the contracts are right and stay:
publication-aware macros, label-mature folds, and unlabelled handling each
add ~0.5pp. The information-integrity program worked exactly as designed:
it found which part of the edge was real (none of it, at this construction)
before more capital or research was spent on it.

## Pre-registered consequences (executing as written)

- Frozen V6 paper book CONTINUES as **process data only** (its ledger is a
  forward chronology test of process, not of edge). No promotion reviews.
- v6c / v7c / v4c are NOT promoted to shadow. phase_g_v6c_rc16 /
  phase_g_v7c_rc16 model dirs remain research artifacts.
- Live script semantics stay on the corrected contracts (F1 live since
  2026-09-14, F2 live since 2026-09-15): forward ledgers score under honest
  windows. The V6/V4/V7 paper picks from here on are the corrected-contract
  picks — if any future edge exists, it must show up in these forward
  ledgers, not in rebuilt history.
- Amendment B (V7 dual-scoring) runway: the v7c-vs-v6c paired diff (+1.07pp)
  is noted but NOT actionable — its base arm failed G1.
- Reopening drift-type features requires a NEW registration: shorter honest
  windows (e.g. 20-30 sessions), maturity-masked by construction, evaluated
  one-factor against the no-drift baseline. No such experiment is run under
  RC-16.

## Mechanics notes (for auditors)

- All V6-family arms restricted to the 16,587 common (permaTicker,
  report_date) keys so run differences are contract effects only.
- One-factor arms: in-memory column swaps from v6c onto the old matrix
  (F1: vix/fed_funds/unemployment_roc21; F2: car_drift_historical_q1),
  label_end derived exactly as T+11 IJH sessions for the old matrix (F3),
  gate-input validity filter for F4 (102 rows dropped of 16,587).
- Simulator: matrix `pregap_return` (stop embedded) for every arm;
  v7c uses its own entry/exit/pregap_return (labels net of side costs;
  no resim-with-stop — consistent across arms).
- Artifacts: `gauntlet.json`, `executed.h5` (per-arm ledgers) in
  `archive/experiments/rc16_r3/`; retrains in `03_model/models/*_rc16`;
  delta reports in `archive/experiments/rc16_r1/report.json`.
