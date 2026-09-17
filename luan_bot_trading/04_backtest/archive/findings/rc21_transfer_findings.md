# RC-21 Closure — Pure Transfer (Strongest Result Yet, Uncertified by 0.33pp)

**Status:** CLOSED 2026-09-16. T1 FAIL per registration. No promotion.
**Spec:** `rc21_transfer_pre_registration.md` (9f14a85). Firewall: 10/10
truncation spot-check PASS on SP600 rows (combined-pool recompute; one
index bug found and fixed during checking — the checker caught its own
bug, which is the checker working).

## Result

```text
TRANSFER — SP400 rc19 calibration applied UNCHANGED to SP600:
  193 trades | 50.8% win | +2.060% avg trade | NAV +115.0% (91 weeks)
  weekly mean +1.092% | week-block CI [−0.331, +2.618]
  g1_rate of picks 33.7% (population 28.6%) | picks>=0.33: 2,431 raw

T1: avg > 0 ✓ but CI includes 0 ✗  → FAIL (lower bound −0.331)

SANITY — same fold models on SP400 test slices: 1,031 picked, +0.118%
avg — consistent with RC-19/RC-20 home-turf behavior (small positive).
```

## Cause of death (named per registration)

**Positive-but-uncertified.** Not negative, not zero: the SP400
calibration transfers — and transfers WELL. +2.06% avg trade on 193
trades over 91 weeks is the largest and most persistent point estimate
of the entire honest program, and its CI lower bound (−0.33) came
within 0.33pp of certification. But the gate is the gate. Fourth
consecutive positive-point-estimate/CI-includes-zero.

## What the last four programs jointly establish

```text
construction                       avg trade   CI            weeks
SP400, pure drift (RC-19)           +0.718    [−0.69,+1.40]    79
COMB,  mixed training (RC-20)       +0.253    [−1.25,+1.55]    75
SP600, TRANSFERRED SP400 model      +2.060    [−0.33,+2.62]    91   ← best
label effect vs 3-gate              +1.6 to +2.3pp, 3-for-3, never reversed
```

1. The pure-drift label redesign is directionally certain.
2. The SP400 calibration does NOT overfit SP400 — it transfers; SP600
   is the stronger universe under it (small caps drift slower = more
   PEAD to capture, consistent with the literature's size effect).
3. No single historical test can certify: power is exhausted.

## Why no amendment is offered this time

RC-18's amendment re-specified a condition whose intent the data showed
was mis-specified. T1 is not that: it is exactly the right question,
and the result is exactly what the gate is for. Amending now would be
gate-shopping. The clean path forward requires NO gate arithmetic:

**Pre-registered forward shadow book** — freeze the SP400 rc19
calibration (fold-4 weights or the exact fit protocol), run nightly in
paper on both universes, log picks forward, pre-register the evaluation
(e.g. >= 26 fresh OOS weeks, avg trade > 0 with week-block CI
excluding 0) BEFORE the first shadow trade. New data only; no history
is re-read. This converts the 0.33pp miss into a waiting problem.

Artifacts: `archive/experiments/rc21/gauntlet.json`, `executed.h5`.
