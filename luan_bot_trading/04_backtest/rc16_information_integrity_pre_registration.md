# RC-16 Pre-Registration — Information-Integrity Contract Fixes and Revalidation

**Status:** REGISTERED 2026-09-14, before any corrected result is computed.
**OUTCOME:** CLOSED 2026-09-15 — G1/G2/G3 all FAIL. Cause of death:
car_drift_historical_q1 information artifact (F2 = 91.2% of damage).
See archive/findings/rc16_information_integrity_findings.md.
**Trigger:** independent audit (`PEAD_TRADING_AUDIT_EN.md`, 2026-09-14).
Four confirmed data-contract defects + one packaging defect. This
registration pre-specifies the fixes, the retrain protocol, the
evaluation gauntlet, and the decision rules. No corrected number is
inspected before this document is committed.

**Scope:** V6 (SP400 home) and V7 (combined) — both inherit the defects.
**Live policy during RC-16:** frozen V6 continues the paper book
unchanged (its ledger is a forward chronology test of process).
Corrected models enter shadow first, per doctrine.

---

## 1. The four contract fixes (exact semantics)

```text
F1  MACRO PUBLICATION AWARENESS
    All macro joins (UNRATE, DFF, VIXCLS) switch from observation-date
    backward joins to PUBLICATION-date joins. Implementation: FRED
    release-date tables (or ALFRED vintages) per series; a value is
    usable at cutoff C only if first_released(C0) <= C. UNRATE roc21
    keeps its 21-observation definition over released observations.
    Affects training AND live scoring identically.

F2  LAGGED-CAR MATURITY MASK
    car_drift_historical_q1 is NaN unless the prior event's 60-session
    window COMPLETED on or before the feature cutoff C. No partial
    windows, no future bars. (Audit: 338/773 H1 rows violated this.)

F3  LABEL-MATURITY PURGE IN FOLDS
    Fold splits move from report_date to label-end date (report_date +
    max(label horizon) = T+12 sessions). A fold's training set contains
    only events whose labels completed before the fold's test window
    begins. Applied identically in DEV/sweep/test partitions.

F4  MISSING-FUTURE LABEL HANDLING
    Events with missing future observations for a gate input become
    UNLABELLED for that gate (excluded from that gate's fit), never
    negative. (Audit: 185 rows.) Document count per matrix.

PACKAGING (non-modeling): skeleton README feature count (24 vs 23),
T+5-label wording, missing-helper references — corrected in docs only.
```

## 2. Retrain protocol (the answer to "do we retrain")

```text
R1  Rebuild matrices with F1-F4: /features/train_matrix_v6c,
    /features/train_matrix_v7c. Report row/cell deltas vs originals
    (expected scale: thousands of macro cells, ~4.8k CAR cells,
    ~185 label exclusions, fold-boundary shifts).

R2  Retrain gate models on corrected matrices with FROZEN
    hyperparameters (V6: gamma/mcw/depth (8,20,3)(12,50,3)(1,50,3),
    300 trees, lr .05, seed lineage; V7: uniform V4_HP). NO re-search,
    NO threshold change (V6 0.33; V7 0.33). This isolates the contract
    effect: same procedure, same parameters — only the information
    contracts change.

R3  The comparison is pre-specified as the ONLY experiment family:
      (a) v6c vs frozen-V6 on identical folds (contract-fix effect)
      (b) v6c vs V4c baseline (does the gate decomposition survive)
      (c) v7c vs v6c (does universe expansion survive)
    No other comparisons, no sweeps, no BMO/AMC rule adoption (the
    audit's BMO/AMC split is recorded as a future hypothesis, not a
    rule; testing it requires its own registration).
```

## 3. Evaluation and decision rules

Historical folds (label-purged) are DIAGNOSTIC — H1 2026 and all prior
windows are consumed by prior inspections (audit finding; we concur).
The only clean tests are the forward ledgers. Pre-registered gates:

```text
G1 (contract survival)  v6c DEV mean trade > 0 AND bootstrap CI
                        (week-block, 10k resamples) excludes 0 by the
                        same standard applied to frozen V6.
G2 (decomposition)      v6c beats V4c on mean trade AND paired
                        week-block CI of the difference excludes 0.
G3 (universe, V7 only)  v7c satisfies G1/G2 analogues.
G4 (damage audit)       The v6c-vs-frozen-V6 delta is decomposed:
                        which fix (F1/F2/F3/F4) contributed what, via
                        one-factor-at-a-time rebuilds. Registered as a
                        reporting requirement, not a selection step.

Verdicts:
  G1+G2 pass -> v6c to shadow (paper, alongside frozen V6) with a
      pre-registered promotion review at >= 8 fresh OOS weeks.
  G1 fails   -> the historical edge was substantially an information
      artifact; V6 live paper continues for process data only; program
      returns to research with the audit's framework as the standard.
  Any gate   -> closure memo names the cause, per doctrine.
```

## 4. Stated limitations (inherited from the audit)

- Corrected results still train on CURRENT vendor vintages — not the
  historical as-known dataset (no point-in-time vendor archive exists
  for earnings/grades). All corrected numbers carry this label.
- The audit's variant-E sign-flip is one draw of a chaotically
  sensitive correction; RC-16's one-factor decomposition (G4) is the
  disciplined version of that question.
- FLEX-class calendar discrepancies (issuer actual vs vendor record)
  remain unrepairable historically; only forward logging fixes them
  (the nightly scripts already log decision-time events; RC-16 adds
  decision-time score/candidate snapshots to the ledgers, per the
  audit's "log candidates at decision time" requirement).

## 5. Operational changes concurrent with RC-16

```text
- Nightly candidates.json already persists all scored events;
  ADD decision-time macro-as-joined values per event (publication-
  aware from day one of F1's live implementation).
- Documentation language: scores are selection statistics, never
  probabilities; synthetic weekly curves are labeled synthetic; the
  stop is fixed, not trailing. (Audit language corrections adopted.)
```
