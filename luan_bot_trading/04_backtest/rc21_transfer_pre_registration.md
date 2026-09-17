# RC-21 Pre-Registration — Pure-Transfer Test: SP400 rc19 Calibration Applied Unchanged to SP600

**Registered:** 2026-09-16, before any RC-21 computation. Unamendable gates.

## Purpose

RC-20 showed universe-mixing destabilizes the SP400 calibration
(SP400-exec −1.245%) while the SP600 side was strong (+1.513%) under a
mixed-trained model. RC-21 answers the one clean hypothesis left
standing WITHOUT mixing: does the SP400-trained rc19 model, applied
unchanged, carry an edge to SP600?

## Frozen specification

```text
training side   /features/train_matrix_v6c restricted to the 16,587
                common keys (identical to RC-19), pass_g1 labels,
                24 features (22 retired + F1_sb_h3 + A2), frozen g1 HPs
                (gamma=8, mcw=20, depth=3, 300 trees, lr=0.05,
                subsample=0.7, colsample=0.7, seed=42)
fold protocol   per DEFAULT_FOLDS on label_end: train SP400 rows with
                label_end <= sve; apply to SP600 rows in (sve, tse].
                No SP600 row ever enters training. Model refit per fold
                exactly as RC-19's machinery.
application     /features/train_matrix_sp600_pt_v7c (17,038), features
                computed by the RC-20 declared pipeline (T:=report_date;
                is_bmo from earnings_full time; beat actual>estimate;
                IJR drift benchmark; embedding/A2 identical code),
                score=p1, selection threshold 0.33, N_SLOTS=4,
                select_weekly, EXCLUDE_SECTORS
bootstrap       week-block 10,000 draws, seed 20260807
firewall        10-event truncation spot-check on SP600 rows (same
                machinery that passed 30/30 in RC-20); any mismatch
                aborts
```

## Gates (unamendable)

```text
T1  transferred SP600 avg trade > 0 AND week-block CI excludes 0

PASS  -> user decides shadow; promotion review >= 8 fresh OOS weeks
FAIL  -> closes; closure names whether SP600 under the SP400
         calibration is negative, zero, or positive-but-uncertified
```

## Declared non-gating diagnostics

Executed count, win rate, g1-rate of picks, per-fold stats; comparison
to rc20's SP600-executed (+1.513%) and v7n's SP600-executed (−0.462%)
as context only; SP400-side replication of the same fold models on the
SP400 test slices (sanity: should resemble RC-19's behavior).

Execution: `rc21_pure_transfer.py`, artifacts to `archive/experiments/rc21/`.
