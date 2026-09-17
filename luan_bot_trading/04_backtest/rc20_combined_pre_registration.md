# RC-20 Pre-Registration — Combined-Universe Replication of the RC-19 Stack

**Registered:** 2026-09-16, before any RC-20 computation. Unamendable gates.

## Purpose

Power extension + transfer test in one registered run. RC-19 produced the
program's first positive expectancy (+0.718% avg trade, +21.4% NAV, 79
weeks) but its CI ([−0.685, +1.400]) could not exclude zero. RC-20
replicates the EXACT frozen RC-19 stack on the combined SP400+SP600
matrix (~2x events, more paired weeks). Replication certifies or kills;
nothing else changes.

## Frozen specification (identical to RC-19 where possible)

```text
matrix          /features/train_matrix_v7c (33,598 rows) — FULL, both arms
                on the same matrix (no old-matrix restriction: no
                cross-matrix comparison is being made)
arm rc20 (PRI)  24 features (22 retired + F1_sb_h3 + A2), SINGLE pass_g1
                classifier, score=p1, selection threshold 0.33
arm v7n (base)  22 features, 3-gate trio, min-gate 0.33 (paired in-script)
hyperparams     frozen V6 g1 HPs (gamma=8, mcw=20, depth=3, 300 trees,
                lr=0.05, subsample=0.7, colsample=0.7, seed=42); v7n uses
                the frozen per-gate trio as in RC-16/17
folds           DEFAULT_FOLDS partitioned on label_end
simulator       entry_date/exit_date/pregap_return from v7c, N_SLOTS=4,
                select_weekly, EXCLUDE_SECTORS
bootstrap       week-block 10,000 draws, seed 20260807
```

## Declared deviations, SP600 feature side only (labels/simulator untouched)

The SP600 matrix lacks event-level T/is_bmo/sue. Feature computation
uses:

1. **T := report_date** for SP600 rows (on the SP400 side, stored T
   equals report_date 99.4% of the time; positions resolved on the IJH
   calendar).
2. **is_bmo** from `/sp600/earnings_full/{ticker}` time field, joined via
   the matrix's ticker column. Missing time -> feature NaN for that row
   (no assumption made).
3. **Peer beat flag**: SP400 events `sue_score > 0` (as registered in
   RC-18); SP600 events `actual > estimate` (both non-NaN). The two
   definitions coexist in one peer pool / embedding — declared, not
   silently unified.
4. **Drift benchmarks**: IJH for SP400 stocks, IJR for SP600 stocks
   (each universe's own benchmark; the embedding's weekly
   cross-sectional centering removes systematic level differences).
5. **Coverage reported**; XGBoost consumes NaN natively.

## Firewall carry-over

F1_sb_h3/A2 passed truncation-equivalence on the SP400 side (120/120,
240/240). RC-20 adds a **30-event truncation spot-check on the combined
universe** (prices truncated at each consumer's cutoff; F1 must
reproduce exactly). A single mismatch = run aborted, no verdict.

## Gates (unamendable)

```text
G1  rc20 avg trade > 0 AND week-block CI excludes 0
G2  rc20 beats v7n: avg trade greater AND paired weekly diff CI excludes 0

PASS  -> user decides shadow paper book; promotion review >= 8 fresh
         OOS weeks
FAIL  -> RC-20 closes; closure names whether the SP400 point estimate
         failed to replicate or transfer is the fault; no archaeology
```

## Declared non-gating diagnostics

Per-universe executed splits (SP400 vs SP600 trades/win/avg), feature
coverage by universe, executed-count comparison vs RC-19's 158.

Execution: `rc20_combined_replication.py`,
artifacts to `archive/experiments/rc20/`.
