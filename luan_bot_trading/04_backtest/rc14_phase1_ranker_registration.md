# RC-14 Phase 1 Pre-registration — LambdaMART ranker (2026-09-09)

Phase 0 PASSed (capacity 46%, Spearman 0.03-0.08 thin, oracle gap 3.6x).
This document registers the design BEFORE any training.

## Model

- Objective: XGBoost `rank:pairwise` (LambdaMART), group = ISO week.
- Training pairs: weeks with >= 2 ELIGIBLE candidates (eligibility =
  current frozen policy: score_real >= 0.33, XLF-excl, ADV>=10M SP600).
- Relevance label: realized ret_cost (stop-truncated, cost-adjusted)
  as continuous relevance within the week.
- Features (26): the 23 raw + is_sp400 + the three OOS gate
  probabilities (p_g1/p_g2/p_g3, computed by the fold's gate models —
  the current rankers min-gate/mean-gates are functions of these, so
  the learned ranker can absorb or dominate them).
- Hyperparameters FIXED, no sweep (Amendment A lesson): depth 3,
  lr 0.05, n_estimators 300, subsample 0.7, colsample 0.7, seed 42.

## Time discipline

Walk-forward identical to the house folds: for each fold f in
{2024H2, 2025H1, 2025H2, 2026H1}, train the ranker on eligible-week
pairs from periods strictly before f's start; rank f's weeks.
Holdout 2026H1 never touches training.

## Promotion gates (pre-registered; baselines recomputed on identical
## windows at eval time)

- G1 SIGNAL: ranker mean weekly Spearman (weeks with >=2 eligible,
  holdout 2026H1) >= +0.10 (must materially beat mean-gates +0.075 and
  min-gate +0.031; a hair above baseline is noise, not promotion).
- G2 TOP-1: ranker top-1-by-score weekly mean ret_cost on holdout
  >= min-gate top-1 + 1.0pp, AND >= min-gate top-1 on DEV folds.
- G3 SIM: replacing slate ordering in the 108-style portfolio sim
  (4 slots, stop, force-refresh, all frozen) must not degrade the
  aggregate: DEV cost-adj mean >= baseline V7's; holdout mean >=
  baseline's - 0.5pp (noise tolerance).
- KILL: any gate missed -> RC-14 CLOSED, cause of death recorded,
  ad-hoc min-gate ranking retained. No threshold/rank-shopping.

## Honest risk (recorded up front)

Ranking groups are small (~64 competition weeks total, ~2-11 eligible
per week) — pairwise training on tiny groups is high-variance. The
fixed-hyperparameter, no-sweep design is the mitigation; if G1-G3 fail
on variance, that is a legitimate kill, not a reason to iterate.

## Scope

No change to eligibility, gates, thresholds, costs, or the live V6
book. Adoption (if gates pass) is shadow-first: the ranker re-orders
the V7 shadow slate alongside the frozen ordering, outcomes accrue in
the ledger, promotion decided on that record. If Amendment B adopts
first, the ranker retrains/revalidates under B-eligibility before any
use (training distribution would change).
