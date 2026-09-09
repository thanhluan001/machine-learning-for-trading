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

---

## AMENDMENT (2026-09-10, registered before any training; user design review)

1. ACTIVATION: ranker scores all eligible candidates weekly; decision
   impact only under slot competition (first-order: eligible > 4 —
   46% of eligible weeks, A-eligibility). G3's full sim prices the
   exact interaction including mid-week slot freeing.
2. FEATURE ARMS (fixes the 26/27 arithmetic error; reopens the
   gate-prob inclusion as an explicit fork):
   - ARM P (pure): 24 inputs = 23 raw + is_sp400.
   - ARM G (gates-in): 27 inputs = ARM P + p_g1/p_g2/p_g3 (OOS-fold
     scores; causal chain verified).
   - PRE-STATED TIEBREAK: both arms run once, identical protocol and
     gates. Both pass -> ARM P adopts (parsimony). One passes -> that
     arm. Neither -> RC-14 closed. No post-hoc arm preference.
   Training pairs: weeks with >=2 eligible (~64), relevance = ret_cost.

---

## PHASE 1 RESULT (2026-09-10): NEITHER PASSES — RC-14 CLOSED

Arm P: holdout Spearman −0.047 (gate ≥ +0.10), top-1 +3.86% vs min-gate
+7.18%. G1/G2/G3 all FAIL.
Arm G: holdout Spearman −0.103, top-1 +2.67%. All FAIL — gate-prob
features AMPLIFIED the overfit, not the signal.
Training reality: ranker for fold 2 had 90 events / 10 weeks; holdout's
had 579 / 45. Within-week realized returns as relevance = extreme label
noise on tiny groups.

Cause of death: learned ranking infeasible at current data scale —
label noise overwhelms the weak within-pool ordering signal; both arms
anti-informative out-of-sample. The pre-registered "dies on variance =
legitimate kill" clause fired exactly as written.

Salvage finding: the incumbent min-gate ordering is STRONGER than
Phase 0 suggested — on competition weeks: DEV Spearman +0.156, top-1
+8.55% vs pool mean +2.82% (Phase 0's +0.031 was a pooling artifact).
The ad-hoc ranker keeps the job, now with evidence it deserves it.
Reopening condition: materially more competition weeks (multi-year
ledger) AND a demonstrated stable within-pool signal.
