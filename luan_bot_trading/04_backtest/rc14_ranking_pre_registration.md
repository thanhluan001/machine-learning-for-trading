# RC-14 Pre-registration — Ranking architecture (2026-09-09)

## Motivation

Amendment B decomposed the policy into eligibility (pure PEAD
prediction, no-flag bar) and ranking (flagged score). The ranking pass
uses min-of-gates — an ad-hoc conservative rule inherited from the
gate-decomposition era, never trained to rank. User question: should
ranking be a dedicated trained model (LambdaMART rank:pairwise on
within-week eligible pairs) instead?

## Phase 0 diagnostics (no training; fold-OOS scores, sim-consistent
## filters: XLF-excl, ADV>=10M SP600, costs on returns)

  a) CAPACITY: distribution of weekly eligible counts (eligible =
     score_real >= 0.33, current policy). Slot competition exists only
     when eligible > 4. Also reported for B-eligibility (no-flag bar).
  b) SIGNAL: mean within-week Spearman(score_real, ret_cost) among
     eligible candidates, pooled over weeks with >= 2 eligible.
  c) TRIVIAL ALTERNATIVES: same statistic for g1-only ranking and
     mean-of-gates ranking; plus top-1-by-each weekly mean return vs
     all-eligible mean vs oracle top-1 (upper bound).

## Pre-registered kill gates (evaluated on DEV folds 1-3 pooled with
## holdout 2026H1)

  KILL 1 (capacity): weeks with eligible > 4 are <= 25% of weeks with
       >= 1 eligible  ->  ranking barely matters; close RC-14.
  KILL 2 (headroom): min-gate top-1 weekly mean >= (best trivial
       alternative top-1 mean - 0.5pp) AND mean Spearman(min-gate)
       >= 0.05  ->  current ranker captures the available signal;
       close RC-14.
  PASS to Phase 1: neither kill fires  ->  LambdaMART ranker program
       with full validation bar (Phase 1 design registered separately
       before any training).

## Scope

No policy change in Phase 0. Diagnostic only. The B runway's dual
scores and outcomes continue accumulating regardless.

---

## PHASE 0 RESULT (2026-09-09): PASS to Phase 1

Capacity: 46% of eligible weeks have >4 candidates (mean 11.2/week, max
81); B-eligibility 37%. KILL1 no (competition is real).
Signal: mean weekly Spearman among eligible — min-gate +0.031, g1-only
−0.028, mean-gates +0.075. All thin: eligibility separates good from
bad (all-eligible mean +2.82%), but ordering within the pool is barely
informed. Top-1 means: min-gate +5.41% vs g1 +4.24% / mean-gates
+4.11%; oracle top-1 +19.48% (selection-biased ceiling, but 3.6x headroom).
KILL2 no (Spearman 0.031 < 0.05 — the pre-registered "current ranker
captures the signal" condition FAILED).

Honest caveats: oracle top-1 of up-to-81 candidates is heavily inflated
by selection; n=64 competition weeks; min-gate's top-1 win over
trivial alternatives is small-n.

Phase 1 design to be registered before any training.
