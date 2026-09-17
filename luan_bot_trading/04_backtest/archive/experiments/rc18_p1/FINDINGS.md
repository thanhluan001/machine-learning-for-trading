# RC-18 P1 — Family F diagnostics (2026-09-16)

Beats only (n=10,670), CAR-first quintile spreads (top-bottom, fixed
boundaries), week-block bootstrap 10k seed 20260807, orthogonalized vs
rel_ret_5d/20d, sector_adjusted_ret_20d, revision_momentum_30/60/90d.

## Results (raw | orthogonalized)

| feature | n | raw spread % [CI] | ortho % [CI] |
|---|---:|---|---|
| F1_sb_h3 (SVD sim, 3-day peer partial) | 9,136 | +0.66 [+0.12, +1.21] | **+0.64 [+0.10, +1.19] ✓** |
| F1_sc_h1 (taxonomy SIC-4, day-0) | 3,296 | +1.02 [+0.08, +1.96] | +0.92 [−0.01, +1.86] (misses by 0.015) |
| F1_sc_h3 | 2,638 | +0.37 ns | +0.26 ns |
| F1_sa_h1 / h3 (shrunk) | 6,492/5,450 | −0.17 / +0.08 ns | −0.22 / −0.00 ns |
| F1_sb_h1 | 9,327 | +0.33 ns | +0.33 ns |
| F2/F3/F4 (all) | — | nothing significant | — |

- h=3 truncation-equivalence completed post-run: 120/120 PASS (P0 had
  tested only h=1; the winner is leak-clean).
- Learned vs taxonomy on COMMON defined sets: h=1 S-c WINS (+1.02 vs
  −0.01 S-a; S-b also below). h=3 TIE (+0.45 S-b vs +0.44 S-c, n=2,367).
  S-b's value = coverage extension (31% -> 86% of beats), not superiority.
- Dispersion moderator (F3 interaction): inconsistent across estimators
  (S-c: low-disp 0.78 > high 0.41; S-b: 0.02 < 0.54) — NOT supported.

## P2 gate (as registered)

FAILS as written: (1) only ONE feature (F1_sb_h3) passes the orthogonalized
CI bar, from ONE family (F) — the gate requires >=2 features from >=2
families; (2) "learned beats taxonomy" fails at h=1 and only ties at h=3.

Note: the gate fails on STRUCTURE, not on signal absence — the read-across
concept carries a real orthogonalized effect on its own domain.

## Open items
- Fast/p0 instance-builder count discrepancy (38,324 vs 34,567) unresolved;
  affects only S-a (the weakest estimator), not the F1_sb_h3 result.
