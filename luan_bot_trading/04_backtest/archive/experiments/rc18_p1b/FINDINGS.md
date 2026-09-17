# RC-18 P1(b) — families A and D (2026-09-16)

Completing the registered >=2-family P2 test. Same machinery as P1
(beats-only, quintile spreads, week-block bootstrap 10k seed 20260807,
orthogonalized vs six confounders). Truncation-equivalence: 240/240 PASS.

| feature | n (coverage on beats) | raw | ORTHO |
|---|---:|---|---|
| A2 gap_followthrough_hist | 10,254 (96.1%) | +0.57 [+0.02, +1.12] | **+0.56 [+0.005, +1.11] ✓** |
| A1 drift_prop_beat_m4     |  9,952 (93.3%) | −0.17 ns | −0.16 ns |
| D2 sector_peer_car_8w     |  9,964 (93.4%) | +0.34 ns | +0.30 ns |
| D1 sector_pead_rate_8w    |  skipped: discrete-valued (rate of binary pead over few peers) — quintile degeneracy |

A2 = mean(r11 · sign(r1)) over the last ≤8 label-matured prior episodes of
the SAME ticker: "does this ticker extend or fade its day-0 reaction?"
Notable symmetry: this is the honest same-ticker descendant of the RC-16
leak feature — the personality survives, the leak's specific construction
did not. CI lower bound +0.0054: a razor-thin pass.

## P2 gate tally after P1 + P1(b)
  condition 1 (>=2 features from >=2 families, ortho CI excl 0):
      SATISFIED — F1_sb_h3 (family F, [+0.10, +1.19]) and A2 (family A,
      [+0.005, +1.11]). Both marginal but both positive.
  condition 2 (a learned estimator beats taxonomy S-c):
      NOT satisfied under the common-set reading (h=1: S-c wins; h=3: tie
      +0.448 vs +0.440). Satisfied only under the own-domain reading
      (S-b h3 +0.64 CI-positive vs S-c h3 +0.37 ns) — which the coverage
      difference (86% vs 25%) confounds.
