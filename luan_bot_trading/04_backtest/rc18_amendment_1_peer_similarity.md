# RC-18 Amendment 1 — Family F: Learned PEAD-Behavior Similarity (Read-Across)

**Status:** FROZEN 2026-09-16 (user-approved; committed before any computation). Supersedes nothing; extends `rc18_new_features_pre_registration.md`
with a new family F. All firewall rules (F0a/b/c) apply unchanged.

**Trigger:** (a) the RC-18 threshold probe found the score predicts the 3-gate
*label* (PEAD rate 6%→15% across deciles) but not the *return* (top decile mean
CAR −1.5%); (b) the user's design insight: the missing variable is the market's
*outlook* read, which is partly observable through peers that already reported;
(c) spec arithmetic showing the naive "peer drift over 1-2 weeks" is infeasible
(0.0% coverage at K=10 — a peer's completed 10-session drift cannot exist in a
1-2 week window).

## 1. The construct (definitional, so implementation cannot drift)

```text
SIMILARITY IS LEARNED, NOT TAXONOMIC.
  Two stocks are similar if their POST-BEAT early drifts historically co-move.
  Industry is an INPUT/PRIOR to the similarity, never the definition.
  ("Same industry helps but not always.")

FEATURE (observable at my decision time C_i = T_i - 1 AMC / T_i - 2 BMO):
  the similarity-weighted EARLY drift of peers j that already reported AND
  BEAT, within the trailing window K (1-2 weeks), where the peer's early
  window has fully printed by C_i.

TARGET: my own early drift (excess CAR over [T_i+1, T_i+h]), conditional on
  my beat — where "my beat" is the part the 22 existing features predict.

LEARNING DATA (this is where completion lives): historical pairs
  (peer early drift, my early drift), all windows matured before the
  consuming cutoff. Similarity is a MODEL and obeys fold discipline.
```

Two clocks, never conflated:

```text
peer signal  -> as FRESH as possible (day-0/1). Completion NOT required.
similarity   -> as MUCH HISTORY as possible, used point-in-time.
                Completion REQUIRED (labels for learning must be matured).
```

## 2. Firewall application (F0a/b/c, unchanged)

```text
F0a  PROVENANCE SPEC per feature: peer eligibility (beat + window K),
     observability inequality T_j + h <= C_i, similarity training window,
     and the overlap check vs the label window [T_i+1, T_i+11].
F0b  TRUNCATION-EQUIVALENCE extended to the similarity estimator itself:
     recompute both w_ij and the feature from history truncated at C_i;
     must equal the full-history value exactly. (>=100 seeded events.)
F0c  DECLARED PARTIALS: h=1 and h=3 windows are partial by design; each
     feature states its h, and no undeclared partial is permitted.
     Print-time data still forbidden (the peer's print IS allowed: it
     happened before my cutoff — that is the entire point).
```

Coverage arithmetic must be recomputed per exact gap requirement at P0
(the earlier run measured "any peer in window", gap >= 1):

```text
h=1 (day-0 reaction)   needs gap >= 3 sessions (AMC)
h=3 (3-session partial) needs gap >= 5 sessions
reference (gap>=1, K=10): SIC-4 43.9% | SIC-3 53.6% | SIC-2 76.5% | sector 96.5%
```

## 3. Features (Family F)

Notation: r_j = peer j's excess CAR over [T_j+1, T_j+h]; w_ij >= 0 similarity.

```text
F1  sim_peer_drift    Sum_j w_ij r_j / Sum_j w_ij          PRIMARY (signed
                      weighted mean = net direction x magnitude)
F2  sim_peer_agree    Sum_j w_ij sign(r_j) / Sum_j w_ij     consensus in [-1,1]
F3  sim_peer_disp     Sum_j w_ij |r_j - F1| / Sum_j w_ij    disagreement
F4  sim_peer_n        (Sum w)^2 / Sum w^2                    effective # peers
V1  sim_peer_drift_h3 same as F1 with h=3 (declared partial)
V2  sim_peer_drift_x  extremity/attention weighting:
                      w'_ij = w_ij * |r_j| (most informative peer dominates)
V3  sim_peer_drift_m  same as F1 including MISS peers, weighted by their SUE
                      (misses are off-distribution for a beat-similarity, so
                      this is a variant, not the primary)
```

### 3.1 The disagreement case (NFLX +4%, MSFT −2%, GOOGL Thursday)

This is the design decision the example forces, and the answer is: **do not
collapse it into a single average and hide it.**

```text
equal-weight mean      -> +1% net  (a confident-looking number from an
                                    incoherent peer set -- wrong)
similarity-weighted    -> the WEIGHTS decide: if GOOGL's post-beat drift
                          historically co-moves with MSFT's more than
                          NFLX's, the net is NEGATIVE. This is exactly what
                          learned similarity is for; taxonomy (both NFLX and
                          GOOGL are Communication Services) would get it
                          wrong by construction.
exposed statistics     -> F1 (net), F2 (agreement = 0.0 here), F3 (dispersion
                          = high), F4 (n = 2). The model then learns the
                          MODERATOR: high dispersion -> firm-specific news,
                          industry read-across unreliable -> shrink.
```

PRE-REGISTERED HYPOTHESIS (P1 diagnostic): peer disagreement (F3) moderates
the F1→drift relationship; specifically, the F1 effect is weaker when F3 is
high. Test: interaction term, week-block CI. This is a hypothesis, not an
assumption — if it fails, F1 stands alone.

Aggregation variants are capped at TWO schemes (F1+F2+F3+F4 as primary;
V2 extremity weighting as the alternative). No hand-crafted aggregation
search.

## 4. Similarity estimators (capped at three, pre-specified)

```text
S-c  TAXONOMY BASELINE (the thing to beat): w_ij = 1{same SIC-4} equal
     weights. Cheap, no estimation, no leak risk.

S-a  SHRUNK CO-MOVEMENT: sample correlation of post-beat early drift over
     joint matured events, shrunk toward a taxonomy prior:
        w = (n/(n+k)) * corr_ij + (k/(n+k)) * prior_ij,   prior = 1{same SIC-2}
     k fixed (e.g. 5) BEFORE results; sparse top-k per stock (k_top = 10).

S-b  LOW-DIM EMBEDDING (recommendation-system form): attention over peers
     with a learned dot-product similarity,
        r_i ~= Sum_j softmax_j( (u_i . u_j) / tau ) * r_j
     u in R^d, d in {8,16}, strong L2, trained per fold on matured history
     ONLY. This is the Netflix analogy made literal: users = stocks,
     items = earnings events, rating = post-beat drift.
```

PIT DISCIPLINE (non-negotiable): every estimator is trained inside the fold
on events whose windows matured before the fold's test window begins
(RC-16 F3 semantics). The similarity used for a test-window event may not
have seen any test-window outcome.

## 5. Phases and gates

```text
P0  Firewall: F0a specs committed; F0b truncation test passes for features
    AND estimators; coverage recomputed at exact gap requirements
    (h=1 -> gap>=3; h=3 -> gap>=5) by grain. Commit before any diagnostic.

P1  Diagnostics on BEATS (SUE>0), CAR-first (the mechanism targets returns,
    not the label): per-feature quintile spread in car_10d (primary) and
    pead_pass (secondary), week-block bootstrap CIs (10k, seed 20260807).
    REQUIRED ADDITIONAL TESTS:
      (i)  orthogonalization vs rel_ret_5d/20d, sector_adjusted_ret_20d,
           revision_momentum_30d/60d/90d -- report raw AND orthogonalized;
           the GATE uses the orthogonalized effect.
      (ii) F3 (dispersion) interaction test.
      (iii) S-a and S-b must each be compared against S-c on the same
            diagnostic: "learned" must earn its keep over taxonomy.

P2  GATE: >=2 features from >=2 distinct families with orthogonalized
    quintile-spread CIs excluding 0, AND at least one learned estimator
    (S-a or S-b) beats the taxonomy baseline S-c on that diagnostic.
    Otherwise CLOSE: "learned PEAD-behavior similarity adds nothing over
    taxonomy / orthogonalized information."

P3  Combined model: surviving features added to the 22 retired features,
    frozen HPs and thresholds (min-gate 0.33), identical folds/bootstrap/
    machinery. Gates: G1 mean trade > 0 with week-block CI excluding 0;
    G2 beats the v6n baseline (paired weekly diff CI excludes 0).
    Pass -> shadow (paper), promotion review >= 8 fresh OOS weeks.
```

## 6. Stated limitations (must be repeated in any result)

- Literature effect sizes for read-across/spillovers are typically tens of
  basis points; a standalone signal of that size cannot pay our costs. The
  feature's job is to improve the COMPOSITE ranking, not to trade alone.
- RC-14 lesson applies: learned structures die when effective sample per
  group is tiny. This is why S-a/S-b are low-dimensional and shrunk, and
  why S-c (no estimation) is the baseline that must be beaten.
- Coverage is seasonal and grain-dependent; every arm declares NaN share
  and carries a has_peer flag.
- Vendor-vintage caveat inherited (FMP/Tiingo current vintages, not
  historical as-known).
- The GWRE/OKTA case (peer OKTA +18.4% five sessions before GWRE −39.7%)
  is filed as a COUNTEREXAMPLE to taxonomy similarity, not as motivation
  for the feature's efficacy.

## Appendix — Literature map (to be verified/expanded before freeze)

```text
A. Read-across / information transfer (the finance core)
   Foster (1981) JAE -- intra-industry information transfers on earnings.
   Clinch & Sinclair (1987) JAE -- recursive intra-industry releases.
   Ramnath (2002) JAR -- investor AND analyst reactions to related firms.
   Cohen & Frazzini (2008) JF -- economic links (customer-supplier) and
       predictable returns: the strongest documented read-across.
   Hou (2007) RFS -- industry information diffusion and lead-lag.
   Hong & Stein (1999) JF; Hong, Lim & Stein (2000) JF -- gradual
       information diffusion, the mechanism this family tests.
   Hong, Torous & Valkanov (2007) JFE -- industries lead stock markets.
   Moskowitz & Grinblatt (1999) JF -- industry momentum.
   Menzly & Ozbas (2010) JF -- diffusion across segmented firms.

B. Similarity construction beyond SIC (directly answers "industry helps
   but not always")
   Hoberg & Phillips (2016) JPE -- text-based network industries (TNIC):
       dynamic peer networks from 10-K text; peers SIC misses.

C. ML / relational prediction
   Feng et al. (2019) ACM TOIS -- temporal relational ranking (graph +
       attention over firm relations) for stock prediction.
   Kim et al. (2019) -- HATS hierarchical graph attention for stock movement.
   Gu, Kelly & Xiu (2020) RFS / (2021) J.Econometrics -- ML asset pricing;
       autoencoder latent factors: methodological precedent for S-b.
   General GNN/stock-embedding literature (Stock2Vec and successors).

D. The gap
   No canonical paper found that (i) LEARNS similarity from PEAD-conditioned
   drift co-movement, (ii) uses the peer's FRESH early reaction as the
   inference-time signal, and (iii) targets the drift conditional on beat.
   Ingredients exist across A-C; the combination appears novel. Consequence:
   no prior effect size to borrow -- treat the expected effect as small and
   let P2 decide.
```

---

## Amendment 2 (post-hoc, 2026-09-16, user-approved) — P2 condition 2 re-specified

```text
ORIGINAL:     "at least one learned estimator (S-a or S-b) beats the
              taxonomy baseline S-c on that diagnostic."
RE-SPECIFIED: "at least one learned estimator is orthogonalized
              quintile-spread CI-positive on its own domain, AND the
              common-set comparison against S-c is reported alongside."
```

Disclosure: made AFTER the P1 / P1(b) results were inspected, with the
user's explicit decision. Reasoning: P1 showed S-b(h=3) is CI-positive on
its domain (+0.64pp [+0.10,+1.19], 86% coverage) but only TIES S-c on the
~25%-coverage common set (+0.448 vs +0.440). The original condition
conflated per-event superiority with justifying the learned complexity:
S-b's justification is coverage extension (86% vs 25% of beats), which a
common-set comparison cannot express by construction. The per-event
question stays answered (tie) and reported. The decisive test remains
P3's pre-registered gates (mean trade > 0 with CI excluding 0; paired
weekly diff vs v6n with CI excluding 0), which no amendment touches.

P2 verdict under the amended condition: **PASS** — F1_sb_h3 (family F),
A2 (family A), both orthogonalized-CI-positive; S-b CI-positive on its
own domain; common-set tie reported. P3 is now legal to run.
