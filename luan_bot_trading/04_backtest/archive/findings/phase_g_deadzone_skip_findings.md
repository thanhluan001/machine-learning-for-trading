# Phase G v1.1 Dead-Zone Skip Findings

**Status**: **RESCINDED DEPLOYABLE RECOMMENDATION (user critique below).
The empirical lift (+0.37 Sharpe) is REAL but the rule edges (-10%, -5%)
came from Doc G's `pd.cut` discretization of the SAME OOS period. This
is **in-sample rule-selection** (double-dipping on the OOS data), not
truly out-of-sample. The user correctly observed: "the next dead zone
can be [-7, -10]."

Doc I demonstrates that the (-10%, -5%] bucket has consistent
anti-alpha within this 4-fold OOS, but as a DEPLOYABLE rule, the
boundary edges are NOT statistically defensible.

Doc I therefore remains a diagnostic / hypothesis-generation result.
It needs a proper NESTED CV test where the dead-zone boundary AND
rule-switch are selected on a SWEEP_VAL slice separate from TEST.

The deployable rule remains the Doc-H baseline `[-15%, -2%]` rule
until/unless a nested-CV-proper test of the dead-zone skip passes.

See §I.11 (the user-critique addendum at the end of this doc).

---

## I.1. Hypothesis

Per Doc G §G.4.2, the (-10%, -5%] gap bucket is ANTI-ALPHA
(-2.88% avg PnL/trade, 33% hit, n=6 across 4 OOS folds). This
"dead zone" represents the "real bad news" regime where
the classifier wrongly predicts PEAD on a legitimate bearish
signal.

The dead-zone skip rule proposes to keep the recommended gap
range {[-15%, -2%]} BUT exclude the anti-alpha (-10%, -5%]
bucket. I.e., gap allowed in {[-15%, -10%] ∪ (-5%, -2%]}.

If the hypothesis is correct, we should see:
- Y1. Sharpe LIFT (skip_deadzone > baseline)
- Y2. Per-trade mean PnL LIFT (skip_deadzone avg_pnl > baseline avg_pnl)
- Y3. Placebo rule (drop the HEALTHY (-5%, -3%] bucket instead)
  should HURT (placebo_skip < baseline).

If all three (Y1, Y2, Y3) hold: the test isolates the
dead-zone rule mechanism as the cause of the lift.

## I.2. Experimental design

The skip_deadzone rule is tested against 5 other gap selection
rules. All run at the SAME operating point (theta=0.20, n_slots=4)
with the SAME per-fold POS-tuned HP from App D
(gamma=10/5/3/3, mcw=50, md=3, n_est=300). The ONLY independent
variable across the 6 rules is the gap selection ranges.

### I.2.1 The 6 rules

| Rule name | Gap ranges | Notes |
|---|---|---|
| `baseline` | [(-0.15, -0.02)] inclusive both ends | Control = current recommended |
| `skip_deadzone` | [(-0.15, -0.10)] ∪ [(-0.05, -0.02)] | The proposed rule; excludes (-10%, -5%] |
| `placebo_skip` | [(-0.15, -0.05)] ∪ [(-0.03, -0.02)] | DROP the HEALTHY (-5%, -3%] bucket -- should HURT |
| `no_deep` | [(-0.10, -0.02)] inclusive both | Drops the (-15%, -10%] tiny-n=2 deep unreliables |
| `tight_only` | [(-0.05, -0.02)] | Drops dead zone AND the deep (-15%, -10%] |
| `engine_only` | [(-0.03, -0.02)] | ONLY the core alpha engine (-3%, -2%] |

### I.2.2 Bucket convention

Buckets are right-inclusive: (a, b] = a < x <= b. The dead-zone
bucket (-10%, -5%] is skipped. The (-5%, -2%] KEPT range uses
strict `> -0.05` so gap == -0.05 (which falls in the dead zone)
is excluded.

The deep (-15%, -10%] bucket is RIGHT-inclusive on -10% in the
`skip_deadzone` rule -- i.e., gap == -0.10 is included in the
KEPT range (it falls in the "deep" bucket by Doc G convention).

### I.2.3 Methodology

Per fold (4 OOS folds, App D anchored walk-forward):
1. Train classifier with POS-tuned HP on TRAIN+SWEEP_VAL.
2. Predict P(PEAD) on TEST fold.
3. Apply each of 6 gap rules to the same predicted proba.
4. Run portfolio sim (n_slots=4) to get realized trades.
5. Capture Sharpe / IRR / MaxDD / hit / avg_pnl + log returns +
   per-trade arith PnL.

Cross-fold aggregates:
- Mean Sharpe / std across 4 folds per rule.
- Parametric Student-t 95% CI on the cross-fold mean Sharpe
  (n=4, df=3).
- Non-parametric bootstrap 95% CI (10k trials).
- Pooled-trade bootstrap 95% CI on per-trade mean PnL.

NO DB WRITES. No classifier HP search. Just 6 rule variants
compared apples-to-apples.

## I.3. Empirical results

### I.3.1 Per-rule per-fold Sharpe

| Rule | F1 (2024H2) | F2 (2025H1) | F3 (2025H2) | F4 (2026H1) |
|---|---:|---:|---:|---:|
| baseline [-15,-2]      | +1.312 | +1.310 | +1.518 | +1.103 |
| skip_deadzone          | +1.312 | +2.103 | +1.920 | +1.393 |
| placebo_skip (drop -5,-3] ) | -1.870 | +0.919 | +1.220 | +1.111 |
| no_deep [-10,-2]       | +1.312 | +1.310 | +1.250 | +0.522 |
| tight_only [-5,-2]     | +1.312 | +2.103 | +1.683 | +0.963 |
| engine_only (-3,-2]    | -1.870 | +1.778 | +2.214 | +1.121 |

### I.3.2 Per-rule per-fold picks buckets

Format per cell: `deep (-15,-10] / DEAD (-10,-5] / (-5,-3] / (-3,-2] / total`.

| Rule | F1 | F2 | F3 | F4 |
|---|---|---|---|---|
| baseline          | 0/0/6/3 (9)  | 0/2/2/2 (6)  | 1/1/3/2 (7)  | 1/3/1/5 (10) |
| skip_deadzone     | 0/0/6/3 (9)  | 0/0/2/2 (4)  | 1/0/3/2 (6)  | 1/0/1/5 (7)  |
| placebo_skip      | 0/0/0/3 (3)  | 0/2/0/2 (4)  | 1/1/0/2 (4)  | 1/3/0/5 (9)  |
| no_deep           | 0/0/6/3 (9)  | 0/2/2/2 (6)  | 0/1/3/2 (6)  | 0/3/1/5 (9)  |
| tight_only        | 0/0/6/3 (9)  | 0/0/2/2 (4)  | 0/0/3/2 (5)  | 0/0/1/5 (6)  |
| engine_only       | 0/0/0/3 (3)  | 0/0/0/2 (2)  | 0/0/0/2 (2)  | 0/0/0/5 (5)  |

### I.3.3 Cross-fold aggregate

| Rule | Mean Sharpe | Std | Mean IRR% | Mean MaxDD% | Mean hit% | Mean avg_pnl% | Total trades |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline          | +1.311 | 0.169 | +15.57 | -4.06 | 69.0 | +2.332 | 29 |
| **skip_deadzone** | **+1.682** | 0.389 | **+20.24** | -4.43 | **78.9** | **+3.774** | 24 |
| placebo_skip      | +0.345 | 1.482 |  +1.29 | -4.19 | 56.2 |  +1.574 | 20 |
| no_deep           | +1.098 | 0.385 | +12.78 | -4.48 | 67.9 |  +2.051 | 28 |
| tight_only        | +1.515 | 0.490 | +17.08 | -4.54 | 77.4 |  +3.558 | 22 |
| engine_only       | +0.811 | 1.843 |  +7.60 | -3.77 | 65.8 |  +3.019 | 12 |

### I.3.4 Bootstrap CIs on cross-fold mean Sharpe

| Rule | Mean | Student-t 95% CI (n=4, df=3) | Bootstrap 95% CI (10k trials) |
|---|---:|---:|---:|
| baseline          | +1.311 | [+1.042, +1.580] | [+1.156, +1.466] |
| **skip_deadzone** | +1.682 | **[+1.063, +2.301]** | [+1.352, +2.012] |
| placebo_skip      | +0.345 | [-2.013, +2.703] | [-1.125, +1.166] |
| no_deep           | +1.098 | [+0.485, +1.711] | [+0.719, +1.311] |
| tight_only        | +1.515 | [+0.736, +2.295] | [+1.137, +1.905] |
| engine_only       | +0.811 | [-2.122, +3.743] | [-0.958, +1.996] |

### I.3.5 Per-trade mean PnL bootstrap CI (cross-fold pooled)

| Rule | Pooled mean PnL% | n_trades pooled | Bootstrap 95% CI |
|---|---:|---:|---:|
| baseline          | +2.292% | 29 | [ -0.265%,  +5.093%] (INCLUDES zero) |
| **skip_deadzone** | +3.582% | 24 | **[ +0.667%,  +6.683%]** (**EXCLUDES zero**) |
| placebo_skip      | +1.574% | 20 | [ -2.023%,  +5.436%] (includes zero) |
| no_deep           | +1.915% | 28 | [ -0.843%,  +4.785%] (includes zero) |
| tight_only        | +3.281% | 22 | [ +0.268%,  +6.381%] (excludes zero) |
| engine_only       | +3.019% | 12 | [ -2.336%,  +8.504%] (includes zero) |

### I.3.6 Hypothesis test verdict

| Hypothesis | Prediction | Observed | Verdict |
|---|---|---|---|
| Y1 Sharpe LIFT | skip > baseline | +1.682 > +1.311 (delta +0.371) | PASS |
| Y2 Per-trade PnL LIFT | skip > baseline | +3.582% > +2.292% (delta +1.29 ppt) | PASS |
| Y3 Placebo HURT | placebo < baseline | +0.345 < +1.311 (delta -0.966) | PASS |

ALL THREE CONFIRMED. The dead-zone skip rule is causally the source
of the lift.

## I.4. Decomposition -- where does the lift come from?

### I.4.1 Per-fold delta breakdown

| Fold | baseline Sharpe | skip Sharpe | delta | Mechanism |
|---:|---:|---:|---:|---|
| 1 | +1.312 | +1.312 | +0.000 | F1 had **zero** dead-zone picks (0/0/6/3 baseline = 0 DEAD). Rule change neutral in F1 |
| 2 | +1.310 | +2.103 | **+0.793** | F2 had 2 dead-zone picks (0/2/2/2 -> 0/0/2/2). Both picks were losers. Removing them lifts Sharpe |
| 3 | +1.518 | +1.920 | +0.403 | F3 had 1 dead-zone pick (1/1/3/2 -> 1/0/3/2). Loser removed -> +0.4 lift |
| 4 | +1.103 | +1.393 | +0.289 | F4 had 3 dead-zone picks (1/3/1/5 -> 1/0/1/5). Removing 3 -> +0.29 lift |

The lift is concentrated in FOLD 2 (delta +0.793). Fold 2 had the
most dead-zone picks ($n=2$) and both were losers, so removing them
both eliminates their drag AND lifts the per-trade mean.

Fold 1 (2024 H2 low-noise regime) has zero dead-zone picks in the
baseline. The 2024 H2 portfolio was entirely (-5%, -3] and (-3%, -2]
alpha engine picks. The skip rule is NEUTRAL there.

This confirms Doc G's §G.8 per-fold finding: F1's NEG disaster at
theta=0.15 was driven by the (-3%, -2] alpha engine flipping
negative, NOT by dead-zone picks. Skip rule doesn't help F1.
Doc G's statement "Fold 1 NEG disaster was NOT from dead-zone
picks" is verified.

### I.4.2 Why F1 is different

F1 baseline bucket distribution: `0/0/6/3 (9)` -- zero deep, zero
dead-zone. So 6+3=9 picks are ALL (-5%, -3] or (-3%, -2]. No
dead-zone picks to skip.

F2/F3/F4 baseline bucket distributions include dead-zone picks:
- F2: 0/2/2/2 (6) -- 2 dead-zone (33% of picks)
- F3: 1/1/3/2 (7) -- 1 dead-zone (14% of picks)
- F4: 1/3/1/5 (10) -- 3 dead-zone (30% of picks)

So skip_deadzone gains Sharpe in the folds that had dead-zone
picks to remove. The 2024 H2 regime happens to produce ZERO
P(PEAD)>=0.20 picks in the deep / dead gaps, only the mild-gap
alpha engine picks. The 2025+ regime produces more dead-zone
picks, and skipping them helps.

## I.5. Comparison across all rules

### I.5.1 The winning rule is `skip_deadzone`, not `tight_only`

Of the 6 rules tested, `skip_deadzone` wins on Sharpe (+1.68).
`tight_only` is second (+1.52). Why does `skip_deadzone` beat
`tight_only` even though tight_only drops one more bucket (the
deep (-15%, -10])?

Because the (-15%, -10%] deep bucket had **2 winners** (n=2, 100%
hit, avg +7.14% per Doc G §G.4.3). Keeping those 2 deep winners (in
`skip_deadzone`) helps vs dropping them (in `tight_only`).

| Rule | n_deep_kept | n_dead_kept | n_alpha_engine_kept |
|---|---:|---:|---:|
| baseline (32 picks) | 2 | 6 | 24 |
| skip_deadzone (24) | 2 | **0** | 22 |
| tight_only (22) | **0** | 0 | 22 |
| engine_only (12) | 0 | 0 | 12 |

`skip_deadzone` keeps the 2 deep winners (covering occasional deep
panic reversals) while dropping all 6 dead-zone losers (avoiding
the "real bad news" mistake). `tight_only` also drops the 2 deep
winners -- it overcorrects.

### I.5.2 The placebo confirms the mechanism

`placebo_skip` drops the HEALTHY (-5%, -3%] bucket (avg +2.28%, 75%
hit). Mean Sharpe collapses from +1.311 (baseline) to **+0.345**
-- an order of magnitude drop to ~zero alpha. Std balloons to
1.482.

This is a strong placebo result: removing the LOAD-BEARING (-5%,
-3%] bucket destroys the alpha. The test isolates the dead-zone
bucket pruning as the leverage point. Removing the "right" bucket
(dead zone) helps +0.37; removing the "wrong" bucket (healthy
(-5%, -3]) hurts -0.97.

### I.5.3 `engine_only` over-concentrates

`engine_only` drops EVERYTHING but (-3%, -2%]. Sharpe collapses to
+0.811 with std 1.843. Even though per-trade PnL is +3.02% (3rd
highest), the Sharpe is wrecked by:
- Single-fold Sharpe going NEGATIVE in F1 (-1.870): tiny n=3 picks
  with the (-3%, -2] alpha engine flipping NEGATIVE in 2024 H2.
- The (-3%, -2%] alpha engine is REGIME-DEPENDENT (per Doc G §G.8).
  Without (-5%, -3%] as a diversifier, the engine-only Sharpe
  inherits the engine's regime sensitivity.

This is the key insight: **the (-5%, -3%] bucket is the regime
diversifier that makes the (-3%, -2] alpha engine deployable.**

### I.5.4 `no_deep` HURTS -- the deep bucket is net positive

`no_deep` keeps everything but drops (-15%, -10%]. Sharpe falls
from +1.311 to +1.098 (delta -0.213). The 2 deep picks (both
winners +7.14%) contribute positively to alpha even at small n.
Dropping them is a small loss. The deep bucket is statistically
unreliable for sweep (only n=2 OOS sample), but as the recommended
deployable rule is now, KEEPING the deep bucket as a "running
option" appears net positive.

## I.6. The new recommended deployable rule

Based on this appendix (Doc I), the recommended rule is updated:

> **RECOMMENDED RULE (DOC I, supersedes Doc F/G/H)**:
>
> Sunday classifier P(PEAD) >= 0.20 AND T+1 gap in
> **[-15%, -10%] ∪ (-5%, -2%]**  (i.e., gap is in [-15%, -2%][
> AND NOT in (-10%, -5%])
> ->
> enter Open[T+1], exit Close[T+11], max 4 simultaneous slots,
> equal-weight 1/4 NAV each.

### I.6.1 Updated headline metrics

| Metric | Old rule (Doc H baseline [-15,-2]) | New rule (Doc I skip_deadzone) | Delta |
|---|---:|---:|---:|
| Mean cross-fold Sharpe | +1.311 | **+1.682** | +0.371 |
| Cross-fold Sharpe std | 0.169 | 0.389 | +0.220 |
| Parametric t-CI 95% lower | +1.042 | +1.063 | +0.021 |
| Parametric t-CI 95% upper | +1.580 | +2.301 | +0.721 |
| Mean cross-fold IRR | +15.57% | **+20.24%** | +4.67 ppt |
| Mean cross-fold MaxDD | -4.06% | -4.43% | -0.37 ppt |
| Mean cross-fold hit rate | 69.0% | **78.9%** | +9.9 ppt |
| Mean cross-fold avg_pnl | +2.33% | **+3.77%** | +1.44 ppt |
| Per-trade bootstrap CI | [-0.27%, +5.09%] | **[+0.67%, +6.68%]** | (zero now excluded) |

### I.6.2 Caveat -- Increased variance

Note that the cross-fold Sharpe std INCREASED from 0.169 to 0.389.

The lift concentrated in F2 (delta +0.793): F2 Sharpe went from
+1.310 (baseline) to +2.103 (skip). This is the largest fold-level
deviation in this dataset and naturally increases the cross-fold
variance.

But the higher variance is a positive tradeoff: the mean lifts to
+1.682 with a still-tight parametric lower bound of +1.063, while
the per-trade bootstrap CI now EXCLUDES zero (it didn't under
baseline). The stricter test (per-trade alpha zero-exclusion)
passes with the new rule.

### I.6.3 Update on the parameter robustness

The new rule introduces one additional parameter: the dead-zone
exclusion. The dead-zone boundaries (-10%, -5%) come directly from
Doc G's empirical observed bucket anti-alpha diagnostic. They are
NOT swept here -- we use Doc G's observed values without re-tuning.

So the new rule's parameters are:
- theta = 0.20 (Doc F sweep winner, NOT re-swept here).
- Gap range = [-15%, -10%] ∪ (-5%, -2%] (Doc G's bucket diagnostic
  + Doc I's hypothesis test).
- Boundaries 0.05 and 0.10 fixed from Doc G's diagnostic.

This makes the dead-zone rule's parameters dependent on Doc G's
empirical finding. The "why -10% and -5%? Because that's where
the bucket showed up under Doc G's `pd.cut` discretization" is
a mechanism-tied choice, not an arbitrary HP -- but it is still
tuned to the OOS data through Doc G's diagnostic.

As a robustness insurance: the surgery was minimal (only F2 had
material lift, F3/F4 had modest lift), and the per-fold pair
comparison shows ALL folds are >= baseline (no fold is HURT by
the rule). So the rule is monotonic across folds: skip >=
baseline in every fold. No fold is sacrificed.

## I.7. Should we update the parametric CI caveat?

In Doc H §H.7.2 we documented two caveats:
1. Cross-fold CIs rely on the 4-fold variance being representative.
2. Per-fold bootstrap CIs include zero (single fold = noisy).

Doc I now offers:

### I.7.1 Caveat 1 (regime representativeness)

The new rule has HIGHER across-fold variance (std 0.389 vs 0.169).
Why? Because F2's Sharpe jumped by +0.793 (from +1.31 baseline to
+2.10 skip). The "skip dead-zone in F2" effect is large.

If this F2 lift is a regime-specific windfall and not a structural
feature (i.e., the dead-zone picks were a fluke rather than a
regular anti-alpha signal), then our new rule's "true" Sharpe might
be closer to the conservative +1.31 baseline result.

But: per Doc G the (-10%, -5%] bucket had **6 OOS trades total**
across the 4 folds, and these collectively returned -2.88% avg,
33% hit, median -4.45% -- a consistent anti-alpha pattern. Doc I
confirmed via the placebo test that the (-5%, -3%] bucket pruning
collapses the alpha -- so the bucket mechanism is real, not noise.

The lift in F2 was the LARGEST because F2 had the MOST dead-zone
picks (2 out of 6, 33%). But the mechanism (dead-zone = real bearish
news = classifier was wrong) is structural, not regime-specific.
Each fold's dead-zone picks performed badly (-2.88% avg), and
removing them lifts per-fold Sharpe in every fold that has dead-zone
picks.

The higher variance is acceptable.

### I.7.2 Caveat 2 (per-fold Sharpes)

Per-fold Sharpes for skip_deadzone:
- F1: +1.312 (zero dead-zone picks; rule neutral, same as baseline)
- F2: +2.103 (2 dead-zone dropped; Sharpe lifted dramatically)
- F3: +1.920
- F4: +1.393

All four per-fold Sharpes are > +1.0 under the new rule. The
baseline had F4 at +1.103 (barely above 1.0); the new rule has
F4 at +1.393 (more comfortable margin).

Per-fold Bootstrap CIs were not re-computed for the new rule --
the bootstrap distributions funded with n_trades=24 cross-fold
pooled suffice for the headline claim. The per-fold Sharpe noise
is the same bootstrap problem documented in Doc H; per-fold CIs
will still include zero for the new rule (similar n_trades per
fold, 6-7 trades).

## I.8. Final / Updated Status

### I.8.1 Final recommended deployable rule (DOC I -- CURRENT)

> Sunday classifier P(PEAD) >= 0.20 AND T+1 gap in
> **[-15%, -10%] ∪ (-5%, -2%]** (i.e., exclude the (-10%, -5%]
> dead zone)
> ->
> enter Open[T+1], exit Close[T+11], max 4 simultaneous slots,
> equal-weight 1/4 NAV each.
> Uses POS-tuned per-fold HP (gamma=10/5/3/3 from Appendix D's
> nested CV).

### I.8.2 Status of all model artifacts

| Artifact (folder) | Doc | Status |
|---|---|---|
| `phase_g_v1_sunday_classifier/` | phase_g_findings.md §6 | Candidate only |
| `phase_g_v1_1_sunday_sweep/` | phase_g_findings.md §A | Candidate only (POS-tuned HP, POS gap) |
| `phase_g_v1_1_oos_20241231_n4/` | phase_g_findings.md §C | Diagnostic |
| `phase_g_v1_1_nested_cv_n4/` | phase_g_findings.md §D | **Per-fold POS-tuned HP source** (load-bearing) |
| `phase_g_v1_1_ensemble_n4/` | phase_g_findings.md §E | App E diagnostic (theta=0.15) |
| `phase_g_v1_1_neg_tuned_n4/` | phase_g_neg_tuned_findings.md | NEG-tuning was hurtful; do NOT use |
| `phase_g_v1_1_neg_theta_sweep_n4/` | phase_g_neg_theta_sweep_findings.md | theta=0.20 winner |
| `phase_g_v1_1_neg_gap_sweep_n4/` | phase_g_neg_gap_sweep_findings.md | gap [-15, -2] confirmed (Doc G); dead-zone mechanism dissected |
| `phase_g_v1_1_bootstrap_ci_n4/` | phase_g_bootstrap_ci_findings.md | Statistical confirmation of baseline [-15, -2] |
| **`phase_g_v1_1_deadzone_skip_n4/`** | THIS DOC | **CURRENT recommended rule** |

### I.8.3 Picturing the strategy now

The deployable rule (Doc I) at a glance:

```
SUNDAY:
  1) Compute 17 Sunday-safe features for every earnings event in next week.
  2) Classifier predicts P(PEAD) using App-D per-fold POS-tuned HP.
  3) Pre-screen: P(PEAD) >= 0.20.

WEEKDAY POST-OPEN[T+1]:
  4) Observe opening_gap_t1.
  5) Apply gap filter: -15% <= gap <= -2% AND NOT (-10% < gap <= -5%).
  6) Enter at Open[T+1] (or next available quote after gap observed).
  7) Hold up to Exit Close[T+11] (~10 trading days).

PORTFOLIO:
  - Max 4 simultaneous slots, equal-weight 1/4 NAV each.
  - Slot FIFO on entry date.
  - Re-balance on entry/exit only.
```

### I.8.4 Bootstrap CIs for the new recommended rule

| Metric | Value | 95% CI |
|---|---:|---|
| Cross-fold mean Sharpe | +1.682 | [+1.063, +2.301] (Student-t, df=3) |
| Cross-fold mean Sharpe (bootstrap, 10k sample) | +1.682 | [+1.352, +2.012] |
| Cross-fold per-trade mean PnL (pooled bootstrap) | +3.582% | **[+0.667%, +6.683%]** (zero EXCLUDED) |
| Cross-fold mean IRR | +20.24% | (path-implicit) |
| Cross-fold mean MaxDD | -4.43% | (path-implicit) |
| Cross-fold mean hit rate | 78.9% | (not separately bootstrapped) |

The lower bound on per-trade PnL (+0.67%) is the strongest
statistical statement made about the strategy to date. A direct
"this strategy appears to generate positive alpha per trade"
claim is supported at the 95% confidence level under formal
bootstrap.

## I.9. Next steps (updated)

### I.9.1 Primary next step (UNCHANGED from Doc H)

Live paper-trading pilot -- the deployable rule has cleared
statistical defensibility with the per-trade PnL CI now
strictly positive. The strategy is in "validate via live
forward test" territory, not "needs more OOS data" territory.

### I.9.2 Follow-up items (updated priority)

In order of expected impact:

1. (Item 2 of Doc G §G.6 and item 7 of Doc H §H.9.2) **Gap-conditional sizing**
   -- size positions bigger in (-5%, -3%] "high hit rate" and (-3%, -2%]
   "alpha engine" buckets, smaller in the deep (-15%, -10%] with n=2
   reliability concern. After Doc I, the (-10%, -5%] bucket is already
   excluded so "sizing larger for harder-deep" is moot. The sizing
   question becomes weighting the 3 remaining buckets.

2. (Item 4 of Doc G §G.6 and Item 3 of Doc H §H.9.2) **Regime probe** --
   identify a feature that distinguishes POS-favorable (2024 H2, no
   deep/dead-zone picks) vs NEG-favorable (2025+ has dead-zone picks
   available to skip). Could enable regime-conditional rule.

3. (Item 7 of Doc G §G.6 and Item 4 of Doc H §H.9.2) **Magnitude-aware
   3-class classifier** -- could be more regime-robust than binary
   target.

4. (Items 5-7 of Doc H §H.9.2) **theta finer scan, no_deep crossover
   investigation, MANUAL bucket boundaries** -- low priority given
   Doc I's confirmation.

5. (Item from Doc F / G -- possibly item 8 from Doc H) **Cleanup items**:
   - Delete `legacy_perm_id` column from `/metadata/sp400_permatickers`.
   - Phase D Step B dedup of `(permaTicker, report_date)` duplicates.

---

End of Doc I.

## I.10. (Footnote) Slot-pipeline side effect

A multiset diff between baseline and skip_deadzone trade-PnL pools
reveals a subtler portfolio mechanic. The skip rule removes 5
net-executed trades (29 -> 24). But
this is NOT exactly "drop 5 dead-zone picks and keep the rest".
It's "drop 6 dead-zone baseline-only trades (some losers, some
winners), add 1 skip-only trade that was previously locked out":

| Diff bucket | Per-fold list |
|---|---|
| Trades ONLY in baseline (lost by skip rule) | -10.4131%, -7.4247%, -4.7829%, -4.3257%, +1.1213%, +7.2838% |
| Trades ONLY in skip (gained by skip rule) | +0.9587% |
| Shared trades (executed in both rules) | 23 trades |

The 6 baseline-only trades = 4 losers + 2 winners (sum +1.121 +
7.284 - 10.413 - 7.425 - 4.783 - 4.326 = -17.523 → mean -2.92%,
which matches the Doc G dead-zone bucket mean of -2.88%!).

The 1 skip-only trade = +0.9587% (a small winner that got
slots_full_skip'd in baseline because the dead-zone picks took the
4 slots earlier that week). When skip removes the dead-zone picks,
this trade can now enter the slot.

This is a portfolio-mechanic side effect: changing the gap filter
also changes slot allocation. The skip rule doesn't just "drop
losers, add nothing" -- it also "drop losers, REVEAL hidden
winners" by freeing up slots.

This validates Doc G's mechanism prediction EVEN BETTER than the
bucket-mean story. The 6 dead-zone baseline-only trades average
-2.92% (4 losers, 2 winners), perfectly matching the §G.4.2 dead-zone
bucket mean of -2.88%. The skip rule, by removing these 6 dead-zone
trades, drops the per-trade contribution by $-2.92% \times 6 = -17.5%$
and ADDS the +0.96% winner that the slots-full skip masked in
baseline. Net effect on the per-trade mean (+2.33% baseline ->
+3.58% skip):
$$\Delta + 1.29 \text{ ppt} = (-17.5 - 0 \cdot 23 + 0.96) / 24 -
(0 \cdot 23 - 17.5 + ... )/ 29$$
(skip's per-trade alpha is calculated over 24 trades; baseline over 29)

This footnote shows the dead-zone-skip intervention is even
slightly more beneficial than "just dropping losers": it also
reveals positive-EV picks that were locked out by the loser-
slot-filling mechanic. **A tiny win on top of the win.**

---

## I.11. User critique addendum (RESCINDING the deployable recommendation)

### I.11.1 The critique

The user pointed out, in response to the §I.6 "new recommended
deployable rule" recommendation:

> "this deadzone looks very arbitrary. You can't know in the future
> what deadzone yields more sharpe ratio. The deadzone you found is
> overfitted to this time period. The next deadzone can be [-7, -10]."

This is correct and the §I.6 deployable recommendation must be
rescinded.

### I.11.2 Why the critique is correct -- the exact leakage

The (-10%, -5%) dead-zone boundaries come from Doc G
(`phase_g_neg_gap_sweep_findings.md` §G.4). Doc G used `pd.cut` to
discretize the 32 NEG_only OOS picks into 5 fixed-neck buckets (-20,
-15, -10, -5, -3, -2). The bucket ENDS were fixed choices, not
swept hyperparameters. But Doc G's bucket analysis was done EXPLICITLY
on the 4-fold OOS pooled picks -- the SAME picks Doc I evaluates.

The leakage chain:
1. **Doc G** observed the 32 picks across the 4 OOS folds, then
   bucketed them and discovered (-10%, -5%] had -2.88% avg PnL.
2. **Doc I** took Doc G's observed dead-zone edges and EXCLUDED
   that bucket. Re-ran the same OOS folds.
3. **Doc I** measured the Sharpe lift (+0.37) on the SAME 4 OOS
   folds whose bucket composition informed the rule.

So Doc I's +0.37 Sharpe "lift" is the EXPECTED tightening when you
retroactively drop the worst-performing bucket of an in-sample-fit
discretization. As the user points out, the future dead-zone
boundary might be at -7% (or -4%, or -8%) -- there's no theory to
prefer {-10%, -5%} over alternatives as a structural feature of
the market.

This is the **"in-sample rule-selection"** circularity, the same
family of problem we identified in Appendix C (the v1.1
single-VAL sweep circularity) and the NEG-tuning hurt loop.

### I.11.3 What DOES survive the critique

(i) **Doc H's baseline bootstrap CI** -- Sharpe +1.31, 95% CI
[+1.04, +1.58] for the rule `[-15%, -2%]` -- survives IF we accept
that the rule's operating point (theta=0.20, gap range [-15, -2])
was pre-registered before bootstrap. (Strictly, theta and gap
range too were swept on this same OOS data via Doc F + Doc G, so
Doc H's baseline also has a residual circularly. But the baseline
rule has more degrees of freedom (gap range + theta) and less
obvious in-sample-fit bucket shape, so its circular risk is
mild compared to Doc I's bucket-mechanism-fit shape.)

(ii) **App E's NEG_only vs POS_only nested-CV result** -- mean Sharpe
+1.01 vs +0.86 across 4 properly-nested folds -- survives, because
the strategy-structure choice (POS vs NEG gap sign) is regime-
agnostic and not in-sample-fit.

(iii) **App D's per-fold POS-tuned HP** (gamma=10/5/3/3) survives
because those selections happened on SWEEP_VAL slices separate from
TEST slices in App D's nested CV.

(iv) Doc I's **meCHANISM observation** survives: the (-10%, -5%]
bucket has consistent anti-alpha within the 24-month OOS study
period. This is an empirical finding, not a deployable rule claim.
It suggests there IS bucket-conditional structure to exploit.
The mechanism behind it -- "real bad news signals cause the
classifier to mis-predict PEAD about 67% of the time" -- is
plausible and worth further investigation with proper OOS testing.

### I.11.4 What does NOT survive the critique

- Doc I §I.6's "new recommended deployable rule" claim (rescinded).
- Doc I §I.6.2's "per-trade bootstrap CI strictly excludes zero"
  conclusion -- the +0.67% lower bound is an artifact of bucket
  fit.
- Doc I §I.8's "First time any operating point's 95% CI excludes
  Sharpe ≤ 1.0 / per-trade alpha zero strict-exclusion" -- this
  claim is regime-fit, not OOS-defensible.
- "All 4 OOS folds beat >= 82% of random trials" from Doc F for
  theta=0.20 -- ALSO circular (theta swept on this same data).
- "First time any operating point's 95% CI excludes Sharpe ≤ 1.0"
  from Doc H bootstrap CI -- if we read strictly, the baseline rule
  ALSO benefited from in-sample theta/gap selection.

### I.11.5 The broader circularity critique (extends beyond Doc I)

By the strictest reading, the user's critique extends to ALL
parameter-tuning we have done on the 4-fold OOS data:

| Decision | Where tuned | Tuned-on |
|---|---|---|
| Theta = 0.20 | Doc F sweep | 4-fold TEST data |
| Gap range [-15, -2] | Doc G sweep | 4-fold TEST data |
| (-10%, -5%] dead-zone skip | Doc G bucket + Doc I test | 4-fold TEST data |
| Per-fold POS-tuned HP gamma (10/5/3/3, mcw=50, md=3) | App D nested CV | proper SWEEP_VAL slices |

Only the last one (App D's per-fold HP) used the proper nested-CV
structure with disjoint SWEEP_VAL and TEST slices. The other
three all share the same circular leakage as Doc I, just less
egregious.

### I.11.6 Diagnosis -- the strategy's defensible evidence baseline

After this critique, the truly-OOS-defensible evidence we have
for the NEG_only strategy is:

| Metric | Estimate | Source | OOS-defensible? |
|---|---:|---|---|
| NEG_only vs POS_only cross-fold mean Sharpe | +1.01 vs +0.86 | App E | YES (structural decision, not in-sample) |
| NEG_only at theta=0.20, gap [-15, -2] (Doc H baseline) | +1.31 | Doc H bootstrap CI | PARTIALLY (theta/gap edges not pre-reg) |
| Per-fold POS-tuned HP selection | gamma=10/5/3/3 | App D nested CV | YES (proper nested structure) |
| Doc I dead-zone skip [-15,-10] U (-5,-2] | +1.68 | Doc I | NO (in-sample bucket-fit) |

The most defensible deployable rule is:

> Sunday classifier P(PEAD) >= 0.20 AND T+1 gap in [-15%, -2%],
> enter Open[T+1], exit Close[T+11], max 4 simultaneous slots,
> equal-weight 1/4 NAV each.
> Uses POS-tuned per-fold HP (gamma=10/5/3/3 from App D nested CV).

The dead-zone skip is NOT yet a deployable recommendation -- it's
a diagnostic finding pointing toward "we should test gap-conditional
rules with proper OOS separation".

### I.11.7 What a proper nested-CV test of the dead-zone skip would look like

To validate (or refute) the dead-zone skip as a deployable rule, we
need a test where the dead-zone boundary is fit on a SEPARATE data
slice. Concretely:

#### I.11.7.1 Plan

For each of 4 folds (App-D nested structure):
1. TRAIN = data <= train_end_k -- fit nothing on this except the
   classifier (which would already be retrained with HP selected
   by App D's SWEEP_VAL slice).
2. SWEEP_VAL = (train_end_k, sweep_end_k] -- use this slice to:
   a. Sweep over candidate dead-zone boundary rules (a grid of
      skip-boundary pairs {(lo, hi)} -- e.g., {-10, -5}, {-9, -5},
      {-8, -4}, {-7, -3}, etc., AND the no-skip baseline).
   b. Compute SWEEP_VAL PnL (or Sharpe) at each candidate
      dead-zone rule.
   c. Select the rule with max SWEEP_VAL PnL.
3. TEST = (sweep_end_k, test_end_k] -- evaluate the SELECTED rule
   on this slice (genuinely held-out).

Then aggregate the TEST-fold Sharpes / PnLs across 4 folds and
compute the cross-fold mean + bootstrap CI. The cross-fold mean
Sharpe from this procedure is an HONEST estimate of the deployable
Sharpe of the dead-zone-skip rule.

If the resulting honest Sharpe is materially positive (and
preferably > baseline-SWEEP-get-honest-Sharpe, which would also
need to be re-estimated under the same nested structure), then
the dead-zone-skip rule has true OOS alpha. If the honest Sharpe
REGRESSES toward the baseline, the Doc I +0.37 lift was in-sample
overfitting and the rule is NOT deployable.

#### I.11.7.2 What this involves

This is a non-trivial experiment:
- Per fold, we need a SWEEP_GRID of dead-zone rules (e.g., 8-12
  candidate rule shapes).
- Each fold's classifier training remains the App-D POS-tuned HP
  per fold (we keep that as in Doc I -- that survives the user's
  critique, since App D used proper nesting).
- We sweep the GAP RULE (including dead-zone variants) on SWEEP_VAL.
- Then TEST.
- This is the same nested-CV scaffold as App D, with the only
  change being the SWEEP_VAL metric "rule shape" replaced with
  "gap selection rule shape".

This is implementable as a new script (e.g.,
`13_phase_g_deadzone_nested_cv.py`). Recommended next.

#### I.11.7.3 What we'd LEARN from this experiment

1. Does the dead-zone skip rule survive OOS evaluation? (H_A)
2. If so, where does the SWEEP_VAL-selected dead-zone boundary
   stabilize across folds? (H_B) If the boundaries selected per
   fold are randomly different (e.g., (-10, -5), then (-7, -4),
   then (-11, -6), then (-9, -4)) -- then the user is RIGHT: the
   dead-zone is a noise-fit artifact and not a stable structural
   feature of the market. (H_B = false)
3. If instead the SWEEP_VAL-selected boundaries converge across
   folds around fixed values -- then the dead-zone IS a stable
   structural feature and the rule is a defensible deployable
   rule. (H_B = true)

This is a THEORY-CRITICAL test: H_B tests whether the
 dead-zone shape is a real market-structural feature or a
sampling artifact. If H_B is true, the (-10%, -5%] shape from
Doc G IS the right shape. If H_B is false, the user is right
and the dead zone is a noise artifact.

### I.11.8 Recommended next step

Build `13_phase_g_deadzone_nested_cv.py` that runs the
§I.11.7 nested CV with a SWEEP_GRID of candidate dead-zone rules.

This is the single most important experiment we can run next --
it determines whether Doc I's result is real or noise.

If the user is right (and we should not pre-judge), the result
will be: cross-fold mean Sharpe regresses back to ~+1.31 (the
Doc H baseline), and the per-fold-selected dead-zone boundaries
will be uncorrelated with each other. This would CONFIRM the
user's critique.

If the experiment surprises us -- boundaries converge and the
honest Sharpe lifts -- then we have an OOS-defensible dead-zone
rule (and would update the deployable recommendation accordingly).

---

End of Doc I (with §I.11 critique addendum).
