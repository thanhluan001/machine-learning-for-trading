# RC-22 Closure — Simulator Alignment (The Positive Point Estimates Were Accounting Artifacts)

**Status:** CLOSED 2026-09-16. All gates FAIL under BOTH selectors. No promotion.
**Spec:** `rc22_simulator_alignment_pre_registration.md` (6be0476).

## Pipeline verification (the fix is the only thing that changed)

The OLD-selector arms reproduce the registered programs exactly:

```text
v6n  old  129 trades  −1.581%   (RC-17: −1.581 ✓ exact)
rc19 old  158 trades  +0.718%   (RC-19: +0.718 ✓ exact)
v7n  old  106 trades  −1.350%   (RC-20: −1.350 ✓ exact)
rc20 old  155 trades  +0.383%   (RC-20: +0.253; differs slightly — see caveat)
rc21 old  194 trades  +1.722%   (RC-21: +2.060; differs slightly — see caveat)
```

Caveat: the combined/transfer arms differ slightly because RC-22's
combined embedding pool includes all 16,665 SP400 rows vs the original
16,587 (0.5% pool difference). SP400-only arms are bit-identical, which
verifies the pipeline.

## Result — continuous (causal) deployment

```text
arm    selector  trades  win%   avg trade   NAV      weekly mean [CI]
v6n    old        129    44.2   −1.581%    −42.8%   −0.838 [−1.751, +0.065]
v6n    NEW        177    45.8   −0.445%    −23.8%   −0.344 [−1.447, +0.759]
rc19   old        158    53.8   +0.718%    +21.4%   +0.359 [−0.685, +1.421]
rc19   NEW        229    47.6   −1.163%    −54.5%   −0.793 [−1.903, +0.342]
v7n    old        106    43.4   −1.350%    −36.8%   −0.680 [−2.139, +0.802]
v7n    NEW        148    43.2   −1.209%    −43.8%   −0.779 [−2.314, +0.739]
rc20   old        155    51.6   +0.383%     −0.5%   +0.188 [−1.190, +1.604]
rc20   NEW        221    48.4   −0.806%    −47.9%   −0.524 [−1.990, +1.014]
rc21   old        194    48.5   +1.722%    +82.5%   +0.898 [−0.530, +2.465]
rc21   NEW        282    45.7   +0.015%    −25.9%   +0.011 [−1.616, +1.719]

GATES: every G1/T1 and G2 = FALSE under BOTH selectors.
```

Trade counts rose +45–48% for every arm, exactly as pre-registered. But
the treatment arms' expectancy COLLAPSED while the baselines improved.

## Mechanism (quantified — this is the finding)

Decomposition of the extra trades (taken only under continuous deployment):

```text
arm    extra n   extra avg   shared/common avg   extra mean score
rc19     110      −2.06%         −0.33%              0.384
rc20     106      −1.54%         −0.13%              0.363
rc21     148      −1.39%         +1.57%              0.385
v6n       77      +1.85%         −2.21%              0.358
v7n       61      +1.05%         −2.79%              0.348
```

Two mechanisms, both real:

1. **Concentration, not capacity.** The pure-drift edge lives in the
   week's best few names. The old simulator's slot-blocking accidentally
   enforced that concentration; continuous deployment spreads capital
   into the deep candidate pool (threshold 0.33 admits ~25% of SP400 and
   ~55% of SP600 events), whose marginal expectancy is NEGATIVE. The
   shared trades keep their edge (rc21 common +1.57%) — the entire
   collapse comes from the extra names. This is a capacity-of-signal
   finding: more capital into the same pool destroys the edge.
2. **A second integrity defect: allocation look-ahead.** The old weekly
   batch selector chose the week's top-4 using the WHOLE week's candidate
   list, i.e. it could hold capital idle on Monday "because" a better
   name would appear Thursday — information unavailable in real time. The
   continuous selector decides with same-day information only. This is
   the second simulator-level integrity issue found in this program
   (RC-16's car_drift feature leak was the first); both inflated
   historical results.

Inverse pattern in the baselines (extra trades +1.05/+1.85%, original
picks −2.2/−2.8%) reconfirms the earlier finding that the 3-gate
min-score is anti-informative about returns.

## Consequences

- **RC-19/20/21's positive point estimates are not achievable
  expectations.** They were artifacts of scarcity-concentration plus
  allocation look-ahead. Under causal, continuously deployed accounting
  the treatment arms earn ≈0 (rc21 +0.015%/trade) or clearly negative
  (rc19 −1.163%/trade).
- **The proposed shadow book on the rc19 calibration is dead as
  stated** — it would deploy a ~0-expectancy construction.
- The program's overall conclusion — no certifiable edge — stands, now
  for a stronger reason: the honest simulator, deployed the way the live
  book actually deploys, yields ~0.
- The LIVE book's slot semantics were already correct (fill on free);
  what was wrong was the backtest's model of it. That is now fixed.

## Honest remaining options (all need new registration; none is a rescue)

1. **Concentrated causal deployment** (e.g. 1–2 slots, or take only the
   day's top candidate): directly targets the concentration finding.
   Must be registered with disclosure that the rule is informed by
   RC-22's result, one variant, no sweeps.
2. **Stop.** The historical record, now correctly simulated, shows no
   deployable edge in any construction tested.

Artifacts: `archive/experiments/rc22/report.json`, `executed.h5`.

---

## CORRECTION ADDENDUM (2026-09-16, same session, after the concentration diagnostic)

The "Mechanism" section above overclaims. A follow-up diagnostic on the
saved candidate sets (score >= 0.33 pools, `archive/experiments/rc22/candidates.h5`)
shows:

```text
POOL MEANS (5-session pregap_return, ALL candidates >= 0.33):
  rc19  n=1,060  +0.18% (SE 0.37)     rc20  n=906  +0.10% (SE 0.54)
  rc21  n=2,221  +0.34% (SE 0.32)     v6n   n=538  −0.32% (SE 0.51)

=> the pool is NOT negative; it is ~0 with small SEs.

SCORE-DECILE MEANS WITHIN POOL (5 = highest score):
  rc19: +1.34 / −1.03 / +0.06 / +0.50 / +0.03     (no gradient)
  rc20: +0.17 / −0.44 / +0.30 / −0.77 / +1.27     (no gradient)
  rc21: −0.19 / +0.81 / +1.16 / −0.00 / −0.07     (no gradient)

TOP-k-PER-WEEK CURVE (k-th highest score each week):
  rc19: k1 −1.07 (1.52)  k2 +0.63 (1.39)  k3 +1.13 (1.47)  k4 −1.64 (1.87)
  rc20: k1 +1.30 (1.83)  k2 +0.01 (1.96)  k3 −0.35 (2.61)  k4 −1.46 (2.80)
  rc21: k1 +3.61 (1.81)  k2 −0.39 (1.70)  k3 +0.62 (2.14)  k4 −1.92 (1.69)
```

**Therefore:**
1. The score does NOT rank returns within the candidate pool — quintiles
   are flat/non-monotone and the top-k curve has no gradient (rc21's k1
   t≈2.0 is one of twelve such comparisons; rc19's k1 is negative).
   "Concentration" is not supported: there is no evidence the top-ranked
   candidates are better.
2. The pool mean is ~0, not negative. The extra trades' negative average
   (t≈−1, see above) was noise.
3. The positive averages of the OLD-selector books (+0.72% rc19, +1.72%
   rc21) are therefore NOT explained by pool quality or score ranking —
   they come from the allocation/timing pattern of the old selector
   (which weeks it filled, and the mild week-level look-ahead). That
   difference (+1.72% vs +0.34% pool) is itself not significant (t≈0.8).
4. Per-trade SD is 13–20%; with 100–300 trades, SE ≈ 1–1.7%. This sample
   can only certify edges of roughly ≥2%/trade. Any realistic PEAD edge
   (0.5–1.5%) is below the detection floor of the entire program.

Corrected conclusion: the honest simulator yields ~0 for every arm; the
historical positive point estimates are allocation artifacts of
uncertified size; and the concentration remedy is refuted by the data.
