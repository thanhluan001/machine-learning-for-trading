# Feature-Set Audit (back to basics) — 2026-09-16

Question: after the RC-16 leak removal, does ANY feature in the deploy set
carry signal about (a) the label `pass_g1 = car_10d > +3%` or (b) the
5-session harvest `pregap_return`?

Method: univariate only — coverage, AUC, Spearman, top-minus-bottom
quintile spread with week-block bootstrap (10k, seed 20260807), era split
(first vs second half), Benjamini-Hochberg FDR across the family.
Sample: 27,191 events (SP400+SP600, XLF excluded), full history.
No model selection. Artifacts: `archive/experiments/feature_audit/`.

## Result: the feature set is EMPTY

```text
                             AUC(pass_g1)   AUC(ret>0)   spread(ret)  t      q_bh
median over 24 features          0.4993       0.4990       ~0        0.82   0.69
best                             0.5409       0.5198     +1.178%     1.52   0.69
                              (idiosyncratic_vol)  (vix)
best |t|                         0.4977       0.4975     −0.521%    −1.72   0.69
                                                     (rel_ret_5d)

features with AUC > 0.52: vix, A2, pre_event_idiosyncratic_vol  (3 of 24)
features with |t| > 1.5:  rel_ret_5d, rel_ret_30d, vix          (3 of 24)
features surviving BH-FDR at q<0.10:  NONE
```

Median AUC is 0.4993 — a coin flip. The best single feature in the entire
deploy set reaches AUC 0.541. Not one feature survives multiple-testing
correction. Era splits flip sign for most of the "best" names
(vix +1.55 → +0.86; unemployment_roc21 +1.21 → −0.36; rel_ret_20d
−0.83 → +0.18).

## The 22 "retired" features are also redundant

12 pairs with |Spearman| > 0.7, e.g.:

```text
revision_momentum_90d ~ revision_ordinal_momentum_90d   0.963
grade_dispersion_90d  ~ n_analysts_covering             0.914
rel_ret_20d           ~ sector_adjusted_ret_20d         0.901
n_analysts_covering   ~ last_action_days_before_earnings −0.798
revision_momentum_60d ~ revision_momentum_90d           0.781
```

So the deploy set is ~14 independent weak variables, not 22 — and each
one is individually indistinguishable from noise.

## The model was fitting noise

XGBoost gain is nearly UNIFORM across all 24 features (0.031–0.061) —
the signature of a model partitioning noise when no feature carries
signal. Its AUC on its own label is 0.53, exactly what combining 24
coin-flip features produces.

## Honest correction on the peer feature

F1_sb_h3 (the RC-18 survivor) has, in the deploy configuration,
spread +0.088% (t=0.27) and AUC 0.5014 — no detectable univariate
signal. Its RC-18 result (+0.64pp, CI [+0.10,+1.19]) was measured
beats-only, out-of-sample, on car_10d, orthogonalized. Either that edge
is smaller than the deploy-configuration noise floor, or it is specific
to that measurement. It should not be treated as established signal.

## What this means

The post-leak program (RC-16 → RC-22) has been searching an EMPTY feature
space. That explains every result coherently:

- the model's AUC ≈ 0.53 on its own label;
- raw picks averaging ≈ 0 (indistinguishable from random and from the
  rejected group);
- positive backtest numbers only ever appearing through allocation
  artifacts (RC-22);
- every CI including zero.

The "baggage" from the look-ahead era is not only stale thresholds and
threshold-tuned machinery — it is that **the feature set itself has no
content once the leaked feature is removed.** No amount of modeling,
label redesign, feature combination, portfolio design, or threshold work
can extract signal that is not present.

The only productive direction is NEW INFORMATION (not new models on these
inputs): e.g. earnings-call transcripts/guidance text, options-implied
expectations, intraday/order-flow reaction, short interest, supply-chain
or peer-text signals. Each is a data-acquisition program, not a modeling
program.
