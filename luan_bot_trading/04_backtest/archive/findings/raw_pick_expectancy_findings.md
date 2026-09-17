# Raw-Pick Expectancy (no portfolio) — 2026-09-16

Question (user): forget slots/weeks — for all events scoring >= 0.33,
is the mean return distinguishable from 0?

## Answer: no. And it is indistinguishable from random events.

```text
group                        n     pregap(5sess)   SE     t    win%   car_10d
UNCONDITIONAL (all events in the same OOS test windows)
  SP400 all               2,364     +0.126%     0.218  0.58   50.8   −0.504%
  v7c SP400 all           2,380     +0.115%     0.218  0.53   50.7   −0.524%
  v7c SP600 all           3,636     +0.289%     0.218  1.32   49.4   −0.462%
  v7c COMBINED all        6,016     +0.220%     0.158  1.40   49.9   −0.486%

PICKS >= 0.33
  rc19  SP400 24feat      1,060     +0.180%     0.372  0.48   51.7   −0.862%
  rc20  COMBINED 24feat     906     +0.105%     0.535  0.20   49.3   −0.685%
  rc21  SP600 transfer    2,221     +0.341%     0.317  1.07   49.0   −0.771%
  v6n   SP400 3-gate        538     −0.318%     0.510 −0.62   48.3   −0.770%
  v7n   COMBINED 3-gate     391     −0.579%     0.718 −0.81   47.1   −0.546%

REJECTED (score < 0.33), backed out of universe minus picks
  rc19 rejected           1,304     +0.082%
  rc20 rejected           5,110     +0.240%
  rc21 rejected           1,415     +0.207%
```

The 24-feature models' picks are statistically indistinguishable from
(a) zero, (b) the unconditional mean, and (c) the REJECTED group. rc20's
picks are, if anything, WORSE than the unconditional mean and worse than
the events it rejected.

## Why — the label is informative, the MODEL is not

```text
THE LABEL IS STRONGLY INFORMATIVE about the 5-session harvest:
  pass_g1 = 0 (car_10d <= +3%):  pregap mean  −1.944%   (n=4,201)
  pass_g1 = 1 (car_10d >  +3%):  pregap mean  +5.229%   (n=1,815)
  corr(pregap_5sess, car_10d) = 0.404

THE MODEL BARELY ENRICHES THE LABEL:
  base rate pass_g1 = 30.2%
  rc19 picks 36.7%   rc20 picks 37.6%   rc21 picks 28.9% (no enrichment)
  within-pool quintiles (rc20): 35.7 / 32.0 / 39.2 / 40.9 / 40.3 %
  AUC(score, pass_g1) ≈ 0.53 (SP400 0.534, SP600 0.529) — near coin-flip

AND THE FAILING PICKS LOSE MORE THAN THE UNIVERSE'S FAILURES:
  rc20 picks: g1=1 → +6.006% (universe +5.229%) but g1=0 → −3.456% (universe −1.944%)
  => 37.6% × 6.006 + 62.4% × (−3.456) = +0.101%  (vs universe +0.220%)
```

The model selects higher-variance events with a slightly enriched success
rate; the enriched success does not cover the deepened failures. Net ≈ 0,
slightly negative.

## Conclusion

There is no deployable edge at the raw-pick level, so the portfolio
question is indeed moot (user's framing is correct). But the failure is
NOT the label (it separates returns by ~7pp) and NOT the portfolio layer —
it is **predictive skill**: AUC ≈ 0.53 on the model's own label. Any
future work must raise the predictor's discrimination on pass_g1; nothing
downstream can compensate for a coin-flip ranker.
