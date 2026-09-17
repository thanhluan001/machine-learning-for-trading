# Beat-Prediction Diagnostic (intermediate target) — 2026-09-16

User proposal: stop targeting CAR T+10 >= +3% (too hard); target the
intermediate goal — BEAT the earnings — and see how predictable it is.

## 1. Base rates (combined SP400+SP600, n=24,151 with a defined beat)

```text
overall beat rate            68.1%
SP400 (sue>0)                ~68%      SP600 (eps>actual)  ~68%
by year                     2015:67% 2017:69% 2019:63% 2021:75% 2023:66% 2025:68%
by sector (SP400)           XLU 54%  XLRE 63%  IJS 66%  XLB 68%  IJJ 73%  IJH 74%  IJK 75%
SUE (SP400)                 mean 0.690  sd 1.227  (q10/50/90: -0.75 / 0.59 / 2.37)
```

## 2. Economics of the beat (5-session harvest)

```text
             beat                     miss                    spread
SP400   +1.762% (n=8,290)      -3.022% (n=3,249)          +4.78pp
SP600   +2.217% (n=8,149)      -3.688% (n=4,463)          +5.91pp
ALL     +1.988% (n=16,439)     -3.408% (n=7,712)          +5.40pp
```

The beat is very strongly related to the harvest — a 5.4pp spread.

## 3. Predictability: the first real model of the program

```text
XGBoost, 22 deploy features, label-matured folds:
  OOS AUC(beat) = 0.6731   (n=5,647)
  SP400 0.6936 (n=2,279)   SP600 0.6490 (n=3,368)
  fold AUCs 0.654 / 0.672 / 0.705 / 0.664
top decile: predicted p_beat 0.878 -> actual beat rate 90.6%
top-2 deciles: 86.2% beat rate (base 68%)
top-8 gain: consecutive_surprises_pre 0.191, sue_lag_1 0.169, sue_lag_2 0.073
```

## 4. But the predictable beat is the PRICED beat

```text
harvest by p_beat decile, split by actual outcome:
dec      n_beat   beat harvest    n_miss   miss harvest
 1         259      +3.63%          306      -3.99%
 5         372      +2.90%          193      -4.67%
10         512      +0.76%           53      -5.36%

=> within beats, HIGHER predicted probability = LESS drift (+3.6% -> +0.8%)
   within misses, higher predicted probability = worse (-4.0% -> -5.4%)
   the two effects cancel: aggregate harvest is FLAT across deciles
   (top-2 deciles +0.236% vs unconditional +0.095%)
```

The model predicts the beat by knowing that serial beaters beat again —
and the market knows that too.

## 5. Ablation: the predictability IS the serial (priced) component

```text
with    sue_lag_1, sue_lag_2, consecutive_surprises_pre:  AUC 0.6849
without those three features:                             AUC 0.5286
```

Removing three features destroys the model. The beat is predictable only
through information the market already prices.

## 6. Conditional-on-beat feature spreads (the surprise effect)

```text
conditional on a BEAT (n=8,290, mean harvest +1.762%):
  sue_lag_2  q1 +2.50% -> q5 +0.82%   spread -1.68pp  t=-5.22
  sue_lag_1  q1 +2.47% -> q5 +1.31%   spread -1.15pp  t=-3.43
  vix        q1 +1.24% -> q5 +2.97%   spread +1.72pp  t=+2.85
  idio_vol   q1 +1.32% -> q5 +2.58%   spread +1.26pp  t=+2.82
  consec_surprises_pre  q1 +2.50% -> q5 +1.58%  t=-2.71
```

The strongest conditional signals in the program's history. A beat after a
WEAK surprise history drifts ~2.5%; a serial beater's beat drifts ~0.8%.
Consistent with the priced-beat mechanism.

## 7. Why it is NOT tradable (honest negative)

The conditional information cannot be used: you cannot condition on an
outcome you have not observed, and the pre-print proxy (low sue_lag_2)
brings many misses whose losses cancel the beat gains.

Attempts to combine the two layers (high p_beat x weak prior surprise)
are NOT ROBUST:

```text
combined sample (n=5,647):  q5-q1 gradient strongest in LOW  tercile (+2.93, CI [+0.35,+5.48])
SP400-only sample (n=2,279): q5-q1 gradient strongest in HIGH tercile (+2.76); low = +0.18
```

The cell flips between samples — an artifact of small extreme cells, not a
structure. Removing the serial features from the beat model (the
"principled" version) gives AUC 0.529 and no coherent tercile structure.

## Conclusion

The user's hypothesis is confirmed in the prediction sense and refuted in
the economic sense: **the beat IS predictable (AUC 0.67) — but exactly to
the extent that it is already priced.** The drift lives in the surprise,
and the surprise is not predictable from this feature set (AUC 0.53
without the serial features). This is the same conclusion as the feature
audit, reached from a different direction and with a sharper mechanism:
the deploy features measure EXPECTED performance (revisions, past
surprises, momentum), all public and priced; PEAD pays for UNEXPECTED
performance. New information (transcripts/guidance, options-implied
expectations, intraday reaction) is required — not new models on these
inputs.

---

## ADDENDUM — accuracy of the beat model, and the precision→return gradient

### Accuracy is dominated by the 68% base rate

```text
OOS n=5,647; base rate 67.2%  =>  naive "always beat" = 67.2% accuracy,
                                  67.2% precision, 100% recall, AUC 0.500

thr   pred_beat   TP    FP   FN    TN   acc%   prec%  rec%  spec%
0.5      5,013  3,500 1,513  292   342   68.0   69.8   92.3   18.4
0.6      3,766  2,809   957  983   898   65.6   74.6   74.1   48.4
0.7      2,290  1,840   450 1,952 1,405   57.5   80.3   48.5   75.7
0.8        914    805   109 2,987 1,746   45.2   88.1   21.2   94.1
0.9        147    140     7 3,652 1,848   35.2   95.2    3.7   99.6
```

Accuracy adds +0.8pp over the naive rule at thr 0.5 and falls BELOW it at
higher thresholds — accuracy is the wrong lens for a 68%-base-rate target.
The informative view is precision/recall: 74.6%/74.1% at 0.6;
80.3%/48.5% at 0.7; 88.1%/21.2% at 0.8; 95.2%/3.7% at 0.9. Top decile:
90.6% precision, 13.5% of all beats captured; top-2 deciles 25.7%.

### Does precision buy return? Monotone — and that is new

```text
thr    n     prec%   harvest%    SE     t      car_10d%
all  5,647   67.2    +0.095    0.16   0.59    -0.488
0.5  5,013   69.8    +0.146    0.17   0.86    -0.425
0.6  3,766   74.6    +0.268    0.19   1.41    -0.389
0.7  2,290   80.3    +0.394    0.24   1.64    -0.284
0.8    914   88.1    +0.597    0.38   1.57    -0.149
0.9    147   95.2    +1.328    1.09   1.22    -0.333
```

The harvest rises monotonically with the beat-probability threshold —
the first monotone economic gradient this program has produced, measured
out-of-sample. But NO slice is individually significant (t = 0.6–1.6),
and the top slice has only 147 events.

### The source of the gradient is miss-avoidance, not drift-finding

```text
WITHIN beats (n=3,792), p_beat quintiles:
  q1 +2.816%  q2 +2.702%  q3 +1.897%  q4 +1.885%  q5 +1.195%
  (AUC p_beat -> harvest>0 = 0.481; -> car_10d>0 = 0.511; -> car_10d>3% = 0.513)
```

Within beats the model cannot rank drift at all (AUC ≈ 0.48–0.51) and the
gradient is NEGATIVE — the confident beats are the priced ones. The
aggregate gradient therefore comes entirely from cutting the miss rate
(45.8% → 4.8% across deciles), not from finding bigger drifters.

### Status

A pre-registered "high-confidence beat" strategy (threshold fixed in
advance, e.g. 0.8) is the first candidate in this program with a
monotone, out-of-sample, economically sensible gradient. Power is the
binding constraint: thr 0.8 gives 914 trades at +0.597% (SE 0.38, t=1.57);
certification needs roughly 1.5–2x more trades at that effect size.
Threshold selection here is post-hoc and must be frozen before any test.
