# Short-Side Diagnostic (user idea: short the miss) — 2026-09-16

Premise: misses are heavily punished (-3.4% avg; even expected misses
-5.4%), misses are fewer (32%), so shorting them might pay more per event
than going long the beat.

Method note (important): `pregap_return` embeds a LONG-side 10% stop, so
negating it would overstate short returns. All numbers below recompute
returns from prices, with a +10% upside stop modelled separately for the
short side. Sanity: corr(long_stop, pregap_return) = 0.984.

## The model's p_beat is compressed — few "predicted miss" events exist

```text
p_beat quantiles (1/5/10/25/50/75/90): 0.387 0.458 0.495 0.566 0.661 0.764 0.834
events with p_beat < 0.5: 634 (11.2%)    < 0.4: 79 (1.4%)    < 0.3: 3
```

## Long vs short at their operating points (recomputed, gross)

```text
side   rule     n     hit%   mean%    SE    t     tail
LONG   p>=0.8   914   51.2   +0.646  0.38  1.69  P(loss>10%)=15.5%  min=-44%
LONG   p>=0.7  2,290  50.8   +0.415  0.24  1.73  P(loss>10%)=15.6%  min=-54%
SHORT  p<0.5    634   53.5   +0.277  0.56  0.49  P(loss>10%)=16.9%  min=-66%
SHORT  p<0.4     79   60.8   +2.663  1.75  1.52  P(loss>10%)=13.9%  min=-40%

COMBINED equal-weight (long p>=0.8 + short p<0.5):
  nL=914 nS=634  combined +0.494%/trade (SE 0.32, t=1.54)
```

## Why the short side underperforms despite the bigger payoff

The model cannot isolate misses. Precision on the short side is
54-61%, versus 88-95% on the long side. The reason is structural: the
model is calibrated to a 68% beat base rate, so genuinely low
probabilities are rare and imprecise. The asymmetry of the payoff
(miss magnitude > beat magnitude) does not compensate for the asymmetry
of the precision.

## Tail risk and costs

```text
shorted names' 5-session returns (p_beat<0.5): median -0.6%,
  75th +6.8%, 90th +15.4%, 95th +23.2%, 99th +40.3%, max +65.9%
=> 16.9% of shorted names rise >10%; a +10% stop is mandatory
   (it costs 0.016pp: short mean +0.261% with stop vs +0.277% without)
borrow cost over a 5-session hold: 3%/yr -> 0.06%; 20%/yr (hard-to-borrow
   small caps) -> 0.39%  => at realistic SP600 borrow rates the +0.277%
   edge is erased; availability is also a practical constraint
```

## Verdict

The user's observation about payoff asymmetry is correct, but the short
side is not the better expression of it: the model identifies misses too
weakly (54-61% precision), the gross edge is +0.28%/trade (t=0.49) versus
+0.65% (t=1.69) for the long high-confidence side, and borrow costs plus
tail risk consume what remains. Adding it to the long book does not
improve the t-statistic (1.54 combined vs 1.69 long-only).

The single best candidate remains LONG p_beat >= 0.8: n=914, +0.646%/trade
(t=1.69), still short of certification (needs ~1.5-2x the sample at that
effect size), threshold to be frozen before any test.
