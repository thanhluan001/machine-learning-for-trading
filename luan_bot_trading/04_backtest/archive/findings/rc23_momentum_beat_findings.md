# RC-23 Closure — Momentum-Conditioned High-Confidence Beat (FAIL on power, as pre-stated)

**Status:** CLOSED 2026-09-16. G1 FAIL per registration (3c4f5b4). No
conditioning variants, no threshold search.

## Result

```text
rule: p_beat >= 0.80 AND rel_ret_20d >= 0.0369 (frozen constants)
  n=302 | mean EXCESS +0.969% | SE 0.71 | t=1.36 | CI95 [−0.427, +2.358]
  G1: mean > 0 ✓ but CI excludes 0 ✗  → FAIL

declared non-gating diagnostics:
  beat rate of picks 89.4%
  SP400 slice: 190 trades, +0.663% excess
  SP600 slice: 112 trades, +1.490% excess
  raw mean +1.671% (vs excess +0.969% — beta +0.70pp, partially
    time-clustered again)
  unconditioned p>=0.8 baseline: +0.022% excess (reproduced exactly)
```

The registered rule reproduced the exploratory estimate bit-for-bit
(+0.969%, t=1.36) — the execution pipeline is verified.

## Cause of death: power, exactly as pre-stated

The registration said: "expected n ≈ 302; detection floor ≈ 1.6pp/trade
at t=2; the exploratory estimate (+0.97%) is below the floor; an
inconclusive FAIL is the expected outcome." That is what happened.

Forward accumulation cannot rescue it: at ~60 qualifying trades/year,
reaching n≈650 (t=2 at a true +0.97%) takes ~11 years. If the effect is
real, it is below any practical detection floor at this trade frequency.

## Final state of the research line

Every candidate constructed in this program is now closed, and the
closures form one coherent picture:

```text
1. FEATURE AUDIT:  the 22 deploy features carry no signal (median
                   AUC 0.4993; zero survive FDR); the leak-era edge was
                   one feature (car_drift) that smuggled the answer.
2. BEAT MODEL:     the beat IS predictable (AUC 0.673) — entirely via
                   serial-surprise information the market already prices;
                   excess expectancy of its picks ≈ 0.
3. PAYOFF STRUCTURE: excess drift lives in SURPRISING beats (low p_beat);
                   surprises are not predictable from these features
                   (AUC 0.53 without the serial block).
4. MEASUREMENT:    three integrity issues found and fixed — the feature
                   leak (RC-16), the simulator's allocation look-ahead
                   (RC-22), and raw-return measurement vs the excess
                   label (the beta catch).
5. LAST CANDIDATE: momentum-conditioned high-confidence beat fails its
                   gate on power and cannot be revived by patience.
```

The honest conclusion remains: no deployable edge exists in the current
feature universe, and the only productive direction is new information
sources, not new models on these inputs.
