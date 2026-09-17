# Market-Adjustment Check — the high-confidence edge was BETA (2026-09-16)

## The catch

`pregap_return` (the trade return used by every backtest in this program)
is a RAW return: `xp/ep - 1`. The model's LABEL (`car_10d`) is
benchmark-EXCESS. So all strategy returns were measured with market beta
included while the model was trained on excess drift.

Recomputed with excess returns (stock minus its own benchmark — IJH for
SP400, IJR for SP600 — over the identical entry->exit window):

```text
                                   RAW        EXCESS
all OOS events (n=5,647)         +0.115%     -0.279%    (beta +0.394pp)
beat  (n=3,792)                  +2.109%     +1.671%
miss  (n=1,855)                  -3.961%     -4.265%
```

The beat/miss asymmetry survives market adjustment (it is real), but the
strategy candidates do not:

```text
condition                    n      RAW EV     EXCESS EV    excess t   CI95
p>=0.8                     914     +0.646%     +0.022%       0.07    [-0.62,+0.69]
p>=0.8 & vix hi            308     +1.988%     -0.146%      -0.24    [-1.33,+1.06]
p>=0.8 & mom20 hi          302     +1.649%     +0.969%       1.36    [-0.43,+2.34]
p>=0.8 & mom10 hi          302     +1.426%     +0.773%       1.16    [-0.59,+2.04]
p>=0.8 & vix hi & mom20 hi 107     +3.170%     +1.050%       0.95    [-1.13,+3.19]
all & vix hi              1878     +2.156%     +0.053%       0.24    [-0.38,+0.49]
```

**The entire "high-confidence beat" edge (+0.646%) was market beta.** In
excess terms it is +0.022% (t=0.07). The vix conditioning is definitively
beta (raw +1.99% -> excess -0.15%). Only the momentum conditioning
survives partially (+0.97% excess) and it is not significant.

Why the beta was so large here: the model's high-confidence events
cluster in time, and those periods had strong market returns. The RC-22
program arms were less time-clustered, so their beta contribution was
only +0.07 to +0.34pp per trade:

```text
arm         n     RAW       EXCESS     beta pp
v6n_new    177   -0.445%   -0.566%    +0.121
rc19_new   229   -1.163%   -1.505%    +0.342
v7n_new    148   -1.209%   -1.156%    -0.053
rc20_new   221   -0.806%   -0.879%    +0.073
rc21_new   282   +0.015%   -0.194%    +0.210
```

So the historical conclusions are unchanged (all still fail), but this is
the THIRD integrity issue found in the program's measurement layer:

```text
1. RC-16: a feature (car_drift) leaked future information
2. RC-22: the simulator's slot allocation had look-ahead
3. now:    the trade return measure included market beta while the label
           was excess  -> any strategy evaluation must report EXCESS returns
```

## Consequence for the beat strategy

The pattern is now complete and coherent in excess terms:

- the model predicts the beat (AUC 0.673) using priced information;
- expected beats earn ~0 excess (p>=0.8 bucket: +0.022%);
- unexpected beats earn a lot (low-p_beat beats +3.6% raw / positive excess);
- unexpected beats cannot be predicted pre-print from these features.

The high-confidence beat rule is therefore not a candidate. The only
residual is the momentum conditioning (+0.97% excess, t=1.36, n=302),
which is not significant and must not be promoted.
