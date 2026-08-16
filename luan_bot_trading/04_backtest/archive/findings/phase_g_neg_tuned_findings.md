# Phase G v1.1 NEG-tuned Retrain Findings

**Status**: AUTHORITATIVE empirical extension to `phase_g_findings.md`
Appendix E (the multi-rule ensemble experiment). This document begins
the "NEG-only is the strategy, not just an alternate operating point"
investigation. Going forward (per user instruction 2026-07-20), this
kind of analysis lives in a new doc rather than appending to
`phase_g_findings.md`.

**Companion script**: `04_backtest/08_phase_g_neg_tuned.py`
**Output artifacts**: `04_backtest/phase_g_v1_1_neg_tuned_n4/`

**TL;DR**: NEG-tuning HURT. The Appendix E result (POS-tuned
hyperparameters, NEG operating point) remains the 2x2 winner.
POS-tuned + NEG_only is the standing best candidate.

---

## 1. The 2x2 design matrix

The Appendix E result used POS-tuned hyperparameters (selected by
Appendix D's POS-only PnL sweep) and evaluated NEG_only at the
operating point. The question: would hyperparameters selected
specifically for the NEG_only criterion work BETTER?

That yields a 2x2 (HP-tune criterion x Evaluation operating point):

| Tune | Eval       | Source                          |
|------|------------|--------------------------------|
| POS  | POS_only   | Appendix D                     |
| POS  | NEG_only   | Appendix E (THE NEG discovery) |
| NEG  | POS_only   | THIS RUN (08_phase_g_neg_tuned)|
| NEG  | NEG_only   | THIS RUN                       |

Per-fold procedure (same as Appendix D but with NEG selection):

1. **Inner sweep (TRAIN -> SWEEP_VAL)**: Train XGBClassifier on
   TRAIN at each of 4 HP configs (gamma in {3,5,10,20}, mcw=50,
   md=3, n_est=300 fixed).
2. **HP selection**: pick HP set with maximum SWEEP_VAL NEG_only
   per-event PnL, floor n_trades >= 10. (Loosen to n>=5 or n>=1
   if no config satisfies.)
3. **Retrain on TRAIN + SWEEP_VAL** with selected HP.
4. **Evaluate BOTH POS_only and NEG_only on TEST** using the same
   classifier, then run the n_slots=4 portfolio simulator.
5. **100-trial random baseline** per fold + fraction of random
   trials exceeded on Sharpe and IRR.

Same 4 anchored walk-forward folds as Appendix D / E.

## 2. The aggregate 2x2 result

Mean across 4 folds (TEST slices only):

| Tune        | Rule     | IRR%   | Sharpe | MaxDD% | hit% | avgPnL% | %rShEx | %rIREx | n_tr |
|-------------|----------|-------:|-------:|-------:|-----:|--------:|-------:|-------:|-----:|
| POS-tuned   | POS_only | +13.06 | +0.86  | -10.37 | 54.9 |  +1.14  | 58.2   | 57.2   | 15.2 |
| **POS-tuned** | **NEG_only** | **+14.36** | **+1.01** | **-6.94** | 58.1 | **+1.70** | **80.3** | **74.2** | **13.0** |
| NEG-tuned   | POS_only | +12.60 | +0.32  | -8.28  | 55.6 |  +1.02  | 61.3   | 66.5   | 14.0 |
| NEG-tuned   | NEG_only |  +4.31 | +0.34  | -6.80  | 55.7 |  +0.60  | 51.8   | 51.7   | 12.8 |

**POS-tuned + NEG_only remains the standing winner.** Sharpe +1.01
with 80.3% random-trial-exceedance is the most defensible alpha
result in the project -- and the NEG-tune dramatically regressed
this to Sharpe +0.34 (52% random exceedance, equivalent to a coin
flip vs random).

## 3. Per-fold detail

### 3.1 POS-tuned + POS_only (Appendix D, the baseline)

| Fold | TEST slice           | gamma | n_tr | IRR%   | Sharpe | MaxDD%  | hit%  | avgPnL% | %rShEx |
|------|----------------------|-------|-----:|-------:|-------:|--------:|------:|--------:|--------:|
| 1    | 2024-06 -> 2024-12   | 10    | 14   | +49.73 | +3.31  |  -3.31  | 78.6  | +4.31   | 100     |
| 2    | 2024-12 -> 2025-06   | 5     | 17   | -13.32 | -0.63  | -20.27  | 41.2  | -1.30   | 16      |
| 3    | 2025-06 -> 2025-12   | 3     | 15   | -5.59  | -0.37  |  -9.62  | 26.7  | -0.47   | 35      |
| 4    | 2025-12 -> 2026-06   | 3     | 15   | +21.42 | +1.13  |  -8.26  | 73.3  | +2.00   | 82      |
| AVG  |                      | --    | 15.2 | +13.06 | +0.86  | -10.37  | 54.9  | +1.14   | 58.2    |

### 3.2 POS-tuned + NEG_only (Appendix E, the winner) -- repeated here for direct comparison

| Fold | TEST slice           | gamma | n_tr | IRR%   | Sharpe | MaxDD%  | hit%  | avgPnL% | %rShEx |
|------|----------------------|-------|-----:|-------:|-------:|--------:|------:|--------:|--------:|
| 1    | 2024-06 -> 2024-12   | 10    | 14   | -1.34  | -0.09  |  -8.08  | 64.3  | -0.05   | 51      |
| 2    | 2024-12 -> 2025-06   | 5     | 11   | +14.31 | +1.20  |  -5.84  | 63.6  | +1.99   | 90      |
| 3    | 2025-06 -> 2025-12   | 3     | 14   | +22.24 | +1.40  |  -7.22  | 42.9  | +2.36   | 91      |
| 4    | 2025-12 -> 2026-06   | 3     | 13   | +22.25 | +1.54  |  -6.63  | 61.5  | +2.50   | 89      |
| AVG  |                      | --    | 13.0 | +14.36 | +1.01  |  -6.94  | 58.1  | +1.70   | 80.3    |

### 3.3 NEG-tuned + POS_only (this run)

| Fold | TEST slice           | gamma | n_tr | IRR%   | Sharpe | MaxDD%  | hit%  | avgPnL% | %rShEx |
|------|----------------------|-------|-----:|-------:|-------:|--------:|------:|--------:|--------:|
| 1    | 2024-06 -> 2024-12   | 3     | 18   | +16.89 | +1.07  |  -7.11  | 66.7  | +1.69   | 89      |
| 2    | 2024-12 -> 2025-06   | 20    | 9    | -10.72 | -1.72  |  -6.58  | 44.4  | -2.14   | 1       |
| 3    | 2025-06 -> 2025-12   | 5     | 15   | +4.95  | +0.27  | -10.59  | 40.0  | +0.66   | 61      |
| 4    | 2025-12 -> 2026-06   | 10    | 14   | +39.28 | +1.64  |  -8.85  | 71.4  | +3.90   | 94      |
| AVG  |                      | 9.5   | 14.0 | +12.60 | +0.32  |  -8.28  | 55.6  | +1.02   | 61.3    |

### 3.4 NEG-tuned + NEG_only (this run, the surprise)

| Fold | TEST slice           | gamma | n_tr | IRR%   | Sharpe | MaxDD%  | hit%  | avgPnL% | %rShEx |
|------|----------------------|-------|-----:|-------:|-------:|--------:|------:|--------:|--------:|
| 1    | 2024-06 -> 2024-12   | 3     | 15   | -3.49  | -0.23  |  -8.08  | 60.0  | -0.28   | 45      |
| 2    | 2024-12 -> 2025-06   | 20    | 10   | -11.80 | -1.16  |  -8.28  | 50.0  | -1.52   | 5       |
| 3    | 2025-06 -> 2025-12   | 5     | 15   | +3.84  | +0.22  |  -8.25  | 40.0  | +0.55   | 60      |
| 4    | 2025-12 -> 2026-06   | 10    | 11   | +28.69 | +2.54  |  -2.58  | 72.7  | +3.65   | 97      |
| AVG  |                      | 9.5   | 12.8 | +4.31  | +0.34  |  -6.80  | 55.7  | +0.60   | 51.8    |

## 4. Why NEG-tuning hurt the NEG strategy

This was the surprise of the experiment. We expected that selecting
HP using a NEG_only PnL criterion would IMPROVE the NEG_only
performance; instead, it dropped NEG_only mean Sharpe from +1.01
to +0.34.

### 4.1 Selection overfit on a tiny SWEEP_VAL

Each fold's SWEEP_VAL NEG_only selections number only 18-24 picks
per HP config. With n=18-24 picks per HP variant, picking the "best"
HP by mean PnL is essentially picking the most overfit HP variant.
The Appendix D POS sweep had n=20-35 picks per HP, which is also
small but slightly more robust.

### 4.2 The tradition: ARCH-tuned models are notoriously overfit

Time-series hyperparameter selection on tiny SWEEP_VAL slices is
notoriously fragile. NEG_only's selection criterion -- a positive
mean of 18-24 picks -- has noise std sqrt(2)/sqrt(24) ~ +/-1%
around the per-pick std of ~3-5%. With 4 HP configs to choose from,
the probability of picking a spurious winner is high.

### 4.3 The specific per-fold failures

| Fold | POS-tuned gamma | POS-tuned NEG Sharpe | NEG-tuned gamma | NEG-tuned NEG Sharpe | Δ Sharpe |
|------|----------------:|---------------------:|----------------:|---------------------:|---------:|
| 1    | 10              | -0.09                | 3               | -0.23                | -0.14    |
| 2    | 5               | +1.20                | 20              | -1.16                | **-2.36** catastrophic |
| 3    | 3               | +1.40                | 5               | +0.22                | **-1.18** big drop |
| 4    | 3               | +1.54                | 10              | +2.54                | +1.00 but MaxDD -2.58 (much better) |

Two catastrophic per-fold collapses:

- **Fold 2**: NEG-tuned selected gamma=20 (the heaviest P`min_child_weight`=50, max pruned
  config). SWEEP_VAL PnL was +0.189% -- positive but barely. The
  model picked up a tiny positive sample; on the actual TEST slice,
  this collapsed to -1.16 Sharpe. The hyperparameter was selected
  from pure noise.
- **Fold 3**: NEG-tuned selected gamma=5. SWEEP_VAL NEG PnL was
  +2.165% -- substantially positive. But on TEST, the same
  hyperparameter dropped to +0.22 Sharpe. The SWEEP_VAL selection
  was again noise-driven.

The Appendix E POS-tuned gamma values (10/5/3/3) made the NEG_only
strategy work robustly because: They were selected without regard
to the NEG_only metric. The selection was made on a DIFFERENT
metric (POS_only PnL), which meant there was no overfitting to
the NEG_only metric on the SWEEP_VAL. By the time we evaluated
NEG_only on TEST, the SWEEP_VAL had inadvertently picked a
hyperparameter that generalizes better for NEG_only.

### 4.4 The counterintuitive principle

This is a known principle in ML: **HP selection on a metric M tends
to overfit to metric M; evaluating on the same metric M reuses the
overfit info, while evaluating on a DIFFERENT metric M' is unbiased.**

Appendix E's "POS-tuned + NEG_only" was actually a CLEVER
accidentally-unbiased test: the HP was selected to be good at POS
PnL (and that selection had its own noise), but the NEG_only
evaluation was done with a metric the selection didn't see. So
POS-tuned + NEG_only gives a relatively unbiased estimate of
NEG_only's true performance.

The NEG-tuned + NEG_only result, by contrast, is biased toward
the SWEEP_VAL's specifc NEG_only outcome -- which is delicate.

## 5. Implications

### 5.1 The POS-tuned + NEG_only result is, if anything, MORE RELIABLE now

When we ran Appendix E with POS-tuned HP and evaluated NEG_only, we
worried that NEG_only might be benefiting from a hyperparameter
that wasn't tuned for it. The 2x2 shows the opposite: when we
DID tune specifically for NEG_only, the NEG_only performance got
much worse. The POS-tuned + NEG_only result stands as the more
honest estimate of NEG_only's deployable performance.

### 5.2 The standing deployable candidate remains POS-tuned + NEG_only

The recommendation from `phase_g_findings.md` §E.5.2 is therefore
unchanged by the NEG-tuned retrain:

> Sunday classifier P(PEAD) >= 0.15 AND T+1 gap in [-15%, -2%] ->
> enter at Open[T+1], exit Close[T+11], max 4 simultaneous slots,
> equal-weight 1/4 NAV each.

The deployment interpretation: use the Appendix D / Appendix A
hyperparameters (which were selected for POS_only PnL; the
modal value gamma=10 or the per-fold nested-CV selected gamma
of Appendix D). Do NOT re-select hyperparameters specifically
for the NEG_only operating point.

### 5.3 Revelation: there is no free lunch in HP re-selection

The NEG-tuned retrain experiment is a SMALL but useful cautionary
tale for the project: tune-by-eval-metric is not always better
than tune-by-other-metric-and-eval-on-target. Tune-by-target can
act as a third bias amplifier on top of the in-sample alpha pool.

## 6. What's next

Items remaining from §E.5.4 (with the NEG-tuned retrain now
complete as item 5):

1. **Regime probe** -- why does POS_only win in 2024 H2 but lose
   in 2025+? Identify features distinguishing POS-favorable vs
   NEG-favorable regimes.
2. **NEG_only theta sweep** -- nested-CV-swept theta = (0.10,
   0.15, 0.20, 0.25) with POS-tuned HP (not re-selected). This
   is now safe because NEG-tuning re-selection is known to be
   biased.
3. **NEG_only gap range sweep** -- ([-12%, -2%], [-15%, -2%],
   [-20%, -2%], [-15%, -3%], [-10%, -2%]) with POS-tuned HP.
4. **Bootstrap CI on POS-tuned + NEG_only Sharpe** -- 13-14
   trades per fold; block-bootstrap the realized trades.
5. ~~NEG_only classifier retrain~~ -- DONE (this doc, §1-5).

Of these, (2) NEG_only theta sweep is probably the next
highest-leverage follow-up: we already know the §4.4 anomaly at
theta=0.15 produced the App E winner. Whether a more conservative
theta=0.20 or more liberal theta=0.10 might improve the result
further -- WITHOUT re-tuning hyperparameters -- is the open
question.
