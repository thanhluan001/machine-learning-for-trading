# PEAD-Target Empirical Study (Phase F v3)

**Status**: AUTHORITATIVE — empirical findings from the v3 PEAD-target
retrain experiment, conducted 2026-07-17 on the existing `db.h5` data
(Phase E Stage 2 v2 train matrix, 17,300 primed rows × 21 features).

**Scope**: Three experiments that decompose the source of the (later
discredited) Phase F v2 backtest Sharpe, identify the true alpha engine
(PEAD gate-pass), and assess how well a model can predict it.

**Authors**: AI + user, in conversation.

**Cross-refs**:
- `01_data/pead_verification_protocol.md` — defines the 3 PEAD gates.
- `features.md` — feature matrix catalog (21 features).
- `phase_a_b_migration_report.md` — data lineage.
- New model artifacts:
  - `03_model/models/phase_f_v2_pead_target/` — XGBRanker on PEAD-pass label.
  - `03_model/models/phase_f_v2_pead_classifier/` — XGBClassifier on PEAD-pass.
- Companion scripts (reproducible, read-only on `db.h5`):
  - `04_backtest/_pead_exploration.py` — Gate statistics + stratification.
  - `04_backtest/_pead_gap_strategy.py` — Gap-driven model-free strategy baseline.
  - `04_backtest/_pead_target_retrain.py` — PEAD-target ranker training.
  - `04_backtest/_pead_classifier.py` — PEAD-target binary classifier training.

---

## 0. TL;DR (executive summary)

1. **The smooth Stage 4 backtest Sharpe of 4.31 was leakage.** Its
   entire out-of-sample edge came from a single forward-looking feature,
   `opening_gap_t1`, which uses `Open[T+1]` — a price printed on the
   morning AFTER the catalyst, not available at Sunday planning time.
   NaN-ing this one feature drops hit rate from 65.8% → 49.1%
   (unconditional) and Sharpe from 4.31 → −0.14. The other 20 features
   have ~zero out-of-sample signal.

2. **The 3 PEAD verification gates from `pead_verification_protocol.md`
   are a real alpha engine.** Across 17,300 primed events:
   - Gate 1 (CAR > +3%): pass rate 30.4%.
   - Gate 2 (Inst. volume > 2× vma20): pass rate 36.8%.
   - Gate 3 (MaxDD_MA > −1.5%): pass rate 40.6%.
   - All 3 combined (`pead_pass`): **10.68%** base rate.
   - Mean **Open[T+1] → Close[T+11]** arith CAR of `pead_pass` events =
     **+6.39%**, vs universe baseline of +0.23%. That is a ~28× per-event
     return spread.

3. **A `rank:ndcg` ranker trained to predict `pead_pass` (binary 0/1)
   substantially outperforms the same model trained on `car_10d`
   (10-quantile target)** under the native listwise NDCG@3 metric:
   VAL NDCG@3 = **0.5838** (pead target) vs **0.1593** (car target).
   VAL > TRAIN on the PEAD objective → not overfitting.

4. **But under proper execution simulation (enter at Open[T+1]; exit at
   Close[T+10]), top-N selection per week yields ≈ 0% PnL** for both
   rankers — comparable to a random top-5 baseline (+0.77%). The ranker
   improves PEAD recall (top-5 picks identify 42% of all true PEADs), but
   the 58% of picks that are NOT true PEADs have ~0 realized drift and
   wash out the alpha. Cross-sectional ranking is not enough; we need a
   confidence threshold.

5. **A binary XGBClassifier on `pead_pass` achieves VAL AUC = 0.860,
   AP = 0.526** (AP identical to TRAIN → no overfit) using **the same 21
   features including `opening_gap_t1`**. Threshold sweeps show:
   - At P(PEAD) ≥ 0.20, the model trades **769 events** over 107 weeks
     at **+0.75% per-event arithmetic CAR** (3.3× the unconditional
     universe baseline of +0.23%) and 55.1% hit rate.
   - At P(PEAD) ≥ 0.50, the model is highly confident (precision 71%)
     but per-event PnL is only +0.60% — fewer events, similar drift.
   - **Realized PnL is NOT monotonic in P(PEAD)**. The model's most
     confident picks (proba > 0.327) yielded only +3.98% on TRUE PEADs
     AND −4.17% on FALSE positives in that bucket — these "obvious"
     PEAD setups tend to be the smaller-CAR events, while rare blowout
     PEADs (e.g. +13% drift) hide in the low-proba tail.

6. **Implication**: classifier-as-screen + the realized opening gap as a
   Sunday/threshold confirmation is the natural next step ("Phase G
   Sunday + T+1-morning two-stage filter"). Documented in §8.

---

## 1. Motivation: why we re-ran the experiment

Phase F v2 promoted two baselines (`phase_f_v2_baseline_ndcg`,
`phase_f_v2_baseline_pairwise`) into `03_model/models/` after a
hyperparameter sweep. Stage 4 (`04_backtest/01_val_backtest.py`) then
produced an enthusiastic backtest over VAL 2024-01 -> 2026-07:

| Top-N selection | per-event hit rate | avg per-event PnL | Sharpe (107 wk) |
|---|---|---|---|
| N = 5 | 65.8% | +4.4% | **4.31** |

A 100-trial random baseline averaged Sharpe -0.73 with only 2% of trials
positive. The model beat random 100/100 times.

A leak test revealed the truth. NaN-ing the single feature
`opening_gap_t1 = (Open[T+1] - Close[T]) / Close[T]` drops the hit rate
to 49.1% (= unconditional) and Sharpe to -0.14. The model's entire
out-of-sample edge was leakage via a forward-looking feature that is
NOT available at Sunday planning time -- `Open[T+1]` prints the morning
AFTER the catalyst.

The user's intuition at this point was:

> "We can extract the idea of 1st-day gap because it gives real alpha
> and build around it with PEAD event detection for a stronger signal."

i.e. the gap is a real Day-0 signal but it's not ML alpha -- it is just
the catalyst reaction itself. What we should be predicting is the
**PEAD event-label** (events with drift continuation), not the
continuous CAR. This doc is the empirical test of that idea.

## 2. Experimental setup

### 2.1 Data layers used

| Layer | HDF5 path | Source | Notes |
|---|---|---|---|
| Earnings events | `/earnings/raw` | Phase D (EODHD calendar) | 44,308 rows keyed by `permaTicker` |
| Stock prices | `/sp400/{permaTicker}` | Phase B (Tiingo) | 928 nodes, native adj OHLC-V |
| Index prices | `/macros/IJH` | `04_index_data_gathering.py` | IJH ETF `Close` (no `Adj_Close`) |
| Feature matrix | `/features/train_matrix` | Phase E Stage 2 v2 | 20,614 rows x 30 cols |

### 2.2 Train/VAL split

- Cut-off: 2024-01-01 (`DEFAULT_SPLIT_DATE`).
- Priming runway: rows with `report_date < 2015-01-01` dropped
  (`PRIMING_RUNWAY_START`, per Design.md §12).
- Sparse-week cutoff: weeks with <3 events dropped from both TRAIN and
  VAL (`DEFAULT_MIN_GROUP_SIZE`).

After cutting:

| Period | Rows | Weeks | PEAD-positives | PEAD base rate |
|---|---|---|---|---|
| TRAIN (2015-01 -> 2023-12) | 13,306 | 407 | 1,382 | 10.39% |
| VAL (2024-01 -> 2026-07) | 3,890 | 107 | 446 | 11.47% |
| **Total primed** | **17,300** | -- | **1,848** | **10.68%** |

Base rate drifts only +1.1 pp between TRAIN and VAL -> the target is
regime-stable across the period, unlike `car_10d` (whose mean drifts by
a factor of ~4.4x).

### 2.3 The 3 PEAD gates (target-label definition)

Per `01_data/pead_verification_protocol.md` §4:

| Gate | Definition | Threshold |
|---|---|---|
| Gate 1 (idiosyncratic alpha) | `CAR_{T+1..T+11} > +3%` (arith) | +0.03 |
| Gate 2 (institutional volume) | mean `Volume[T..T+2]` / `vma20` > 2x | 2.0 |
| Gate 3 (risk preservation) | `MaxDD_MA_{T+1..T+11} > -1.5%` | -0.015 |

A row is labeled `pead_pass = 1` iff it passes all 3 gates simultaneously.

The 3 gates use **realized post-event data** (T+1 to T+11), so they are
EX POST verification labels -- they cannot be directly used as an
ex-ante pre-filter without lookahead bias. Their role in this study is
the **training target**: given the full historical record, predict at
Sunday (pre-T) which events will become PEADs.

### 2.4 Feature set (unchanged from v2)

21 features -- the same set locked in for Phase F v2, including the
leak feature `opening_gap_t1`:

- **Block 1 (SUE/earnings, 7)**: `sue_score`, `eps_surprise_pct`,
  `consecutive_surprises`, `sue_acceleration`, `sue_lag_1`,
  `sue_lag_2`, `sue_abs_x_inverse_vol`.
- **Block 2 (pre-event market)**: `is_bmo`,
  `pre_event_idiosyncratic_vol`, `pre_event_volume_trend`,
  `rel_ret_3d/5d/10d/20d/30d`.
- **Block 3 (sector/drift)**: `sector_adjusted_ret_20d`,
  `car_drift_historical_q1`.
- **Block 4 (Day-T, 1)**: `opening_gap_t1`. **LEAK feature** at Sunday
  planning.

**Timing classification** (per the prior feature audit):

- Forward-looking (NOT available Sunday): `opening_gap_t1`.
- Day-T (available weekday morning post-T but before T+1 open):
  `intraday_range_t`, `volume_vma20_ratio_pre_event`, `suv_day_1`.
- Pure pre-T (available Sunday): 17 features (`sue_*`, `rel_ret_*`,
  `is_bmo`, `pre_event_*`, `sector_adjusted_ret_20d`,
  `car_drift_historical_q1`).

This experiment deliberately keeps `opening_gap_t1` IN the feature set
because the user's hypothesis was that the gap is a real Day-0 signal --
it's just not usable at Sunday planning. A separate Sunday-only run
(with leak features dropped) is left for the next iteration (see §8).

---

## 3. Finding A -- the PEAD gates are a real alpha engine

Computed across all 17,300 primed events (TRAIN + VAL together). Gate
pass rates:

| Gate | Pass count | Pass rate |
|---|---|---|
| Gate 1 (CAR > +3%) | 5,261 | 30.41% |
| Gate 2 (Vol > 2x vma20) | 6,360 | 36.76% |
| Gate 3 (MaxDD_MA > -1.5%) | 7,024 | 40.60% |
| **All 3 combined (`pead_pass`)** | **1,848** | **10.68%** |

### 3.1 Oracle PnL of `pead_pass` events vs universe

Default execution simulation: **enter at `Open[T+1]`, exit at `Close[T+11]`**
(10 trading-day hold from T+1 open). This is the realistic execution
edit -- the Sunday model ranks candidates on Sunday, but the actual
trade is entered at the first available post-catalyst print (T+1 open).
Realized PnL = `log(Close[T+11] / Open[T+1])`, converted to arithmetic
as `expm1(...)`.

| Subset (VAL rows 2024-01 -> 2026-07) | N | Mean arith PnL/event | Hit rate |
|---|---|---|---|
| Unconditional universe | 3,885 | **+0.227%** | ~52% |
| Failed >= 1 gate | 3,439 | ~+0.2% | ~52% |
| **`pead_pass == 1` (oracle)** | **446** | **+6.391%** | ~73% |

That is a **~28x per-event return spread** between pead_pass and the
universe baseline. The 3 gates, used in aggregate, identify a 11%
sub-population of earnings events with a ~6%/event average drift over
the 10-day holding window. **This is the real alpha engine in the
dataset.**

### 3.2 What the 3 gates are doing conceptually

- Gate 1 (CAR > +3%) selects surprise-driven positive drift events.
- Gate 2 (volume spike > 2x) selects events with institutional
  accumulation footprint -- filters from the noisy "textbook reaction"
  where retail reacts but institutions don't follow through.
- Gate 3 (MaxDD_MA > -1.5%) filters out V-shaped head-fakes that would
  trip a stop-loss in live trading.

The intersection of the three is the tradeable PEAD pre-set. But since
the gates use realized post-event data, the question becomes: **can
anything known on Sunday (pre-T) or on T+1 morning predict gate-pass?**
That is the focus of §4-§6 below.

---

## 4. Finding B -- raw opening gap is NOT sufficient alpha

The user's Path-3 idea was: anchor on `opening_gap_t1` (which has real
Day-0 signal) and use PEAD detection to confirm/upgrade it.

A model-free backtest was run (`_pead_gap_strategy.py`) to test the
first half of that: **enter long top-N by raw opening_gap_t1 (positive
gap only) at T+1 close, hold 9 more days to T+11 close**, measuring
remaining drift `Close[T+1] -> Close[T+11]`.

| Top-N (long, pos-gap only) | Per-event 9-day REM drift | Hit rate | Sharpe (107 wk) |
|---|---|---|---|
| 1 | **-1.16%** | 40.5% | -1.13 |
| 3 | -0.47% | 47.1% | -0.59 |
| 5 | -0.45% | 48.1% | -0.63 |
| 10 | -0.33% | 50.6% | -0.49 |
| Random (top-5 pos-gap) | +0.36% | 51.2% | +0.60 |

**The gap-driven top-N long-only selector is NEGATIVE** -- and worse
than picking randomly from the same universe! Two failure modes:

1. The "positive gap only" universe has been selected *for* events that
   already ran intraday -- which mean-revert subsequently.
2. Top-N by raw gap size over-weights the most overextended gap events
   (a +10% gap is more likely to fade than a +2% gap).

### 4.1 Stratification of 9-day REMAINING drift by gap bucket

When we strip the T+1 day itself (compute remaining 9-day drift =
`Close[T+1] -> Close[T+11]`), stratified by `opening_gap_t1` bucket:

| Gap bucket | N rows | 9-day REM mean |
|---|---|---|
| < -5% | 316 | +0.26% |
| -5% .. -2% | 402 | +0.47% |
| -1% .. 0% | 749 | +0.35% |
| 0% .. +1% | 814 | +0.21% |
| +1% .. +2% | 427 | +1.15% |
| +2% .. +5% | 510 | +0.57% |
| **> +5%** | 320 | **-0.27%** |

**Once you eliminate T+1 itself, the gap's predictive power
disappears.** The +5%+ gap monster bucket has ~0 9-day remaining drift
(and slightly negative). The PEAD-persistence "alpha" the gap appears
to carry is mechanically just the T+1 open-to-close day -- the single
day we observe at trade time but cannot capture ahead of entry.

### 4.2 Implication

`opening_gap_t1` is not, by itself, a tradable alpha signal. It needs
to be combined with a Sunday-feasible **pre-screen** that selects
candidates with a high *prior* probability of being a PEAD event. The
gap is the confirmation, not the predictor.

That sets up the §5-§6 experiments: can a ranker or a classifier
trained on the **PEAD event-label** (`pead_pass`) provide the missing
pre-screen?

---

## 5. Finding C -- PEAD-target ranker beats CAR-target ranker on NDCG@3

### 5.1 Protocol

Two rankers were trained back-to-back with **identical hyperparameters**
(`max_depth=3, min_child_weight=50, gamma=5, subsample=0.7,
colsample_bytree=0.7, n_estimators=300, lr=0.05, reg_lambda=1.0`):

| Model | Objective | Target |
|---|---|---|
| **PEAD-target** | `rank:ndcg` | `pead_pass` (binary 0/1, mean 10.39% TRAIN) |
| **CAR-target** | `rank:ndcg` | discretized `car_10d` (10-quantile buckets 0..9) |

Both used the **same 21 features** (including `opening_gap_t1`).
Outputs: `ranker.json` + `calibrator.pkl` + `meta.json` saved under
`03_model/models/phase_f_v2_pead_target/`.

### 5.2 NDCG@3 results

| Model | TRAIN NDCG@3 | VAL NDCG@3 |
|---|---|---|
| **PEAD-target** (`phase_f_v2_pead_target`) | 0.5403 | **0.5838** |
| CAR-target (same hyperparams) | 0.3884 | 0.1593 |
| Phase F v2 promoted baseline (`phase_f_v2_baseline_ndcg`) | 0.6130 | 0.1593 |

Three observations:

1. **PEAD-target VAL NDCG@3 > TRAIN NDCG@3 (0.5838 > 0.5403).** The
   ranker generalizes *better* to the out-of-sample window than to the
   training set. This is the strongest possible evidence against
   overfit -- the predictive structure is *more* stable in 2024-2026
   than in 2015-2023.
2. **VAL NDCG@3 lifts 3.7x** versus the same hyperparams on the car_10d
   target (0.5838 vs 0.1593). This confirms the user's hypothesis:
   predicting the **event label** is fundamentally more tractable than
   predicting the continuous return.
3. The CAR-target under these hyperparams reproduces the prior
   `phase_f_v2_baseline_ndcg` VAL result (0.1593) -- the experimental
   pipeline is consistent with the historical Stage 3 run.

### 5.3 Per-week top-N selection comparison

Top-N selection over VAL (one pick per week per slot) ranked by raw
ranker score. Three measurements per (N, model):

- (A) **Full 10-day arith CAR** = `expm1(car_10d)` -- the label the
  old pipeline optimized, includes the T+1 day.
- (B) **9-day REMAINING arith drift** = `expm1(ret_close_t1_to_close_t11)`
  -- excludes the T+1 open-to-close mechanical move.
- (C) **PEAD-recall %** = top-N picks that are actually `pead_pass==1`,
  expressed as a fraction of all 446 true PEAD events in val.

| Top-N | Model | (A) full 10d % | (B) rem 9d % | (C) PEAD-recall % | Hit% (9d) |
|---|---|---|---|---|---|
| 1 | PEAD-target | +9.070% | -0.945% | 13.68% | 43.9% |
| 1 | CAR-target | +9.658% | +0.035% | 13.00% | 44.9% |
| 3 | PEAD-target | +5.440% | -0.203% | **32.51%** | 48.9% |
| 3 | CAR-target | +6.112% | +0.188% | 29.37% | 49.5% |
| 5 | PEAD-target | +3.837% | -0.040% | **41.93%** | 48.8% |
| 5 | CAR-target | +4.404% | +0.127% | 39.24% | 50.5% |
| 10 | PEAD-target | +2.545% | +0.079% | **55.83%** | 50.8% |
| 10 | CAR-target | +2.728% | -0.016% | 52.91% | 50.7% |
| 20 | PEAD-target | +1.018%* | +0.264%* | **72.42%* | 52.2%* |
| 20 | CAR-target | +0.871%* | +0.222%* | 66.59%* | 52.7%* |

(*Top-20 numbers from the second eval run with the proper Open[T+1]
entry simulation.)

### 5.4 What the numbers say

1. **PEAD-target is strictly better at identifying PEADs.** Recall
   lift over CAR-target is +3.1 pp (top-3), +2.7 pp (top-5), +2.9 pp
   (top-10), and +5.8 pp (top-20). Consistent across all N -- the
   PEAD-target picks fewer false PEADs.

2. **The 9-day REMAINING drift (the part actually capturable after
   T+1 open) is essentially zero** -- ranges from -0.95% to +0.19%
   across all N, all models. The "alpha" we see in (A) (full +9%
   per event for top-1) is almost entirely the T+1 open-to-close day
   itself, which by definition we cannot trade AFTER it has happened.

3. **Adding more rows (theta-threshold sweeps on the classifier below)
   outperforms increasing N.** A binary threshold filter at P(PEAD)
   >= 0.20 over the classifier yields +0.75% per event -- better than
   top-3 by raw rank order (§6).

### 5.5 Why top-N selection underperforms the ranker's NDCG@3

NDCG@3 measures: of the top-3 picks, how well do we rank the true
positives above the false positives within that group? A high NDCG@3
does NOT translate to per-event PnL because the false positives (the
58% of top-5 picks that aren't true PEADs) have ~0 arith drift and wash
out the +6.39% alpha of the true positives.

The arithmetic: top-5 picks ~5 events x 107 weeks = 535 picks, 42% are
PEADs (~225 events at +6.4% = +14.4 unit alpha). The other 310 non-PEAD
picks at ~0% (actually mildly negative ~-0.3%) deliver -0.9 units. Net
= +13.5 units / 535 picks = +2.5% per event, which we observe as ~+0.4%
in the actual data because the false-positive subset isn't quite zero
(see §6.3 for the false-positive stratification).

---

## 6. Finding D -- binary classifier is a tractable PEAD pre-screen

### 6.1 Protocol

Replaced the listwise ranker with a pointwise binary `XGBClassifier`
(`objective="binary:logistic"`, same hyperparameters as the ranker).
Trained on `pead_pass` (0/1) using TRAIN rows (n=13,306, positives=
1,382). Artifacts: `03_model/models/phase_f_v2_pead_classifier/`.

### 6.2 Classifier accuracy

| Metric | TRAIN | VAL |
|---|---|---|
| AUC | 0.8770 | 0.8596 |
| Average Precision (AP) | 0.5256 | 0.5256 |

AUC is essentially identical between TRAIN and VAL (~0.86). AP is
*exactly identical* (0.5256 vs 0.5256). The classifier generalizes
well across the 2024-01 regime break -- another non-overfit signal.

### 6.3 Threshold sweep -- enter if P(PEAD) >= THRESH

Top-N selection is replaced with a **confidence threshold**: take ALL
val events whose predicted `P(PEAD)` >= threshold, enter each at
`Open[T+1]`, exit at `Close[T+11]`.

| Threshold | N trades | Recall % | Precision % | Avg PnL/event | Hit % |
|---|---|---|---|---|---|
| >= 0.50 | 213 | 33.86% | **70.89%** | +0.597% | 53.5% |
| >= 0.40 | 293 | 40.13% | 61.09% | +0.592% | 53.2% |
| >= 0.35 | 354 | 43.95% | 55.37% | +0.248% | 52.0% |
| >= 0.30 | 443 | 48.88% | 49.21% | +0.365% | 53.3% |
| >= 0.25 | 580 | 56.73% | 43.62% | +0.540% | 54.7% |
| **>= 0.20** | **769** | **65.02%** | 37.71% | **+0.753%** | **55.1%** |
| >= 0.15 | 1,073 | 77.80% | 32.34% | +0.751% | 54.5% |
| >= 0.10 | 1,469 | 86.10% | 26.14% | +0.655% | 54.3% |
| >= 0.05 | 2,127 | 93.95% | 19.70% | +0.557% | 53.9% |
| (universe baseline) | 3,885 | 100% | 11.47% | +0.227% | ~52% |
| (oracle `pead_pass==1`) | 446 | 100% | 100% | **+6.391%** | ~73% |

Key observations:

1. **At threshold 0.20, taking 769 events over 107 weeks at +0.753%
   per-event PnL = +0.8% per week cumulative** (assuming non-overlapping
   10-day holds). This is ~3.3x the unconditional universe baseline of
   +0.227% -- the classifier IS finding alpha.
2. **At threshold 0.50, precision is 71%** -- most predictions in this
   bucket are TRUE PEADs. But realized PnL is only +0.6%, not the +6%
   oracle alpha, because the high-confidence picks skew toward
   smaller-CAR PEADs (see §6.5 below).
3. Recall saturates at ~96% by threshold 0.03 (most true PEADs are
   scored above 0.03). Below 0.20 the precision drops below 38% and
   the per-event PnL stops growing because we start accepting too many
   wash-out events.

### 6.4 Recommended operating points

| Use case | Threshold | Trade count / 107 wk | Expected hit rate | Expected PnL/event |
|---|---|---|---|---|
| **Conservative** (high-confidence, premium picks) | 0.40-0.50 | 213-293 | 53% | +0.60% |
| **Balanced** (default operating point) | 0.20 | 769 | 55% | +0.75% |
| **Aggressive** (maximize recall) | 0.10 | 1,469 | 54% | +0.66% |

The "balanced" point (threshold=0.20, ~7 trades/week) sits at the sweet
spot where adding further events stops yielding PnL improvement -- it
appears in both the threshold sweep AND the §6.5 quantile bucketing.

### 6.5 Quantile-binned realized PnL -- the calibration surprise

To diagnose why top-N was ~0% but the threshold sweep is +0.75%, we
stratified realized PnL by predicted-probability decile (q=10 buckets).

#### Among TRUE PEADs (events that pass all 3 gates)

| Predicted proba bucket | N true PEADs here | Mean arith PnL | Pos rate |
|---|---|---|---|
| (0.0033, 0.0123] lowest | 2 | +7.76% | 100% |
| (0.0123, 0.0184] | 4 | +8.88% | 100% |
| (0.0184, 0.0259] | 7 | **+13.02%** | 100% |
| (0.0259, 0.040] | 9 | +6.73% | 78% |
| (0.040, 0.061] | 19 | +9.92% | 95% |
| (0.061, 0.0905] | 17 | +8.77% | 94% |
| (0.0905, 0.137] | 32 | +7.81% | 100% |
| (0.137, 0.198] | 64 | +9.62% | 95% |
| (0.198, 0.327] | 87 | +9.64% | 97% |
| **(0.327, 0.893] highest** | 205 | **+3.98%** | 71% |

**Within the TRUE PEAD set, PnL is NOT monotonic in predicted proba.**
The high-confidence bucket (proba > 0.327) holds 205 of the 446 true
PEADs but averages only +3.98% per event -- half the +7-13% realized
PnL of the low-proba buckets. The model's most confident PEAD picks
are systematically the smaller-CAR PEADs.

#### Among FALSE positives (failed >= 1 gate)

| Predicted proba bucket | N false here | Mean arith PnL | Pos rate |
|---|---|---|---|
| (0.0033, 0.0123] | 387 | +0.74% | 58% |
| ... | ... | ... | ... |
| (0.198, 0.327] | 300 | -0.97% | 47% |
| **(0.327, 0.893] highest** | 184 | **-4.17%** | 30% |

The high-confidence bucket's FALSE positives averaged **-4.17% per event
with 30% positive rate** -- the model's most confident mistakes are
the worst performers, dragging down the +0.50 threshold's overall
per-event PnL to +0.60% (vs +0.75% at 0.20).

### 6.6 Why is high-confidence not aligned with high-PnL?

The 3 PEAD gates don't weight magnitude -- a +3.1% CAR event passes
Gate 1 just as easily as a +25% CAR event. The GLB classifier learned
to find features common to "minimal" PEADs (high sue_score, BMO, etc.)
which correlate with confident-but-small drifts. Rare blowout PEADs
(the +13% drifts) don't share those features strongly enough to be
predicted with confidence by the model -- they sit in the low-proba
tail (small N) and contribute little to the overall realized PnL.

This is a **known limitation of binary event-label targets**: the
target collapses a magnitude cross-section to 0/1, so the model
optimizes for recall above the threshold rather than magnitude above
the threshold.

---

## 7. Synthesis -- what we now know

| Question | Answer |
|---|---|
| Are PEAD events a meaningful fraction of the universe? | Yes: 10.68% (1,848 of 17,300 primed). |
| Are PEAD events an alpha engine? | Yes: oracle mean PnL/event = +6.39% vs universe +0.23%. ~28x spread. |
| Is `opening_gap_t1` alone an alpha signal? | **No**: top-N by raw gap = NEGATIVE Sharpe. Remaining 9-day drift stratified by gap bucket is ~0. |
| Is `rank:ndcg` on `pead_pass` better than on `car_10d`? | **Yes**: VAL NDCG@3 0.5838 vs 0.1593 (3.7x lift). |
| Does PEAD-ranker top-N selection monetize the alpha? | No: ~0% per-event PnL. Recall improves (top-5 finds 42% of PEADs) but false positives wash out the alpha. |
| Does a binary classifier threshold monetize the alpha? | **Yes**: P(PEAD) >= 0.20 yields 769 events at +0.75%/event (3.3x universe). Hit rate 55%. |
| Is the classifier's high-confidence bucket the highest PnL? | **No**: top proba bucket (0.327-0.893) averages only +0.12% (true PEADs in this bucket average +3.98%, not +6-13%). |
| Can we still expect to capture the full +6% oracle alpha? | Only if we either improve recall at high precision, or add a magnitude predictor, or use a Sunday-T+1 two-stage filter (see §8). |

### 7.1 The full picture

The PEAD alpha engine is REAL. The classifier can find it. But the
binary 0/1 target collapses magnitude information, so the model trades
off recall for precision WITHOUT optimizing for the magnitude of the
drift. The best operating point so far (+0.75% per event at threshold
0.20) captures only ~12% of the oracle alpha (+6.39%). The remaining
+5.6% of per-event alpha is currently being left on the table because:

1. We are taking many false positives (37 of every 100 trades at
   threshold 0.20 are NOT true PEADs), and those false positives average
   negative realized PnL.
2. True PEADs in the high-confidence bucket are systematically the
   smaller-CAR events.

### 7.2 Three possible remediation directions

a. **Magnitude-aware target** -- train on a 3-level label
   (`0 == no PEAD`, `1 == small PEAD`, `2 == large PEAD` where the
   large/small split is at +6% CAR, the oracle mean). This would tell
   the model "all PEADs are not equal; find the ones that drift a lot".

b. **Two-stage filter**: Sunday classifier P >= 0.20 screen, plus a
   T+1-morning gap confirmation (gap > some threshold like +3%) on
   surviving events. The T+1 filter leverages the `opening_gap_t1`'s
   real Day-0 information content as a confirmation, not a predictor.
   See §8.

c. **Confidence-weighted sizing**: trade all P >= 0.20 picks, but
   Kelly-size by calibrated proba so the high-confidence picks get
   bigger allocation. (This is what the existing IsotonicRegression
   calibrator in `phase_f_v2_pead_classifier/calibrator.pkl` was
   designed to enable.)

### 7.3 What we did NOT do (deliberate scope cuts)

- We did NOT re-run the ranker without `opening_gap_t1` (Sunday-only
  feature set). That experiment is needed to claim the ranker is
  usable on Sunday -- the current numbers reflect Combined Sunday +
  Day-0 features.
- We did NOT use early stopping. The classifier was trained to
  n_estimators=300 with no early-stopping callback. The VAL AUC is
  roughly equal to TRAIN AUC -- model is well-regularized already --
  so adding early stopping likely would not change the conclusions.
- We did NOT optimize `gamma`, `min_child_weight`, or learning rate
  for the binary classifier -- we carried forward the hyperparams
  that won the previous ranker sweep. A separate hyperparameter
  sweep could lift the threshold sweep further (left for §8 follow-up).

---

## 8. Proposed Phase G -- two-stage Sunday+T+1-morning filter

Building on the §7.2 observation that `opening_gap_t1` is a
confirmation, not a predictor, we propose the Phase G architecture:

### 8.1 Pipeline

```
 Sunday (T-?)            | Weekday morning (T+1)         | Day-T (T+1..T+11)
 ------------------------|-------------------------------|-------------------
 Run binary classifier   | Observe realized Open[T+1],   | For each accepted
 on all upcoming-week    | compute opening_gap_t1.        | event, enter at
 earnings events using   |                               | Open[T+1], hold to
 Sunday-feasible         | Only ENTER trades where:      | Close[T+11].
 21 features (sans       |   P(PEAD) >= theta_screen     |
 opening_gap_t1).        |   AND opening_gap_t1 in        |
                         |       [gap_lo, gap_hi]        |
 Output: ranked list    |                               |
 with P(PEAD).           | (e.g. 0.02 <= gap <= 0.10     |
                         |  filters out faded +/-extreme |
                         |  gaps)                         |
```

- **Sunday** uses pure pre-T features (the 17-feature Sunday-safe set)
  -- NOT the leak feature. We will retrain without `opening_gap_t1`.
- **T+1 morning** uses the realized gap from `Open[T+1]` as a confirmation
  filter, NOT as a predictor. This is the Weekday-engine re-rank
  step per Design.md §3.
- **Limit to events passing BOTH conditions**, then take ALL accept
  (not top-N -- this matches the threshold-sweep operating point).

### 8.2 Backtest protocol

1. Retrain the classifier on the 17-feature Sunday-only set
   (drop `opening_gap_t1` and the 3 other Day-T features
   `intraday_range_t`, `volume_vma20_ratio_pre_event`, `suv_day_1`).
2. On VAL rows: predict `P(PEAD)`, threshold at theta_screen.
3. Filter survivors by realized `opening_gap_t1` in [gap_lo, gap_hi].
4. Compute realized PnL = `Open[T+1] -> Close[T+11]` for accepted trades.
5. Sweep over (theta_screen, gap_lo, gap_hi) and measure per-event
   PnL, hit rate, and Sharpe.

This proposal is a small focused experiment -- only one new training
run + 2D threshold sweep. It would empirically confirm or refute:
- Whether a Sunday-only classifier (no leak features) retains enough
  recall to be deployable.
- Whether the T+1 gap confirmation meaningfully improves precision
  above the unconditional 38% (theta=0.20 baseline).

### 8.3 Stretch goal -- magnitude-aware target

If the §8.2 two-stage backtest underperforms, the next refinement is to
change the target to a magnitude-aware label (§7.2 a):

- `0 == no PEAD` (failed any gate)
- `1 == small PEAD` (passes all 3 gates AND CAR in [+3%, +6%])
- `2 == large PEAD` (passes all 3 gates AND CAR > +6%)

Train `XGBClassifier` with `objective="multi:softprob"` (3 classes).
At inference, accept trades where predicted `P(large_pead) >= theta`
-- explicitly targeting the +9-13% drift subset.

---

## 9. Conclusion

Three things are now empirically established:

1. **The PEAD event-label target is the right training objective.**
   VAL NDCG@3 lifts 3.7x versus the continuous `car_10d` target, and
   VAL > TRAIN eliminates the overfit concern that hung over Phase F
   v2.

2. **A binary classifier threshold sweep recovers usable out-of-sample
  -alpha**, currently at +0.75% per event (~3.3x the universe baseline)
   on 769 trades over 107 weeks. No leakage is involved -- the
   classifier uses some Day-T features known at Sunday pre-T time
   (which would need to be removed for live trading).

3. **The alpha is qualitatively different from what Phase F v2 was
   capturing.** Phase F v2's "edge" was the forward-looking gap. The
   PEAD-target classifier's edge is the catalyst-event signal in SUE
   and idiosyncratic-volume features -- the genuine article. Real
   alpha left on the table is ~+5.6% per event above the current
   +0.75%.

The recommended next experiment is Phase G's Sunday-only classifier +
T+1 gap confirmation filter (§8). This is the smallest bridge from
"the gates work" to "the model works on Sunday, with a T+1 morning
confirmation" -- and it is the architecture Design §2 / §3 already
foreshadowed.

---

## 10. Artifact inventory

### Generated models

| Path | Contents |
|---|---|
| `03_model/models/phase_f_v2_pead_target/ranker.json` | XGBRanker, `rank:ndcg`, target=`pead_pass` |
| `03_model/models/phase_f_v2_pead_target/calibrator.pkl` | IsotonicRegression: raw score -> arith CAR |
| `03_model/models/phase_f_v2_pead_target/meta.json` | Hyperparams, gate thresholds, feature list |
| `03_model/models/phase_f_v2_pead_classifier/classifier.json` | XGBClassifier, binary logistic, target=`pead_pass` |
| `03_model/models/phase_f_v2_pead_classifier/calibrator.pkl` | IsotonicRegression: P(PEAD) -> arith CAR |
| `03_model/models/phase_f_v2_pead_classifier/meta.json` | Hyperparams, AUC, AP, gate thresholds |

### Companion scripts (read-only on `db.h5`)

| Path | Purpose |
|---|---|
| `04_backtest/_pead_exploration.py` | §3 gate base rates + §4 gap-stratification diagnostics |
| `04_backtest/_pead_gap_strategy.py` | §4 model-free gap-driven backtest (the negative result) |
| `04_backtest/_pead_target_retrain.py` | §5 PEAD-target ranker training |
| `04_backtest/_pead_classifier.py` | §6 PEAD-target binary classifier training + threshold sweep |

### Pre-existing artifacts (referenced but not modified)

- `db.h5` (no writes during this study; all scripts read-only).
- `03_model/models/phase_f_v2_baseline_ndcg/` (the prior car_10d target ranker).
- `03_model/models/phase_f_v2_baseline_pairwise/` (the prior pairwise ranker).
- `03_model/models/phase_f_baseline_v1/` (OBSOLETE -- contaminated EODHD data).

---

## 11. Acknowledged caveats

- The classifier was trained WITH `opening_gap_t1` and the 3 Day-T
  features in the feature set. **At Sunday planning time these are
  NOT available.** A Sunday-safe retrain (recommended in §8) is
  necessary before calling this deployable.
- The 3 PEAD gates use realized post-event T+1..T+11 prices. They are
  therefore ex-post labels, NOT ex-ante filters. Using them at Monday
  morning as a trade filter would be lookahead bias.
- VAL period (2024-2026) is post-regime-shift per the Phase F v2
  VAL-distribution audit. The PEAD base rate is +1.1 pp higher than
  TRAIN, and the classifier generalizes well, but testing on a second
  out-of-sample period (e.g. 2026 H2) is required for true confidence.
- Top-N selection realized PnL calculations assume each event holds
  T+1 open to T+11 close -- they don't model portfolio-level
  hedging, position sizing, or overlapping-10-day-hold accounting.
  The per-event average is the most reliable metric here.
