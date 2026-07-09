# Feature Engineering Matrix Protocol: Cross-Sectional Listwise Setup

This document is the **single source of truth** for the feature schema used by the
PEAD trading bot's training (XGBoost **Ranker** — listwise Learning-to-Rank) and
Sunday/weekday inference pipelines. It supersedes any prior feature-spec drafts
(including the legacy `proposed_new_features.md`).

See `Design.md` §17 for the listwise ranking architecture (model objective,
group structure, isotonic calibration, live sizing) this schema feeds into.

---

## 0. Row Granularity & Group Structure

* **Feature Row Granularity (Company / CIK Level):** Each row of the feature
  matrix corresponds to **one earnings event** of one company. A company
  (identified by its SEC CIK — see `Design.md` §9b and `company_merge_design.md`)
  contributes many rows over its S&P  400 membership window, one per earnings
  announcement. Ticker renames/rebrands are collapsed via
  `/metadata/sp400_companies`: the feature builder gathers **all earnings dates
  stored under any alias** of a company in `/earnings/raw`, aligns them onto the
  **canonical ticker's** price series (`/sp400/{canonical_ticker}`, which EODHD
  retro-adjusts across rebrands so it spans the alias periods), and gates each
  event with the merged `combined_intervals` membership span plus the **90-day
  rebalance buffer**. Pre-rebrand earnings events are recoverable because the
  canonical price series already spans the alias periods.

* **When Is a Row Eligible? (Interval Gating):** For each interval
  `{added, removed}` in a company's `combined_intervals`:
  1. Expand to a buffered window `[added + 90 days, removed]` (use "today" if
     `removed` is `null`). The 90-day buffer IS the first-quarter exclusion —
     the skip of the noisy first quarter after SP400 addition is already baked
     in; no extra per-event logic is required.
  2. Keep the earnings event iff `report_date ∈ [added + 90d, removed]`.
  3. **Companies with `price_unavailable=True` produce zero rows.**

* **Feature Lookback May Cross Interval Boundaries:** The 90-day buffer gates
  *which events become rows*, NOT what historical data is used to compute their
  features. Pre-addition prices, sector ETF prices, and the company's full
  earnings history are all fair game as feature inputs.

* **Listwise Group Anchor (`calendar_week_group`):** Two mandatory metadata
  columns are stored alongside the feature matrix (NOT themselves features):
  * `calendar_week_group` — ISO week of `report_date` in `YYYY-Www` form
    (e.g., `2026-W27`). All rows sharing the same group are evaluated by the
    ranker as one cross-sectional cohort.
  * `car_10d` — the continuous 10-day Cumulative Abnormal Return percentage,
    used as the **listwise gain target**. **Do not convert to discrete ordinal
    ranks**; the NDCG gain function reads raw continuous values.

* **Strict Sorting Rule:** Before constructing the `DMatrix`, the training and
  validation frames **must** be sorted via
  `.sort_values(['calendar_week_group', 'canonical_ticker'])` so that rows of
  the same `calendar_week_group` occupy contiguous index blocks. Group sizes
  are then `.groupby('calendar_week_group').size().values` and passed as the
  `group=` argument to `XGBRanker.fit`.

---

## 1. Feature Blocks & Column Mapping

The active training matrix `X` contains **20 features** grouped into 5 blocks.
A 21st potential feature (`rev_growth_yoy`, Block 1) is **SHELVED** — see §2.
Macro features (`vix_close`, `fed_funds_rate`, `yield_curve_spread`,
`spy_momentum_20d`) are **NOT in `X`** — see §3 (Macro Detachment).

### Block 1 — Catalyst Fundamentals
Block 1 features are computed from `/earnings/raw` and require no price lookups.
All per-company sequential features (`sue_*`, `consecutive_surprises`,
`car_drift_historical_q1`) are grouped by `canonical_ticker` and sorted by
`report_date`.

* `sue_score` (Float): Standardized Unanticipated Earnings.
  `sue_score[t] = difference[t] / σ_{12Q}(difference, all prior quarters)`,
  where `difference = actual - estimate` (precomputed by EODHD) and
  `σ_{12Q}(·)` is a **rolling 12-quarter standard deviation over ALL prior
  quarters of the company** (`min_periods = 12`), not restricted to the SP400
  membership window. Where `estimate` is `NaN`, EODHD sets `difference = 0.0`;
  those zeros are **included** in the rolling std denominator (Option B per
  `Design.md` §15), and `sue_score` evaluates to `0.0` for those events.
* `eps_surprise_pct` (Float): Raw percentage deviation of reported EPS from
  consensus — directly from the `percent` column of `/earnings/raw`. No
  recomputation.
* `consecutive_surprises` (Integer): Running count of consecutive quarters
  where `actual > estimate` (strict beat), grouped by `canonical_ticker`.
  Increment by 1 on a beat; reset to 0 otherwise.
* `sue_acceleration` (Float): Q-on-Q change in `sue_score`
  (`sue_score[t] - sue_score[t-1]`), grouped by `canonical_ticker`. Captures
  whether the earnings quality trajectory is improving or deteriorating.
* `sue_lag_1` (Float): `sue_score` from the previous quarter (Q-1) for the same
  `canonical_ticker`. Captures SUE surprise persistence — a series of
  consistent beats may signal systematic analyst mis-calibration.
* `sue_lag_2` (Float): `sue_score` from two quarters ago (Q-2) for the same
  `canonical_ticker`. Extends the persistence check back through two full
  reporting cycles.
* `car_drift_historical_q1` (Float): The **actual index-adjusted CAR generated
  during the stock's previous post-earnings window last quarter** — i.e. the
  CAR (T+1 → T+60, abnormal relative to `IJH`) of the prior event of the same
  `canonical_ticker`, shifted by 1. Tests whether a stock has a repeatable PEAD
  signature: some companies consistently drift while others mean-revert.
  *Implementation note:* requires a two-pass build — compute every event's own
  post-event CAR first, then `shift(1)` per canonical.

### Block 2 — Microstructure & Technical Event Context
Block 2 features encode the execution environment, institutional presence, and
positioning footprint on the event day. `T` is the report-trading-day = the
first trading day in `/sp400/{canonical}` with `Date >= report_date` (roll
**forward** to the next trading day if `report_date` falls on a weekend or
holiday; drop the event if no later trading day exists).

* `is_bmo` (Binary 0/1): `1` if `before_after_market == "Bmo"` else `0`.
* `volume_vma20_ratio_pre_event` (Float): `Volume[T] / mean(Volume[T-20 : T-1])`.
  Pre-announcement positioning intensity.
* `suv_day_1` (Float): Standardized Unexpected Volume —
  `Adj_Volume[T] / mean(Adj_Volume[T-20 : T-1])` — abnormal volume on the event
  day itself (uses adjusted volume to neutralize split noise).
* `pre_event_idiosyncratic_vol` (Float):
  `std(log_ret_stock - log_ret_IJH, ddof=1)` over `T-20 : T-1`. Captures the
  stock's risk profile after stripping out broad mid-cap market volatility.
* `opening_gap_t1` (Float): `(Open[T+1] - Close[T]) / Close[T]` — the overnight
  price gap after the earnings announcement encodes the market's first-pass
  surprise assessment.
* `intraday_range_t` (Float): `(High[T] - Low[T]) / Close[T]` — normalized
  intraday range on the event day. Wide range = high uncertainty, often
  precedes larger drift resolution.
* `pre_event_volume_trend` (Float): OLS slope of `Adj_Volume` over
  `T-10 : T-1` (linear regression of volume vs. day index). Rising pre-event
  volume suggests information leakage.

### Block 3 — Multi-Horizon Market & Sector-Adjusted Technicals
Relative velocity into the event. The sector benchmark is **NOT GICS** — it is
the SEC SIC-mapped `index_ref` from `/metadata/sp400_companies`, computed per
`01_data/SIC_code_to_index.md` (e.g. `IJK`, `IJJ`, `XLB`, `XLF`, `XLU`, `XLRE`).
Companies whose `index_ref` is missing default to the mid-cap blend `IJH`.

* `rel_ret_3d` / `rel_ret_5d` / `rel_ret_10d` / `rel_ret_20d` / `rel_ret_30d`
  (Float): `log(Adj_Close_stock[T-1] / Adj_Close_stock[T-1-h]) -
  log(IJH_Close[T-1] / IJH_Close[T-1-h])` for each `h ∈ {3, 5, 10, 20, 30}`.
  Cumulative log return of the stock minus the cumulative log return of the
  mid-cap proxy `IJH` over the same trailing window.
* `sector_adjusted_ret_20d` (Float):
  `log(Adj_Close_stock[T-1] / Adj_Close_stock[T-21]) -
  log(Adj_Close_sector[T-1] / Adj_Close_sector[T-21])`. Isolates idiosyncratic
  strength from industry-wide momentum.

### Block 4 — Interaction Terms

* `sue_abs_x_inverse_vol` (Float): `abs(sue_score) / pre_event_idiosyncratic_vol`
  — Surprise-to-risk ratio. Filters noise when idiosyncratic volatility is high.
  NaN-safe: if the denominator is NaN or 0, the result is NaN.

### Block 5 — Cross-Product Interactions (SHELVED)

The interactions `sue_x_momentum_5d` and `volume_ratio_x_sue` are EXPLICITLY
SHELVED — they are not part of the v1 active feature matrix. They may be
revisited later if the v1 model underperforms and the next iteration needs
additional signal.

---

## 2. Shelved / Deferred Features

* **`rev_growth_yoy` (SHELVED):** Year-over-year revenue growth for the
  reported quarter is **NOT buildable** from the current `db.h5` — the EODHD
  earnings endpoint (`/earnings/raw`) does not carry revenue, only EPS-level
  fields. We shelve this feature for v1; if the model needs more signal in a
  later iteration, we revisit a dedicated revenue source (e.g. SEC 10-K XBRL
  extraction, or paid fundamental data) and add it then.

* **`short_interest_pct_float` (REMOVED):** Officially deprecated and dropped
  from the matrix (per `Design.md` §15). No 15-year point-in-time short-interest
  feed is available without lookahead contamination. Positioning proxies are
  provided by the cross-sectional interaction of `suv_day_1` (volume shocks)
  and `opening_gap_t1` (opening price gaps).

* **Optional Block 5 cross-products (`sue_x_momentum_5d`,
  `volume_ratio_x_sue`):** Shelved for v1. See §1 Block 5.

---

## 3. Macro Detachment (Macro Features Are NOT in `X`)

* `vix_close`, `fed_funds_rate`, `yield_curve_spread`, `spy_momentum_20d` are
  **not** part of the active training matrix `X`. They are uniform (or
  near-uniform) across all rows within a given `calendar_week_group`, so a
  listwise ranker cannot meaningfully split on them cross-sectionally — any
  attempt to do so wastes tree capacity or injects cross-time noise.

* **Where they belong: the Weekday / Execution Bot.** These macro signals are
  preserved at the **Portfolio Execution Layer** (`08_backtest_execution.py` in
  `Design.md` §17.5) and used downstream to:
  * shift absolute tier hurdles dynamically (e.g., widen the
    `MINIMUM_EXPECTED_CAR` quality handbrake when `vix_close` is in a stress
    regime),
  * cap position size when broad-market stress reduces available risk budgets,
  * gate the live `weekly_schedule_queue` during macro-regime breakdowns.

* For convenience, the macro series are already stored in `db.h5` under
  `/macros/fred_*` (VIX, fed funds, 10Y-2Y spread, WTI, unemployment, CPI) and
  `/macros/SPY`. Pull them at Sunday inference time and thread them into the
  execution layer; **do not** join them into the training matrix.

---

## 4. NaN Handling Policy

* **Never drop rows.** The model — both Regressor and Ranker — handles `NaN`
  natively via optimal default split direction. Features that cannot be
  computed because of insufficient lookback data, missing analyst estimates,
  or unaligned trading days evaluate to `NaN`, and the row survives.
* All feature columns are strictly `float` (or `int` for `consecutive_surprises`
  and `is_bmo`). No `object` / `str` columns enter the `DMatrix`.
* Cases that produce `NaN` by construction:
  * `sue_a_lag_1` / `sue_lag_2` for the 1st / 2nd event of a company.
  * `sue_score` for the first 11 quarters of a company when the rolling-12Q
    std denominator has fewer than 12 prior observations.
  * `pre_event_idiosyncratic_vol` when fewer than 2 valid residual-return
    observations fall inside `T-20 : T-1`.
  * `opening_gap_t1` when `T+1` falls outside the stored price series.
  * Block 2 / Block 3 features for events near the start of a company's price
    series where the required trailing lookback window does not yet exist.

---

## 5. Active Feature Summary Table

| # | Block | Feature | Calculation | Priority |
|---|-------|---------|-------------|----------|
| 1 | 1 | `sue_score` | `difference[t] / σ_{12Q}(diff, all prior quarters)` per company (`min_periods=12`) | **Must** |
| 2 | 1 | `eps_surprise_pct` | `percent` column directly from `/earnings/raw` | **Must** |
| 3 | 1 | `consecutive_surprises` | Running count, +1 on `actual > estimate`, reset 0 otherwise (per canonical) | **Must** |
| 4 | 1 | `sue_acceleration` | `sue_score[t] - sue_score[t-1]` (per canonical) | **Must** |
| 5 | 1 | `sue_lag_1` | `sue_score` from Q-1 (per canonical) | **Must** |
| 6 | 1 | `sue_lag_2` | `sue_score` from Q-2 (per canonical) | **Must** |
| 7 | 1 | `car_drift_historical_q1` | Prior event's post-earnings CAR (T+1→T+60, IJH-adjusted), `shift(1)` per canonical — two-pass build | **Must** |
| 8 | 2 | `is_bmo` | `1 if before_after_market == "Bmo" else 0` | **Must** |
| 9 | 2 | `volume_vma20_ratio_pre_event` | `Volume[T] / mean(Volume[T-20 : T-1])` | **Must** |
| 10 | 2 | `suv_day_1` | `Adj_Volume[T] / mean(Adj_Volume[T-20 : T-1])` | **Must** |
| 11 | 2 | `pre_event_idiosyncratic_vol` | `std(log_ret_stock - log_ret_IJH, ddof=1)` over `T-20 : T-1` | **Must** |
| 12 | 2 | `opening_gap_t1` | `(Open[T+1] - Close[T]) / Close[T]` | **Must** |
| 13 | 2 | `intraday_range_t` | `(High[T] - Low[T]) / Close[T]` | **Must** |
| 14 | 2 | `pre_event_volume_trend` | OLS slope of `Adj_Volume` over `T-10 : T-1` | **Must** |
| 15 | 3 | `rel_ret_3d` / `_5d` / `_10d` / `_20d` / `_30d` | `log_stock_ret(h) - log_IJH_ret(h)` for `h ∈ {3,5,10,20,30}` | **Must** |
| 16 | 3 | `sector_adjusted_ret_20d` | `log_stock_ret(20d) - log_sector_etf_ret(20d)` via SIC-mapped `index_ref` (default `IJH`) | **Must** |
| 17 | 4 | `sue_abs_x_inverse_vol` | `abs(sue_score) / pre_event_idiosyncratic_vol` (NaN-safe) | **Must** |

**Total active features in `X`: 20**
(15 + 5 from `rel_ret_*d` counted as one cell) + `sector_adjusted_ret_20d` + 1
Block 4 interaction = 7 + 5 + 1 + 1 + 6 + 1 = the 20 above.

---

## 6. Stored (Non-Feature) Columns in the Output Table

The output of `build_feature_matrix()` (stored in `db.h5` under `/features/...`)
contains the 20 features above, PLUS these mandatory metadata columns that are
NOT used as inputs to the ranker but are required by the training loop and the
execution layer:

* `canonical_ticker` (str) — joins back to `/metadata/sp400_companies`.
* `cik` (str) — SEC CIK anchor (for audit / dedup).
* `report_date` (datetime) — the earnings announcement date (calendar basis).
* `T` (datetime) — the matched trading day (`Date >= report_date`, rolled
  forward) used for all price-side feature lookups.
* `calendar_week_group` (str) — ISO week (`YYYY-Www`) — the listwise group
  anchor (see §0).
* `car_10d` (float) — the continuous 10-day Cumulative Abnormal Return (target,
  `T+1 → T+11`), kept continuous for NDCG gain (see `Design.md` §17).
* `added` / `removed` (datetime) — the buffered interval that admitted this
  event (audit only).

---

## 7. Builder Implementation Skeleton

The full implementation lives in `02_features/build_feature_matrix.py` (NOT in
this doc). The reference contract is below — the actual function reads directly
from `01_data/db.h5` and writes the matrix back into a new HDF5 node; it makes
**zero external API calls**.

```python
import pandas as pd
import numpy as np
from sklearn.isotonic import IsotonicRegression
import xgboost as xgb

def build_ranking_matrix(event_df):
    """
    Constructs a contiguous, sorted feature matrix with structural group track
    arrays. Both X and the LTR metadata are returned; the persistent table
    written to db.h5 contains X plus the §6 metadata columns.

    Args:
        event_df (pd.DataFrame): gated events table keyed by
            (canonical_ticker, report_date) with all Block 1-4 features
            precomputed.
    Returns:
        X (pd.DataFrame), y (pd.Series), groups (np.array),
        groups_keys (pd.Series)
    """
    # 1. Enforce rigorous LTR chronological grouping layout
    event_df = event_df.sort_values(
        by=['calendar_week_group', 'canonical_ticker']
    ).reset_index(drop=True)

    X = pd.DataFrame(index=event_df.index)

    # Block 1: Catalyst Fundamentals (per-canonical sequential)
    X['sue_score']              = event_df['sue_score']
    X['eps_surprise_pct']       = event_df['eps_surprise_pct']
    X['consecutive_surprises']  = event_df['consecutive_surprises']
    X['sue_acceleration']       = event_df.groupby('canonical_ticker')['sue_score'].diff().values
    X['sue_lag_1']              = event_df.groupby('canonical_ticker')['sue_score'].shift(1)
    X['sue_lag_2']              = event_df.groupby('canonical_ticker')['sue_score'].shift(2)
    X['car_drift_historical_q1']= event_df.groupby('canonical_ticker')['car_10d'].shift(1)

    # Block 2: Microstructure & Technicals (price-derived, precomputed earlier)
    X['is_bmo']                       = event_df['is_bmo'].astype(int)
    X['volume_vma20_ratio_pre_event'] = event_df['volume_vma20_ratio_pre_event']
    X['suv_day_1']                    = event_df['suv_day_1']
    X['pre_event_idiosyncratic_vol']  = event_df['pre_event_idiosyncratic_vol']
    X['opening_gap_t1']               = event_df['opening_gap_t1']
    X['intraday_range_t']             = event_df['intraday_range_t']
    X['pre_event_volume_trend']       = event_df['pre_event_volume_trend']

    # Block 3: Relative Velocity Horizons
    for h in [3, 5, 10, 20, 30]:
        X[f'rel_ret_{h}d'] = event_df[f'rel_ret_{h}d']
    X['sector_adjusted_ret_20d'] = event_df['sector_adjusted_ret_20d']

    # Block 4: Interaction Layer
    X['sue_abs_x_inverse_vol'] = X['sue_score'].abs() / X['pre_event_idiosyncratic_vol']

    # Extract native target label and LTR tracking metadata arrays
    y        = event_df['car_10d']
    groups   = event_df.groupby('calendar_week_group').size().values
    group_keys = event_df['calendar_week_group']

    return X, y, groups, group_keys


def train_and_calibrate_pipeline(X_train, y_train, groups_train,
                                 X_val, y_val, groups_val):
    """
    Trains the cross-sectional NDCG model and fits the monotonic calibration
    bridge (see Design.md §17.3 / §17.4).
    """
    ranker = xgb.XGBRanker(
        objective="rank:ndcg",
        eval_metric="ndcg@3",
        lambdamart_num_threshold=64,
        random_state=42,
    )
    ranker.fit(X_train, y_train, group=groups_train,
               eval_set=[(X_val, y_val)], eval_group=[groups_val])

    # Calibrator maps raw rank scores -> absolute expected CAR (%)
    val_raw_scores = ranker.predict(X_val)
    calibrator = IsotonicRegression(out_of_bounds='clip')
    calibrator.fit(val_raw_scores, y_val)
    return ranker, calibrator
```

---

## 8. Sunday / Weekday Inference Pipeline Requirement

When generating predictions for the upcoming week's cohort (Sunday pre-screen
or weekday live evaluation), the extraction script must populate **this exact
20-feature schema**. Block 1 catalyst features (`sue_`, `consecutive_surprises`,
`car_drift_historical_q1`) are seeded with the most-recently-finalized quarter.
Sunday substitutes a "Perfect Beat Baseline" (`sue_score = 2.0`,
`suv_day_1 = 3.0`) for live catalyst features that are unknown until after the
announcement (see `Design.md` §2 Sunday Pipeline + §3 Weekday Engine live
overwrite).

A missing / miscalculated column will invalidate the `DMatrix` shape and crash
`ranker.predict(X_live)`. If option-implied / live data is unresolvable for a
low-liquidity mid-cap (e.g., `pre_event_idiosyncratic_vol` lacks the historical
window), fall back to `NaN` — the ranker handles native missingness. Do NOT
impute `-999.0` or rolling-mean constants; both leak in cross-sectional
ranking where the cohort's distribution is what matters.
