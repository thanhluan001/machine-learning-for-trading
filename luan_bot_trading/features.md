# Feature Engineering Matrix Protocol: Cross-Sectional Listwise Setup

> **STATUS NOTE (2026-07-23).** This document describes the **21-feature set**
> stored in `/features/train_matrix` PLUS the **v2 planned features** from the
> FMP data expansion (see `feature_sourcing_audit.md` §4A). The feature
> COMPUTATIONS and NaN policy in §1 / §3 / §4 remain authoritative and
> unchanged across Phase F -> Phase G. The DATA-WRANGLING scaffolding (row
> granularity, primary key, group structure, builder code, Sunday/weekday
> inference) has been updated below to reflect:
>
> - **Identity anchor**: `permaTicker` (Tiingo identity-stable) replaces the
>   legacy `perm_id`/`canonical_ticker` collision-disambiguation rule per
>   Phase A migration. See `Design.md §9b` and `database_layout.md`.
> - **Model class**: Phase F's `XGBRanker (rank:ndcg)` is the OBSOLETE
>   leaky design. The DEPLOYABLE Phase G model is `XGBClassifier
>   (binary:logistic)` on **17 Sunday-safe features** (a subset of the 21
>   feature columns; see §1.A). The 4 DROPPED features (`opening_gap_t1`,
>   `intraday_range_t`, `volume_vma20_ratio_pre_event`, `suv_day_1`) are
>   forward-looking at Sunday planning time but become available as live
>   weekday inputs; `opening_gap_t1` is the T+1 morning confirmation gate.
> - **Ranker-target scope**: the `car_10d` continuous CAR target stored in
>   `/features/train_matrix` is still the row-level label per the §0 design.
>   Phase G's binary classifier targets the 3 PEAD gates (`pead_pass`,
>   computed from `car_60d_pass1`) instead -- the 21 features themselves,
>   the per-row computations, and the NaN policy are identical.
> - **Sunday inference**: the §8 "Perfect Beat Baseline" simulation trick is
>   OBSOLETE per Phase G. The Sunday classifier (17 Sunday-safe features)
>   returns `P(PEAD)` directly; the weekday morning overwrites `opening_gap_t1`
>   with the realized T+1 open and applies the single two-stage confirm filter
>   `P(PEAD) >= 0.20 AND opening_gap_t1 in [-15%, -2%]`.
> - **Kelly sizing**: deferred per Phase G -- equal-weight 1/4 NAV per slot.
>   The isotonic calibration bridge in §7's builder skeleton is also dropped
>   from the deployable pipeline (the classifier does not need a calibrated
>   `mu`; sizing is equal-weight). See `future_implementation.md` for the v2
>   roadmap that revives Kelly + the ranker.
> - **FMP data expansion (NEW 2026-07-23)**: FMP $49/mo plan purchased.
>   Adds the "expectation/positioning" feature axis that was identified as
>   the #1 gap in `feature_sourcing_audit.md`. New data sources:
>   `/stable/grades` (analyst upgrade/downgrade history, 14 yrs),
>   `/stable/earnings?includeReportTimes=true` (replaces EODHD -- adds
>   revenue estimates + BMO/AMC + fiscal labels), `/stable/analyst-estimates`
>   (quarterly consensus estimates). See §1.B below for the planned v2
>   feature additions. Doc K (`phase_g_pos_vs_neg_findings.md`) revealed
>   that the current 17 features are all on the "fundamental/momentum" side
>   -- the missing axis is expectation/positioning data. FMP fills this gap.
>
> This file remains the source of truth for the **feature definitions** (what
> is computed, from what data, with what NaN policy). The MODELS consuming
> these features are documented in `Design.md §17 (OBSOLETE) / §17.A (DEPLOYABLE)`.

group structure, isotonic calibration, live sizing) this schema feeds into.

---

## 0. Row Granularity & Group Structure

* **Feature Row Granularity (permaTicker Level).** Each row of the feature
  matrix corresponds to **one earnings event** of one `permaTicker`. A
  permaTicker (Tiingo identity-stable; see `Design.md` §9b, `database_layout.md`)
  contributes many rows over its S&P 400 membership window, one per earnings
  announcement. Ticker renames/rebrands within the same permaTicker are
  collapsed: Tiingo's `/tiingo/daily/{permaTicker}/prices` returns the full
  rebrand-covered price history server-side under a single permaTicker ID, so
  **NO alias concatenation is needed** (the Phase F-era EODHD alias-concat
  workaround documented in the OLD `phase_b_contamination_audit.md` is gone).
  The feature builder gathers **all earnings rows in `/earnings/fmp` keyed by
  that `permaTicker`** (today: 87,609 rows × 18 cols, 839 distinct
  permaTickers, 0 dup groups per `(permaTicker, report_date)`)
  and gates each event with the permaTicker's `wikipedia_intervals`
  membership span plus the **90-day rebalance buffer** described below.
  Pre-rebrand earnings events are directly recoverable because the
  permaTicker's price series spans the rebrand automatically.

* **NO canonical_ticker collision disambiguation (Phase A simplification).**
  PermaTicker is the storage key (`/sp400/{permaTicker}`), so two distinct
  "tracks" NEVER share a key — the Phase A §7.7 disambiguation rule (12
  perm_id pairs sharing `canonical_ticker` with overlapping intervals) is
  **deleted entirely** under permaTicker keying. Each permaTicker is a single
  legal-entity track by construction.

* **When Is a Row Eligible? (Interval Gating).** For each interval
  `{added, removed}` in the permaTicker's `wikipedia_intervals` JSON:
  1. Expand to a buffered window `[added + 90 days, removed]` (use "today" if
     `removed` is `null`). The 90-day buffer IS the first-quarter exclusion —
     the skip of the noisy first quarter after SP400 addition is already baked
     in; no extra per-event logic is required.
  2. Keep the earnings event iff `report_date ∈ [added + 90d, removed]`.
  3. **PermaTickers with `price_unavailable=True` produce zero rows.**

* **GME edge case (multi-interval overlap fix at the gate).** A small number
  of permaTickers (notably GME, US000000000229) carry **multiple overlapping
  Wikipedia intervals** (e.g. `[2016-04-22, null]` AND `[2021-08-04, null]` —
  both still open). The interval loop naively emits the same event TWICE under
  these overlapping intervals. Stage 1 (`01_features_gate_events.py`) applies
  a post-loop dedup by `(permaTicker, report_date)` keeping the row with the
  **earliest `added`** (stable mergesort). Today's verification: GME has 40
  unique events, all `added=2016-04-22`, 0 dup groups in `/features/gated_events`.

* **Feature Lookback May Cross Interval Boundaries.** The 90-day buffer gates
  *which events become rows*, NOT what historical data is used to compute their
  features. Pre-addition prices, sector ETF prices, and the company's full
  earnings history are all fair game as feature inputs.

* **Listwise Group Anchor (`calendar_week_group`).** Two mandatory metadata
  columns are stored alongside the feature matrix (NOT themselves features):
  * `calendar_week_group` — ISO week of `report_date` in `YYYY-Www` form
    (e.g., `2026-W27`). All rows sharing the same group are evaluated by the
    Phase F ranker as one cross-sectional cohort. The Phase G classifier does
    NOT use the LTR group structure directly (binary `pead_pass` target per
    row), but `calendar_week_group` is still stored for compatibility and
    per-week fold splits in nested CV.
  * `car_10d` — the continuous 10-day Cumulative Abnormal Return, stored in
    **log units** (`Σ_{t=T+1}^{T+11} (log R_stock − log R_IJH)`), used as the
    Phase F **listwise gain target**. Phase G's classifier targets the 3 PEAD
    gates `pead_pass` instead (computed from `car_60d_pass1`, see §1.A and
    `_pead_target_retrain.py:compute_pead_gates_full`), but `car_10d` and
    `car_60d_pass1` are BOTH stored in the matrix as label candidates and for
    audit.

* **Strict Sorting Rule.** Before constructing the `DMatrix` for the ranker,
  the training and validation frames **must** be sorted via
  `.sort_values(['calendar_week_group', 'permaTicker', 'report_date'])`
  (permaTicker within-group canonical ordering per
  `03_model/01_train_model.py:SORT_KEYS`) so that rows of the same
  `calendar_week_group` occupy contiguous index blocks. Group sizes are then
  `.groupby('calendar_week_group').size().values` and passed as the `group=`
  argument to `XGBRanker.fit`. The classifier has no contiguity requirement
  but uses the same sort for reproducibility.

* **Training-Horizon Priming Cutoff (§12).** Stage 2
  (`02_build_feature_matrix.py`) stores ALL gated rows at
  `/features/train_matrix` (today: **20,265 rows × 30 cols, 0 dups** —
  includes the 3-year priming window rows, which are needed as
  feature-context neighbors for the `sue_score` 12Q rolling std of
  later training-window rows). The training script (Stage 3 — both the
  OBSOLETE ranker in `01_train_model.py` and the deployable classifier in
  `02_phase_g_sunday_classifier.py`) applies the §12 priming-runway cut at
  **training time, never at storage time**, by filtering
  `train_df = train_df[train_df.report_date >= pd.Timestamp('2015-01-01')]`
  before constructing the model. This drops the first 3 years of the global
  timeline (2012-01-01 → 2014-12-31) from `y` training eligibility — they are
  feature-priming only — and keeps the active training window from
  **2015-01-01 to 2026-12-31 (12 years)**. Apply this filter BEFORE the
  sparse-week (`<3` events, `DEFAULT_MIN_GROUP_SIZE=3` in `01_train_model.py`)
  cutoff so both rules compose cleanly. (See `Design.md §12` for the full
  rationale — the 3-year priming window matches the `sue_score` 12Q
  rolling-std baseline, which is the longest actual feature lookback in the
  v1 schema.)

## 1. Feature Blocks & Column Mapping

The active training matrix `X` contains **21 features** grouped into 4 blocks (B1 7 + B2 7 + B3 6 + B4 1 = 21).
A 21st potential feature (`rev_growth_yoy`, Block 1) is **SHELVED** — see §2.
Macro features (`vix_close`, `fed_funds_rate`, `yield_curve_spread`,
`spy_momentum_20d`) are **NOT in `X`** — see §3 (Macro Detachment).

### Block 1 — Catalyst Fundamentals
Block 1 features are computed from `/earnings/fmp` and require no price lookups.
All per-company sequential features (`sue_*`, `consecutive_surprises`,
`car_drift_historical_q1`) are grouped by **`permaTicker`** (Phase A changed
the grouping key from `canonical_ticker` to `permaTicker` — see `Design.md §9b`)
and sorted by `report_date`.

* `sue_score` (Float): Standardized Unanticipated Earnings.
  `sue_score[t] = difference[t] / σ_{12Q}(difference, all prior quarters)`,
  where `difference = eps_actual - eps_estimated` (precomputed by FMP as
  `eps_difference` in `/earnings/fmp`) and
  `σ_{12Q}(·)` is a **rolling 12-quarter standard deviation over ALL prior
  quarters of the company** (`min_periods = 12`), not restricted to the SP400
  membership window. Where `eps_estimated` is `NaN`, `eps_difference` is also
  `NaN` (FMP returns `null` for both); those rows are **skipped** in the
  rolling std computation (Option B per `Design.md` §15), and `sue_score`
  evaluates to `NaN` for those events.
* `eps_surprise_pct` (Float): Raw percentage deviation of reported EPS from
  consensus — from the `eps_surprise_pct` column of `/earnings/fmp` (derived by
  the FMP fetcher as `(eps_actual - eps_estimated) / |eps_estimated| * 100`).
  **Capped at ±300%** at feature-build time to suppress
  divide-by-tiny-denominator outliers (e.g. -320M% for SUNE 2015-Q1 when the
  estimate was near zero). The cap preserves true large surprises (99th
  percentile raw = +500%) while eliminating nonsense outliers
  (constant `EPS_SURPRISE_PCT_CAP = 300.0` in `02_build_feature_matrix.py`).
`consecutive_surprises` (Integer): Running count of consecutive quarters
where `actual > estimate` (strict beat), grouped by **`permaTicker`**.
Increment by 1 on a beat; reset to 0 otherwise.
* `sue_acceleration` (Float): Q-on-Q change in `sue_score`
  (`sue_score[t] - sue_score[t-1]`), grouped by **`permaTicker`**. Captures
  whether the earnings quality trajectory is improving or deteriorating.
* `sue_lag_1` (Float): `sue_score` from the previous quarter (Q-1) for the same
  **`permaTicker`**. Captures SUE surprise persistence — a series of
  consistent beats may signal systematic analyst mis-calibration.
* `sue_lag_2` (Float): `sue_score` from two quarters ago (Q-2) for the same
  **`permaTicker`**. Extends the persistence check back through two full
  reporting cycles.
* `car_drift_historical_q1` (Float): The **actual index-adjusted CAR generated
  during the stock's previous post-earnings window last quarter** — i.e. the
  CAR (T+1 → T+60, abnormal relative to `IJH`) of the prior event of the same
  **`permaTicker`**, shifted by 1. Tests whether a stock has a repeatable PEAD
  signature: some companies consistently drift while others mean-revert.
  *Implementation note:* requires a two-pass build — compute every event's own
  post-event CAR first, then `shift(1)` per `permaTicker`.
  *Units note:* Stored in **log units** — same as `car_60d_pass1` (which it
  shifts from), since `.shift()` is a unit-preserving linear operation.
  The XGBoost ranker (Phase F) ingests this feature in log units **directly**
  (trees are scale-invariant under monotonic transforms). The Phase G
  `XGBClassifier` likewise ingests log values directly. No `np.expm1` conversion
  is applied to this feature at any stage. The only log→arithmetic conversion
  in the pipeline is on the **target** `car_10d` at the Stage-3 isotonic
  calibration bridge of the OBSOLETE ranker (§17.4 of `Design.md`) — Phase G's
  classifier has no such bridge (it uses equal-weight sizing, see
  `future_implementation.md` §2.1). Log units also incidentally improve training
  quality here: they symmetrize positive vs negative long-horizon (60-day)
  drift magnitudes, giving cleaner tree-split breakpoints.

### Block 2 — Microstructure & Technical Event Context
Block 2 features encode the execution environment, institutional presence, and
positioning footprint on the event day. `T` is the report-trading-day = the
first trading day in `/sp400/{canonical}` with `Date >= report_date` (roll
**forward** to the next trading day if `report_date` falls on a weekend or
holiday; drop the event if no later trading day exists).

* `is_bmo` (Binary 0/1): `1` if the announcement was before market open,
  else `0`. FMP encodes `before_after_market` as **`"bmo"` / `"amc"`**
  (clean lowercase, no CamelCase parsing needed).
  The builder's matching code is:
  `is_bmo = 1 if "bmo" in bam.strip().lower()[:6] else 0`, with a fallback
  1 if `bam.strip().lower().startswith("before")`. Empirically (after the
  2026-07-23 FMP migration): **48% of events are BMO, 100% coverage**
  (was ~41% coverage with EODHD due to CamelCase parsing bug — now fixed).
  *Two-system note.* **Kept in `X` even though it does NOT enter the CAR label**
  window, which is uniform `T+1..T+11` for both BMO and AMC announcements.
  Reasons:
    1. **Label scope.** The label measures post-event drift continuation
       (the PEAD thesis). For BMO events, the day-T intraday market reaction
       is part of the announcement-day repricing, NOT drift — it is captured
       by `opening_gap_t1` and `intraday_range_t` as INPUT features, not in
       `Y`.
    2. **No label-leak bias.** Letting `is_bmo` modify the label window
       (e.g., BMO `T..T+10`, AMC `T+1..T+11`) would have leaked the day-T
       market reaction into the BMO label only — empirically ~3.85 pp of
       systematic bias in BMO labels, or ~73% of the label's expected value
       for a BMO winner. The ranker/classifier would then learn to prefer
       BMO events for a unit-conversion artifact, not because they have
       larger drift.
    3. **Sunday-safe.** Note that `is_bmo` IS known at Sunday planning time
       (it's published in FMP's `/stable/earnings?includeReportTimes=true`
       forward calendar), so this feature is one of the 17 Sunday-safe inputs
       to the Phase G classifier (§1.A).
  See `Design.md` §17.A for the Analysis-Bot /
  Execution-Bot separation.
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
* `pre_event_volume_trend` (Float): OLS slope of **log(`Adj_Volume`)** over
  `T-10 : T-1` (10-day pre-event window). Log-transformed so the slope is
  in units of log(shares)/day (typically [-0.1, +0.1]) instead of raw
  shares/day (which was incomparable across stocks of different liquidity).
  Positive slope = volume increasing into earnings (anticipation/leakage);
  negative slope = volume decreasing (pre-earnings lull, investors
  de-risking before the binary event).

### Block 3 — Multi-Horizon Market & Sector-Adjusted Technicals
Relative velocity into the event. The sector benchmark is **NOT GICS** — it is
the SEC SIC-mapped `index_ref` from **`/metadata/sp400_permatickers`**
(prior to Phase A this was `/metadata/sp400_companies`, PURGED), computed per
`01_data/SIC_code_to_index.md` (e.g. `IJK`, `IJJ`, `XLB`, `XLF`, `XLU`, `XLRE`).
Companies whose `index_ref` is missing default to the mid-cap blend `IJH`.
(Phase v2 plan to make sector-matched CAR the primary benchmark instead of fiat IJH: see `future_implementation.md §2.3`.)

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

### Block 1.A — Sunday-Safe Subset (Phase G classifier)

> The Phase G deployable model trains on a **17-feature subset** of the 21
> feature columns dropped in §1 above. This subset is `SUNDAY_SAFE_FEATURES`
> in `03_model/02_phase_g_sunday_classifier.py`.

The 4 features DROPPED are forward-looking at Sunday planning time or only
available post-T-close (Day-T features); they cannot be computed from
Sunday-side data alone:

* **`opening_gap_t1`** (`Block 2`) — uses `Open[T+1]`, the morning
  AFTER the announcement. This is THE LEAK FEATURE that contaminated Phase F
  v1's `XGBRanker` (Phase G leak-test: NaN-ing `opening_gap_t1` drops hit
  rate 65.8%→49.1% and Sharpe 4.31→-0.14). The Phase G architecture
  consumes it as the **T+1 morning confirmation filter** at weekday morning,
  NOT as a Sunday classifier input.
* **`intraday_range_t`**, **`volume_vma20_ratio_pre_event`**, **`suv_day_1`**
  (Block 2) — all use Day-T market-close entities. Phase G classifies them as
  Day-T features for a weekday re-rank pass (`future_implementation.md §2.5`).

The 17 Sunday-safe features (the classifier's `X`):

```
  Block 1 (7):
      sue_score, eps_surprise_pct, consecutive_surprises,
      sue_acceleration, sue_lag_1, sue_lag_2,
      car_drift_historical_q1
  Block 2 (3 of 7 -- dropped volume_vma20_ratio_pre_event,
                            suv_day_1, intraday_range_t, opening_gap_t1):
      is_bmo,
      pre_event_idiosyncratic_vol,
      pre_event_volume_trend
  Block 3 (6):  rel_ret_3d, rel_ret_5d, rel_ret_10d, rel_ret_20d,
      rel_ret_30d, sector_adjusted_ret_20d
  Block 4 (1):  sue_abs_x_inverse_vol
```

Total: **17 Sunday-safe features** (`assert len(SUNDAY_SAFE_FEATURES) == 17`
in `02_phase_g_sunday_classifier.py`). The other 4 (mostly Day-T) are
not just unused by the Phase G classifier — they are deliberately FORBIDDEN
from the Sunday-side feature matrix to prevent further leak contamination.

For the Sunday classifier target: **`pead_pass`** — boolean 0/1 label =
"event passed all 3 PEAD verification gates" (computed from `car_60d_pass1`
via `_pead_target_retrain.py:compute_pead_gates_full`). Base rate ≈ 10.68% of
gated events pass all 3 gates (per `pead_target_findings.md`). Among the
`cross-fold NEG_only picks at theta=0.20`, recall lifts to ~22-40%.

### Block 5 — Cross-Product Interactions (SHELVED)

The interactions `sue_x_momentum_5d` and `volume_ratio_x_sue` are EXPLICITLY
SHELVED — they are not part of the v1 active feature matrix. They may be
revisited later if the v1 model underperforms and the next iteration needs
additional signal.

### Block 6 — FMP Expectation/Positioning Features (PLANNED, Phase H)

> **STATUS**: PLANNED, not yet built. Data source confirmed (FMP $49/mo,
> see `feature_sourcing_audit.md` §4A). These features address the #1 gap
> identified in Doc K: the current 17 Sunday-safe features are ALL on the
> "fundamental/momentum" side. The missing axis is **expectation/
> positioning** data — what the market expects and how it's positioned heading
> into earnings.
>
> These features are **Sunday-safe** (available before market open on Monday)
> and will be added to `SUNDAY_SAFE_FEATURES` in the v2 classifier.

**6.1 Analyst Revision Momentum** (from FMP `/stable/grades`)

The #1 PEAD predictor in modern literature. FMP returns 14 years of daily-
granularity analyst upgrade/downgrade actions from 111 firms.

| Feature | Calculation | Source |
|---------|-------------|--------|
| `revision_momentum_30d` | `count(upgrades) - count(downgrades)` in 30 days pre-earnings | FMP `/stable/grades` |
| `revision_momentum_60d` | same, 60-day window | FMP `/stable/grades` |
| `revision_momentum_90d` | same, 90-day window | FMP `/stable/grades` |
| `n_analysts_covering` | count of unique `gradingCompany` in last 90 days | FMP `/stable/grades` |
| `last_action_days_before_earnings` | days from last analyst action to `report_date` | FMP `/stable/grades` |
| `revision_velocity` | rate of estimate changes pre-earnings (from `analyst-estimates` history) | FMP `/stable/analyst-estimates` |

**Rationale**: if analysts are revising UP into earnings, the surprise is
more likely to be positive AND the post-earnings drift stronger (market
underreacts to revision momentum itself). Analysts revising DOWN -> even a
"beat" vs lowered bar is weak. This is the most consistently documented
PEAD predictor in post-2010 literature (Truong 2014, Livnat & Petrovits
2009, Zhang 2008).

**6.2 Revenue Surprise Features** (from FMP `/stable/earnings`)

FMP's earnings endpoint returns `revenueActual` and `revenueEstimated` --
fields EODHD did not provide. Revenue surprise is a complementary signal to
EPS surprise.

| Feature | Calculation | Source |
|---------|-------------|--------|
| `revenue_surprise_pct` | `(revenueActual - revenueEstimated) / revenueEstimated` | FMP `/stable/earnings` |
| `revenue_sue_score` | standardized revenue surprise (rolling std baseline, per permaTicker) | FMP `/stable/earnings` |
| `revenue_eps_agreement` | 1 if revenue and EPS surprise agree in direction, 0 otherwise | FMP `/stable/earnings` |

**Rationale**: many PEAD studies find revenue surprises have equal or
greater predictive power than EPS surprises alone. When revenue and EPS
surprise agree (both positive or both negative), the drift is stronger.

**6.3 Still missing (deferred to later phases)**

These features are NOT available from FMP and would need a separate source:

| Feature | Source needed | Priority |
|---------|---------------|----------|
| `short_interest_change` | FINRA (free, register) or Polygon ($29-199/mo) | MEDIUM |
| `options_iv_skew` | Polygon ($29-199/mo) | LOW |
| `institutional_flow_13f` | SEC EDGAR (free, bulk parser) | LOW |
| `insider_buy_intensity` | SEC EDGAR Form 4 (free) | LOW |
| `transcript_sentiment` | SEC 8-K (free) or FMP higher tier | LOW |

---

## 2. Shelved / Deferred Features

* **`rev_growth_yoy` (NOW BUILDABLE via FMP):** Previously shelved because
  EODHD's earnings endpoint did not carry revenue. FMP's `/stable/earnings`
  returns `revenueActual` and `revenueEstimated`, making revenue surprise
  features buildable. See §1.B Block 6.2 for the planned features
  (`revenue_surprise_pct`, `revenue_sue_score`, `revenue_eps_agreement`).

* **`short_interest_pct_float` (STILL DEFERRED):** No free historical
  short-interest feed is available. FINRA's API requires registration for an
  auth token (free but manual). FMP does not serve short interest on the
  stable API. Polygon ($29-199/mo) is the paid alternative. See
  `feature_sourcing_audit.md` §5.3. Positioning proxies in the current model
  are the cross-sectional interaction of `suv_day_1` (volume shocks) and
  `opening_gap_t1` (opening price gaps) -- both are Day-T/leak features,
  NOT Sunday-safe.

* **Analyst revision features (NOW BUILDABLE via FMP):** Previously not
  feasible -- no data source available on our subscriptions. FMP's
  `/stable/grades` endpoint provides 14 years of daily-granularity analyst
  upgrade/downgrade history from 111 firms. See §1.B Block 6.1 for the
  planned features (`revision_momentum_30d/60d/90d`, `n_analysts_covering`,
  `last_action_days_before_earnings`).

* **Optional Block 5 cross-products (`sue_x_momentum_5d`,
  `volume_ratio_x_sue`):** Shelved for v1. See §1 Block 5.

---

## 3. Macro Detachment (Macro Features Are NOT in `X`)

* `vix_close`, `fed_funds_rate`, `yield_curve_spread`, `wti_oil`,
  `cpi`, `unemployment_rate`, plus the sector-ETF / SPY prices stored at
  `/macros/*`, are **not** part of the active Phase G classifier `X` (the
  17-feature `SUNDAY_SAFE_FEATURES` subset in §1.A). They are also not part of
  the OBSOLETE Phase F ranker `X` (the 21-column `FEATURE_COLUMNS`).

### Why they were OUT under Phase F (ranker-era justification, now historical)

Under Phase F's **listwise `XGBRanker (rank:ndcg)`**, the structural constraint
was hard: macros are uniform (or near-uniform) across all rows within a given
`calendar_week_group`, so a cross-sectional listwise objective could not
meaningfully split on them within a group — any attempt wasted tree capacity
or injected cross-time noise into the per-week ordering.

### Why the classifier CHANGES the argument (technical side)

Under Phase G's **per-row `XGBClassifier (binary:logistic)`**, the within-group
invariance no longer blocks training: the classifier makes one prediction per
row, not an ordering within a group. `XGBClassifier` can split on `vix_close`
even if every event in the same week shares the same value — it just needs
*some* leaf-impurity improvement, which it can find from the cross-time
variation in macro state (low-VIX 2017, stress 2020-Q1, hiking 2022-Q3, etc.).
So the **organizational** blocker from Phase F is gone — macros are now legal
classifier inputs from a pure XGBoost-mechanics standpoint.

### Why we STILL recommend keeping them OUT for the deployable Phase G classifier

There are three reasons, none of them purely organizational:

1. **Fold-axis time collinearity.** Nested CV walks forward in time (4 folds,
   anchored per `03_model/02_phase_g_sunday_classifier.py`). Macro regime is
   strongly collinear with the fold axis — e.g. 2020-Q1 stress lives almost
   entirely in fold 1, the 2022-23 hiking cycle in fold 3. Splitting on
   `vix_close` or `fred_funds_rate` therefore partitions rows along fold
   boundaries, not within-fold signal. The classifier will not crash on it,
   but it learns leaves that do not generalize across folds — exactly the
   kind of regime-overfitting that a small-OOS nested-CV harness is designed
   to detect-and-penalize. The same problem the ranker had at the group level
   reappears at the fold level for the classifier.

2. **Small-OOS-sample fragility (curse of dimensionality).** Today the
   deployable rule produces 29 OOS trades with cross-fold Sharpe +1.23
   (std 0.91, per-fold [0.24, 1.31, 0.97, 2.41]). Adding ~6 macro features
   raises `SUNDAY_SAFE_FEATURES` from 17 to 23 — a ~35% column-count
   increase on a sample size that bootstrap CI already places at Sharpe
   95% `[+1.04, +1.58]` (Doc H). Empirical PEAD literature finds only modest
   slow-moving macro modulation in mid-caps; we would be paying real degrees
   of freedom for an unproven, weak signal. Coming out of a leak overfitting
   cycle, conservative feature count seems right.

3. **A better architectural home exists.** The right way to consume a slow,
   soft macro signal is NOT as a deep-tree leaf split inside the classifier.
   It is as a **regime overlay** one layer down — apply it at portfolio
   construction time, where it can adjust slot caps, stop-loss tightness,
   and the confidence-to-size mapping without polluting the per-event
   P(PEAD) estimate. This architectural separation survives the F→G
   migration cleanly.

### Where they belong — the Phase G execution / risk overlay

These macro signals are preserved at the **Portfolio Execution Layer** (in the
Phase G pipeline this is `04_backtest/04_phase_g_portfolio.py:
simulate_portfolio()` and the live execution scripts that mirror it) and used
downstream to:

* ~~shift the **per-trade stop-loss** tightness~~ (UPDATED 2026-07-30:
  a -10% delayed stop is used -- statistically neutral but caps tail
  risk. Tighter stops (3%) HURT expectancy; wider stops (-12%, -14%)
  also hurt. -10% is the only neutral level),
* explore **regime-adaptive entry timing** (pre-gap vs post-gap) in
  low-VIX vs high-VIX regimes,
* cap slot count (`n_slots=4` baseline) or restrict leverage when
  `fred_yield_curve_spread < 0` (inverted yield curve),
* gate the `weekly_schedule_queue` entirely during macro-regime breakdowns
  (e.g., VIX > 35 + WTI > 20% MoM).

### Empirical evaluation path (deferred to Phase v2)

A legitimate question — "does macro state actually lift cross-fold AUC?" — is
**not answered by this policy; it is deferred**. To answer it honestly we
need a dedicated experiment (call it **Phase G PLUS macro probe**):

1. Join the 6 FRED macro series + 10-day rate-of-change features onto the
   20,265-row `/features/train_matrix` keyed by event-day `T`.
2. Re-run the exact `02_phase_g_sunday_classifier.py` nested-CV protocol
   (per-fold POS-tuned HP via `03_phase_g_sweep.py`) on `SUNDAY_SAFE_FEATURES
   ∪ {vix_close, fed_funds_rate, yield_curve_spread, wti_oil, cpi,
   unemployment_rate}`.
3. Compare per-fold AUC, OOS Sharpe, and trade-level win rate vs the Doc-H
   baseline WITH and WITHOUT the macros. **Hypothesis H_macro**: macros lift
   mean OOS Sharpe above +1.31 by a statistically meaningful margin (Δ > 0.15
   with non-overlapping bootstrap CIs).
4. If H_macro is supported → promote macros into `SUNDAY_SAFE_FEATURES` and
   re-tag this section (macros become IN). If not → keep the regime overlay
   architecture and skip them from the classifier `X`.

This is parked as a Phase v2 candidate at `future_implementation.md §3.5
Regime Probe` (separate sub-model / overlay, not input columns) and references
the broader confidence-calibrated sizing question at `future_implementation.md
§3.4`. Don't run it on the deployable artifact today.

### Convenience note (data is already there)

For convenience, the macro series are already stored in `db.h5` under
`/macros/fred_*` (VIX `fred_vix_close`, fed funds `fred_fed_funds_rate`, 10Y-2Y
spread `fred_yield_curve_spread`, WTI `fred_wti_oil`, unemployment
`fred_unemployment_rate`, CPI `fred_cpi`) and `/macros/SPY` plus
`/macros/{sector ETF}`. Pull them at Sunday inference time and thread them into
the execution / risk overlay; **do not** join them into the training matrix
unless H_macro is cleared above.

---

## 4. NaN Handling Policy

* **Never drop rows.** The model — Regressor, Ranker, AND Classifier — handles `NaN`
  natively via optimal default split direction. Features that cannot be
  computed because of insufficient lookback data, missing analyst estimates,
  or unaligned trading days evaluate to `NaN`, and the row survives.
  (The ONE exception: T-match failures — rows where no trading day `Date >=
  report_date` exists for a `T` lookup. These ARE dropped + logged by
  `02_build_feature_matrix.py`; today: 110 T-match drops, dominated by the
  Coherent-legacy `US000000001364` permaTicker whose Tiingo price history ends
  2022-07-01 so no post-2022 events resolve a T.)
* All feature columns are strictly `float` (or `int` for `consecutive_surprises`
  and `is_bmo`). No `object` / `str` columns enter the `DMatrix`.
* Cases that produce `NaN` by construction:
  * `sue_lag_1` / `sue_lag_2` for the 1st / 2nd event of a company. (Typo
    note: previously written as `sue_a_lag_1`; correct column names are
    `sue_lag_1` and `sue_lag_2`.)
  * `sue_score` for the first 11 quarters of a company when the rolling-12Q
    std denominator has fewer than 12 prior observations.
  * `pre_event_idiosyncratic_vol` when fewer than 2 valid residual-return
    observations fall inside `T-20 : T-1`.
  * `opening_gap_t1` when `T+1` falls outside the stored price series.
  * Block 2 / Block 3 features for events near the start of a company's price
    series where the required trailing lookback window does not yet exist.

---

## 4.A. Macro features — EXCLUDED (empirically validated)

**A/B test (2026-07-30, `04_backtest/35_macro_ab_test.py`)**: Tested adding
12 macro features (6 FRED series + 6 20-day rate-of-change) to the
24 Sunday-safe set.

| Metric | A: 24 Sunday-safe | B: 36 (+ macros) |
|--------|-------------------:|----------------:|
| Win rate | 69.7% | 60.7% (-9pp) |
| Total PnL | +636.4% | +482.6% (-24%) |
| PEAD precision | 36% | 30% (-6pp) |

AUC delta per fold: +0.0004, -0.0028, -0.0036, -0.0045 (macros hurt
in 3 of 4 folds, increasingly negative).

**Conclusion**: macros HURT the binary classifier. Despite appearing
as #2 and #4 in feature importance (macro_fed_funds_rate,
macro_unemployment_rate_roc20), this is in-sample overfitting:
macros have very few unique values over time, and with 4 time-ordered
CV folds, each fold sees a different regime. The model can't learn
regime transfer — it learns "this regime = good" then applies it to
a different fold where it fails.

**Why the original LTR rationale was wrong but the conclusion was right**:
The LTR excluded macros because "macros are constant across ranker
candidates." That's technically correct for a ranker but irrelevant for
a binary classifier. However, the conclusion (exclude macros) is correct
for a different reason: PEAD is a stock-specific event driven by
earnings surprise + analyst revisions + volume — not by macro regime.
The macro context doesn't change whether a specific stock will drift
after a positive earnings surprise. Adding macro noise distracts the
model from the stock-specific signals that actually predict PEAD.

**Deployable feature set**: 24 Sunday-safe features (stock-specific only).
Macros are NOT included.

---

## 5. Active Feature Summary Table

| # | Block | Feature | Calculation | Priority |
|---|-------|---------|-------------|----------|
| 1 | 1 | `sue_score` | `difference[t] / σ_{12Q}(diff, all prior quarters)` per company (`min_periods=12`) | **Must** |
| 2 | 1 | `eps_surprise_pct` | `eps_surprise_pct` column from `/earnings/fmp` (derived as `(eps_actual - eps_estimated) / |eps_estimated| * 100` by FMP fetcher), capped at ±300% | **Must** |
| 3 | 1 | `consecutive_surprises` | Running count, +1 on `actual > estimate`, reset 0 otherwise (per `permaTicker`) | **Must** |
| 4 | 1 | `sue_acceleration` | `sue_score[t] - sue_score[t-1]` (per `permaTicker`) | **Must** |
| 5 | 1 | `sue_lag_1` | `sue_score` from Q-1 (per `permaTicker`) | **Must** |
| 6 | 1 | `sue_lag_2` | `sue_score` from Q-2 (per `permaTicker`) | **Must** |
| 7 | 1 | `car_drift_historical_q1` | Prior event's post-earnings **60-day** CAR (T+1→T+60, IJH-adjusted), `.shift(1)` per `permaTicker` — two-pass build; stored in **log units** (inherited from `car_60d_pass1`), fed to ranker/classifier directly with NO arithmetic conversion | **Must** |
| 8 | 2 | `is_bmo` | `1 if "bmo" in before_after_market.lower()[:6] else 0` (FMP uses clean `"bmo"`/`"amc"` format; 48% of events BMO, 100% coverage) | **Must** (also Sunday-safe, see §1.A) |
| 9 | 2 | `volume_vma20_ratio_pre_event` | `Volume[T] / mean(Volume[T-20 : T-1])` | **Must** |
| 10 | 2 | `suv_day_1` | `Adj_Volume[T] / mean(Adj_Volume[T-20 : T-1])` | **Must** |
| 11 | 2 | `pre_event_idiosyncratic_vol` | `std(log_ret_stock - log_ret_IJH, ddof=1)` over `T-20 : T-1` | **Must** |
| 12 | 2 | `opening_gap_t1` | `(Open[T+1] - Close[T]) / Close[T]` — **LEAK FEATURE**, NOT in Sunday classifier `X` (see §1.A); used as T+1 morning confirmation filter at live inference | **Must** (stored but Sunday-excluded) |
| 13 | 2 | `intraday_range_t` | `(High[T] - Low[T]) / Close[T]` | **Must** |
| 14 | 2 | `pre_event_volume_trend` | OLS slope of **log(`Adj_Volume`)** over `T-10 : T-1` (log-transformed for cross-stock comparability; positive = volume ramp-up into earnings, negative = pre-earnings lull) | **Must** |
| 15 | 3 | `rel_ret_3d` / `_5d` / `_10d` / `_20d` / `_30d` | `log_stock_ret(h) - log_IJH_ret(h)` for `h ∈ {3,5,10,20,30}` | **Must** |
| 16 | 3 | `sector_adjusted_ret_20d` | `log_stock_ret(20d) - log_sector_etf_ret(20d)` via SIC-mapped `index_ref` (default `IJH`) | **Must** |
| 17 | 4 | `sue_abs_x_inverse_vol` | `abs(sue_score) / pre_event_idiosyncratic_vol` (NaN-safe) | **Must** |

**Total active features in `X`: 21** (the §5 footer previously said "20"; that
was an off-by-one in prose arithmetic. Block 1 contributes 7 features, Block 2
contributes 7, Block 3 contributes 6 (5 rel_ret + 1 sector-adjusted), Block 4
contributes 1, total = 21 feature columns.)

**Of the 21 stored features, only 17 are Sunday-safe per §1.A** — the Phase G
Deployable classifier trains on the 17-feature subset (see `03_model/02_phase_g_sunday_classifier.py:SUNDAY_SAFE_FEATURES`).
The remaining 4 (`opening_gap_t1`, `intraday_range_t`,
`volume_vma20_ratio_pre_event`, `suv_day_1`) are Day-T or T+1 features reserved
for the weekday pass (or for the NEXT-iteration re-rank model, see
`future_implementation.md §2.5`).

The Phase F `XGBRanker` (OBSOLETE -- see `Design.md §17`) ingested all 21 including
the leak feature `opening_gap_t1`. Its edge was contaminated (Phase G leak-test
showed NaN-ing `opening_gap_t1` drops Sharpe 4.31 -> -0.14).

**v2 planned expansion (Phase H, FMP data):** the 17-feature Sunday-safe set
will be augmented with ~6-9 new features from FMP (§1.B Block 6): analyst
revision momentum (`revision_momentum_30d/60d/90d`, `n_analysts_covering`,
`last_action_days_before_earnings`) + revenue surprise (`revenue_surprise_pct`,
`revenue_sue_score`, `revenue_eps_agreement`). These add the "expectation/
positioning" axis that Doc K identified as the #1 gap. See
`feature_sourcing_audit.md` for the confirmed data plan.

---

## 6. Stored (Non-Feature) Columns in the Output Table

The output of `build_feature_matrix()` (stored in `db.h5` under `/features/train_matrix`)
contains the 21 features above, PLUS these mandatory metadata columns that are
NOT used as inputs to the ranker; they are required by the training loop, the
Sunday classifier, and the execution layer. Today `20,265 rows × 30 cols, 0 dups`
(per `database_layout.md`).

* `permaTicker` (str) — PRIMARY key. Joins back to `/metadata/sp400_permatickers`
  (Phase A identity table; `/metadata/sp400_companies` was PURGED), and joins
  to `/sp400/{permaTicker}` for prices.
* `canonical_ticker` (str) — informational only. Retained for backward
  compatibility with pre-Phase-A code.
* `cik` (str) — SEC CIK (informational/audit only; may be `None`).
* `report_date` (datetime) — the earnings announcement date (calendar basis).
* `T` (datetime) — the matched trading day (`Date >= report_date`, rolled
  forward) used for all price-side feature lookups. **T-match failures are the
  one NaN-policy exception** — Rows with no resolvable `T` (no trading day
  `>= report_date` in the `/sp400/{permaTicker}` price series) are DROPPED +
  logged; today: 110 T-match drops, dominated by the Coherent-legacy
  `US000000001364` permaTicker (price history ends 2022-07-01).
* `calendar_week_group` (str) — ISO week (`YYYY-Www`) — the listwise group
  anchor for the OBSOLETE Phase F ranker. Used by Phase G classifier for
  per-week fold splits in nested CV (see `03_model/02_phase_g_sunday_classifier.py`).
* `added` (datetime) — the permaTicker's earliest-Wikipedia-interval `added`
  date that admitted this event (audit only; used in the GME edge-case dedup).
* `car_10d` (float) — the continuous 10-day Cumulative Abnormal Return stored
  in **log units** (`Σ_{t=T+1}^{T+11} (log R_stock − log R_IJH)`). Used as the
  Phase F LTR target. Phase G's classifier targets the 3 PEAD gates
  (`pead_pass`, computed from `car_60d_pass1`) instead — but `car_10d` is
  still stored as a row label column and used for backtest realized-PnL
  computation (`compute_entry_pnl` in `02_phase_g_sunday_classifier.py`).
  For the OBSOLETE ranker, the conversion to arithmetic percentages at the
  §17.4 isotonic calibration bridge was `y_arith = np.expm1(car_10d_log)`;
  the calibrator's output `mu` was then in true percentage units and was
  consumed directly by Kelly (no further transformation). Phase G drops this
  bridge (equal-weight sizing, see `future_implementation.md §2.1`).
* `car_60d_pass1` (float) — Oracle 60-day post-event CAR (pass-1, computed
  once for the long-horizon drift horizon). Stored as `car_60d_pass1` in log
  units. NOT used by the classifier `X` — used by the **label-derivation
  pass** (`_pead_target_retrain.py:compute_pead_gates_full`) to compute the
  binary `pead_pass` target via the 3 PEAD verification gates (CAR>+3%,
  institutional-vol >2× vma20, MaxDD_MA>-1.5%). Also shifted by 1 as
  `car_drift_historical_q1` (Block 1 feature).

---

## 7. Builder Implementation Skeleton — Phase F reference, partly OBSOLETE

> **STATUS NOTE (2026-07-22).** The code skeleton below is the Phase F-era
> `XGBRanker` + isotonic calibration bridge contract. **Both the `XGBRanker`
> objective and the `IsotonicRegression` calibrator are OBSOLETE in the Phase G
> pipeline** (`01_train_model.py main()` still trains the ranker+calibrator but
> the entire edge was leak contamination — see `Design.md §17`). The Phase G
> pipeline is:
>
> - `02_features/02_build_feature_matrix.py` writes `/features/train_matrix`:
>   reads `/metadata/sp400_permatickers`, `/earnings/fmp` (per-permaTicker),
>   `/sp400/{permaTicker}` (Tiingo prices), `/macros/IJH`, computes the 21
>   features + 8 metadata columns, applies T-match-drop + GME-multi-interval
>   dedup, and writes the 20,265-row x 30-col output to db.h5. **No isotonic
>   calibrator and no ranker in the builder.**
> - `03_model/02_phase_g_sunday_classifier.py` trains `XGBClassifier` on the
>   17-feature `SUNDAY_SAFE_FEATURES` subset with the binary `pead_pass` label
>   (computed via `_pead_target_retrain.py:compute_pead_gates_full`). No
>   isotonic bridge, no `mu` output, no Kelly input. Saves `classifier.json`
>   + `meta.json` + `threshold_sweep.csv` (calibrator.pkl is intentionally
>   empty).
> - `03_model/01_train_model.py` retains the OBSOLETE `main()` ranker trainer as
>   a deprecated block (per the deprecation banner in that file) but its helper
>   API (`load_train_matrix`, `apply_priming_cutoff`, `DB_FILE`,
>   `FEATURE_COLUMNS`, `LABEL_COLUMN`, `GROUP_COLUMN`, `SORT_KEYS`,
>   `DEFAULT_SPLIT_DATE`, `DEFAULT_MIN_GROUP_SIZE`, `N_BUCKETS`,
>   `PRIMING_RUNWAY_START`) IS still load-bearing and imported by all Phase G
>   scripts.
> - Kelly sizing is REPLACED by equal-weight 1/4 NAV per slot per Phase G
>   (= `04_backtest/04_phase_g_portfolio.py:simulate_portfolio(n_slots=4)`),
>   and the isotonic calibration bridge restoration is parked for Phase v2
>   (`future_implementation.md §2.1`).
>
> The skeleton below is preserved as a HISTORICAL reference for what the Phase F
> ranker+calibrator looked like; it is NOT a contract for Phase v2 either — see
> `future_implementation.md §2.2` for the leak-clean ranker plans.

The full implementation lives in `02_features/02_build_feature_matrix.py`
(NOT in this doc). The Phase F reference contract below shows the OLD shape
(kept for historical context — search git history or the archive for context).
The Phase G code is similar in shape but uses `permaTicker` as the grouping
key, drops the leak feature from classifier training, and has no isotonic
calibrator or ranker in the builder proper.

```python
import pandas as pd
import numpy as np
from sklearn.isotonic import IsotonicRegression
import xgboost as xgb


def build_ranking_matrix(event_df):
    """
    Phase F reference builder (OBSOLETE per STATUS NOTE above).
    The Phase G version uses `permaTicker` as the primary key throughout —
    see `02_features/02_build_feature_matrix.py` for the live code.
    """
    # 1. Enforce rigorous LTR chronological grouping layout
    event_df = event_df.sort_values(
        by=['calendar_week_group', 'permaTicker', 'report_date']   # Phase A: permaTicker replaces canonical_ticker
    ).reset_index(drop=True)

    X = pd.DataFrame(index=event_df.index)

    # Block 1: Catalyst Fundamentals (per-permaTicker sequential)
    X['sue_score']               = event_df['sue_score']
    X['eps_surprise_pct']        = event_df['eps_surprise_pct']                          # capped at +/-300% upstream
    X['consecutive_surprises']   = event_df['consecutive_surprises']
    X['sue_acceleration']        = event_df.groupby('permaTicker')['sue_score'].diff().values
    X['sue_lag_1']               = event_df.groupby('permaTicker')['sue_score'].shift(1)
    X['sue_lag_2']               = event_df.groupby('permaTicker')['sue_score'].shift(2)
    X['car_drift_historical_q1'] = event_df.groupby('permaTicker')['car_60d_pass1'].shift(1)   # NB: shift from car_60d_pass1 (60d CAR), NOT car_10d. Units = log.

    # Block 2: Microstructure & Technicals (price-derived, precomputed earlier)
    X['is_bmo']                        = event_df['is_bmo'].astype(int)   # CamelCase 'BeforeMarket'/'AfterMarket' matched upstream
    X['volume_vma20_ratio_pre_event']  = event_df['volume_vma20_ratio_pre_event']
    X['suv_day_1']                     = event_df['suv_day_1']
    X['pre_event_idiosyncratic_vol']   = event_df['pre_event_idiosyncratic_vol']
    X['opening_gap_t1']                = event_df['opening_gap_t1']   # LEAK FEATURE -- not in Sunday classifier's SUNDAY_SAFE_FEATURES (see §1.A)
    X['intraday_range_t']              = event_df['intraday_range_t']
    X['pre_event_volume_trend']        = event_df['pre_event_volume_trend']

    # Block 3: Relative Velocity Horizons
    for h in [3, 5, 10, 20, 30]:
        X[f'rel_ret_{h}d'] = event_df[f'rel_ret_{h}d']
    X['sector_adjusted_ret_20d'] = event_df['sector_adjusted_ret_20d']

    # Block 4: Interaction Layer
    X['sue_abs_x_inverse_vol'] = X['sue_score'].abs() / X['pre_event_idiosyncratic_vol']

    # Phase F target label and LTR tracking metadata arrays (OBSOLETE).
    # Phase G classifier targets `pead_pass` (binary 0/1, computed from
    # `car_60d_pass1` via the 3 PEAD gates -- see _pead_target_retrain.py).
    y          = event_df['car_10d']                 # log-CAR -- Phase F ranker target
    groups     = event_df.groupby('calendar_week_group').size().values
    group_keys = event_df['calendar_week_group']

    return X, y, groups, group_keys


# Phase G deployable classifier training (replaces the OBSOLETE ranker+calibrator):
# clf = xgb.XGBClassifier(
#     objective="binary:logistic",
#     learning_rate=0.05, reg_lambda=1.0, subsample=0.7, colsample_bytree=0.7,
#     eval_metric=["logloss", "auc"], random_state=42, n_jobs=-1,
#     # Plus per-fold POS-tuned HP (gamma=10/5/3/3 from Doc D nested CV).
#     # See `03_model/02_phase_g_sunday_classifier.py` for the exact fit protocol.
# )
# clf.fit(X_train[SUNDAY_SAFE_FEATURES], y_train_pead_pass)   # NO ordinal grid, NO isotonic bridge, NO Kelly


# OBSOLETE Phase F bridge — left for historical reference (CONTAMINATED, DO NOT USE):
def train_and_calibrate_pipeline_obsolete(X_train, y_train, groups_train,
                                          X_val, y_val, groups_val):
    """Phase F: cross-sectional NDCG model + monotonic calibration bridge (§17.3 / §17.4).
    CONTAMINATED by `opening_gap_t1` leak — DO NOT USE."""
    ranker = xgb.XGBRanker(
        objective="rank:ndcg",
        eval_metric="ndcg@3",
        lambdamart_num_threshold=64,
        random_state=42,
    )
    ranker.fit(X_train, y_train, group=groups_train,
               eval_set=[(X_val, y_val)], eval_group=[groups_val])

    val_raw_scores = ranker.predict(X_val)
    y_val_arithmetic = np.expm1(y_val)         # log-CAR  -> arithmetic %
    calibrator = IsotonicRegression(out_of_bounds='clip')
    calibrator.fit(val_raw_scores, y_val_arithmetic)
    return ranker, calibrator
```

---

## 8. Sunday / Weekday Inference Pipeline Requirement

> **STATUS NOTE (2026-07-22).** The "Perfect Beat Baseline" simulation trick
> (Sunday substitutes `SUE_current=2.0`, `SUV_day_1=3.0` for unrealized live
> features) is OBSOLETE under Phase G — see `Design.md §2 STATUS NOTE`. Phase G
> replaces it with a **Sunday classifier on 17 Sunday-safe features** (§1.A)
> that outputs `P(PEAD)` per event directly. There is no Sunday-simulation
> of catalyst features; there is no tier-splitting or hurdle laddering; there
> is no Sunday-vs-weekday two-pass full re-rank. The weekday morning pass
> simply OVERWRITES `opening_gap_t1` with the realized `Open[T+1]` value and
> applies the single two-stage confirmation filter.

### Sunday pass (pre-screen)

For each upcoming-week earnings event for a permaTicker currently in the
S&P 400 universe, populate the **17 Sunday-safe feature columns** (§1.A subset
of `FEATURE_COLUMNS`), then run the saved Phase G classifier:

```python
proba_pead = clf.predict_proba(X_sunday[SUNDAY_SAFE_FEATURES])[:, 1]
sunday_watchlist = X_sunday[proba_pead >= THETA]   # THETA=0.20 per Doc F
```

The Sunday pass produces a watchlist of (permaTicker, report_date, P(PEAD))
candidates whose `P(PEAD) >= theta_screen`. **No "Perfect Beat Baseline"
substitution.** All 17 Sunday-safe features are real — they depend ONLY on
the permaTicker's stored price history in `/sp400/{permaTicker}` and on the
FMP `/earnings/fmp` row for the event.

### Weekday pass (T+1 morning confirmation)

At the weekday morning of `T+1` for each candidate that announced after
Wednesday, the realized `Open[T+1]` equals the morning's actual opening price
of the permaTicker. Compute:

```python
realized_gap = (Open[T+1] - Close[T]) / Close[T]
accepted = sunday_watchlist[
    (sunday_watchlist.P_PEAD >= THETA) &
    (realized_gap >= GAP_LO) & (realized_gap <= GAP_HI)
]   # GAP_LO=-0.15, GAP_HI=-0.02 (Doc G NEG_only operating point)
```

The accepted set enters **pre-gap** (`Close[T-1]` BMO / `Close[T]` AMC, before the earnings announcement), exits `Close[T+5]` (5-day hold), -10% delayed stop, equal-weight. Uses a **binary classifier** (`binary:logistic`, target = `pead_pass`), accepting picks where **P(PEAD) >= 0.20** and sector != XLF
`1/4 NAV` per trade with max 4 simultaneous slots (Doc H baseline).

### Pipeline shape contract

| Stage | Source columns        | DROPPED `opening_gap_t1`? | Output shape           |
|-------|-----------------------|---------------------------|------------------------|
| Sunday | 17 Sunday-safe features (per §1.A) | YES (not yet realized) | `P(PEAD)` per event    |
| Weekday T+1 | realized `opening_gap_t1` only (price read) | realized post-market, not simulated | accepted/rejected filter |

A missing / miscalculated column at Sunday inference time will invalidate the
`DMatrix` shape and crash `clf.predict_proba(X_live)`. If a Sunday-safe feature
is unresolvable for a low-liquidity mid-cap (e.g., `pre_event_idiosyncratic_vol`
lacks the historical 20-day window), fall back to `NaN` — XGBoost handles
native missingness for **classification** too. Do NOT impute `-999.0` or
rolling-mean constants; both leak cross-sectionally when the cohort's
distribution is what matters.

### Live-Inference Implementation Pointers

- `03_model/02_phase_g_sunday_classifier.py` — Sunday classifier trainer +
  threshold sweep (saved artifact at
  `03_model/models/phase_g_v1_sunday_classifier/`).
- `04_backtest/14_phase_g_trade_stats.py` — load the saved classifier +
  apply the operating point filter to compute per-trade win/loss stats at
  test time. The same logic is the deployable Sunday model's run inference.
- `04_backtest/04_phase_g_portfolio.py:simulate_portfolio(n_slots=4)` —
  equal-weight portfolio simulator (no Kelly, no transaction cost).
- For the v2 ambition (Kelly sizing + ranker + Day-T re-rank + tier hurdles
  etc.), see `future_implementation.md`.

