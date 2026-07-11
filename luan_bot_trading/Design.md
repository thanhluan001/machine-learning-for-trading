# Quantitative Architecture Specification: PEAD Algorithmic Trading Bot
## Target Universe: S&P 400 Mid-Cap (approximately 400 Assets)

## 1. Target Variable Definition (The ML Objective)
*   **Core Goal:** Do not predict binary earnings surprises or nominal absolute price moves. Predict institutional drift (Idiosyncratic Alpha / Market-Neutral Abnormal Returns).
*   **Target Metric:** Cumulative Abnormal Return ($CAR$) measured from $T+1$ to $T+11$ (10-day holding horizon), where $T$ is the earnings announcement date.
*   **Mathematical Formula:**
    $$CAR_{i} = \sum_{t=T+1}^{T+11} \left( R_{i,t} - R_{m,t} \right)$$
    *   $R_{i,t}$: Daily log return of stock $i$ on day $t$.
    *   $R_{m,t}$: Daily log return of the stock’s corresponding sector ETF or broad market proxy on day $t$.
*   **Model Type:** **XGBoost Ranker (Listwise Learning-to-Rank)**. See §17 for full architecture. The model evaluates all earnings announcements within a calendar week as a single cross-sectional group, optimizing for relative outperformance via NDCG. Raw rank scores are mapped back to absolute expected return ($\mu$) via Isotonic calibration (§17.3) before being passed to the Kelly sizing engine. Do not use pointwise regression or classification frameworks; the minority-class imbalance (~6% base rate) and cross-sectional macro noise are neutralized structurally by the listwise architecture.

---

## 2. Sunday Pipeline: Preliminary Screening & Feature Calibration
*   **Data Scope:** Process 600 universe assets to extract long-term historical features on Sunday: `SUE_trend_1Y`, `SUE_trend_3Y`, `Turnaround Ratio`, and rolling 60-day historical variance ($\sigma^2$).
*   **The Simulation Trick:** Because real-world catalyst features (`SUE_current`, `SUV_day_1`, `VWAP_Coherence`) are unavailable until weekday market open, the Sunday script evaluates candidates by substituting missing attributes with a "Perfect Beat Baseline" (`SUE_current = 2.0`, `SUV_day_1 = 3.0`).
*   **Scoring Metrics:** The Sunday rank utilizes a Model Signal-to-Noise Ratio to sort candidates:
    $$\text{Sunday Rank Score} = \frac{\text{Simulated Predicted CAR}}{\text{Prediction Standard Error}}$$
*   **Watchlist Constraints:** Filter out the top 10 highest-ranked assets to construct the weekly execution pipeline. Divide these assets into 3 execution tiers:
    *   **Tier 1 (Rank 1-3):** Peak structural receptivity names.
    *   **Tier 2 (Rank 4-7):** Moderate structural configuration profiles.
    *   **Tier 3 (Rank 8-10):** Low-receptivity/High-uncertainty configurations.

---

## 3. Weekday Engine: Tiered Hurdles & Real-Time Inferences
*   **Data Updates:** Upon weekday earnings releases, overwrite the Sunday simulated features with true live broker API readings (`SUE_current`, `SUV_day_1`, and `VWAP_Coherence`).
*   **Dynamic Execution Gates:** Run final inference using the live feature vector. The asset must clear a dynamic hurdle scaled according to its Sunday structural tier to trigger a buy order:
    *   **Tier 1 (Rank 1-3):** Trigger trade if Final Predicted CAR $> 1.5\%$
    *   **Tier 2 (Rank 4-7):** Trigger trade if Final Predicted CAR $> 2.5\%$
    *   **Tier 3 (Rank 8-10):** Trigger trade if Final Predicted CAR $> 4.0\%$ (Forces high-variance entries to show massive expected edge).
*   **Temporal Execution Protocol:** Do not execute assets early based on Sunday metrics. The asset remains dormant on the watchlist until its real-world calendar earnings announcement date triggers the live weekday script.

---

## 4. Portfolio Architecture: Fixed Independent Slot-Pod Structure
*   **Capacity Mapping:** The portfolio uses a strict 5-slot parallel configuration layout. 
*   **Slot Capitalization:** Each virtual slot baseline is allocated exactly 20% of the Net Asset Value (NAV).
*   **Saturation Guardrail:** If all 5 slots are holding active 10-day positions, the weekday engine enters Standby Mode. Any fresh market triggers generated during saturation are discarded to avoid path-dependent liquidity dilution.
*   **No Slot Merging:** Under no circumstances should leftover capital from multiple under-allocated slots be merged into a temporary 6th slot. Maximum absolute capital exposure is hard-capped at 100% NAV.

---

## 5. Capital Allocation: Continuous Fractional Kelly Sizing
*   **Sizing Engine:** The system employs Continuous-Time Kelly (Merton's Fraction) inside individual slots to control idiosyncratic distribution variance.
*   **Mathematical Formula:**
    $$K^* = \frac{1}{\gamma} \cdot \frac{\mu}{\sigma^2}$$
    *   $\mu$: Weekday Final Predicted CAR from the XGBoost model.
    *   $\sigma^2$: Scaled historical daily return variance matching the 10-day holding horizon.
    *   $\gamma$: Risk-aversion coefficient. Hardcoded to $\gamma = 2$ for **Half-Kelly** performance boundaries.
*   **Dynamic Risk Management Handbrake:** High-volatility/high-uncertainty stocks inflate the denominator ($\sigma^2$), automatically scaling down the position size to clean, low-leverage "test" allocations. Reliable, low-variance large caps naturally secure maximum slot capacity allocations.

---

## 6. Capital Compounding & Net Asset Value (NAV) Sweeps
*   **Daily Re-baselining:** Total Equity ($E$) must be updated daily at market close using the following asset loop calculation:
    $$E = \text{Liquid Cash Reserve} + \sum (\text{Unrealized Current Fair Market Value of Open Positions})$$
*   **Compounding Feed:** Every morning, scale the 5 individual virtual slots to match the updated portfolio baseline:
    $$\text{Updated Individual Slot Capacity} = \frac{E}{5}$$
*   **Execution Value:** Position sizing calculations use the latest slot baseline value. Profits from exited trades organically compound into larger subsequent positions, while active open unrealized gains are preserved in baseline calculations to avoid under-leverage decay.

---

## 7. Portfolio Slot Constraints & Sunday Ranking Logic
* **Fixed Architecture Limits:** The portfolio configuration is hard-constrained to exactly 5 virtual parallel slots (each representing a maximum 20% Allocation Capacity of current total NAV).
* **Sunday Watchlist Ceiling:** The Sunday screening program must rank all upcoming weekly earnings events by their simulated Model Signal-to-Noise Ratio. The weekly watchlist handed down to the weekday execution script is strictly capped at the top 10 highest-ranked assets to prevent over-scanning and conserve API quotas.
* **Capital Saturation Protocol:** If all 5 slots are occupied by active 10-day post-earnings drift positions, the weekday execution engine must enter Standby Mode and cease all real-time data ingestion and order generation until an active slot is officially vacated via a target horizon close.

---

## 8. Weekday Execution Tiering & Hurdle Logic
* **Dynamic Activation Gates:** The Weekday execution script must read the Sunday ranking tier to dynamically adjust the final predicted validation barrier required to execute a market entry:
  * **Watchlist Rank 1-3 (Tier 1):** Trigger buy order if Live Weekday Final Predicted CAR > 1.5%
  * **Watchlist Rank 4-7 (Tier 2):** Trigger buy order if Live Weekday Final Predicted CAR > 2.5%
  * **Watchlist Rank 8-10 (Tier 3):** Trigger buy order if Live Weekday Final Predicted CAR > 4.0%
* **Simultaneous Signal Conflict Resolution:** In the event that multiple weekday earnings releases simultaneously clear their respective tier hurdle rates on the same morning, capital slots must be allocated sequentially starting with the highest Sunday-ranked asset first.

---

## 9. Data Pipeline & Survivorship Bias Architecture (REVISED)
* **Storage Framework:** Data must be saved in an unified HDF5 storage format (`.h5`) organized hierarchically by ticker nodes or a unified tabular format optimized for time-series querying.
* **Historical Membership Resolution:** The data ingestion script must parse the Wikipedia S&P 400 Revision History/Changes matrix to map the exact historical entry and exit windows for all historical constituents over a 15-year lookback.
* **Multi-Interval Residency Map:** * Do **NOT** store index membership as flat scalar keys (`added_date`, `removed_date`). 
    * The `/metadata/sp400` table must store a structured **Interval Array** (list of dicts) for every ticker to fully capture stocks that exited and re-entered the index over their lifecycles.
    * *Example Structure:* `intervals = [{"added": "2014-03-15", "removed": "2018-06-20"}, {"added": "2022-09-01", "removed": "None"}]`
* **Pre-2012 Backfill:** Wikipedia change history cuts off before 2012. For all tickers whose initial `added_date` is missing from the changes table (including historically removed constituents), backfill `added_date = 2012-01-01` as a conservative lower bound inside the first interval dict. This prevents survivorship bias by ensuring removed constituents can still contribute to training. Use this backfilled date only for training inclusion eligibility; do not treat it as a precise addition date.
* **SEC SIC Sector Standard:** Use the SEC EDGAR SIC classification as the authoritative sector taxonomy (stored in `/metadata/sp400` as `sic` and `index_ref` columns, replacing the old `/metadata/sp400_sic`). This replaces the Wikipedia GICS fields for constituents added to the index before 2012 or where Wikipedia history is incomplete, ensuring consistent sector labels across the full 15-year lookback.
* **The Rebalance Exclusion Rule:** Implement a strict rolling timeline guardrail. A stock's row data is only valid for XGBoost training or weekday signal generation if the current date is $\ge 90$ days (1 quarter) past the closest active index addition date (`added`) inside its interval log. Mark this eligibility using a boolean flag `in_index_clean` inside the HDF5 matrix.
* **EODHD Delisting Fallback Rule (Overriding Previous Drop Rule):** If a stock has no `removed_date` recorded in the Wikipedia changes table AND is not present in the current S&P 400 constituents table, the bot **must not drop the ticker**. 
    * To prevent active survivorship bias (ignoring bankruptcies, fire sales, or defaults), the bot must look up the ticker's historical end-of-day price data in the local EODHD store.
    * The exact date where the daily adjusted close and trading volume flatlines or permanently ceases to print is the definitive `removed_date` for that interval block.

## 9b. Company-Level Merge & Canonical Ticker Layer (NEW)

The feature matrix has **one row per earnings event** (not per ticker symbol; each company contributes many rows over its S&P 400 membership window). Companies change ticker symbols over time (rebrands, mergers, bankruptcy-Q suffixes) while staying continuously in the index. Treating each ticker symbol as a separate company fragments price history, drops pre-rebrand earnings events, and creates phantom "still in the index" rows.

* **Canonical Anchor: SEC CIK.** CIK is the stable company identifier that survives renames, rebrands, bankruptcy-Q suffix delistings, and delistings. All ticker aliases of the same company map to the same CIK.
    * Build the historical `ticker -> CIK` map by unioning (a) current `sec_cache/ticker.txt` and (b) cached DERA `sub_{year}.txt` snapshots, where ticker is extracted from the `instance` column's leading token.
    * Spinoffs create new CIKs and are treated as new companies in v1 (intentional scope cut).
    * Tickers with no recoverable CIK become singletons (canonical = self).
* **Pipeline Step:** A new `02b_build_company_map.py` runs after `02_SEC_sector_gathering.py` and before `03_data_gathering.py`. See `01_data/company_merge_design.md` for full spec.
* **Outputs in `db.h5`:**
    * `/metadata/sp400` is extended with `cik` and `canonical_ticker` columns (per-ticker view retained; each alias points to the canonical).
    * New `/metadata/sp400_companies` table: one row per company (CIK), with columns `canonical_ticker`, `cik`, `aliases` (JSON), `name`, `sic`, `index_ref`, `combined_intervals` (JSON merged span), `per_ticker_intervals` (JSON audit), `price_unavailable` (bool).
* **Canonical Selection Priority:** (1) ticker in current `ticker.txt` AND verified on EODHD; (2) else, ticker with most-recent `removed` date AND verified on EODHD; (3) else, most-recently-added ticker regardless of EODHD (flagged `price_unavailable=True`).
* **Interval Merge:** For each company, collect all aliases' intervals, sort by `added`, merge overlapping or abutting (gap $\le 7$ days) spans. Real gaps (>7 days) kept as separate spans. Result is a single membership span per company, usually 1 span.
* **Data Fetcher (`03_data_gathering.py`):** Iterates per company (not per ticker). Tries `aliases` in priority order on EODHD (`/api/eod/{ticker}.US`); first non-empty response stored under `/sp400/{canonical_ticker}`. Companies with `price_unavailable=True` are skipped entirely with a log line (v1 keeps it simple; revisit if many pile up). EODHD subscription is effectively unlimited, so there is **no throttle, no batching, and no offset checkpoint** — a single invocation processes the full universe.
* **Adj-OHLCV Derivation:** EODHD's `/api/eod` endpoint returns raw OHLC + `adjusted_close` + raw volume only. The pipeline derives `Adj_Open/Adj_High/Adj_Low/Adj_Volume` locally via the `close/adjusted_close` ratio (cumulative split+dividend factor), and sets `Adj_Close = adjusted_close`. Validated empirically (`validate_eodhd_adjclose.py`, 7/7 probe tickers PASS).
* **Earnings Alignment:** The feature builder reads `/metadata/sp400_companies`, gathers **all earnings dates** stored under any alias in `/earnings/raw`, and collapses them into one company-level earnings timeline. The canonical ticker's `/sp400/{canonical}` price series (which EODHD retro-adjusts across rebrands so it spans the alias periods) is used for all feature calculations, including pre-rebrand earnings events. Combined intervals gate each event with the 90-day exclusion buffer.
* **Result:** One company in `/metadata/sp400_companies` $\rightarrow$ many earnings events $\rightarrow$ many feature rows in the final matrix.

---

## 10. Training Matrix Extraction Protocol (REVISED)
* **The Row Selection Filter:** When constructing the training matrix (`X`, `y`) for the XGBoost model, the dataset generator must iterate through the historical timelines and only select rows where `in_index_clean == True`.
* **Dynamic Historical Matching & Interval Validation:**
    * The feature engine must evaluate the timestamp of each historical earnings event against the *entire* interval array of the corresponding asset.
    * If a company was in the index from 2014 to 2018, and then re-added in 2022, rows from 2014+90 days up to the 2018 removal date **MUST be included** in training. Rows between 2018 and 2022+90 days must be dropped. 
    * If a company was deleted via bankruptcy/merger in 2018 (resolved via the EODHD price-flatline fallback), all rows prior to its 2018 delisting (and after its original inclusion date + 90 days) **MUST be preserved** to teach the model downside tail-risk.

---

## 11. Earnings Surprise Ingestion Protocol
* **Data Sourcing:** The data pipeline queries the **EODHD Earnings API** (`/api/calendar/earnings`) to pull historical quarterly timelines of `Actual_EPS`, `Estimated_EPS`, and `Report_Date`. Earnings events are stored in `/earnings/raw` with `difference = actual - estimate` and `percent` (surprise percentage) precomputed by EODHD.
* **HDF5 Alignment:** Store the parsed earnings metrics in a single consolidated `/earnings/raw` table, keyed by `(canonical_ticker, report_date)` with `cik` for company-level alignment. (The prior per-ticker-node `/earnings_history` design is deprecated.)
* **Feature Calculation Rule:** Do not pass raw surprise dollar values to the model. The pipeline must compute the Standardized Unanticipated Earnings (`sue_score`) by dividing the surprise deviation by the rolling 12-quarter standard deviation of the asset's historical surprises (per §15).


## 12. Feature Lookback vs. Training Horizon Slicing
* **Global History Constraint:** The data engine utilizes a strict 15-year historical data runway.
* **Feature Context Window:** All rolling historical features (`SUE_trend`, momentum vectors, macro baselines) are capped at a maximum lookback window of 3 years (12 quarters) — this matches the `sue_score` 12Q rolling-std baseline, which is the longest lookback in the active feature set. No other feature requires a longer window.
* **Active Training Horizon:** The XGBoost training engine must discard the first 3 years of the global history timeline, using them exclusively to populate the initial feature contexts. The active target training matrix ($y$) must be strictly drawn from the remaining 12-year deep history window (2015-01-01 onward, given the 2012-01-01 backfill boundary).
* **Index Membership vs. Feature Lookback Separation:** The 3-year feature lookback requirement applies strictly to the presence of raw historical pricing and fundamental lines inside the HDF5 data storage node. It is NOT a requirement for historical index membership. If an asset has 3 years of trading history available, its rows are fully eligible for training immediately after the 90-day index stabilization buffer passes, regardless of its pre-inclusion index classification.


## 13. Calendar Scheduling & Live Watchlist Ingestion
* **Schedule Provider:** The Sunday scheduling module queries the **EODHD Earnings Calendar API** (`/api/calendar/earnings`), passing a 5-day forward-looking date array (`from` to `to`).
* **Cross-Sectional Filtering:** The fetched raw global calendar must immediately be cross-referenced against the local HDF5 `in_index_clean` array for that specific calendar date. Any reporting ticker not actively clearing the S&P 400 point-in-time membership filter must be instantly pruned from memory.
* **Pre-Earnings Ingestion Matrix:** For the remaining valid weekly candidates, parse the `epsEstimated` and `date` strings. Save this structure to a temporary runtime matrix (`weekly_schedule_queue`) to dictate the active weekday activation order and seed the placeholder values for Sunday's point-estimation simulator.

## 14. Data Sourcing, Index Proxies, and Market-Adjusted CAR
* **EODHD-Centric Ingestion Architecture:** The data pipeline consolidates **all** data sourcing on **EODHD**. EODHD provides 15+ years of continuous split-adjusted daily equity prices, volume metrics, historical earnings actuals/estimates, forward-looking earnings calendar, and bulk earnings surprise metrics through a single subscription.
* **Broad Market Benchmark Proxy (`IJH`):** To calculate the broad market return baseline ($R_{m,t}$) without dealing with spot index availability bugs, the pipeline must query the historical price series for the **iShares Core S&P Mid-Cap ETF (`IJH`)**. 
* **Adjusted Return Constraint:** The feature engine must strictly use the `Adj_Close` column (EODHD `adjusted_close`, locally derived per §9b) for both individual assets and the `IJH` proxy to eliminate artificial price drops caused by dividend distributions or fund splits.
* **Streamlined Market-Adjusted Calculation:** Replace any complex sector-matching lookups with a flat Market-Adjusted Model against the S&P 400 proxy. The daily abnormal return for an asset is calculated as:
    $$AR_{i,t} = \ln\left(\frac{\text{Asset AdjClose}_{t}}{\text{Asset AdjClose}_{t-1}}\right) - \ln\left(\frac{\text{IJH AdjClose}_{t}}{\text{IJH AdjClose}_{t-1}}\right)$$
  The target variable $CAR_i$ is the simple sum of $AR_{i,t}$ across the $T+1$ to $T+11$ holding horizon.

## 15. Operational Pipeline Interfacing
* **Sunday Schedule Parsing:** Every Sunday, the engine queries the EODHD `/api/calendar/earnings` endpoint for the upcoming week. The resulting array is filtered against the local HDF5 `in_index_clean` array to build the localized `weekly_schedule_queue`.
* **The SUE Normalization Feature:** When loading data from the EODHD `/earnings/raw` table, the data script must not pass raw surprise amounts to the model. It must compute the Standardized Unanticipated Earnings (`sue_score`) by dividing the surprise deviation by the **rolling 12-quarter standard deviation** of the asset's historical earnings surprises (per company CIK, across all prior quarters, `min_periods=12`):
    $$SUE = \frac{\text{Actual EPS} - \text{Estimated EPS}}{\sigma_{12Q}(\text{Historical Surprise})}$$
    Where `difference = actual - estimate` is the per-quarter EODHD-provided surprise. Events where `estimate` is NaN have EODHD-set `difference = 0.0` and are retained in the rolling std window (Option B, per `features.md`).
* **Missing Value Routing:** In cases where an asset has been trading for the required 3-year lookback window but lacks older earnings estimates (e.g., due to limited analyst coverage early in its lifecycle), the feature engine must preserve the resulting missing data as a `NaN`. Do not drop the row; allow the model to learn the optimal default splitting direction during the training phase. XGBoost (both Regressor and Ranker) handles NaN natively.

* **Short Interest Metric Omission:** The feature `short_interest_pct_float` has been officially deprecated and removed from the active feature matrix. To maintain a strict 15-year historical data runway without lookahead data contamination, the feature engine relies entirely on the cross-sectional interaction of `SUV_day_1` (volume shocks) and `opening_gap_t1` (opening price gaps) to proxy institutional short-covering velocity and localized structural supply scarcity.

---

## 16. Implementation Reference: Interval Validation Layer

The bot can use this core logic block to generate the precise boolean array mask for daily historical tracking:

```python
def calculate_in_index_clean_mask(ticker_history_df, intervals):
    """
    Evaluates every historical trading row against an array of index residency 
    intervals to prevent multi-residency blind spots and survivorship bias.
    """
    # Initialize index mask to False
    in_index_clean = pd.Series(False, index=ticker_history_df.index)
    
    for interval in intervals:
        added_dt = pd.to_datetime(interval['added'])
        # If 'removed' is None or active, evaluate up to the current runtime date
        removed_dt = pd.to_datetime(interval['removed']) if interval['removed'] != "None" else pd.Timestamp.now()
        
        # Enforce the 90-day stabilization window post-addition
        buffer_end_dt = added_dt + pd.Timedelta(days=90)
        
        # Create mask for rows falling cleanly inside this specific residency block
        interval_mask = (ticker_history_df.index >= buffer_end_dt) & (ticker_history_df.index <= removed_dt)
        
        # Bitwise OR to combine across all historical residencies
        in_index_clean = in_index_clean | interval_mask
        
    return in_index_clean


> **CRITICAL CODE REVIEW UPDATE:** This document overrides and patches the previous data ingestion and index residency rules in **Section 9** and **Section 10**. 
> - Fixed the single-scalar `added_date`/`removed_date` bug to support multi-interval residency rows (preventing the accidental deletion of historical mid-cap training rows for "boomerang" stocks).
> - Patched the "Missing Removal Date" rule to eliminate active survivorship bias by using EODHD flatline price actions to find actual delisting dates.

## 17. Model Architecture: Cross-Sectional Listwise Ranking (XGBRanker)

> **Impacted Pipeline Steps:** `02b_build_company_map.py`, `03_data_gathering.py`, `07_train_model.py`, `08_backtest_execution.py`  
> **Status:** Approved for Implementation (2026-07-08). Replaces the prior pointwise XGBoost Regressor architecture (which suffered from ~6% base-rate class imbalance and could not neutralize cross-sectional macro noise).

### 17.1 Motivation & Structural Pivot
To completely eliminate severe minority class imbalance (~6% base rate) and mathematically neutralize macro-market directional noise, the machine learning pipeline uses a **Listwise Learning-to-Rank (LTR)** architecture (not pointwise regression). 

Instead of treating each earnings event as an isolated data point, the model evaluates all earnings announcements within a given calendar week as a single cross-sectional group, directly optimizing for relative outperformance.

### 17.2 Label and Feature Matrix Restructuring
*   **The Target Label (`car_10d`):** Stored as the continuous **log CAR** — the sum of daily log excess returns `Σ_{t=T+1}^{T+11} (log R_stock − log R_IJH)` — across the 10-day post-event window. **Do not** convert this to discrete ordinal ranks (1, 2, 3) in the dataset. Maintaining continuous labels is required for the Listwise Gain function. Storing in log units is safe for the ranker because NDCG is invariant to monotonic transforms of the gain values; only the *ranking* within each group matters.
*   **Group Anchor (`calendar_week_group`):** A new structural column must be added to the feature matrix representing the calendar week of the earnings date (e.g., `2026_W27`).
*   **Sorting Requirement:** Before feeding data into the model training loop, the training and validation dataframes **must** be sorted chronologically and clustered explicitly by `calendar_week_group`.

### 17.3 Model Architecture Changes (`07_train_model.py`)
*   **Model Class:** Replace `XGBRegressor` or `LGBMRegressor` with **`XGBRanker`** (or LightGBM equivalent).
*   **Objective Function:** Hardcode `objective="rank:ndcg"`.
*   **Evaluation Metric:** Set `eval_metric="ndcg@3"` (or match your portfolio's target weekly slot capacity) to focus optimization entirely on the top of the funnel.
*   **Group Dimension Array:** Extract group sizes using `.groupby('calendar_week_group').size().values` and pass this array explicitly into the `.fit(X, y, group=groups)` method.

### 17.4 The Sizing Bridge: Isotonic Score Calibration
Because `XGBRanker` outputs arbitrary, unitless ordinal marginal utility scores, these scores cannot be passed directly into the Continuous-Time Kelly (Merton's Fraction) sizing engine. A non-parametric calibration layer is introduced to map scores back to absolute expected return ($\mu$).

*   **Implementation:** Fit a `sklearn.isotonic.IsotonicRegression(out_of_bounds='clip')` on the **validation set** immediately after training.
    *   **Input ($X$):** Raw validation output scores from `ranker.predict(X_val)`.
    *   **Target ($y$):** Actual historical `car_10d` converted from **log units to arithmetic** via `np.expm1(y_log)` before fitting. This conversion is mandatory: Stage 2 (`02_build_feature_matrix.py`) stores `car_10d` in log units (see §17.1), but the isotonic calibrator's output must be in true percentage units so it can be consumed directly by the Kelly sizing engine (§17.5).
*   **Monotonic Guarantee:** Isotonic regression is strictly monotonic; it scales the values into realistic return percentages without altering the model's chosen sorting sequence.

### 17.5 Live Inference & Sizing Execution Engine (`08_backtest_execution.py`)
During live trading weeks or backtest steps:
1.  Assemble features for the week's reporting stocks, assigning them to the same query batch.
2.  Generate raw ranking utility scores: `df['raw_rank_score'] = ranker.predict(X_live)`.
3.  Calibrate scores to define expected return: `df['mu'] = calibrator.predict(df['raw_rank_score'])`. Because the calibrator was fit on arithmetic-converted `car_10d` (per §17.4), `df['mu']` is in true percentage units — feed directly to Kelly with NO further conversion.
4.  Apply an **Absolute Quality Handbrake Filter**:
    ```python
    MINIMUM_EXPECTED_CAR = 0.01  # 1% hurdle rate
    df['is_viable_trade'] = df['mu'] >= MINIMUM_EXPECTED_CAR
    ```
5.  Execute **Continuous-Time Kelly Sizing** (Merton's Fraction) on chosen winners:
    $$K^* = \frac{1}{\gamma} \cdot \frac{\mu}{\sigma^2}$$
    Where $\gamma = 2$ (Half-Kelly boundary). For unviable trades failing the handbrake filter, assign a hardcoded size of `0.0` (or a nominal `0.01` test allocation if explicitly desired).