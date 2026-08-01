# Quantitative Architecture Specification: PEAD Algorithmic Trading Bot
## Target Universe: S&P 400 Mid-Cap (approximately 400 Assets)

> **STATUS NOTE (2026-07-22).** This document is the ORIGINAL Phase F-era
> aspiration architecture. Sections §1, §9 (survivorship/inclusion rules),
> §10, §11, §12, §13, §14, §16 are still authoritative.
>
> Sections describing the WORKING pipeline state have been UPDATED to reflect
> the Phase A permaTicker migration AND the Phase G working backtester+
> Sunday classifier. Specifically the divergences below are LIVE in the code:
>
> - **Identity anchor (§9b).** `permaTicker` (Tiingo identity-stable) is the
>   primary anchor. SEC CIK is a deprioritized informational col. Phase A's
>   synthetic `perm_id` is PURGED.
> - **Data source (§9b, §14).** Tiingo `/tiingo/daily/{permaTicker}/prices`
>   is the price-history source. EODHD `/api/calendar/earnings` is retained
>   ONLY for the earnings calendar (Phase D). Local `adj_OHLCV` derivation
>   is gone — Tiingo returns adjusted columns natively.
> - **Model class (§17).** The deployable model is **`XGBClassifier`
>   (`binary:logistic`)** targeting the 3 PEAD gates (`pead_pass`), trained
>   on **24 Sunday-safe features** (17 base + 8 FMP analyst revision momentum,
>   `is_bmo` removed) — see `03_model/02_phase_g_sunday_classifier.py`.
>   The `XGBRanker` (`rank:ndcg`) in §17rank:ndcg`) in §17 + `01_train_model.py main()` is the
>   OBSOLETE Phase F design; it has known feature-leak contamination from
>   `opening_gap_t1` and is NOT usable for live trading. See `04_backtest/`
>   README + `strategy_v2_synthesis.md` for the working alpha + statistics.
> - **Portfolio slots (§4, §7).** The deployable Phase G rule uses **4 slots**
>   (`n_slots=4`), NOT 5. The 5-slot description in §4/§7 is the original
>   aspiration; 4 slots is the tested design.
> - **Position sizing (§5, §8).** The Phase G baseline uses **equal-weight
>   1/4 NAV per slot** — NO Kelly, NO volatility scaling. Kelly sizing (§5)
>   is an UNTESTED FUTURE enhancement (see `04_backtest/strategy_v2_synthesis.md`
>   Next Priorities §4).
> - **Tiered hurdles (§3, §8).** The Sunday "Perfect Beat Baseline"
>   simulation trick + Tier-1/2/3 hurdle design has been SIMPLIFIED in Phase G:
>   the Sunday classifier outputs `P(PEAD)` from 24 Sunday-safe features. The
>   `opening_gap_t1 ∈ [-15%, -2%]` gap filter (Phase G v1) was **DELETED in v2**
>   (blocked 99.5% of PEAD events). Entry is now **pre-gap** (`Close[T-1]` BMO /
>   `Close[T]` AMC), hold **5 days** (was 10), with **-10% delayed stop**. **XLF
>   (Financials) excluded at inference only.** No tier-1/2/3 hurdle logic.
>
> - **Primary key (§11).**> - **Primary key (§11).** Earnings rows are keyed by
>   `(permaTicker, report_date)` (NOT `(canonical_ticker, report_date)`).
>
> For the live strategy spec see `04_backtest/strategy_v2_synthesis.md`. For
> the live DB schema see `database_layout.md`.

## 1. Target Variable Definition (The ML Objective)
*   **Core Goal:** Predict institutional drift (Idiosyncratic Alpha / Market-Neutral Abnormal Returns) following earnings announcements — classic PEAD (Post-Earnings-Announcement Drift).
*   **Target Metric:** Cumulative Abnormal Return ($CAR$) measured from $T+1$ to $T+11$ (10-day holding horizon), where $T$ is the earnings announcement date. The realized `car_10d` (log) is stored in `/features/train_matrix` per `database_layout.md`.
*   **Mathematical Formula:**
    $$CAR_{i} = \sum_{t=T+1}^{T+11} \left( R_{i,t} - R_{m,t} \right)$$
    *   $R_{i,t}$: Daily log return of stock $i$ on day $t$.
    *   $R_{m,t}$: Daily log return of the broad-market proxy on day $t$. For S&P 400 mid-caps we use **IJH** (iShares Core S&P Mid-Cap ETF) as the flat benchmark for all stocks, per §14.
*   **Model Type (UPDATED FROM PHASE F).** The DEPLOYABLE Phase G model is **`XGBClassifier` (`binary:logistic`)** targeting the 3 PEAD gates (`pead_pass`) — see **§17.A** for the full architecture. The Phase F-era `XGBRanker (Listwise Learning-to-Rank)` design described in §17 below is the OBSOLETE leaky model (edge came entirely from `opening_gap_t1`, a forward-looking feature using Open[T+1]). The Phase G classifier targets the same `car_10d` drift indirectly via the `pead_pass` gates (`car_60d_pass1`-based) instead of as a continuous CAR regression target, and uses a T+1 gap confirmation stage (see §17.A.5) to consume the genuinely realized `opening_gap_t1`. Do NOT deploy the §17 ranker.

---

## 2. Sunday Pipeline: Preliminary Screening & Feature Calibration

> **STATUS NOTE (2026-07-22).** The "Perfect Beat Baseline" simulation trick
> + "Model Signal-to-Noise Ratio" + Tier-1/2/3 watchlist construction described
> below is the ORIGINAL Phase F design intent. **Phase G replaces it with a
> direct XGBClassifier** trained on 24 Sunday-safe features (see §17.A) to
> produce `P(PEAD)` directly from real pre-event features. No Sunday-simulated
> "Perfect Beat" features, no signal-to-noise sorting, no tier splitting. The
> Sunday classifier is the same model that runs at inference time (the live
> weekday morning has no separate confirmation stage — the Sunday classifier
> IS the inference model (pre-gap entry eliminates the gap-filter step). See `03_model/02_phase_g_sunday_classifier.py`.

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

> **STATUS NOTE (2026-07-22, updated 2026-07-30).** The tiered hurdle structure below
> (Tier-1/2/3 with 1.5%/2.5%/4.0% CAR thresholds) is the ORIGINAL
> Phase F aspiration design. **Phase G v2 replaces it with a single
> rule** (see `04_backtest/strategy_v2_synthesis.md`):
> 1. Sunday classifier returns `P(PEAD)` for each event.
> 2. **Accept** iff `P(PEAD) >= 0.20` AND sector != XLF (Financials
>    excluded at inference only — model still trains on XLF).
> 3. **Pre-gap entry:** `Close[T-1]` for BMO, `Close[T]` for AMC.
> 4. **Exit** at `Close[T+5]` or -10% delayed stop (whichever first).
>
> The `opening_gap_t1 ∈ [-15%, -2%]` gap filter (Phase G v1) was DELETED —
> it blocked 99.5% of PEAD events. Pre-gap entry captures the gap instead
> of filtering on it.
>
> Tier-1/2/3-style threshold laddering and "Perfect Beat Baseline"
> simulation are NOT used in Phase G.

*   **Data Updates:** Upon weekday earnings releases*   **Data Updates:** Upon weekday earnings releases, overwrite the Sunday simulated features with true live broker API readings (`SUE_current`, `SUV_day_1`, and `VWAP_Coherence`).
*   **Dynamic Execution Gates:** Run final inference using the live feature vector. The asset must clear a dynamic hurdle scaled according to its Sunday structural tier to trigger a buy order:
    *   **Tier 1 (Rank 1-3):** Trigger trade if Final Predicted CAR $> 1.5\%$
    *   **Tier 2 (Rank 4-7):** Trigger trade if Final Predicted CAR $> 2.5\%$
    *   **Tier 3 (Rank 8-10):** Trigger trade if Final Predicted CAR $> 4.0\%$ (Forces high-variance entries to show massive expected edge).
*   **Temporal Execution Protocol:** Do not execute assets early based on Sunday metrics. The asset remains dormant on the watchlist until its real-world calendar earnings announcement date triggers the live weekday script.

---

## 4. Portfolio Architecture: Fixed Independent Slot-Pod Structure
*   **Capacity Mapping (UPDATED).** The portfolio uses a strict **4-slot** parallel configuration layout. (Original aspiration was 5 slots; tested design uses 4 — see `04_phase_g_portfolio.py`'s `n_slots=4` default + `04_backtest/strategy_v2_synthesis.md` §3.)
*   **Slot Capitalization (UPDATED).** Each virtual slot baseline is allocated exactly **25% of NAV** (1/4 = 25%). Position sizing is **equal-weight** — NO Kelly, NO volatility scaling (this is a Phase G baseline simplification; Kelly sizing in §5 is an untested future enhancement).
*   **Saturation Guardrail.** If all 4 slots are holding active 10-day positions, the weekday engine enters Standby Mode. Any fresh market triggers generated during saturation are discarded to avoid path-dependent liquidity dilution.
*   **No Slot Merging:** Under no circumstances should leftover capital from multiple under-allocated slots be merged into a temporary 5th slot. Maximum absolute capital exposure is hard-capped at 100% NAV.

---

## 5. Capital Allocation: Continuous Fractional Kelly Sizing (FUTURE / untested)

> **STATUS NOTE (2026-07-22).** The Phase G deployable baseline uses **equal-weight 1/4 NAV**
> per slot — NO Kelly, NO volatility scaling. The Kelly sizing engine described
> below is the ORIGINAL aspiration design (Phase F) and is currently
> UNIMPLEMENTED in the working backtest pipeline. It is listed as a
> medium-priority future enhancement (`04_backtest/strategy_v2_synthesis.md`
> §6 Next Priorities item 4 — "confidence-calibrated sizing").

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
* **Fixed Architecture Limits (UPDATED):** The portfolio configuration is hard-constrained to exactly **4** virtual parallel slots (each representing a **25% Allocation Capacity** of current total NAV). (Original Phase F aspiration was 5 slots × 20%; Phase G tested design uses 4 slots × 25%.)
* **Sunday Watchlist Ceiling:** The Sunday screening program must rank all upcoming weekly earnings events by their **real-feature `P(PEAD)` output** from the XGBClassifier (24 Sunday-safe features, no `opening_gap_t1` — see §17.A). The weekly watchlist handed down to the weekday execution script is capped at the top tier-1 candidates whose `P(PEAD) >= theta` (theta=0.20 is the tested operating point).
* **Capital Saturation Protocol:** If all 4 slots are occupied by active 5-day post-earnings drift positions, the weekday execution engine must enter Standby Mode and cease all real-time data ingestion and order generation until an active slot is officially vacated via a target horizon close.

---

## 8. Weekday Execution Tiering & Hurdle Logic

> **STATUS NOTE (2026-07-22, updated 2026-07-30).** OBSOLETE per Phase G — see §3 note. Phase G
> replaces tiered hurdles with a single rule: `P(PEAD) >= 0.20`, exclude XLF,
> pre-gap entry, 5-day hold, -10% delayed stop. No gap filter.

* **Dynamic Activation Gates:*** **Dynamic Activation Gates:** The Weekday execution script must read the Sunday ranking tier to dynamically adjust the final predicted validation barrier required to execute a market entry:
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

## 9b. Company-Level Merge & Identity Layer (Phase A rewrite — permaTicker primary)

The feature matrix has **one row per earnings event** (not per ticker symbol; each company contributes many rows over its S&P 400 membership window). Companies change ticker symbols over time (rebrands, mergers, bankruptcy-Q suffix delistings) while staying continuously in the index. Treating each ticker symbol as a separate company fragments price history, drops pre-rebrand earnings events, and creates phantom "still in the index" rows.

* **Primary Anchor: Tiingo `permaTicker`.** Tiingo's `permaTicker` is the stable company-track identifier that survives renames, rebrands, bankruptcy-Q suffix delistings, mergers, spinoffs, and same-CIK reorgs. Each permaTicker is a single, identity-stable legal-entity track.
    * ~~SEC CIK anchor~~ DEPRECATED as the primary anchor: CIK incorrectly merges same-CIK reorgs (e.g. **CHK → EXE**) and same-CIK spinoffs (e.g. **CFX → ENOV + ESAB**), and Tiingo's permaTickers correctly split those cases. Live probe evidence: `01_data/tiingo_permaTicker_audit.md`.
    * Spinoffs create new permaTickers and are treated as new company tracks (intentional scope cut).
    * Tickers with no recoverable permaTicker are purged from the working set during Phase A (~~`__nocik_*` fallbacks in the v1 transitional era~~ phased out — see `phase_a_b_migration_report.md`).

* **Pipeline Step (`01_data/02b_build_company_map.py`):** Runs after `02_SEC_sector_gathering.py` and before `03_data_gathering.py`. Reads `/metadata/sp400` (per-ticker Wikipedia intervals), then for EACH Wikipedia ticker invokes **Tiingo's `/search` endpoint** (with `includeDelisted=true`, `exactTickerMatch=true`) to discover every permaTicker that has ever answered to that ticker. Disambiguation matches each Wikipedia interval to the correct permaTicker via a Tiingo `/prices` physical-row-count probe (`physical_row_count` primary tiebreaker; `isActive=False` secondary tiebreak for past-closed Wikipedia intervals).

* **Outputs in `db.h5`:**
    * `/metadata/sp400` retains per-ticker Wikipedia intervals + audit columns `cik_at_added` and the (legacy) `perm_id` field.
    * **New PRIMARY identity table** `/metadata/sp400_permatickers` (962 rows × 10 cols): one row per permaTicker, with columns `permaTicker, canonical_ticker, name, isActive, openfigi, cik, sic, index_ref, wikipedia_intervals (JSON), price_unavailable (bool)`.
    * ~~`/metadata/sp400_companies`~~ PURGED. ~~`/metadata/sp400_perm_ids`~~ PURGED.

* **Canonical Selection Priority:** (1) chosen Tiingo search hit per Wikipedia interval + verified via physical-row-count probe; (2) `isActive=False` preference for past-closed Wikipedia intervals; (3) `price_unavailable=True` flag when no probe returns data.

* **Interval Merge:** `02b_build_company_map.py` imports each permaTicker's `wikipedia_intervals` directly from `/metadata/sp400`'s per-ticker interval arrays (interval merging happened upstream in `01_metadata_gathering.py`, where aliases-during-overlap-ranges are deduplicated on per-ticker boundaries under the point-in-time `cik_at_added` rule).

* **Data Fetcher (`03_data_gathering.py`):** Iterates per permaTicker (1 fetch per permaTicker, not per ticker). For each `permaTicker`, requests **Tiingo `/tiingo/daily/{permaTicker}/prices`** — a single endpoint that back-merges the rebrand history server-side under the permaTicker ID. Stores under `/sp400/{permaTicker}`. Companies with `price_unavailable=True` are skipped entirely with a log line. The Phase B EODHD alias-concatenation (the contamination source — see `phase_b_contamination_audit.md`) is **eliminated entirely**.

* **Adj-OHLCV:** **NO local derivation** (the Phase F-era local `close/adjusted_close` ratio derivation is removed). Tiingo returns adjusted columns natively (`adjOpen, adjHigh, adjLow, adjClose, adjVolume`), which are split + dividend back-adjusted. They are stored in db.h5 snake-cased as `Adj_Open, Adj_High, Adj_Low, Adj_Close, Adj_Volume`. Tiingo's response also includes `divCash` and `splitFactor` but **these are dropped** at write-time (not used downstream). See `database_layout.md` for the full 11-col stored schema.

* **Earnings Alignment:** The feature builder reads `/metadata/sp400_permatickers`, then for each permaTicker iterates the per-permaTicker rows in `/earnings/raw` (keyed by `(permaTicker, report_date)`). The permaTicker's `/sp400/{permaTicker}` price series (which Tiingo back-merges across any rebrands under the same permaTicker) is used for all feature calculations, including pre-rebrand earnings events. `wikipedia_intervals` gates each event with the 90-day exclusion buffer per the §9 rules.

* **Result:** One row in `/metadata/sp400_permatickers` $\rightarrow$ many earnings events $\rightarrow$ many feature rows in the final matrix.

---

## 10. Training Matrix Extraction Protocol (REVISED)
* **The Row Selection Filter:** When constructing the training matrix (`X`, `y`) for the XGBoost model, the dataset generator must iterate through the historical timelines and only select rows where `in_index_clean == True`.
* **Dynamic Historical Matching & Interval Validation:**
    * The feature engine must evaluate the timestamp of each historical earnings event against the *entire* interval array of the corresponding asset.
    * If a company was in the index from 2014 to 2018, and then re-added in 2022, rows from 2014+90 days up to the 2018 removal date **MUST be included** in training. Rows between 2018 and 2022+90 days must be dropped. 
    * If a company was deleted via bankruptcy/merger in 2018 (resolved via the EODHD price-flatline fallback), all rows prior to its 2018 delisting (and after its original inclusion date + 90 days) **MUST be preserved** to teach the model downside tail-risk.

---

## 11. Earnings Surprise Ingestion Protocol
* **Data Sourcing:** The data pipeline queries the **EODHD Earnings API** (`/api/calendar/earnings`) to pull historical quarterly timelines of `Actual_EPS`, `Estimated_EPS`, and `Report_Date`. Earnings events are stored in `/earnings/raw` with `difference = actual - estimate` and `percent` (surprise percentage) precomputed by EODHD. EODHD remains the source for the earnings calendar — Tiingo has no equivalent endpoint.
* **HDF5 Alignment (UPDATED).** Store the parsed earnings metrics in a single consolidated `/earnings/raw` table, keyed by **`(permaTicker, report_date)`** (NOT `(canonical_ticker, report_date)` — permaTicker is the primary anchor per §9b). The `cik` and `canonical_ticker` columns are retained as informational/audit only. Today, `/earnings/raw` is 43,682 rows × 12 cols, 862 distinct permaTickers, **0 dup groups** (the 2024-07-21 cleanup applied dedup by `(permaTicker, report_date)` keeping the smallest-cik row per dup group).
* **Feature Calculation Rule:** Do not pass raw surprise dollar values to the model. The pipeline must compute the Standardized Unanticipated Earnings (`sue_score`) by dividing the surprise deviation by the rolling 12-quarter standard deviation of the asset's historical surprises (per §15).


## 12. Feature Lookback vs. Training Horizon Slicing
* **Global History Constraint:** The data engine utilizes a strict 15-year historical data runway.
* **Feature Context Window:** All rolling historical features (`SUE_trend`, momentum vectors, macro baselines) are capped at a maximum lookback window of 3 years (12 quarters) — this matches the `sue_score` 12Q rolling-std baseline, which is the longest lookback in the active feature set. No other feature requires a longer window.
* **Active Training Horizon:** The XGBoost training engine must discard the first 3 years of the global history timeline, using them exclusively to populate the initial feature contexts. The active target training matrix ($y$) must be strictly drawn from the remaining 12-year deep history window (2015-01-01 onward, given the 2012-01-01 backfill boundary).
* **Index Membership vs. Feature Lookback Separation:** The 3-year feature lookback requirement applies strictly to the presence of raw historical pricing and fundamental lines inside the HDF5 data storage node. It is NOT a requirement for historical index membership. If an asset has 3 years of trading history available, its rows are fully eligible for training immediately after the 90-day index stabilization buffer passes, regardless of its pre-inclusion index classification.


## 13. Calendar Scheduling & Live Watchlist Ingestion
* **Schedule Provider:** The Sunday scheduling module queries the **FMP Earnings Calendar API** (`/stable/earning-calendar`), passing a 5-day forward-looking date array (`from` to `to`). EODHD is CANCELLED — FMP replaced it for earnings data.
* **Cross-Sectional Filtering:** The fetched raw global calendar must immediately be cross-referenced against the local HDF5 `in_index_clean` array for that specific calendar date. Any reporting ticker not actively clearing the S&P 400 point-in-time membership filter must be instantly pruned from memory.
* **Pre-Earnings Ingestion Matrix:** For the remaining valid weekly candidates, parse the `epsEstimated` and `date` strings. Save this structure to a temporary runtime matrix (`weekly_schedule_queue`) to dictate the active weekday activation order and seed the placeholder values for Sunday's point-estimation simulator.

## 14. Data Sourcing, Index Proxies, and Market-Adjusted CAR
* **Tiingo-Primary Ingestion Architecture (UPDATED).** The data pipeline sources daily split-adjusted equity prices and identity data from **Tiingo** (`/tiingo/daily/{permaTicker}/prices` and `/search` endpoints) — Tiingo is irreplaceable for historical prices (FMP has no historical OHLCV). **Earnings data** comes from **FMP** (`/stable/earning-calendar` and `/stable/historical-rating` for analyst grades). **EODHD is CANCELLED** — FMP $49/mo replaced it (strictly superior: BMO/AMC coverage, revenue estimates, 14-year analyst revision history).
    * ~~EODHD-Centric ingestion~~ DEPRECATED per Phase A+B migration (see `phase_a_b_migration_report.md`). EODHD alias-concatenation has been eliminated because Tiingo back-merges rebrand history server-side under the permaTicker.
* **Broad Market Benchmark Proxy (`IJH`):** To calculate the broad market return baseline ($R_{m,t}$) without dealing with spot index availability bugs, the pipeline must query the historical price series for the **iShares Core S&P Mid-Cap ETF (`IJH`)** (stored at `/macros/IJH`). IJH is also used as the market-neutralization baseline for `car_drift_historical_q1` and the 24 Sunday-safe features.
* **Adjusted Return Constraint:** The feature engine must strictly use the `Adj_Close` column (now Tiingo native `adjClose`, stored as `Adj_Close` per §9b) for both individual assets and the `IJH` proxy to eliminate artificial price drops caused by dividend distributions or fund splits.
* **Streamlined Market-Adjusted Calculation:** Replace any complex sector-matching lookups with a flat Market-Adjusted Model against the S&P 400 proxy. The daily abnormal return for an asset is calculated as:
    $$AR_{i,t} = \ln\left(\frac{\text{Asset AdjClose}_{t}}{\text{Asset AdjClose}_{t-1}}\right) - \ln\left(\frac{\text{IJH AdjClose}_{t}}{\text{IJH AdjClose}_{t-1}}\right)$$
  The target variable $CAR_i$ is the simple sum of $AR_{i,t}$ across the $T+1$ to $T+11$ holding horizon.

## 15. Operational Pipeline Interfacing
* **Sunday Schedule Parsing:** Every Sunday, the engine queries the FMP `/stable/earning-calendar` endpoint for the upcoming week. The resulting array is filtered against the local HDF5 `in_index_clean` array to build the localized `weekly_schedule_queue`.
* **The SUE Normalization Feature:** When loading data from the FMP earnings table (`/earnings/fmp`), the data script must not pass raw surprise amounts to the model. It must compute the Standardized Unanticipated Earnings (`sue_score`) by dividing the surprise deviation by the **rolling 12-quarter standard deviation** of the asset's historical earnings surprises (per company CIK, across all prior quarters, `min_periods=12`):
    $$SUE = \frac{\text{Actual EPS} - \text{Estimated EPS}}{\sigma_{12Q}(\text{Historical Surprise})}$$
    Where `difference = actual - estimate` is the per-quarter EODHD-provided surprise. The 12Q rolling std is computed **per permaTicker** (not per-CIK — the permaTicker is the primary anchor; CIK is informational only, see §9b) across all prior quarters, `min_periods=12`. Events where `estimate` is NaN have EODHD-set `difference = 0.0` and are retained in the rolling std window (Option B, per `features.md`).
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

## 17. Model Architecture — DEPRECATED: was Cross-Sectional Listwise Ranking (XGBRanker)

> **STATUS NOTE (2026-07-22).** The §17 XGBRanker (`rank:ndcg`) + isotonic
> calibration + Kelly sizing architecture below is the **OBSOLETE Phase F**
> design. Confirm the divergence in `03_model/01_train_model.py main()`: the
> script STILL trains this ranker + isotonic calibrator, but the **entire edge
> was feature leakage from `opening_gap_t1`** (forward-looking, uses Open[T+1]).
> Leak test (Phase G): NaN-ing `opening_gap_t1` drops hit rate 65.8% → 49.1%
> and Sharpe 4.31 → -0.14. Do NOT deploy this model.
>
> The DEPLOYABLE replacement is **§17.A Phase G Sunday-safe classifier**
> below: `XGBClassifier` (`binary:logistic`) on 24 Sunday-safe features,
> targeting the 3 PEAD gates (`pead_pass`) instead of the continuous CAR.
>
> See `03_model/02_phase_g_sunday_classifier.py` for the working trainer;
> `04_backtest/strategy_v2_synthesis.md` for the working strategy spec,
> OOS trade-level statistics, and the Kelly-free equal-weight design.

> **Impacted Pipeline Steps:** `02b_build_company_map.py`, `03_data_gathering.py`, `07_train_model.py`, `08_backtest_execution.py`  
> **Status:** APPROVED THEN SUPERSEDED. Originally approved 2026-07-08 to
> replace the prior pointwise XGBoost Regressor architecture (which
> suffered from ~6% base-rate class imbalance and could not neutralize
> cross-sectional macro noise). SUPERSEDED by the Phase G classifier in §17.A
> below due to feature-leak contamination.

### 17.1 Motivation & Structural Pivot (Phase F historical)
To completely eliminate severe minority class imbalance (~6% base rate) and mathematically neutralize macro-market directional noise, the machine learning pipeline uses a **Listwise Learning-to-Rank (LTR)** architecture (not pointwise regression). 

Instead of treating each earnings event as an isolated data point, the model evaluates all earnings announcements within a given calendar week as a single cross-sectional group, directly optimizing for relative outperformance.

### 17.2 Label and Feature Matrix Restructuring (Phase F historical)
*   **The Target Label (`car_10d`):** Stored as the continuous **log CAR** — the sum of daily log excess returns `Σ_{t=T+1}^{T+11} (log R_stock − log R_IJH)` — across the 10-day post-event window. **Do not** convert this to discrete ordinal ranks (1, 2, 3) in the dataset. Maintaining continuous labels is required for the Listwise Gain function. Storing in log units is safe for the ranker because NDCG is invariant to monotonic transforms of the gain values; only the *ranking* within each group matters.
*   **Group Anchor (`calendar_week_group`):** A new structural column must be added to the feature matrix representing the calendar week of the earnings date (e.g., `2026_W27`).
*   **Sorting Requirement:** Before feeding data into the model training loop, the training and validation dataframes **must** be sorted chronologically and clustered explicitly by `calendar_week_group`.

### 17.3 Model Architecture Changes (Phase F historical — `03_model/01_train_model.py`)
*   **Model Class:** Replace `XGBRegressor` or `LGBMRegressor` with **`XGBRanker`** (or LightGBM equivalent).
*   **Objective Function:** Hardcode `objective="rank:ndcg"`.
*   **Evaluation Metric:** Set `eval_metric="ndcg@3"` (or match your portfolio's target weekly slot capacity) to focus optimization entirely on the top of the funnel.
*   **Group Dimension Array:** Extract group sizes using `.groupby('calendar_week_group').size().values` and pass this array explicitly into the `.fit(X, y, group=groups)` method.

### 17.4 The Sizing Bridge: Isotonic Score Calibration (Phase F historical)
Because `XGBRanker` outputs arbitrary, unitless ordinal marginal utility scores, these scores cannot be passed directly into the Continuous-Time Kelly (Merton's Fraction) sizing engine. A non-parametric calibration layer is introduced to map scores back to absolute expected return ($\mu$).

*   **Implementation:** Fit a `sklearn.isotonic.IsotonicRegression(out_of_bounds='clip')` on the **validation set** immediately after training.
    *   **Input ($X$):** Raw validation output scores from `ranker.predict(X_val)`.
    *   **Target ($y$):** Actual historical `car_10d` converted from **log units to arithmetic** via `np.expm1(y_log)` before fitting. This conversion is mandatory: Stage 2 (`02_build_feature_matrix.py`) stores `car_10d` in log units (see §17.1), but the isotonic calibrator's output must be in true percentage units so it can be consumed directly by the Kelly sizing engine (§17.5).
*   **Monotonic Guarantee:** Isotonic regression is strictly monotonic; it scales the values into realistic return percentages without altering the model's chosen sorting sequence.

### 17.5 Live Inference & Sizing Execution Engine (Phase F historical)
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

---

## 17.A. DEPLOYABLE: Phase G v2 Binary Sunday-Safe Classifier

> **AUTHORITATIVE deployable model spec.** This is what
> `03_model/04_freeze_binary_model.py` freezes, what `04_backtest/`
> evaluates, and what `04_backtest/strategy_v2_synthesis.md` synthesizes.
> §17 above is preserved only as historical context for future
> ranking-model enhancements.

### 17.A.1 Motivation
The Phase F `XGBRanker` edge was contaminated by `opening_gap_t1` — a forward-looking feature using `Open[T+1]` (NOT available at Sunday planning time). NaN-ing it collapses the strategy Sharpe 4.31 → -0.14. The Phase G rewrite trains a **Sunday-safe** binary classifier on features available strictly before the earnings announcement. The POS/NEG gap filter (`opening_gap_t1 ∈ [-15%, -2%]`) was tested in Phase G v1 but **DELETED in v2** because it blocked 99.5% of PEAD events — see §17.A.6.

### 17.A.2 The 24 Sunday-Safe Features (X for the classifier)
`sue_score, eps_surprise_pct, consecutive_surprises, sue_acceleration, sue_lag_1, sue_lag_2, car_drift_historical_q1, pre_event_idiosyncratic_vol, pre_event_volume_trend (log-transformed), rel_ret_3d, rel_ret_5d, rel_ret_10d, rel_ret_20d, rel_ret_30d, sector_adjusted_ret_20d, sue_abs_x_inverse_vol, revision_momentum_30d, revision_momentum_60d, revision_momentum_90d, revision_ordinal_momentum_90d, revision_intensity_90d, grade_dispersion_90d, n_analysts_covering, last_action_days_before_earnings`

**Excluded from model** (tested and rejected):
- `opening_gap_t1` — forward-looking (uses Open[T+1]), caused Phase F leakage
- `is_bmo` — operational scheduling choice, not a PEAD predictor; caused OOS overfitting
- 12 macro features (FRED levels + ROC) — A/B test shows they HURT (total PnL drops 24%). PEAD is stock-specific, not regime-dependent.

The 8 `revision_*` and `grade_*` features come from FMP analyst grades (`/stable/historical-rating`), spanning 14 years of daily analyst actions from 111 firms.

### 17.A.3 The 3 PEAD Gates (`pead_pass` label y)
The binary classification target is `pead_pass`, which requires ALL THREE gates to pass:
1. **Gate 1 (CAR):** `car_10d >= 3%` — abnormal return vs IJH over T+1→T+10 exceeds 3%
2. **Gate 2 (volume):** `inst_vol_ratio >= 2.0` — institutional volume on event day >= 2× the 20-day average
3. **Gate 3 (MaxDD):** `maxdd_ma >= -1.5%` — the abnormal return never drops below -1.5% during the hold window

Base rate ~10.7% of earnings events pass all 3 gates. See `_pead_target_retrain.py:compute_pead_gates_full` for the full gate logic.

### 17.A.4 Model Trainer (`03_model/02_phase_g_sunday_classifier.py`)
*   **Model Class:** `xgboost.XGBClassifier`.
*   **Objective:** `binary:logistic`. Outputs `P(PEAD)` per event.
*   **NaN policy:** Do NOT drop rows. XGBoost handles NaN natively.
*   **Fixed HP (no per-fold tuning):** `gamma=3, min_child_weight=50, max_depth=3, n_estimators=300, learning_rate=0.05, reg_lambda=1.0, subsample=0.7, colsample_bytree=0.7, eval_metric=["logloss", "auc"], random_state=42, n_jobs=-1`.
*   **Split convention** (Nested CV — anchored walk-forward, 4 folds):
    - Fold 1: TRAIN ≤2023-12-31 | SWEEP 2024 H1 | TEST 2024 H2
    - Fold 2: TRAIN ≤2024-06-30 | SWEEP 2024 H2 | TEST 2025 H1
    - Fold 3: TRAIN ≤2024-12-31 | SWEEP 2025 H1 | TEST 2025 H2
    - Fold 4: TRAIN ≤2025-06-30 | SWEEP 2025 H2 | TEST 2026 H1
*   **Final classifier fit** (deployable): train on **TRAIN + SWEEP** concatenated.
*   **Frozen artifact:** `03_model/models/phase_g_v2_binary/{classifier.json, meta.json}`. VAL AUC=0.6396.

### 17.A.5 Entry/Exit Rules (the tested operating point)
1.  **Sunday planning** — compute the 24 Sunday-safe features for each upcoming-week earnings event, run `clf.predict_proba(X)` → `P(PEAD)`.
2.  **Accept rule:** `P(PEAD) >= 0.20` (theta=0.20, the sweet spot from `17_theta_sweep.py`).
3.  **Sector exclusion:** Exclude **XLF (Financials)** at inference only (model still trains on XLF). Financials have 13% PEAD precision vs 41% for the rest — structural, not overfit (`41_exclude_xlf_test.py`). This is the only precision lever that improves BOTH precision AND total return.
4.  **Pre-gap entry:** Enter at `Close[T-1]` for BMO announcements (day before), `Close[T]` for AMC announcements (same-day close). PEAD drift is front-loaded into the overnight gap; pre-gap captures it. Entering post-gap (Open[T+1]) gets eaten by the gap (`22_bmo_amc_pregap.py`).
5.  **Position sizing** — equal-weight `1/4 NAV` per slot. NO Kelly, NO volatility scaling.
6.  **Weekly batch selection** — sort each week's accepted picks by `P(PEAD)` descending, take top N = free slots. This fixes a global-sort bias (user-identified).
7.  **Exit** — `Close[T+5]` (5-day hold from report date). Frees slots weekly, cuts losses short. 5-day dominates 10-day on every metric (`30_hold_comparison_bootstrap.py`).
8.  **Stop-loss** — `-10% delayed` (skip gap day, check days 1+). Statistically neutral but caps worst case from -37% to -34%. All 13 stopped trades were losers (`37_wider_stop_test.py`).

### 17.A.6 Why the Gap Filter Was Deleted
Phase G v1 used `opening_gap_t1 ∈ [-15%, -2%]` as a T+1 confirmation filter. Phase G v2 **DELETED** it because:
- It blocked **99.5% of PEAD events** — the filter was so restrictive it eliminated nearly all true positives.
- Pre-gap entry (§17.A.5.4) captures the gap instead of filtering on it.
- The "NEG_only gap regime" theory from Doc E was regime-dependent noise, not market structure (Doc J confirmed).

### 17.A.7 Macros Excluded (empirical A/B test)
Adding 12 macro features (6 FRED levels: DGS10, DGS2, T10YIE, VIXCLS, FEDFUNDS, BAA; plus 6 month-over-month ROC of each) was tested via `35_macro_ab_test.py`:
- Total PnL (raw sum) drops from +636% → +483% (-24%). NAV-compounded: +338% → +241%.
- AUC delta per fold: 0 to -0.005 (no improvement).
- Conclusion: **PEAD is stock-specific, not regime-dependent.** Macros cannot transfer across time-ordered CV folds. Excluded from the deployable model.

### 17.A.8 Alternative Models Tested and Rejected
- **3-class softprob classifier** (`26_three_class_classifier.py`): Degenerate argmax (predicts class 0 for 100% of VAL events). Binary theta=0.20 beats 3-class P(any)>=0.20 on total return (+636% vs +607%) and win rate (69.7% vs 64.6%). See `34_binary_vs_3class_deep.py`.
- **2-stage model** (binary + CAR regression, `33_two_stage_model.py`): Stage 2 CAR regression has correlation ~0 with returns. CAR magnitude is unpredictable from Sunday-safe features. The features that predict PEAD occurrence cannot predict PEAD magnitude.
- **eps_surprise_pct secondary filter** (`39_eps_filter_test.py`): Improves precision (+4.7pp at eps>=20%) but drops total PnL (-13%). Non-PEAD picks are profitable (+2.11%), filtering them loses alpha.

### 17.A.9 OOS Performance (4-fold nested CV, exclude XLF, pre-gap, -10% delayed stop)
From `42_xlf_excluded_detailed_stats.py`:
- **101 OOS trades across 4 folds** (~51 trades/year)
- **Win rate: 75.2%** (76W / 25L)
- **PEAD precision: 38.6%** (39 true PEAD / 62 false positives)
- **Expectancy per trade: +6.72%** (median +5.99%, std 13.03%)
- **Avg win: +11.66%**, **avg loss: -8.28%**, payoff 1.41, profit factor 4.28
- **Total PnL (raw sum):** +679.1% (with stop), +672.4% (no stop)
- **Total PnL (NAV-compounded):** **+391.3%** (4.91x NAV, 4 slots at 1/4 NAV, weekly compounding). Raw sum treats each trade as 100% NAV; NAV-compounded is the realistic portfolio return. See `44_slot_sweep_nav_sizing.py`.
- **Best/worst trade: +51.82% / -30.54%**
- **Max drawdown (NAV-compounded): -7.1%** (raw-sum cumulative: -32.0%)
- **Annualized Sharpe (approx): 3.71**
- **No losing fold** (per-fold totals: +154%, +134%, +280%, +111%)
- Bootstrap CI: expectancy [+3.46%, +8.87%], total [+342%, +878%]. All CIs exclude 0.
- **Large PEAD (CAR>=10%):** 22 trades, 86% win, +18.23% avg, contribute +401% raw sum (+59% of total raw PnL).

### 17.A.10 How to Run the Deployable Model
1.  Train + freeze: `conda run -n trading python 03_model/04_freeze_binary_model.py`
2.  Detailed stats: `conda run -n trading python 04_backtest/42_xlf_excluded_detailed_stats.py`
3.  Bootstrap CI: `conda run -n trading python 04_backtest/30_hold_comparison_bootstrap.py`
4.  **Live paper-trading:** `conda run -n trading python 05_live/01_live_fold_pull.py`

Artifacts: `03_model/models/phase_g_v2_binary/{classifier.json, meta.json}`.

### 17.A.11 Acknowledged Gaps in Evidence
- **No live paper-trading fold #5** — the first true forward-looking OOS data point. The live script (`05_live/01_live_fold_pull.py`) is ready; run around 2026-09-30 for 2 months of forward-looking data.
- **Precision at 38.6%** — every precision lever tested (theta increase, 3-class, eps filter, CAR regression) improves precision but hurts total PnL because non-PEAD picks are profitable (+2.11%). Better FEATURES (not filters) are needed. Options: FMP estimate revision trajectory, SEC Form 4 insider trading, FINRA short interest.
- **Transaction costs + slippage not modeled** — must be added before live capital deployment.
