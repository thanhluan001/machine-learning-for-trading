# Quantitative Architecture Specification: PEAD Algorithmic Trading Bot
## Target Universe: S&P 400 Mid-Cap (approximately 400 Assets)

> **AUTHORITATIVE CURRENT STATUS — 2026-08-07.** The paper-executable system is
> **phase_g_v6_gate_decomposition**. V4 remains the frozen comparison baseline;
> historical Phase F/v3/v4 sections below are retained for provenance. This
> current-status block and **§17.B.7** override older conflicting text.
>
> **Current paper model:** three `XGBClassifier` (`binary:logistic`) gate models
> using the 23-feature timing-correct information set:
>
> ```text
> phase_g_v6_gate_decomposition/{pass_g1,pass_g2,pass_g3}/classifier.json
> /features/train_matrix_v4_timing_correct
> ```
>
> Gate labels are `CAR > +3%`, event volume ratio `> 2x` baseline, and
> market-adjusted MaxDD `> -1.5%`. The executable score is
> `min(p_pass_g1, p_pass_g2, p_pass_g3)` and the threshold is `0.33` (raised
> from `0.30` on 2026-08-13 after bootstrap validation; see §17.B.7).
> V6 is a paper-trading candidate, not a live-capital production promotion.
>
> **V6 validation reference:** final frozen 2026 H1 holdout produced 47
> executed trades, 63.8% win rate, +3.88% average trade, and +53.4% compounded
> holdout NAV. The four-week block bootstrap for average trade was +0.19% to
> +7.93%; block-NAV uncertainty remained wide. Earlier nested OOS results were
> 158 trades, 63.3% win rate, +3.82% average trade, +314.7% NAV, but the fixed
> policy was selected from the broader research comparison and must not be
> treated as an untouched final estimate.
>
> **Point-in-time timing contract:** for an earnings event on trading day T,
> AMC features use the latest completed daily close through T-1 (entry at T
> close); BMO features use the latest completed daily close through T-2 (entry
> at T-1 close). Price/volume, analyst revisions, and macro observations use
> the same executable cutoff. Prior reported earnings history is historical
> information and remains valid.
>
> **Plan routing:** Script 01 writes the V6 executable plan to
> `05b_alpaca_live/plan.json`; Script 02 reads only that file and refuses to
> trade a plan whose model is not V6. Script 01 also writes the V4 comparison
> plan to `v4_plan.json` and records V4 hypothetical events in
> `v4_shadow_trades.json`. V4 comparison files are never executed.
>
> **Fresh actionable inference:** remove events whose entry date has passed.
> Tiingo paid daily OHLCV is refreshed through the latest completed close.
> Script 01 **also self-refreshes benchmark ETFs (IJH + needed sector ETFs) and
> FRED macros (VIX, fed funds, unemployment)** before feature computation — a
> stale benchmark silently corrupts every `rel_ret` / `car_drift` feature (the
> 2026-08-08 stale-IJH incident). FMP uses
> `/stable/earnings-calendar?from=today-1&to=today+weeks*7&includeReportTimes=true`;
> the per-ticker fallback is bounded by the same window. No partial intraday
> bars are used.
>
> **Universe maintenance:** `01_data/refresh_sp400_membership.py` re-parses
> Wikipedia monthly to close graduated stocks and flag new constituents;
> `02b_build_company_map.py --tickers <new> --merge` maps only the new
> permaTickers incrementally (no full re-disambiguation). This prevents the
> AMD-style failure mode of trading a stock that left the S&P 400.
>
> **Current execution:** Alpaca paper trading, four equal-weight slots,
> whole-share immediate-market (`DAY`) orders for both buys and sells —
> manual position management,
> no bracket orders. Script 02 reserves only V6 candidates whose entry date is
> today through today + 7 calendar days; farther candidates are reconsidered
> on later runs. Dry-run mode does not modify `positions.json`.
>
> **Current paper state:** WTS 2 shares active; DBX 9 shares active from a
> partial MOC fill; no pending orders at the last verified run. V4 shadow
> outcomes are recorded separately and do not consume slots.
>
> **Legacy status:** Sections §1, §9 (survivorship/inclusion rules), §10, §11,
> §12, §13, §14, and §16 remain useful where not contradicted by this block.
> Phase F ranker text and older Phase G/v4 model text are historical.

> **HISTORICAL STATUS NOTE (2026-07-22).** The following legacy block is
> retained for provenance. Where it conflicts with the 2026-08-06 authority
> block above or §17.B, the current block/§17.B wins.
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
>   on **23 honest features** (19 base + consecutive_surprises_pre + 3 macros;
>   5 SUE look-ahead features DROPPED 2026-07-31)
>   — see `03_model/05_freeze_honest_model.py`.
>   The `XGBRanker` (`rank:ndcg`) in §17rank:ndcg`) in §17rank:ndcg`) in §17 + `01_train_model.py main()` is the
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
*   **Model Type (UPDATED FROM PHASE F).** The current paper-executable V6
    architecture uses three **`XGBClassifier` (`binary:logistic`)** gate models
    — see **§17.B.7**. The V4 single classifier is retained as the comparison
    baseline in §17.C. The Phase F-era `XGBRanker (Listwise Learning-to-Rank)`
    design described in §17 below is the OBSOLETE leaky model (edge came
    entirely from `opening_gap_t1`, a forward-looking feature using Open[T+1]).
    Do NOT deploy the §17 ranker.

---

## 2. Sunday Pipeline: Preliminary Screening & Feature Calibration

> **STATUS NOTE (2026-07-22).** The "Perfect Beat Baseline" simulation trick
> + "Model Signal-to-Noise Ratio" + Tier-1/2/3 watchlist construction described
> below is the ORIGINAL Phase F design intent. **Phase G replaces it with a
> direct timing-correct classifiers** trained on 23 features (see §17.B.7)
> to produce V6 gate probabilities from real pre-event features. There is no Sunday
> simulation stage. Script 01 is the live inference engine and uses the
> freshest completed daily data for the actionable event set.

*   **Data Scope:** Process 600 universe assets to extract long-term historical features on Sunday: `SUE_trend_1Y`, `SUE_trend_3Y`, `Turnaround Ratio`, and rolling 60-day historical variance ($\sigma^2$).
*   **The Simulation Trick:** Because real-world catalyst features (`SUE_current`, `SUV_day_1`, `VWAP_Coherence`) are unavailable until weekday market open, the Sunday script evaluates candidates by substituting missing attributes with a "Perfect Beat Baseline" (`SUE_current = 2.0`, `SUV_day_1 = 3.0`).
*   **Scoring Metrics:** The Sunday rank utilizes a Model Signal-to-Noise Ratio to sort candidates:
    $$\text{Sunday Rank Score} = \frac{\text{Simulated Predicted CAR}}{\text{Prediction Standard Error}}$$
*   **Watchlist Constraints:** Filter out the top 10 highest-ranked assets to construct the weekly execution pipeline. Divide these assets into 3 execution tiers:
    *   **Tier 1 (Rank 1-3):** Peak structural receptivity names.
    *   **Tier 2 (Rank 4-7):** Moderate structural configuration profiles.
    *   **Tier 3 (Rank 8-10):** Low-receptivity/High-uncertainty configurations.

---

## 3. Weekday Engine: Actionable Daily Inference

> **STATUS NOTE (2026-07-22, updated 2026-07-30).** The tiered hurdle structure below
> (Tier-1/2/3 with 1.5%/2.5%/4.0% CAR thresholds) is the ORIGINAL
> Phase F aspiration design. **Phase G v2 replaces it with a single
> rule** (see `04_backtest/strategy_v2_synthesis.md`):
> 1. V6 gate classifiers return three gate probabilities for each event.
> 2. **Accept** iff `min(p_pass_g1, p_pass_g2, p_pass_g3) >= 0.33` and sector
>    != XLF (Financials
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
*   **Saturation Guardrail.** If all 4 slots are holding active 5-day positions
    or pending entry orders, the engine does not open another slot. P(PEAD)
    priority reserves free slots for the highest-ranked candidates.
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
*   **Compounding Feed:** Each slot targets approximately E/4, subject to
    whole-share MOC sizing and buying-power constraints. Leftover cash is
    allowed; fractional/notional MOC orders are not used.
*   **Execution Value:** Position sizing calculations use the latest slot baseline value. Profits from exited trades organically compound into larger subsequent positions, while active open unrealized gains are preserved in baseline calculations to avoid under-leverage decay.

---

## 7. Portfolio Slot Constraints & Sunday Ranking Logic
* **Fixed Architecture Limits (UPDATED):** The portfolio configuration is hard-constrained to exactly **4** virtual parallel slots (each representing a **25% Allocation Capacity** of current total NAV). (Original Phase F aspiration was 5 slots × 20%; Phase G tested design uses 4 slots × 25%.)
* **Actionable bench:** Script 01 ranks all actionable events in its bounded
  FMP lookahead window by the V6 min-gate score, writes the executable bench to
  `plan.json`, writes the V4 comparison bench to `v4_plan.json`, and records
  V4 hypothetical events in `v4_shadow_trades.json`. Script 02 applies
  four-slot V6-score-priority reservation only within a seven-calendar-day
  entry horizon.
* **Capital Saturation Protocol:** If all 4 slots are occupied by active 5-day post-earnings drift positions, the weekday execution engine must enter Standby Mode and cease order generation until an active slot is vacated. A future event more than seven calendar days away does not reserve a free slot.

---

## 8. Weekday Execution Tiering & Hurdle Logic

> **STATUS NOTE (2026-07-22, updated 2026-07-30).** OBSOLETE per Phase G — see §3 note. Phase G
> replaces tiered hurdles with the V6 rule: `min(p_pass_g1, p_pass_g2,
> p_pass_g3) >= 0.33`, exclude XLF, pre-gap entry, 5-day hold, -10% delayed
> stop. No gap filter.

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


## 13. Calendar Scheduling & Live Actionable-List Ingestion
* **Schedule Provider:** Script 01 queries FMP `/stable/earnings-calendar`
  with `from=today-1`, `to=today+weeks*7`, and
  `includeReportTimes=true`. Starting one day early prevents FMP from
  dropping events whose report date equals the query start. The per-ticker
  `/stable/earnings?symbol=...` fallback obeys the same window. EODHD is
  cancelled.
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

## 17.B. CURRENT PAPER-EXECUTABLE: Phase G v6 Gate-Decomposition Candidate

### 17.B.7 V6 paper-execution specification

V6 is the current paper-executable research candidate. It does not replace the
V4 baseline as a validated production artifact yet.

#### Artifacts and labels

```text
03_model/models/phase_g_v6_gate_decomposition/pass_g1/classifier.json
03_model/models/phase_g_v6_gate_decomposition/pass_g2/classifier.json
03_model/models/phase_g_v6_gate_decomposition/pass_g3/classifier.json
03_model/models/phase_g_v6_gate_decomposition/meta.json
/features/train_matrix_v4_timing_correct
```

All three models use the same 23 timing-correct features and independently
predict:

```text
pass_g1: CAR > +3%
pass_g2: event volume ratio > 2x baseline
pass_g3: market-adjusted MaxDD > -1.5%
```

The frozen executable score is:

```python
v6_score = min(p_pass_g1, p_pass_g2, p_pass_g3)
accept if v6_score >= 0.33
```

XLF remains excluded at inference. The historical V6 policy does not apply the
V4 LEG-like filter. Entry, exit, stop, sizing, and universe rules otherwise
remain the same as the V4 execution contract.

#### Plan and execution routing

```text
01_fetch_and_predict.py
    ├── plan.json              V6 default; consumed by Script 02
    ├── v4_plan.json           V4 comparison-only plan
    └── v4_shadow_trades.json  hypothetical V4 event/entry/exit ledger

02_paper_trade.py reads only plan.json and rejects a non-V6 model plan.
```

The V4 ledger stores hypothetical entry and planned exit information and is
updated on later inference runs. It does not create Alpaca orders, consume
slots, or alter `positions.json`.

#### Weekly slot-refresh entry policy (force-refresh, mh=4 guard)

Script 02 refreshes all four slots each ISO week with that week's top V6 picks.
For each pick DUE TODAY (in `p_pead` priority order):

```text
slate = this ISO week's threshold-passing picks (not held), top-4 by p_pead
for each due-today slate pick:
    if a slot is free            -> buy into it
    elif a force-sellable slot   -> force-sell it, buy the new pick
    else                         -> skip
```

A slot is **force-sellable** iff the occupying position was entered in a PRIOR
ISO week AND has been held >= `MIN_FORCE_HOLD` (=4) business days — i.e. it is
near its T+5 exit. The mh=4 guard means only near-T+5 positions are displaced,
so negligible drift is sacrificed.

Rationale: front-loaded PEAD (§17.B.8) means a position held since last week
has already banked most of its drift, while a fresh this-week pick has the
full front-loaded drift ahead of it. Refreshing slots weekly captures picks
that the prior conviction-priority policy skipped when all slots were full —
which is EV-positive.

This **replaces conviction-priority** (revised 2026-08-13), which skipped
due-today picks when all slots were full and held them for future
higher-conviction names. Validation (`04_backtest/63_force_refresh_backtest.py`,
`64_force_refresh_guard_bootstrap.py`) on the nested OOS folds + 2026 H1 holdout:

```text
                       conviction(skip)   force_refresh mh=4
DEV folds 1-3  NAV%         90.6               132.5
2026 H1 holdout NAV%        63.0                87.2
2026 H1 holdout maxDD%      -4.3                -8.6
2026 H1 holdout win%        68.6                69.4
```

Force-refresh mh=4 wins on NAV in every fold and on the untouched holdout,
with similar win rate, at the cost of somewhat deeper drawdown (more
aggressive slot usage). Honest caveat: the bootstrap statistical edge over
conviction-skip is **suggestive, not decisive** (overlapping per-trade CIs;
basic force-refresh's holdout NAV CI dips negative, but the mh=4 guard
restores a reliably-positive NAV CI). Approved for paper execution;
live-capital promotion awaits corroborating live paper evidence. The three
gate classifiers, the threshold (0.33), and `db.h5` are unchanged — this is a
portfolio-construction policy revision. Dry-run mode does not write local
position state.

#### V6 validation status

The original 0.30-threshold holdout (2026 H1) produced 47 executed trades, 63.8%
win rate, +3.88% average trade, and +53.4% compounded NAV; four-week block-NAV
bootstrap interval was -5.9% to +175.1%. The current policy threshold is 0.33
(raised 2026-08-13; the 0.33-threshold holdout: 40 trades, 70.0% win, +3.91%
avg, per-trade CI [0.17, 8.00] excludes zero — see the Threshold revision note
below). V6 is approved for controlled paper execution and shadow comparison
against V4, but is not promoted to live-capital production.

#### Threshold revision (2026-08-13): 0.30 → 0.33

The min-gate threshold was raised from `0.30` to `0.33` after bootstrap
validation (`04_backtest/61_v6_threshold_bootstrap.py`). On the DEV OOS surface
(nested folds 1-3, the selection surface), the per-trade avg-return CI at `0.30`
was `[-0.77, 3.66]` — it **included zero**, i.e. the per-trade edge was not
statistically reliable. At `0.33` the CI is `[0.56, 5.22]` (excludes zero), win
rate rose 57.3 → 59.3% and avg trade 1.46 → 2.84%; the 2026 H1 holdout confirms
direction (70.0% win, CI `[0.17, 8.00]`). Two honest caveats: (1) the
border-band negativity (scores 0.30-0.33 averaged -1.0%) was a **point
estimate only** — its bootstrap CI `[-4.81, 2.79]` includes zero, so border
trades are *not* proven losers; (2) NAV CIs at these trade counts are too wide
to be decisive. The defensible gain is that `0.33` makes the per-trade edge
statistically reliable where `0.30` did not. `0.35` was data-stronger (tighter
positive CI, higher win rate) but was beyond the requested 0.30-0.33 range and
cuts to 66 trades; retained as a candidate if more aggression is later wanted.
The three gate classifiers are unchanged — this is a policy-threshold revision,
not a model change. Gate classifiers and `db.h5` are untouched.

### 17.B.8 Label horizon vs. execution horizon (train-10, hold-5); persistence-gate path closed

This subsection records two durable design conclusions that are not obvious
from the entry/exit rules alone. It is the authoritative record after the
transient G4 research scripts/artifacts were cleaned up.

#### Train-10 / harvest-5 is a deliberate label/instrument mismatch

The classification labels (`car_10d`, and the G1 CAR gate) are measured over
the 10-trading-day window T+1→T+10 (§17.A.3), but execution holds only 5 days
and exits at `Close[T+5]` (§17.A.5.7). This is intentional, not legacy:

- **The 10-day horizon is a denoising target.** It forces the classifiers to
  learn the *shape* of a sustained post-earnings drift rather than a one-day
  gap-and-pop that reverts. A shorter label would conflate true PEAD with
  transient overnight noise.
- **The 5-day hold is the harvest window.** The drift is front-loaded
  (`52_hold_period_comparison.py`: 5-day hold ≈ 2.5× the NAV of 10-day); the
  back half of the horizon (days 6–10) is where drift fades and mean-reverts,
  adding noise. Executing at 5 days captures the high-SNR front window and
  exits before the tail fade.

Learn the full shape, harvest the cleanest part. This mismatch is load-bearing
and is part of why the models hold up out-of-sample across the 2014–2024 train
/ H2 2024–H1 2026 test span, which spans multiple regimes (QT cycle, COVID,
the 2021 mania, the 2022 rate-shock bear, the 2023–24 recovery).

#### Why a fourth (persistence) gate is structurally closed

A drift-persistence gate — "does abnormal drift continue in the back half of
the horizon?" (`CAR[T+6:T+10] > 0`, the researched "G4") — was investigated
and rejected. It cannot improve the executable PnL for a structural reason,
not a tuning one:

- **The orthogonal definition is irrelevant.** Late-window persistence
  (days 6–10) measures drift in the portion of the horizon the strategy does
  not hold. Predicting returns in the discarded window does not affect
  realized PnL. This was the tested G4 definition; it was non-binding across a
  0.30–0.70 threshold sweep and did not improve the untouched 2026 H1 holdout
  NAV under nested selection.
- **The relevant definition is redundant.** Front-loaded persistence
  (days 1–5) measures the monetized window, but that signal is already
  captured by G1: G1's 10-day CAR label is dominated by the front-loaded
  contribution, so an early-window gate would be near-duplicate to G1.

Orthogonal definition is irrelevant; relevant definition is redundant. The
gate-decomposition search is therefore closed at three gates. Future precision
work should add **structurally different edges** (e.g. timestamped consensus
revisions) rather than a fourth persistence gate on the same post-earnings
event.

#### Ongoing PEAD posture

V6 remains the paper-executable model. With ~4 slots × ~25 turn/year against a
~100-trade/year supply, the strategy is near slot capacity, so additional
filtering discards trades that would fill slots anyway rather than freeing
capacity. The marginal value of further gates is below the effort threshold;
the binding open question is not "add signal" but whether PEAD itself persists
forward. Live paper execution doubles as a decay monitor; a pre-registered
rolling kill rule (e.g. rolling 30-trade win rate < 50% or avg trade < +1%)
should gate any live-capital promotion.

## 17.C. V4 Timing-Correct Comparison Baseline

### 17.C.1 Model and artifact
The V4 comparison baseline is an `XGBClassifier` with
`objective="binary:logistic"`, trained on 23 features and the same `pead_pass`
three-gate target used by v3. The frozen artifact is:

```text
03_model/models/phase_g_v4_timing_correct/classifier.json
03_model/models/phase_g_v4_timing_correct/meta.json
/features/train_matrix_v4_timing_correct
```

The trainer is `03_model/06_freeze_timing_correct_model.py`. It writes a
separate matrix and model; v3 is not overwritten.

### 17.C.2 Information-set contract
For an event on trading day T:

| Event | entry | Latest completed daily data at decision | v4 training/inference cutoff |
|---|---|---|---|
| AMC on T | Close[T] | Close[T-1] | T-1 |
| BMO on T | Close[T-1] | Close[T-2] | T-2 |

The corrected BMO rows shift price/volume features back one daily bar. Analyst
revision and macro features are also evaluated at the same cutoff. Prior
reported earnings features remain historical. No partial intraday data is used.

### 17.C.3 Features and filters
The 23 features are:

```text
sue_lag_1, sue_lag_2, car_drift_historical_q1,
pre_event_idiosyncratic_vol, pre_event_volume_trend,
rel_ret_3d, rel_ret_5d, rel_ret_10d, rel_ret_20d, rel_ret_30d,
sector_adjusted_ret_20d,
revision_momentum_30d, revision_momentum_60d, revision_momentum_90d,
revision_ordinal_momentum_90d, revision_intensity_90d,
grade_dispersion_90d, n_analysts_covering, last_action_days_before_earnings,
consecutive_surprises_pre, unemployment_roc21, fed_funds, vix
```

The V4 comparison plan requires `P(PEAD) >= 0.20`, current S&P 400 membership,
sector not XLF, and not the LEG-like profile:
`sue_lag_1 < -0.5 AND consecutive_surprises_pre == 0 AND rel_ret_20d < -0.05`.

### 17.C.4 Walk-forward reference statistics
The v4 four-fold walk-forward test is approximately **1.92 years of executed
out-of-sample time**, not one year:

```text
Executed-entry span: 2024-07-23 -> 2026-06-24
Fold 1 test: 2024 H2
Fold 2 test: 2025 H1
Fold 3 test: 2025 H2
Fold 4 test: 2026 H1
```

Detailed recomputation using the v4 matrix, theta=0.20, four slots, weekly
P(PEAD)-priority selection, pre-gap entry, five-day hold, and delayed -10%
stop:

```text
99 executed trades
57 wins / 42 losses
57.6% win rate
+2.78% average trade
+1.59% median trade
11.93% trade-return standard deviation
+10.43% average winning trade
-7.61% average losing trade
1.37 payoff ratio
1.86 profit factor
+39.74% best trade
-20.77% worst trade
+274.97% raw sum of trade returns

Final NAV: 1.897x
Compounded NAV return: +89.73%
Approximate CAGR over executed span: +39.61%
Maximum weekly portfolio drawdown: -12.46%
Weeks with trades: 52
PEAD precision: 24.3%
```

Per-fold details:

```text
Fold 1: 16 trades, 75.0% win, +7.78% avg, +34.35% NAV
Fold 2: 30 trades, 60.0% win, +2.95% avg, +23.61% NAV
Fold 3: 26 trades, 42.3% win, +0.47% avg,  +1.94% NAV
Fold 4: 27 trades, 59.3% win, +1.85% avg, +12.08% NAV
```

These replace v3's +293.8% NAV headline for deployment decisions. The result
is positive in every fold but materially weaker and less stable than v3; this
is the cost of removing the unavailable BMO day. The maximum drawdown above is
computed from the weekly equal-weight portfolio NAV curve; it is not the same
as the worst fold return.

### 17.C.5 Fresh actionable-list inference
Script `05b_alpaca_live/01_fetch_and_predict.py` is an as-of-now engine:

* It always refreshes Tiingo daily data through the latest completed close.
* **It refreshes benchmark ETFs (IJH + needed sector ETFs) from Tiingo** before
  loading them — stale benchmarks corrupt every `rel_ret` / `car_drift`
  feature. (Added 2026-08-08 after the stale-IJH incident: IJH was a month
  behind, silently biasing all candidates.)
* **It refreshes FRED macros (VIXCLS, DFF, UNRATE) via REST API** before
  loading them — stale macros corrupt the `vix` / `fed_funds` /
  `unemployment_roc21` features.
* It queries FMP with `from=today-1` to avoid the boundary omission observed for
  ROKU, and bounds the per-ticker fallback by the same `--weeks` horizon.
* It drops events whose entry date has passed.
* During market hours, today's AMC/BMO entries remain actionable. After the
  close, only entries after today remain actionable.
* The list is therefore provisional for future events and must be refreshed
  again before their entry date.

Examples:

```text
Aug 6 market hours: AMC Aug 6+ and BMO Aug 7+
After Aug 6 close: AMC Aug 7+ and BMO Aug 8+
```

### 17.C.6 Execution and order-state rules
Script `05b_alpaca_live/02_paper_trade.py` manages four equal-weight slots and
uses whole-share immediate-market (`TimeInForce.DAY`) orders for both buys
and sells — fills are reliable in paper and live, same-day sell→buy pairs
execute against cleared buying power, and there is no dependence on Alpaca
MOC/CLS support (which needs elite smart-router routing live and is simulated
with injected partial fills in paper). No bracket orders are used; exits are
managed through the delayed -10% stop and T+5 rules.

Order reconciliation must use actual broker quantities, not only terminal
status. Exit orders have a separate `pending_exits` lifecycle:

* A T+5 or stop exit is submitted as an immediate DAY market sell and removed
  from `active` only after the sell is broker-verified.
* Filled sells move to `closed` using the actual filled quantity and price.
* Partial sells close the filled quantity and restore the residual quantity to
  `active`.
* Rejected/canceled/expired zero-fill sells restore the position to `active`.
* A broker-accepted pending sell may free a same-run replacement slot, but the
  sell remains in `pending_exits` until the actual fill is confirmed on the next
  run.
* Replacement buys are blocked if the same-run sell cannot be broker-verified.

Entry order reconciliation must use actual broker quantities, not only terminal
status:

* Entry `filled_qty > 0` plus `expired`/`canceled` means a real active
  partial-fill position; move only the filled quantity to `active`.
* Record `planned_qty`, `filled_qty`, `unfilled_qty`, `fill_status`, and final
  order status.
* A terminal zero-fill order is recorded as unfilled/expired and frees its slot.
* A manual close sets `exit_date` to the actual close date and preserves the
  original planned date as `planned_exit_date`.

**Order type (2026-08-12): both buys and sells use immediate `DAY` market
orders, not MOC (`CLS`).** Two independent lines of reasoning converged here:

*Sell side (2026-08-08, front-loaded-PEAD rationale):* the execution-side
expression of the same front-loaded-PEAD principle as the train-10/harvest-5
label choice (§17.B.8) — the tail of the horizon is low-value, so do not spend
fill-fidelity or lock capital protecting it. At T+5 a slot is earning ~0
marginal drift (deep in the harvest tail), while a fresh pick entering that
slot has the full front-loaded drift ahead of it, so accelerating turnover is
EV-positive. An unfilled MOC sell also used to leave a dead position holding
into T+6/T+7 (no drift left + slot locked + blocks the next entry); immediate
market fills eliminate that failure mode.

*Buy side (2026-08-12) — DAY market is the correct PAPER choice; live is
deferred and may revert to MOC.* Alpaca developer relations confirmed two
paper-specific facts: (1) paper trading intentionally injects a higher
partial-fill rate (partial fills "are rare" in live), so the observed MOC-buy
pattern (HRB 22/22 filled; TWLO 0/5, DBX 9/29, ENS 0/5 expired) is a paper
artifact, not a live reliability signal; (2) paper `CLS` orders are simulated
as plain market orders at the close that cross the spread — they do NOT fill
at the actual closing-auction price, so the close-price fidelity MOC was
chosen for was never delivered in paper anyway. Both make DAY market the right
paper choice: reliable fills, and no fidelity is lost (paper MOC never
delivered auction fidelity to begin with). A live caveat also surfaced: live
MOC/CLS is not generally supported unless the account uses elite smart-router
order handling. For LIVE, the buy order type is deferred to promotion time:
real closing-cross liquidity makes partial fills rare and real MOC fills at
the actual auction price (delivering the Close[T] fidelity the backtest
assumes), so live MOC may be preferable if elite routing is available; the
decision rests on forum/Alpaca findings still in progress.

*Timing implication (operational, under DAY-market execution):* unlike MOC —
which fills at the close regardless of submission time — a DAY market order
fills at script-run time. To approximate the Close[T] entry the backtest
assumes, Script 02 must run near the close (~3:45 PM ET), not at market open;
the prior market-open runs were harmless under MOC (orders sat until the
close) but would produce morning entries (~6h early) under DAY market. If live
reverts to MOC, this discipline relaxes again — MOC fills at the close
regardless of run time.

Net effect: same-day sell→buy pairs execute against cleared buying power with
reliable fills on both legs; the `ENTRY_PRICE_BUFFER` (a 1% qty cushion for the
MOC close-overshoot) was removed because market fills execute at ~the sizing
price. Fill quantity/price are still reconciled by `sync_exit_orders` on the
next run.

## 17.A. HISTORICAL: Phase G v3 Honest Binary Classifier (No Look-Ahead)

> v3 is retained as the prior deployable baseline for comparison. It is not
> the current live artifact. The current timing-correct v4 specification is
> §17.B.7 (V6) and §17.C (V4 comparison) below.

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
7.  **Exit** — `Close[T+5]` (5-day hold from report date). Frees slots weekly, cuts losses short. 5-day dominates 10-day on every metric (`30_hold_comparison_bootstrap.py`). The 10-day *label* is retained on purpose; see §17.B.8.
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

### 17.A.9 Historical v3 OOS Performance (not the current benchmark)
From `42_xlf_excluded_detailed_stats.py`:
- **102 OOS trades across 4 folds** (~51 trades/year)
- **Win rate: 62.7%** (64W / 38L)
- **PEAD precision: 30.4%** (31 true PEAD / 71 false positives)
- **Expectancy per trade: +5.71%** (median +3.40%, std 13.14%)
- **Avg win: +13.06%**, **avg loss: -6.68%**, payoff 1.96, profit factor 3.30
- **Total PnL (raw sum):** +679.1% (with stop), +672.4% (no stop)
- **Total PnL (NAV-compounded):** **+293.8%** (3.94x NAV). Honest model — no look-ahead.
- **Best/worst trade: +51.82% / -30.54%**
- **Max drawdown (NAV-compounded): -5.9%**
- **Annualized Sharpe (approx): 3.71**
- **No losing fold** (per-fold NAV: +43%, +41%, +31%, +49%)
- Fold range: only 18.3pp (min +30.9%, max +49.2%) — most stable model yet
- **Large PEAD (CAR>=10%):** 20 trades, 85% win, +20.66% avg.

### 17.A.10 Historical v3 Artifacts
The v3 trainer and artifact remain available for comparison only:

```text
03_model/05_freeze_honest_model.py
03_model/models/phase_g_v3_honest/{classifier.json, meta.json}
```

Do not use v3 for new plans. Use V6 §17.B.7 for paper execution and V4 §17.C for comparison.

### 17.A.11 Acknowledged Gaps in Evidence
- **No live paper-trading fold #5** — the first true forward-looking OOS data point. The live script (`05_live/01_live_fold_pull.py`) is ready; run around 2026-09-30 for 2 months of forward-looking data.
- **Precision at 38.6%** — every precision lever tested (theta increase, 3-class, eps filter, CAR regression) improves precision but hurts total PnL because non-PEAD picks are profitable (+2.11%). Better FEATURES (not filters) are needed. Options: FMP estimate revision trajectory, SEC Form 4 insider trading, FINRA short interest.
- **Transaction costs + slippage not modeled** — must be added before live capital deployment.

---

## 18. RESEARCH BACKLOG (post-freeze candidates)

> Full narrative record of the 2026-08 edge search and the governing doctrine
> ("we compete on depth of analysis, not speed"): see
> `archive/findings/edge_landscape_memo.md` — the authoritative summary of
> all 10 candidates, the institutional-advantage map, and the pattern laws.

Gated behind the model freeze (§17.B): nothing here is actionable until the
paper-evidence period closes and a research cycle is explicitly approved.
Each item records hypothesis, mechanism, data source, validation bar, and
revisit trigger — so a closed investigation never needs re-litigating.

### Background: the slow-week edge search (2026-08, closed)

Seven candidates were tested with honest event studies (execution from first
tradeable close, CAR vs IJH, tail-vs-dispersion test, ex-ante conditioning):

| Candidate | Script | Result | Status |
|---|---|---|---|
| Analyst upgrade drift | 68 | Day-0 +3.27% (untradeable), post drift +0.09% | CLOSED |
| S&P 400 index additions | 45 | Abnormal ret −0.00% vs IJH post-2020 | CLOSED |
| Insider cluster buying | 69 | 5–10d CAR +0.11% (49% win); only 60d has drift | CLOSED (as edge) |
| Pre-ex dividend run-up (raw) | 70 | Mean ≈ 0; 35% tail is pure dispersion | CLOSED |
| Pre-ex dividend run-up (filtered: yield>2.5%, SMA50, ADV) | 70 | −0.10%..−0.22%; yield anti-predictive | CLOSED |
| Senate trades (disclosure-date entry) | probe | Real 10–20d drift (+2.13% CAR) but ~1.7 Sep events/mo, 28d staleness | PARKED |
| S&P 500 PEAD expansion | 66/67 | Transfer works (+663% NAV) but home universe wins (61.4% vs 55.5% win) | PARKED, low pri |
| FMP fundraisers (Form D/C) | probe | Private-market, untradeable; sp400 supply dead since 2016 | CLOSED |

Pattern established: every scheduled or instantly-public event is arbitraged
to zero; edge survives only where information requires slow interpretation.
Composite-event testing of these candidates is pre-rejected: sparse ∩ sparse
(senate 0.23%/stock-wk × insider cluster 1.2% → ~1 event/yr universe-wide)
cannot be validated, and both legs share the same mechanism (informed-capital
optimism) so they substitute rather than compound.

### RC-1: Insider-accumulation pre-event features for PEAD

- **Hypothesis (mechanism-backed):** Piotroski & Roulstone (2005) — insider
  trades predict future earnings surprises; informed capital accumulates
  ahead of good quarters. PEAD after a beat should be LARGER when the beat
  was preceded by net insider buying.
- **Proposed features (join to gated earnings events):**
  - `insider_net_buy_90d` (flag: net P-purchase dollars > 0 in [T−90d, T))
  - `insider_cluster_90d` (flag: ≥2 distinct insiders bought, ≥$50k total)
  - `insider_dollars_90d` (log net dollars)
- **Data:** already cached — `01_data/db_insider.h5` (Script 69, 71,663 Form 4
  records, 959 tickers, P-purchases only, filing dates for leak-free joins).
- **Supply estimate:** ~690 material purchases/yr → feature non-zero for
  ~15–25% of earnings events. Adequate for tree learning.
- **Precedent:** analyst grades failed as an edge (V5, script 68) yet function
  as features — features need conditional information, not standalone alpha.
  Caveat: revision-momentum features rank low in importance; weak signals
  usually stay weak as features. Mechanism here is stronger (informed capital
  front-running fundamentals vs repackaged opinions).
- **Validation bar (same as everything):** walk-forward retrain with feature
  vs frozen V6 baseline; bootstrap CI on per-trade edge; accept only if DEV
  folds show lift without holdout degradation.
- **Trigger:** paper-evidence period closes and freeze lifts; requires its own
  approved research cycle (no silent feature additions to a frozen model).

### RC-5: Guidance pre-announcement mini-events via 8-K detection (CLOSED 2026-08-22 at the hand-label gate)

> **Endpoint probes (our tier, 2026-08-22):**
> - `/stable/sec-filings-search/symbol?symbol=X` — WORKS, filters properly
>   (ANF: 79 rows 2026-YTD, single symbol, 2015+ verified). Returns ALL form
>   types (8-K, 4, 10-Q, SC 13G, DEF 14A...) — filter formType locally.
>   One-time backfill: ~13 pages/symbol × 419 ≈ 5.4k requests.
> - `/stable/sec-filings-search/form-type?formType=8-K` — WORKS for the
>   universe 8-K firehose (daily operational pull; filter locally by cik).
> - `/stable/sec-filings-8k` (bulk date-window) — works but `symbol` param
>   IGNORED; superseded by the two search endpoints above.
> - Fields (all three): symbol, cik, filingDate, acceptedDate (INTRADAY —
>   BMO/AMC entry timing derivable), formType, link, finalLink (SEC doc).
> - STILL MISSING: item classification + content. Item 2.02/7.01 vs 1.01 vs
>   5.02 must be read from the linked SEC document (lexicon classifier).
> - Bonus: by-symbol returns Form 4s → same endpoint can refresh RC-1
>   insider features. FMP guidance endpoints remain 403/legacy.

- **Zero-NLP flag probe (2026-08-22, 60 tickers / 694 events 2023-26 / 2,311
  8-Ks):** does a mere ANY-8-K flag in [T-14, entry] carry directional info?
  Result: flag fires on 61% of events (routine filings dominate — not a
  selective guidance detector); flagged events show HIGHER SUE (+10.7% vs
  +5.1% — well-run companies file more 8-Ks; quiet ≠ warning-free) and
  slightly better r15 (+0.06% vs -1.02%); but among beaters the flag adds
  only +0.27pp (+0.97 vs +0.70) — mostly SUE-composition artifact. Verdict:
  the raw flag is a weak health proxy, NOT a guidance signal. The raise/
  warning information lives only in Item 2.02/7.01 content → the hand-label
  + lexicon step is unavoidable and remains the gate. (Known contamination:
  evening-accepted 8-Ks on BMO entry day counted as known; minor.)
- **Hypothesis:** off-cycle guidance updates (warnings, raises, conference
  reaffirmations) are unscheduled mini-events with genuine unresolved
  uncertainty, requiring interpretation (guide revision -> fundamental
  re-rating), in our thinly-covered universe. Same doctrinal class as PEAD:
  slow diffusion, multi-day horizon, no timestamp race (8-K accepted-time
  detection is minutes — irrelevant vs a 5-20d harvest).
- **Required build (the real cost):** daily 8-K index pull → cik filter →
  fetch SEC doc via finalLink → lexicon classifier for Item 2.02/7.01 +
  raise/warning/reaffirm (validated against a hand-labeled sample; classic
  keyword NLP, no paid data). Then event study: CAR vs IJH by classification
  and recency-vs-next-scheduled-earnings, 2015-2026, same method bar as
  every closed candidate.
- **Cheap probe first (gate before building):** hand-label ~100 historical
  SP400 Item-2.02/7.01 filings pulled via finalLink; eyeball drift by class.
  If warnings/raises don't show visibly fat CAR tails on raw eyeball, the
  pipeline never gets built.
- **Relation to RC-2 kill:** RC-2 closed post-event extensions (info already
  priced by day 5). RC-5 is a *new pre-event uncertainty supply* — the
  pre-announcement is the event itself, entered at first close after
  acceptedDate. Doctrinally distinct, not a zombie of RC-2.
- **Priority:** behind RC-1 (insider features, frozen-gate) and the August
  month-end megatrend panel. September dead-zone candidate if the cheap
  probe passes.

> **HAND-LABEL GATE RESULT (2026-08-22; 656 pre-earnings-window 8-Ks fetched,
> item headers extracted, 49 strictly-pre-earnings Item-2.02/7.01 docs read
> incl. exhibit retrieval): CLOSED — event supply is fatally sparse.**
>
> **FULL-DOC CENSUS (2026-08-23, user-requested before final close; corpus =
> all 656 in-window docs, content-classified end-to-end):** FMP's finalLink
> points at the 8-K SHELL for ~179 docs but straight at the EX-99 press
> release for 477 — the earlier item-header read covered only the shells;
> the exhibit side was content-classified by regex + eyeball verification.
> Full-corpus breakdown (strictly-pre = 188 docs, ≥1 day before earnings):
> earnings releases 210 (338 of 477 exhibits same-day = the event itself;
> 6 "pre" ones are ±1d date mismatches — OGE/PII evening filings);
> board/mgmt 95+57shell; M&A 52+shells; conferences/presentations 37+2;
> dividends/buybacks 30; debt/offerings; other. GENUINE pre-earnings
> guidance documents: 3 of 656 (0.5%) / 3 of 188 strictly-pre (1.6%) —
> PII separation+prelim-results preview, GEF "reaffirms FY23 guidance" at
> Baird conf, VSAT "reaffirms FY25 guidance" with exec reorg — and 2 of 3
> are REAFFIRMATIONS (weakest class); raises/warnings: zero. Plus VOYA's
> 23 recurring monthly prelim disclosures (insurer idiosyncrasy, one
> company). Scaling: 3 events / 60 tickers / 3.2y = 1.6% of companies/yr
> → SP400 ≈ 6-7 events/yr; ALL US filers (8-K volume measured 608/day on
> 2026-08-13, ~4x SP400-membership count) ≈ 60-70 events/yr universe-wide
> (~1.2/week), concentrated in larger caps (classic pre-announce behavior);
> small caps even quieter. Verdict UNCHANGED for SP400 PEAD (feature fires
> ~1.5% of events, dominated by reaffirm-noise); universe-wide it is a
> low-frequency event supply (~1/wk) needing whole-market screening
> infrastructure (608 docs/day × content classification) for a handful of
> mostly-reaffirmation trades/yr — capacity fine, supply and signal quality
> are not. RC-5 stays CLOSED; census archived as the permanent basis.
> - Of 49 pre-earnings 2.02/7.01 filings, classification: 1 true
>   pre-announcement (PII 2025-10-14: separation + preliminary Q3 results
>   preview), 23 recurring monthly preliminary-metrics disclosures from ONE
>   company (VOYA: "alternative investment income above/below expectations" +
>   AUM — an insurer idiosyncrasy, directional but not guidance), 2 unreadable
>   shells, and the rest routine corporate actions (board ×7, M&A/transaction
>   ×9, debt refinancing ×3, litigation ×1, partnership ×1, segment-report
>   change ×1, presentation posting ×1). Ex-VOYA base rate: ~1-3 genuine
>   guidance pre-announcements per 656 filings / 60 tickers / 3.2y ≈ 0.5%.
> - Design A (PEAD feature) dead: feature non-zero for ~2-5% of events
>   universe-wide, 23/24 of which are one company — unlearnable by the gates,
>   unvalidatable. Worse than senate-trade sparsity (parked at 4.5/mo).
> - Design B (standalone event supply) dead: ~1-2 events/yr ex-VOYA.
> - Why so rare (mechanism): Reg-FD + mid-cap practice puts guidance updates
>   AT the quarterly release, not mid-quarter; the classic warning/raise
>   pre-announcement is an S&P-500-era behavior that SP400 companies mostly
>   don't do. The window [T-30, T-1] is largely the quiet period.
> - Salvage: (a) 8-K fetch/item-extraction infrastructure works and is
>   reusable; (b) FMP report_date vs actual 8-K dates have ±1d mismatches
>   (OGE earnings filed T-1) — never join filings to events by date alone
>   without a tolerance; (c) recurring-preliminary-metrics disclosures
>   (insurers/financials) exist as a niche pattern — not a general edge.
> RC-5 CLOSED at the gate, per doctrine: no pipeline was built.

### RC-6: 8-K event-edge census & buyback-announcement test (CLOSED 2026-08-23: no slow-window alpha in the 8-K channel)

> **Full-universe test (user-directed; S&P 500, 503 names, 2023-01..2026-08):**
> 23,567 8-Ks indexed via /stable/sec-filings-search (by-symbol); 16,087
> standalone (non-earnings-day) docs fetched from EDGAR (15,169 OK, 6%
> fetch-loss noted) and content-classified. Taxonomy of standalone 8-Ks
> (~17/day universe-wide): board/mgmt 45.7%, M&A 19.2%, conferences 13.4%,
> other 12.5%, div declarations 2.4%, off-cycle earnings-style releases
> 1.4%, offerings 0.8%, div changes 0.6%, buyback auth/increase 0.1% (22
> events in 3.6y — most buyback programs are announced by press release
> only and never 8-K'd; the filed subset is the 'material' minority).
>
> **Buyback auth/increase CAR vs SPY (n=22, entry=first close after
> acceptedTime):** gap +0.26% (untradeable instant); T+1→T+5 −0.57% (win
> 41%); T+1→T+10 +0.64% (t=0.8). NO drift at our horizon — the Ikenberry
> buyback premium is a 2-4 YEAR effect, fully priced at the gap for anyone
> slower than the announcement print. Consistent with doctrine: instantly
> public + instantly interpretable = no slow window.
>
> **Secondary classes tested:** dividend cuts (n=27): c15 +0.97% (t=1.4,
> 2024-driven +2.4%, other years ~0) — the same expectations-reset/
> mean-reversion echo found with analyst downgrades (script 46); does not
> clear any bar. Dividend raises (n=46): t=0.8, nothing. Off-cycle
> earnings-style releases (n=218): c15 +0.29% (t=1.3), nothing.
> 'Non-reliance' regex class was boilerplate-contaminated (restated
> certificates/XBRL) — dropped.
>
> **What remains untested in this channel:** (a) M&A definitive agreements
> (19.2% of standalone docs) — TESTED 2026-08-23, see below; (b) Russell-2000
> small caps — user-assessed as equally buyback-poor (smaller capital bases),
> not queued.
>
> **M&A decomposition (3,095 docs → 2,642 acquirer-side / 274 target-side /
> 179 other):**
> - ACQUIRER side (n=797 deduped): gap −0.17% (t=−2.5), T+1→5 −0.24%
>   (t=−1.8), T+1→10 −0.52% (t=−2.8), negative in every year. The classic
>   ACQUIRER'S CURSE — the only statistically stable finding in the entire
>   8-K census — but it is a SHORT-side effect (needs borrow, short-risk
>   profile; wrong lane for a long-only PEAD book; parked).
> - TARGET side: 99% classification contamination (merger completions,
>   subsidiary mergers, JV/spin agreements — no announcement jump).
>   Validation via pre-jump screen (T−1→T close must show >+10% for a true
>   fresh target): only 3 of 260 docs qualify (NFLX-26, TPR-24, PSKY-26;
>   ~1-2 fresh SP500 take-private 8-K events/yr). Their post-entry drift
>   T+1→10 +3.44% (win 3/3) = classic deal-spread convergence — the ONE
>   positive slow-burn found in the whole 8-K channel, but n=3/3.6y and it
>   is deal-break-risk arbitrage (selling legal-event insurance; −30..−50%
>   tail on a broken deal), a different business, not a PEAD slot-filler.
> Verdict: M&A channel cannot fill slow weeks for a long-only book; the
> census is complete. RC-6 CLOSED (final).
>
> RC-6 CLOSED. Infrastructure retained: filing index + EDGAR fetch +
> classification stack (reusable, ~2.5h wall-clock for full SP500 3.6y).

### RC-7: Merger-arb sleeve for slow weeks (PROBED & CLOSED 2026-08-23: naive sleeve is mean-negative; the lane requires a different business model)

- **Origin:** RC-6's M&A decomposition found target announcements +3.4%/10d
  with 3/3 wins; user flagged the +100% win rate as encouraging and asked
  about non-8-K sources. Sources probed: FMP M&A endpoints DEAD (403
  legacy); EDGAR Form 425 (1,185 symbols in 2025), SC 14D9 (52-70 distinct
  tender targets/yr), DEFM14A (deal terms) all WORK via our verified
  sec-filings-search endpoints; Tiingo news API works (fast detection).
- **Survivorship confession:** RC-6's n=3 was an artifact — db_sp500 holds
  current members only; acquired targets delist and vanish. True deal flow
  is ~50-70 tender deals/yr plus hundreds via 425.
- **Naive-strategy measurement (SC 14D9 targets 2024-25, n=111, 38 priced
  via Tiingo incl. delisted; entry = first close after first SC 14D9):**
  c5 mean +3.15% / MEDIAN 0.00% / win 47% (a few fast convergences = the
  arb desks' capture); c20 +0.78%/median −0.18%; c40 mean **−3.95%** /
  median +0.74% / win 53%. The mean is destroyed by the break tail:
  SQNS −74.5%, ALLGF −43.2%, LU −33.2%, MANU −29.0%, RHEP −27.8%.
  Distribution = sell-insurance: most deals converge ~0-2%, breaks
  destroy the mean. n=3 SP500 slice was the cleanest-regime cherry.
- **Why we close:** profitable merger arb REQUIRES deal-terms screening
  (financing/regulatory conditions, termination fees from DEFM14A),
  broad concurrent diversification (break tail 1/N'd), typically acquirer
  hedging (borrow — retail-unfriendly), and capacity competes with
  dedicated desks in exactly the liquid deals worth doing. The remaining
  unpriced 73 tickers skew foreign/microcap (SQNS, MANU ADR...) — the
  part of the distribution that IS accessible to retail is the worst part.
- **Salvage:** SC 14D9/425/DEFM14A feeds + Tiingo news all verified and
  reusable; the break-tail measurement is a permanent reference for why
  "sell legal-event insurance" ≠ slow-window drift.

### RC-8: Post-announcement DIP-ENTRY merger convergence (CLOSED 2026-08-23 at the regime gate)

- **Origin:** user observed (a priori, before any test — the best kind of
  hypothesis): after a deal announcement jump, targets often dip for 2-3
  days, then drift up into the deal's formal process. Testing found RC-7's
  mean-negative result was an ANCHORING artifact: the SC 14D9 acceptance
  used as entry is typically 20-60 days AFTER announcement — i.e., we had
  been measuring AFTER the convergence window closed.
- **Method:** announcement day0 = largest single-day jump (>8%) in the 90d
  before SC 14D9 (price-detected, no document ambiguity); Tiingo daily
  (delisted included); SC 14D9 tender targets 2023-25; n=77 clean.
- **Honest implementable rule (no look-ahead):** IF day+2 close is BELOW
  day0 close (observable at entry) → buy at day+2 close.
  - Hold +10d:  mean +3.28%  median +0.58%  win 71%  t=+2.2  n=34
  - Hold to 14D9: mean +5.35%  median +1.42%  win 87%  t=+3.7  n=31
  - Control (NO dip): +0.40% t=+0.3 / −0.02% t=0.0 — alpha is ENTIRELY in
    the dipped subset; no-dip deals are already converged at day0.
  - Hindsight 'dip bottom' entry (look-ahead, upper bound): +4.12%/84%/t=3.9.
- **Mechanism (doctrinally clean):** the dip identifies deals where flow
  (retail exit, index funds, tax sellers unloading into the pop) — not deal
  risk — pushed price under the initial close, leaving residual spread the
  market slowly re-captures as deal process confirms. Slow interpretation,
  long-only, no borrow, no speed: retail-compatible. Fills idle PEAD slots
  (~25 dipped deals/yr universe-wide, killable by PEAD per slot policy).
- **Open risks (validation gates before ANY sleeve):**
  1. Regime: 2024-25 friendly M&A window; 2022 financing winter unsampled.
     GATE: extend targets to 2019-2023 via sec-filings-search (endpoint
     supports historical windows); require dipped-subset positive in the
     break-heavy year.
  2. Break tail inside hold: 6 deals <−15% in full sample (APLT −90%,
     BLUE −57%, NHHS −56%, RVNC −48%); mean survives in-sample; sizing
     must assume −50% tails (half-slot or quarter-slot per deal).
  3. Multiple-testing: ~6 variants were run before landing on the user's
     rule; the a-priori origin mitigates but the 2019-23 extension is the
     true out-of-sample test.
  4. Deal-terms conditioning (DEFM14A cash/stock, financing terms,
     residual spread) is the professional upgrade — later, only if the
     regime gate passes.

> **REGIME-GATE RESULT (2026-08-23, same day — the 2019-2023 extension was
> the decisive test and it FAILED):** SC 14D9 targets backfilled to 2019
> (241 new symbols, 168 priced via Tiingo incl. delisted); combined
> 2019-2025 clean sample n=124 with price-detected announcements. The
> dipped-subset rule (+3.3%/t=2.2, +5.4%/t=3.7 in 2024-25) COLLAPSES out
> of window: pooled 2019-2025 h10 mean −0.50% (t=−0.6), to-14D9 +1.70%
> (t=+0.9); 2021 −3.69% mean / 55% win; 2022 (break winter, n=17)
> −1.94%/47% win, −3.72% to 14D9 — the entire in-sample edge was the
> friendly 2024-25 M&A regime, not the dip mechanic. The 'alpha entirely
> in dipped subset' conditioning also weakens (no-dip control only
> mildly worse pooled). Verdict: regime-dependent, no persistent edge;
> RC-8 CLOSED per the pre-registered gate (2022-23 stress requirement
> not met). Kill-honesty note: the 2024-25 result stood at t=3.7 with an
> a-priori rule — exactly why out-of-window validation outranks in-sample
> significance; the friendly regime, not the pattern, was the alpha.

> **APPENDIX — adaptive-entry variant (user follow-up, tested 2026-08-23):
> also closed.** Hypothesis: fixed T+2 misses the true bottom in contested
> deals (SAVE: announcement Apr 5, bottom May 12, 14D9 May 19); wait for
> drift CONFIRMATION instead (first close > max of prior 3 closes, only
> after a dip below day0). Result on the same 2019-23 sample: triggered
> 78% (median wait 10d, entry ≈ day0 close, −0.21% median); entry→14D9
> pooled −0.28% (t=−0.2); by year 2023 −3.54%, 2022 −2.02%; head-to-head
> vs fixed T+2 on the same 53 deals: median improvement −0.49% per deal,
> better in only 21%. Mechanism of failure: in contested deals the first
> bounce FAILS — SAVE's confirmation fired Apr 19 ~$23.1 on the initial
> bounce, then the stock collapsed to $14.77 (−36% from trigger); deeper
> confirmation thresholds = more parameters = pure mining. Diagnosis:
> 'wait for the real bottom' requires distinguishing real bounces from
> dead-cat bounces, which IS deal-terms underwriting (the professional's
> edge), not a price-only pattern. SAVE full anatomy: fixed T+2 −28.0%,
> confirmation −25.4%, hold-to-completion from T+2 +40% — the only winning
> path required sitting through a −38% drawdown on one deal. Every
> price-only entry rule tested (fixed, dip-bottom, confirmation) converges
> to the same verdict. RC-8 CLOSED (final, with appendix).

> **APPENDIX 2 — risk-management variant (user follow-up: −10% PEAD-style
> stop + no-microcap filter; tested 2026-08-23): also closed.** 2×2 on the
> 61 dipped deals (to-14D9 exit): no-stop/all +1.70% (win 70%) → stop/all
> **−1.67%** (win 51%); liquid-only (ADV≥$10M, n=16) no-stop +0.17%
> (win 88%, 2 neg years of 5) → stop+liquid −1.63%. Three findings:
> (1) the stop is anti-moat here: 21 deals breached −10% mid-trade, 71%
> RECOVERED (MEDP ended +6.1%, GNMK +10.4% — both chopped at −10%);
> foregone recovery avg +6.6%; (2) stops don't cap risk anyway — fills
> ranged −10.0% to −29.8% (deal-break gaps through the trigger);
> (3) the liquidity filter works mechanically (AUTO $0.03M out, SAVE $88M
> in) and yields exactly the professional universe: 16 deals/7y of +0.2%
> to +6% convergence (DNKN +5.7, WH +4.3, LOXO +1.0, MIK, ABMD...) with
> one SAVE −13.2% eating a whole regime year. Diagnosis: a −10% stop is a
> THESIS-INVALIDATION device; PEAD's thesis lives in price (drift failed
> = exit), merger convergence's thesis lives in documents, where an
> interim −10% dip carries ~zero survival information. Price-only risk
> management of a document-driven trade fails for the same reason
> price-only entry timing did. RC-8 CLOSED (final, appendices 1-2).

### RC-9: Undecided-state detector for the megatrend watcher (PROMOTED 2026-08-23 to panel §[13]; advisory)

- **Problem (user-stated):** during undecided phases money is thrown across
  many trends simultaneously (e.g. 2020-21: clean energy / SPACs / EV /
  crypto / WFH; 2017 melt-up) and the winner is unknowable ex-ante. Partial
  entry/exit (config D) is the right POLICY for that state — but it was
  validated unconditionally and carried −43% DD partly from running in
  states where it doesn't belong. Missing layer: a classifier for WHICH
  policy applies now.
- **Design (3 metrics, monthly, from cached /mt series; thresholds fixed,
  no tuning):**
  - P = Spearman rank autocorr of theme 3m relative returns (vs SPY), lag 1m
    (leadership persistence)
  - B = fraction of theme series above MA10 with positive 6m relative
    strength (theme breadth)
  - C = avg pairwise 60d correlation of theme daily returns (bloc vs
    differentiation)
  - State map: UNDECIDED (B↑ P↓ C↑ → fractional+rotation posture) /
    DIFFERENTIATING (P rising through ~0.5, B narrowing → winner emerging;
    the actionable transition) / CONCENTRATED (P↑ C↓ → hold leader) /
    DISPERSAL (B↓ C↑ → de-risk advisory; recession evidence still decides
    absolute risk per doctrine, not this classifier).
- **What it is NOT:** winner selection (AI vs clean energy unknowable from
  breadth); auto-allocation (advisory section [13] + log field in the
  month-end panel); a timing overlay (n of transitions ≈ 5-6 in 12y;
  fixed thresholds, pre-registered before any backfill).
- **Validation gate (episode classification, NOT NAV):** using only
  prior-month data, the map must call 2020-21 UNDECIDED, late-2021
  DIFFERENTIATING, 2023+ CONCENTRATED, 2018 DISPERSAL. Pre-registered
  thresholds: P=0.5, B=50%, C=60th pct of own history. Any post-hoc
  threshold chosen after seeing the backfill = mining, reject.
- **Honest priors:** classification is a lower bar than prediction, so
  this is more likely to survive than the warning filters (all dead);
  but the DIFFERENTIATING signal is inherently lagging — we pay for not
  marrying losers by giving up the winner's first leg. The fractional
  design already accepts this trade.

> **RESULT (2026-08-23, same day):** built (86_rc9_undecided_state_detector.py),
> backfilled 2014-2026, validation 2 clean passes (2021-H2 DIFFERENTIATING
> called in real time; 2018-Q4 DISPERSAL), 1 partial (2023-25 mostly
> CONCENTRATED; 2024-08..10 DISPERSAL = yen-carry window), 1 PREMISE FAILURE
> (2020-21 read CONCENTRATED-bloc, not UNDECIDED: P stayed 0.77-0.94 —
> the pandemic cohort WAS decided; "many trends bid" was one liquidity
> factor, which C correctly flagged). No thresholds changed (per
> pre-registration). Promoted to month-end panel §[13] + rc9_state log
> field. Current reading 2026-08: CONCENTRATED (bloc) after a genuine
> UNDECIDED spell in Apr-May (P~0.45, B 50-60%, C 91st pct). Findings:
> archive/findings/rc9_undecided_state_findings.md.

### RC-10: Deep-crash post-earnings mean reversion ("reverse PEAD") (CLOSED 2026-08-23 at the probe)

- **Hypothesis (user):** the same event machinery in reverse — buy stocks that
  CRASHED after earnings, harvest the bounce.
- **Probe (10,422 events, 258-ticker sample, 2015-2026; honest entry = day+1
  CLOSE after the reaction is known; CAR vs IJH):**
  crash≥15% (n=441): rev5 +0.26% (t=0.6), rev10 −0.01%, rev20 +0.12% — flat;
  −15..−10%: negative everywhere; all other buckets ≈ −0.1..−0.3%, t<1.5.
  Yearly rev10 for crash≥10% oscillates around zero (no regime where it works).
  Quality interaction INVERTED: crashed-despite-beating (sue>0, the classic
  overreaction candidate) rev10 −0.46% (t=−1.3) — the quality screen makes it
  WORSE; crashed-and-missed +0.17% (t=0.4, nothing).
- **Reconciliation with Doc K's NEG-side bounce (script 15):** that bounce was
  real but is an Open[T+1]-entry artifact — the intraday open→close recovery
  on day 1 after a shallow gap (−2..−5%), completed by the first close. With
  close-based entry (our lane) there is nothing multi-day. Deep buckets were
  negative even at open entry (entry_pnl −3.71% for −5..−10%).
- **Mechanism:** short-term reversal is the most-arbitraged anomaly in the
  literature (microcap weekly, costs-eaten); SP400 midcaps are liquid enough
  for reversal desks to have cleaned it. Post-miss drift is the documented
  mirror of PEAD — continued selling (downgrade cascades, index rules), not
  bounce. Our own book shows the asymmetry live: winners drift (HRB +9.7%,
  ANF +29%), losers just bleed (DBX −2.4%, BILL −3.6%).
- **Doctrine closure:** post-event information classes now fully mapped —
  winners day5+ (RC-2: dead), losers day1+ (RC-10: dead). The only lane that
  ever paid is pre-print entry. RC-10 CLOSED at the probe; no pipeline built.

### RC-2: Post-event hold extension (CLOSED 2026-08-22: both confirmation legs fail)

> **Coverage probe (2026-08-22, 33,587 events × 807 grade nodes):** the
> analyst-upgrade trigger fires on only 5-6% of SP400 events (any action:
> 10-15%); 86% have NO grade action within 9 days post-earnings and 62% have
> zero actions in the prior 90 days. Research-measurable (~1,700 confirmed
> events) but operationally sparse (~3-4 extensions/yr on our pick volume).
> FMP transcripts are paywalled on our tier (402) — transcript tone not
> testable without a data decision. Architecture amended: price-based
> reaction is the primary trigger; upgrades demoted to secondary condition;
> transcript/8-K guidance NLP deferred.

> **KILL PROBE (2026-08-22, event study, 10,421 events / 310-ticker random
> sample, 2015-2026):** the price-based primary trigger ALSO fails. Day5->20
> abnormal return (vs IJH) by SUE tercile x entry->day5 reaction tercile:
> the "most confirmed" cohort (SUE_high x react_high, n=1,581) shows mean
> -0.39% / median -0.24% / 49% win; yearly means unstable and mostly
> negative (2016 -1.2%, 2020 -2.1%, 2026 -1.4%); best cell anywhere is
> SUE_mid x react_mid at +0.20% (noise; SE ~0.25%). The drift is EXHAUSTED
> by day 5 in this universe — extending confirmed winners past T+5 earns
> less than nothing. Survivorship bias in the sample (current members only)
> flatters continuation, so the true numbers are worse. Verdict: confirms
> the front-loaded-capture finding (script 52) and the force-refresh
> architecture (script 63/64); both extension triggers dead; transcript-NLP
> variant left untested (paywalled) with zero supporting evidence — not
> worth the data spend. RC-2 CLOSED.

- **Hypothesis (interaction, not raw):** our T+5 exit is unconditional, but the
  front-loaded capture average hides conditioning. The re-rating scenario —
  big earnings surprise (high SUE) followed by post-event analyst revision
  confirmation (≥1 upgrade in days 0–5) — should fatten the day 5→20 tail
  enough to justify holding confirmed winners past T+5. Raw revision momentum
  alone is only +0.46%/21d (script 46), far below force-refresh economics;
  only the surprise×revision interaction has a plausible shot.
- **Mechanism:** guidance revisions arrive in the earnings release itself
  (inside the surprise the gates score blind) and then ignite the analyst
  cascade; the extender harvests the slow-diffusion tail of the same PEAD
  event, not a separate edge. Survives the analysis-depth doctrine: requires
  interpretation, mid-caps, multi-day diffusion.
- **Architecture under test (shape 1 of 3):**
  ```text
  normal:    enter T close → exit T+5
  extension: enter T close → IF confirmation by day 5 (≥1 upgrade days 0–5,
             stop never hit) → hold to ~T+21; else exit T+5 as usual
  ```
  Alternatives if shape 1 fails: (2) exit T+5, re-enter on accelerating
  revision cascade day 5–10 (pays spread twice, cleaner slot math);
  (3) guidance-momentum as pre-event feature (weakest; duplicates
  revision_momentum — V5 precedent says standalone analyst data dies).
- **Honest baseline:** extension must beat FORCE-REFRESH at T+5 (mh=4, script
  64 methodology), not buy-and-hold. Fresh picks average +6.66%/5d, so the
  confirmed cohort's day 5→20 drift must beat a fresh pick's first 15 days.
- **Data:** already cached — 14y grades with dates (807 nodes, script 46),
  earnings/SUE from db.h5. No new FMP endpoints. Blocked cousin noted:
  consensus-revision timestamps (FMP /analyst-estimates lacks observation
  timestamps — point-in-time detection impossible; company-issued guidance
  endpoint untested, optional extension).
- **Method:** event study first (sort PEAD events 2015-2026 into
  confirmed/unconfirmed cohorts, day 5→20 CAR vs IJH per cohort per year);
  if fat + stable → full portfolio sim extension vs frozen force-refresh
  baseline; walk-forward + bootstrap CI; same promotion bar as everything
  else.
- **Kill criteria:** confirmed-cohort day 5→20 CAR ≈ unconfirmed (revision
  adds no conditioning info), or portfolio sim loses to force-refresh after
  costs at 50bps.

### RC-3: Polymarket whale-lifecycle tracker (design notes, venue-gated)

Design-stage only (2026-08-15, no data pulled). Killed at the Phase-0 legal
gate: France (ANJ) ISP-blocks Polymarket; venue risk not worth the gray zone
for the operator. Design is sound and preserved here for revival if the legal
state changes (or an EU-licensed public-ledger equivalent emerges; Kalshi
lacks public account data, which is disqualifying for this design).

Structure (slow institutional-style lifecycle tracker — competes on analysis
depth, not speed):

```text
Phase A WATCH     observe new wallets as they trade; log per-bet fill +
                  resulting crowd size (small same-direction order flood); no capital
Phase B VALIDATE  statistical skill bars (CLV persistence, shrinkage, min fills)
                  → monitored list
Phase C COPY      mirror validated wallets' bets ONLY while per-bet crowd size
                  is below a measured threshold band (the load-bearing param —
                  backtestable ex ante from on-chain crowd curves)
Phase D RETIRE    crowd size/acceleration crosses the alpha-exhausted line →
                  stop copying, return to watchlist
```

Key principles captured:
- crowd size is a STATE VARIABLE, not a race — enter where thin, exit on its
  acceleration (level is lagging; rate is leading); the whole crowd watches
  the same signal so exit thresholds race downward
- validation gates skill-vs-luck (a small crowd around a lucky wallet is a
  mini-bubble; flow profit + negative informational drift = bag holder)
- entry-price discipline: profit = whale's informational drift (only if entry
  near their fill) + crowd flow carrying the position; miss the fill wave and
  expectancy flips negative
- wallet rotation handled by wallet-graph clustering (funding sources,
  correlated timing) — track operators, not addresses
- legal-market analog: small-activist 13D coattailing in micro/small caps
  (obscure filers, untracked by the Ackman/Icahn crowd, positive abnormal
  returns documented around early 13D positions)

See `archive/findings/edge_landscape_memo.md` §2 #10 and §4 for the full
discussion (including the crowd-as-feature insight and the institutional
advantage map).

### RC-4: Megatrend watcher (design notes; OVERLAY for the 90% core book, not the PEAD sleeve)

Design-stage (2026-08-15). Long-horizon trend-following / time-series momentum
on sector clusters. NOT a slot-filler for the PEAD book — months-long holds are
architecturally incompatible with the 5-day 4-slot sleeve. Role: a WARNING /
ALLOCATION indicator governing a small carve-out of the core 90% portfolio
(real estate + S&P 500/blue-chip buy-and-hold, e.g. TotalEnergies, Danone).

**Premise (verified 2026-08-15, Tiingo):** megatrends persist for years because
institutional repositioning is constrained (career risk, benchmark hugging,
capacity, redemptions) — underreaction at macro scale. Being late is fine.

```text
NVDA 2022->2026: +650%, ATH-days/yr 1/17/47/28/6, max DD from ATH -62.7%
TSM  2022->2026: +256%, ATH-days/yr 4/0/24/28/23, max DD from ATH -56.5%
200d-MA rule beat buy&hold on BOTH (+228% vs +201% logret NVDA; +142% vs
+127% TSM) while invested only ~70% of days — the trivial-rule bar the
machinery must clear.
```

**Design skeleton:**
- L1 TREND DETECTION — cluster-level scoring (12m momentum, 200d-MA distance,
  ATH cadence/breadth) on sector ETFs + baskets. Price/breadth only; narratives
  ("AI bubble" chatter) are the LAST signal to arrive — never inputs.
- L2 LEADERSHIP — rank constituents within cluster (momentum + breadth).
- L3 RISK CLUSTERING — cluster by RETURN CORRELATION, not GICS labels (solves
  the multi-sector-company nuance automatically); correlated active trends =
  ONE risk position (AI compute + shovels/TSM/Samsung/memory = one cluster,
  one crash).
- L4 ENTRY — slow build / pyramiding (thirds on confirmation; adds on
  pullback-to-trend or new ATH bases).
- L5 EXIT — cluster-composite trend break (composite < 150/200d MA, breadth
  rollover). Fading is slow at CLUSTER level (distribution over months) but
  violent at stock level (halvings in weeks) — detect on the composite only.

**Honest classification:** this is documented trend-following (200 years of
evidence, survives publication) — a RISK PREMIUM, not a hidden crack. Expect
long flat periods (CTA lost decade 2011-2019) and momentum crashes
(Daniel-Moskowitz) when sharp reversals hit crowded trends. Long-only equity
variant: crashes = exit to cash, no short leg. Size assuming -50% cluster
drawdowns are NORMAL, not tail events.

**Validation bar (kill tests):**
- A — SURVIVORSHIP: run 2020-2026 across ALL candidate megatrends incl. the
dead (ARK-style innovation, crypto-adjacent, China tech, biotech, cleantech
2021): must exit dead ones with acceptable loss AND ride the live ones.
A backtest on winners only is a eulogy written in advance.
- B — REGIME TABLE: 2008, 2011, 2015, 2018Q4, 2020 crash, 2022, 2023 chop.
- C — TRIVIAL RULE BAR: machinery must beat 200d-MA-on-ETFs after costs; the
  trivial rule already beat buy&hold on the motivating examples.

**Deployment plan (user, 2026-08):** core 90% stays real estate + index/blue-
chip buy-and-hold; a SMALL carve-out of the core (not the PEAD sleeve) tilts
toward active megatrend clusters on signal, sized so cluster-crash scenarios
do not threaten the core. PEAD sleeve remains 5-8% of net worth, unchanged.

**Data:** in hand (Tiingo daily bars; sector ETFs cached in db.h5). Turnover
low — whipsaw, not friction, is the enemy.

**Phase-1 probe (when a research cycle opens):** the dead-trends exit test
(Kill test A) on ETF proxies + leaders, 2015-2026.

**Phase 1 RESULT (2026-08-16): KILL TEST A PASSED** — see
`archive/findings/megatrend_phase1_findings.md` + script `71_megatrend_phase1_dead_trends.py`.
Headline: trivial MA200 state machine exits 13 dead megatrends at −28.6% mean
giveback-from-peak (vs −62..−98% B&H drawdowns; MSTR +2157% vs +653%), keeps
80–88% of live AI-complex upside, cuts maxDD 17–26pp; MONTHLY-cadence variant
passes all criteria (13/13 DD-wins, 80% capture) → month-end decision cadence
is operationally viable. Caveats: ranging assets whipsaw (Phase-2 cluster
ranking handles), SPY/QQQ capture only 66–72% (overlay for trending clusters,
not whole portfolio). Phase 2 next: correlation clusters + composite scoring,
must beat this floor (Kill test C).

**Phase 2 RESULT (2026-08-16): KILL TEST C FAILED — cluster-selection layer
REJECTED; the simple floor is the strategy.** See
`archive/findings/megatrend_phase2_findings.md` + script
`72_megatrend_phase2_clusters.py`. Headline: cluster machinery +251% vs floor
+458% (risk win only: DD −17.4% vs −29.7%, 2022 +1%); the decisive
survivorship stress test — strip the 7 hand-picked mega-winners — collapses
the cluster strat (−11%, DD −49.3%) while the floor stays robust (+192%, DD
−21.6%). The floor's diversification across simultaneous trends is the robust
structure; top-cluster concentration is the fragile one. Retained: MA10m
equal-weight trend basket as the carve-out vehicle; cluster labels demoted to
descriptive semantics (which-trend reporting for the warning role).
**Phase 3 (redirected):** regime table + BROAD deployment universe (GICS
sectors + theme ETFs, structurally less survivor-flattered) for the FLOOR —
open question: does it beat SPY B&H with lower DD on the real universe?

**Phase 3 RESULT (2026-08-16): FAILED as strategy — SALVAGED as warning
indicator.** See `archive/findings/megatrend_phase3_findings.md` + script
`73_megatrend_phase3_broad_regime.py`. On the 26-asset category universe
(2006-2026): floor +245%/maxDD −33.4% vs SPY +784%/−50.8% — crash insurance
(2008: −11% vs −46%) but 13/20 years trail SPY, worst −20pp (2022). 2022
diagnosis: exit machinery works (basket empties at every break) but re-entry
whipsaw into bear rallies; dual-confirm improves 2008 (−7%) and worsens 2022
(−43%) — regime-structural, no in-family fix. ALSO: equal-weight-17-names is
NOT megatrend exposure (Phase-2's concentration WAS the thesis expression —
universe choice inverted the structural conclusion).

**RC-4 FINAL STATUS: panelized warning-indicator role adopted; strategy role closed.**
Deployed component = `05c_megatrend_watcher/monthly_panel_report.py`, a manual
monthly panelized dashboard: separate equity breadth, theme breadth,
cross-asset context, relative-capex, full-market insider, and news context.
Equity breadth remains the timing panel; all others are context only. Month-end
cadence, zero capital, no automatic orders.

**RC-4 Step 3 status (2026-08-16): normalization research started.** Script
`80_megatrend_normalize_insider_news.py` produces point-in-time monthly
features separately from Script 74. Insider uses exact open-market Form 4
P/S rows, deduplication, seller/buyer breadth, and within-company historical
percentiles; raw dollars are audit-only because no point-in-time market-cap
series is available. News uses company-day/category deduplication, a fixed
operational taxonomy, company-relative event percentiles, denominators,
coverage gates, and explicit missing provenance. Latest month passes neither
normalized warning candidate rule. No signal or allocation action is approved;
the next gate is a fixed-episode false-reentry test.

**RC-4 Step 4 result (2026-08-16): insider/news timing confirmation failed.**
The corrected expanding-panel test (`81_megatrend_false_reentry_test.py`) found
9 non-overlapping recovery episodes. Baseline relapse was 33.3%; the normalized
insider filter blocked 3 episodes with the same 33.3% relapse rate and blocked
66.7% profitable recoveries. The normalized news filter blocked zero episodes.
The 2008 observation is explicitly unavailable because the cache has fewer than
five valid equity names; 2020/2022 have only five. No insider/news filter is
promoted. `05c_megatrend_watcher/monthly_panel_report.py` remains the sole
deployed RC-4 component; normalized insider/news remain descriptive research
artifacts.
Three-phase funnel shape (exit pass → selection fail → deployment fail) is
the expected honest arc.

**RC-4 Cycle 1 reopened (2026-08-17): partial theme exposure research.** The
user's refined hypothesis is accepted as a new research question: 2022 should
be treated as an undecided, non-recessionary theme-discovery regime rather than
an automatic cash signal. Keep absolute exposure invested, rotate gradually
toward sustained price support, and reserve absolute de-risking for a separate
recession condition. Script `82_megatrend_partial_exposure_cycle1.py` tests
fixed equal-theme, price-only rotation, bounded price+point-in-time-capex
rotation, and a recession-overlay variant. Theme weights use a 10% floor, 70%
cap, and 10pp monthly movement limit; capex is a bounded sponsorship prior, not
an exit trigger. Preliminary gross proxy results are archived in
`archive/findings/megatrend_partial_exposure_cycle1_findings.md`; they are not
approved for operations. The first run also caught and corrected an invalid
benchmark comparison before interpretation. Next gate: walk-forward, theme
floor/cap/step sensitivities, gradual absolute-exposure ladder, costs, bootstrap
and regime stability. Operational Script 74 replacement remains manual and
unchanged.

**RC-4 Cycle 2 result (2026-08-17): robustness grid informative, promotion gate
not cleared.** Script `83_megatrend_partial_exposure_cycle2_robustness.py`
tested floors 5/10/15%, caps 50/70/90%, monthly steps 5/10/15%, price-only vs
price+point-in-time-capex, and 0/25/50/100bp turnover costs across fixed
2014-19, 2020-22, and 2023-26 blocks. The 50% cap was generally more stable
than 70–90%; 10–15% steps were more responsive than 5%; price+capex remained
better than price-only in the 2020–22 block. The fixed recession overlay still
sacrificed too much 2020 recovery. No cell cleared the strict all-block gate;
no operational allocation rule changed. Findings:
`archive/findings/megatrend_partial_exposure_cycle2_findings.md`.

**RC-4 short-list gate (2026-08-17): descriptive stability passed; promotion
still deferred.** Script `84_megatrend_partial_exposure_short_gate.py` tested
only four fixed candidates (price/cap 50/step 10, price/cap 50/step 15,
price+capex/cap 50/step 10, price+capex/cap 70/step 10) at 0/50/100bp costs.
All four were positive in each fixed calendar block at 50bp and all survived
through 100bp; 2022 losses remained severe at approximately -37% to -39%.
The price+capex 70% cap had the strongest gross sample profile but is not
selected post hoc. Bootstrap was run on absolute return only and is not
superiority or forward-validation evidence. Next gate: benchmark excess-return
and drawdown bootstrap, rolling chronology, proxy/missing-capex stress, and
leave-one-theme-out testing. Manual operational watcher remains unchanged.

**RC-4 final benchmark/stress gate (2026-08-17): return advantage, risk failure.**
Script `85_megatrend_partial_exposure_final_stress_gate.py` compared A–D with
SPY/static 60/40, 50/100bp costs, 3/6/12-month capex delays, missing capex,
rolling 36-month excess return, block bootstrap, and leave-one-theme-out stress.
All four beat SPY on cumulative return in the 2014–2026 proxy sample, but all
had materially worse active drawdown (~-43% to -46% vs SPY ~-24%). D remained
strongest; its edge survived capex delays but degraded with missing capex. The
AI leave-one-theme-out test made D underperform SPY and showed worse drawdown.
Conclusion: partial rotation is an interesting high-risk thematic portfolio,
not a validated low-risk core-book overlay. No automatic or manual allocation
rule is promoted; the watcher remains the operational endpoint. Findings:
`archive/findings/megatrend_partial_exposure_final_stress_gate_findings.md`.

**Phase 2b/2c (2026-08-16, post-closure refinements): gradual momentum pivot
REJECTED + capex tilt REJECTED.** See
`archive/findings/megatrend_phase2b2c_findings.md` + scripts `75`/`76`.
Pivot (k-swept momentum tilt): +449%/−33.0% vs floor +390%/−29.7% full; worse
on BOTH dims ex-winners (+141%/−26.8%) — maximal trailing momentum = maximal
weight immediately before death; gradualness does not fix fattest-at-top.
Capex probe: user's scale hypothesis CONFIRMED (AI capex 174–401x crypto,
$574B TTM 2026) but the natural experiment kills the tilt role — clean-energy
capex ROSE +13% into ICLN's −43%..−58% (peaked 3 years after price; rolled
over only 2024) = capex LAGS at death (capital-cycle oversupply mode).
Salvage: capex as descriptive context in the breadth report (never selection).
General law added: for long-only trend participation, ANY confirmation signal
is procyclical at trend death — the confirmation IS the crowding. Pivot path
closed at every layer (binary P2 / gradual 2b / fundamental 2c).
`archive/findings/megatrend_phase1_findings.md` + script `71_megatrend_phase1_dead_trends.py`.
Headline: trivial MA200 state machine exits 13 dead megatrends at −28.6% mean
giveback-from-peak (vs −62..−98% B&H drawdowns; MSTR +2157% vs +653%), keeps
80–88% of live AI-complex upside, cuts maxDD 17–26pp; MONTHLY-cadence variant
passes all criteria (13/13 DD-wins, 80% capture) → month-end decision cadence
is operationally viable. Caveats: ranging assets whipsaw (Phase-2 cluster
ranking handles), SPY/QQQ capture only 66–72% (overlay for trending clusters,
not whole portfolio). Phase 2 next: correlation clusters + composite scoring,
must beat this floor (Kill test C).

### RC-2: Senate-trade composite (parked, unlikely)

Real drift exists (10–20d CAR +2.13% from disclosure close) but supply
(~4.5 events/mo universe-wide, ~1.7 in September), 28-day median staleness,
and horizon mismatch (5-day architecture vs 10–20d drift) park it.
As a PEAD *feature* it also fails: ~1–2% row coverage → XGBoost learns noise.
Revisit only if disclosure rules shorten (real-time filing legislation) or a
10–20d horizon variant is ever approved.
