# Quantitative Architecture Specification: PEAD Algorithmic Trading Bot
## Target Universe: S&P 400 Mid-Cap (approximately 400 Assets)

## 1. Target Variable Definition (The ML Objective)
*   **Core Goal:** Do not predict binary earnings surprises or nominal absolute price moves. Predict institutional drift (Idiosyncratic Alpha / Market-Neutral Abnormal Returns).
*   **Target Metric:** Cumulative Abnormal Return ($CAR$) measured from $T+1$ to $T+11$ (10-day holding horizon), where $T$ is the earnings announcement date.
*   **Mathematical Formula:**
    $$CAR_{i} = \sum_{t=T+1}^{T+11} \left( R_{i,t} - R_{m,t} \right)$$
    *   $R_{i,t}$: Daily log return of stock $i$ on day $t$.
    *   $R_{m,t}$: Daily log return of the stock’s corresponding sector ETF or broad market proxy on day $t$.
*   **Model Type:** XGBoost Regressor (Continuous point estimation of $CAR$). Do not use classification frameworks; optimization gradients must reflect the precise magnitude of structural anomalies.

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

## 9. Data Pipeline & Survivorship Bias Architecture
* **Storage Framework:** Data must be saved in an unified HDF5 storage format (`.h5`) organized hierarchically by ticker nodes or a unified tabular format optimized for time-series querying.
* **Historical Membership Resolution:** The data ingestion script must parse the Wikipedia S&P 400 Revision History/Changes matrix to map the exact historical entry and exit windows for all historical constituents over a 15-year lookback.
* **The Rebalance Exclusion Rule:** Implement a strict rolling timeline guardrail. A stock's row data is only valid for XGBoost training or weekday signal generation if the current date is $\ge 270\text{ days}$ (3 quarters) past its official index addition date. Mark this eligibility using a boolean flag `in_index_clean` inside the HDF5 matrix.

----

## 10. Training Matrix Extraction Protocol
* **The Row Selection Filter:** When constructing the training matrix (`X`, `y`) for the XGBoost model, the dataset generator must iterate through the historical timelines and only select rows where `in_index_clean == True`.
* **Dynamic Historical Matching:** * If a company was added in 2022, its rows prior to 2022 + 270 days must be excluded from training.
  * If a company was deleted in 2022, its rows prior to 2022 (and after its original inclusion date) must be included in training, while all rows post-2022 must be completely ignored.

---

## 11. Earnings Surprise Ingestion Protocol
* **Data Sourcing:** The data pipeline must query a structured estimates API (e.g., Finnhub or Financial Modeling Prep) to pull historical quarterly timelines of `Actual_EPS`, `Estimated_EPS`, and `Report_Date`.
* **HDF5 Alignment:** Store the parsed earnings metrics under a secondary dataset named `/earnings_history` inside each respective ticker node.
* **Feature Calculation Rule:** Do not pass raw surprise dollar values to XGBoost. The pipeline must compute cross-sectional rolling features (`SUE_trend_1Y`, etc.) by dividing the historical deviations by the rolling 4-quarter standard deviation of past surprises to stabilize the feature variance.