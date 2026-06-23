# Feature Engineering Matrix Protocol: 3-Year Memory

This document outlines the strict feature schema for training and Sunday inference pipelines.

**Important:** The **3-year rolling eligibility window** means that when constructing a training example for an event in 2024-Q2, you may only use historical data back to 2021-Q2. This is a **data availability constraint** (to avoid regime-contaminated old data), **not** a feature lookback. Most features use only the most recent quarter or a short trailing window within that 3-year boundary.

---

## 1. Feature Categories & Column Mapping

### Block 1: Catalyst Fundamentals
These features quantify the intensity and surprise vectors of the earnings catalyst. All are calculated from data within the 3-year eligibility window.

* `sue_score` (Float): Standardized Unanticipated Earnings. Calculated as: (Actual EPS - Consensus EPS) / Standard Deviation of Analyst Estimates.
* `rev_growth_yoy` (Float): Year-over-Year revenue growth percentage change for the reported quarter.
* `eps_surprise_pct` (Float): Raw percentage deviation of reported EPS from the consensus estimate.
* `consecutive_surprises` (Integer): The running count of consecutive quarters the company has beaten consensus EPS estimates leading up to this event.
* **`sue_acceleration`** (Float): **`Q-on-Q change in SUE score`** (`SUE_current` - `SUE_prev_quarter`). Captures whether the earnings quality trajectory is improving or deteriorating - **promoted to core**.
* **`sue_lag_1`** (Float): **SUE score from the previous quarter (Q-1)** for this same ticker. Captures SUE surprise persistence - a series of consistent beats/surprises may signal systematic analyst mis-calibration.
* **`sue_lag_2`** (Float): **SUE score from two quarters ago (Q-2)** for this same ticker. Extends the persistence check back through two full reporting cycles.
* **`car_drift_historical_q1`** (Float): **Actual index-adjusted CAR generated during the stock's previous post-earnings window last quarter.** Tests whether a stock has a repeatable PEAD signature - some companies consistently drift while others mean-revert.

### Block 2: Microstructure & Technical Context
These features identify the execution environment, institutional presence, and positioning constraints on the day of the release.

* `is_bmo` (Binary: 0 or 1): Execution timing tag. `1` if the report dropped Before Market Open. `0` if it dropped After Market Close.
* `volume_vma20_ratio_pre_event` (Float): Trading volume on Day T relative to its rolling 20-day average. Measures pre-announcement positioning intensity.
* `short_interest_pct_float` (Float): The percentage of the company's floating shares currently held short. Captures short-squeeze potential during the drift.
* **`suv_day_1`** (Float): **`[ Volume(T) / mean(Volume T-20 to T-1) ]`** - the Standardized Unexpected Volume, measuring abnormal volume on the event day itself. A proxy for hidden institutional flow - **promoted to core**.
* **`pre_event_idiosyncratic_vol`** (Float): **`std( stock_ret - IJH_ret )`** over the 20-day window `(T-20 to T-1)`. Captures the stock's risk profile after stripping out broad mid-cap market volatility - **promoted to core**.
* **`opening_gap_t1`** (Float): **`(Open_{T+1} - Close_T) / Close_T`** - the overnight price gap after the earnings announcement. Encodes the market's first-pass surprise assessment.
* **`intraday_range_t`** (Float): **`(High_T - Low_T) / Close_T`** - the normalized intraday range on the event day. Wide range = high uncertainty, often precedes larger drift resolution.
* **`pre_event_volume_trend`** (Float): **Slope of volume from T-10 to T-1** (via linear regression). Rising pre-event volume suggests information leakage.

### Block 3: Multi-Horizon Market & Sector-Adjusted Technicals
These features measure the stock's absolute and relative velocity across multiple timeframes leading into the event. Sector metrics are relative to the company's mapped GICS Sector ETF.

* `rel_ret_3d` / `5d` / `10d` / `20d` / `30d` / `60d` (Float): Cumulative log returns of the stock minus the cumulative log returns of the mid-cap index proxy (`IJH`) over the respective lookback days.
* `sector_adjusted_ret_20d` (Float): The stock's 20-day cumulative return minus its corresponding GICS Sector ETF return (e.g., `XLK`, `XLF`, `XLI`) over the exact same window. Isolates idiosyncratic strength from industry-wide momentum.

### Block 4: Macro Environment & Regime Filters
These features provide broad market context, allowing the classifier to adapt its probability output based on prevailing systemic risks.

* `vix_close` (Float): The closing price of the CBOE Volatility Index on Day T-1. Represents market fear levels.
* `fed_funds_rate` (Float): The effective Federal Funds Rate on the day of the event (sourced via FRED).
* `yield_curve_spread` (Float): The 10-Year Treasury Yield minus the 2-Year Treasury Yield. Captures macroeconomic growth expectations.
* `spy_momentum_20d` (Float): The 20-day rolling return of the S&amp;P 500 index proxy. Captures the primary market trend vector.

### Block 5: Derived Interaction Terms
*Optional. Add only if base model underperforms.*

* `sue_x_momentum_5d` (Float): `sue_score * rel_ret_5d` - Tests whether strong earnings surprise amplifies or is offset by pre-existing price momentum.
* **`sue_abs_x_inverse_vol`** (Float): **`|sue_score| / pre_event_idiosyncratic_vol`** - Surprise-to-risk ratio. Filters noise when idiosyncratic volatility is high - **promoted to core**.
* `volume_ratio_x_sue` (Float): `suv_day_1 * sue_score` - High surprise + abnormal volume = high conviction. Currently optional.

---

## 2. Programmatic Feature Matrix Builder (Python Blueprint)

```python
import pandas as pd
import numpy as np

# --- DEBUGGING SUPPORT ---
# NaN handling policy for XGBoost:
#   - Do NOT drop rows with missing features.
#   - XGBoost natively handles NaN via optimal default split direction.
#   - Ensure all data types are float or int (no object/str columns in DMatrix).
#   - If a feature is completely unresolvable (e.g., no historical short interest),
#     backfill with a systematic constant (e.g., rolling 4Q mean, or -999.0).


def build_feature_matrix(event_df, market_prices, sector_prices, macro_df):
    """
    Combines corporate event logs with market, sector, and macro data arrays.
    Enforces a strict historical feature construction format.
    
    Args:
        event_df (pd.DataFrame):            Columns include ['Date', 'ticker',
                                             'sue_score', 'eps_surprise_pct', 
                                             'consecutive_surprises', 'rev_growth_yoy',
                                             'is_bmo', 'vol_ratio_t0', ...]
        market_prices (pd.DataFrame):       IJH daily OHLCV for all dates in scope.
        sector_prices (dict[str, pd.DF]):   Mapping of GICS Sector ETF ticker to 
                                             daily OHLCV.
        macro_df (pd.DataFrame):            FRED / CBOE macro data indexed by date.
    
    Returns:
        pd.DataFrame: Feature matrix with the exact column order specified above.
    """
    features = pd.DataFrame(index=event_df.index)

    # -----------------------------------------------------------------------
    # 1. Block 1: Catalyst Fundamentals
    # -----------------------------------------------------------------------
    features['sue_score']              = event_df['sue_score']
    features['rev_growth_yoy']         = event_df['rev_growth_yoy']
    features['eps_surprise_pct']       = event_df['eps_surprise_pct']
    features['consecutive_surprises']  = event_df['consecutive_surprises']

    # SUE Acceleration: Q-on-Q change (use .shift() within grouped ticker)
    features['sue_acceleration'] = (
        event_df.groupby('ticker')['sue_score'].diff().values
    )

    # SUE lag features (lagged by 1 and 2 quarters)
    features['sue_lag_1'] = event_df.groupby('ticker')['sue_score'].shift(1)
    features['sue_lag_2'] = event_df.groupby('ticker')['sue_score'].shift(2)

    # Historical PEAD signature (previous quarter's actual CAR)
    features['car_drift_historical_q1'] = event_df.groupby('ticker')['car'].shift(1)

    # -----------------------------------------------------------------------
    # 2. Block 2: Microstructure & Technical Context
    # -----------------------------------------------------------------------
    features['is_bmo']                           = event_df['is_bmo'].astype(int)
    features['short_interest_pct_float']           = event_df['short_interest_float']
    features['volume_vma20_ratio_pre_event']      = event_df['vol_ratio_t0']

    # SUV Day 1: event-day volume spike vs trailing 20-day avg
    #   Assumes event_df has columns ['volume_t', 'volume_avg_20d']
    features['suv_day_1'] = (
        event_df['volume_t'] / event_df['volume_avg_20d']
    )

    # Pre-event idiosyncratic vol: std(residual return) over T-20 to T-1
    #   Residual = stock_ret - IJH_ret
    #   Assumes event_df has column 'residual_std_20d'
    features['pre_event_idiosyncratic_vol'] = event_df['residual_std_20d']

    # -----------------------------------------------------------------------
    # 3. Block 3: Multi-Horizon Market & Sector-Adjusted Technicals
    # -----------------------------------------------------------------------
    # IJH-relative momentum at multiple lookbacks
    horizons = [3, 5, 10, 20, 30, 60]
    for h in horizons:
        features[f'rel_ret_{h}d'] = (
            event_df[f'stock_ret_{h}d'] - event_df[f'ijh_ret_{h}d']
        )

    # Sector-relative strength (20d)
    features['sector_adjusted_ret_20d'] = (
        event_df['stock_ret_20d'] - event_df['sector_etf_ret_20d']
    )

    # -----------------------------------------------------------------------
    # 4. Block 4: Macro Environment & Regime Filters
    # -----------------------------------------------------------------------
    features = pd.merge_asof(
        features.sort_index(), 
        macro_df[['vix', 'fed_rate', 'yield_spread', 'spy_ret_20d']].sort_index(),
        left_index=True, 
        right_index=True, 
        direction='backward'
    )

    # -----------------------------------------------------------------------
    # 5. Block 5: Derived Interaction Terms (Optional)
    # -----------------------------------------------------------------------
    # These are calculated *after* the base blocks to avoid leakage.
    # Uncomment / add only if the base model underperforms in backtesting.
    
    # features['sue_x_momentum_5d']       = features['sue_score'] * features['rel_ret_5d']
    features['sue_abs_x_inverse_vol']   = features['sue_score'].abs() / features['pre_event_idiosyncratic_vol']
    # features['volume_ratio_x_sue']      = features['suv_day_1'] * features['sue_score']

    return features
```

---

## 3. Sunday Inference Pipeline Requirement

When generating predictions on Sunday afternoons, the extraction script must populate this identical feature vector matrix for the upcoming week's cohort.

Any missing column will invalidate the DMatrix state and cause the `model.predict_proba()` scoring module to throw a structural size exception. If option-implied data is unresolvable via API for low-liquidity mid-caps, the bot must fallback to calculating the asset's rolling 20-day historical standard deviation (`historical_volatility_20d`) as a systematic imputation anchor.

## 4. Feature Summary Table

| # | Block | Feature | Calculation | Priority |
|---|-------|---------|-------------|----------|
| 1 | 1 | `sue_score` | (Actual EPS - Consensus) / std(Analyst Estimates) | **Must** |
| 2 | 1 | `sue_acceleration` | `sue_score_current` - `sue_score_prev_q` | **Must** |
| 3 | 1 | `sue_lag_1` | SUE score from Q-1 | **Must** |
| 4 | 1 | `sue_lag_2` | SUE score from Q-2 | **Must** |
| 5 | 1 | `car_drift_historical_q1` | Previous quarter's actual CAR | **Must** |
| 6 | 1 | `consecutive_surprises` | Running count of consecutive EPS beats | **Must** |
| 7 | 1 | `eps_surprise_pct` | Raw surprise % | **Must** |
| 8 | 1 | `rev_growth_yoy` | YoY revenue change | **Must** |
| 9 | 2 | `is_bmo` | 1 if report BMO, 0 if AMC | **Must** |
| 10 | 2 | `volume_vma20_ratio_pre_event` | `Vol(T)` / MA20(Vol) | **Must** |
| 11 | 2 | `short_interest_pct_float` | Short interest / float | **Must** |
| 12 | 2 | `suv_day_1` | `Vol(T)` / trailing 20d avg | **Must** |
| 13 | 2 | `pre_event_idiosyncratic_vol` | std(stock_ret - IJH_ret, 20d) | **Must** |
| 14 | 2 | `opening_gap_t1` | `(Open_{T+1} - Close_T) / Close_T` | **Try** |
| 15 | 2 | `intraday_range_t` | `(High_T - Low_T) / Close_T` | **Try** |
| 16 | 2 | `pre_event_volume_trend` | Slope of volume T-10 to T-1 | **Try** |
| 17 | 3 | `rel_ret_{3,5,10,20,30,60}d` | Stock log return - IJH log return | **Must** |
| 18 | 3 | `sector_adjusted_ret_20d` | Stock return - Sector ETF return | **Must** |
| 19 | 4 | `vix_close` / `fed_funds_rate` / `yield_curve_spread` / `spy_momentum_20d` | Macro snapshots at T-1 | **Must** |
| 20 | 5 | `sue_abs_x_inverse_vol` | `abs(sue_score) / pre_event_idiosyncratic_vol` - promoted to core | **Must** |
| 21 | 5 | `sue_x_momentum_5d` / `volume_ratio_x_sue` | Cross-product interactions (remaining optional) | **Optional** |

*Total base features: **24**. With optional Block 5: **26**.*
