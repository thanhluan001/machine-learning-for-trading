# Strategy Verification & Backtesting Protocol: PEAD Anomaly

This document establishes the technical blueprint for **Step 2 (Verify)** of the algorithmic trading system development cycle. The objective is to programmatically isolate, measure, and verify the existence of true Post-Earnings Announcement Drift (PEAD) alpha within the S&P 400 universe before writing live automation scripts.

## 1. Data Requirements & Sample Framework
* **Universe Constraint:** S&P 400 Mid-Cap Index constituents (`IJH` proxy).
* **Historical Timeframe Horizon:** 2 to 3 years of continuous historical data (yielding approximately 3,200 to 4,800 total corporate earnings events).
* **Data Fields Required:**
  * **Earnings Calendar Matrix:** Historical ticker symbols, actual reporting dates ($T$), actual reported EPS, and consensus analyst estimates.
  * **Pricing Matrix:** Split/dividend-adjusted Daily Open, High, Low, Close (OHLC), and Volume data for both the target tickers and the benchmark proxy (`IJH`).

## 2. Event-Window Vectorization Schema
For every historical earnings event row extracted from the calendar API, the verification script must map out a rolling execution window array using a dual index timeline anchor:

```text
       [ PRE-EVENT MOMENTUM ]               [ POST-EARNINGS DRIFT TARGET WINDOW ]
  T-5 ──────────────────────► T (Report) ────────────────────────────────────────► T+11
   │                          │          │                                         │
   └──────── 5 Days ──────────┘          └─────────────── 10 Trading Days ─────────┘
```

* **$T$ (Event Anchor Day):** The physical trading date of the earnings release. If a company reports post-market, $T$ must automatically roll forward to the next active trading day.
* **Pre-Event Momentum Array:** $[T-5 \rightarrow T]$
* **Post-Event Execution Tracking Array:** $[T+1 \rightarrow T+11]$ (Standard 10-day active holding period).

## 3. Core Feature & Label Math Calculations

### A. Pre-Event Momentum ($Mom_{5D}$)
Captures the percentage change of the stock relative to its event anchor point to identify speculative "buy the rumor" run-ups or depressed markdown cycles:
$$Mom_{5D} = \frac{\text{Close}_{T}}{\text{Close}_{T-5}} - 1$$

### B. Standardized Unanticipated Earnings ($SUE$)
Establishes the normalized catalyst surprise scale score (if analyst standard deviations are unavailable, substitute with a 4-quarter rolling historical surprise deviation matrix denominator $\sigma_{\Delta \text{EPS}}$):
$$SUE = \frac{\text{Reported EPS}_T - \text{Estimated EPS}_T}{\sigma_{\text{Surprise}(4Q)}}$$

### C. Cumulative Abnormal Return ($CAR$)
The structural target training label ($y$). Strips out basic market beta by subtracting the broad mid-cap index proxy log returns from the stock asset log returns across the holding timeline:
$$CAR_{T+1 \rightarrow T+11} = \sum_{t=T+1}^{T+11} \left[ \ln\left(\frac{\text{Close}_{t}}{\text{Close}_{t-1}}\right) - \ln\left(\frac{\text{IndexClose}_{t}}{\text{IndexClose}_{t-1}}\right) \right]$$

### D. Market-Adjusted Maximum Drawdown ($MaxDD_{MA}$)
A critical capital preservation metric designed to explicitly flag and discard stop-loss traps and immediate post-print "head-fake" liquidations:
$$MaxDD_{MA} = \min_{t \in [T+1, T+11]} \left( \left[\frac{\text{Close}_{t}}{\text{Close}_{T}} - 1\right] - \left[\frac{\text{IndexClose}_{t}}{\text{IndexClose}_{T}} - 1\right] \right)$$

---

## 4. The Three Mathematical Verification Gates
A historical event row is classified as a **"True PEAD Anomaly Setup"** if and only if it simultaneously clears three strict numerical constraints. If it fails any gate, the verification script flags it as structural noise or a trading loss.

### Gate 1: The Idiosyncratic Alpha Hurdle
The asset must deliver clear, uncorrelated outperformance over the benchmark proxy within the 10-day drift horizon.
$$\text{Condition: } CAR_{T+1 \rightarrow T+11} > +3.0\%$$

### Gate 2: The Institutional Accumulation Volume Footprint
Volume must confirm that institutional execution algorithms are active across the market tape, breaking past normal background noise levels.
$$\text{Condition: } \left(\frac{\text{Volume}_{T} + \text{Volume}_{T+1} + \text{Volume}_{T+2}}{3}\right) > 2.0 \times \text{Rolling 20-Day Volume Moving Average}$$

### Gate 3: The Risk & Capital Preservation Barrier
The position must not have violated protective risk boundaries during its holding pathway. This filters out the V-shaped delayed reactions that would cause live stop-loss account liquidations.
$$\text{Condition: } MaxDD_{MA} > -1.5\%$$

---

## 5. Reference Verification Script Skeleton (Python)

```python
import pandas as pd
import numpy as np

def run_pead_verification_pipeline(calendar_df, hdf5_price_store):
    """
    Vectorized verification loop designed to parse historical datasets 
    and establish base-rate success statistics for the PEAD anomaly.
    """
    verification_results = []
    
    for idx, event in calendar_df.iterrows():
        ticker = event['symbol']
        t_zero = pd.to_datetime(event['reportDate'])
        
        # Pull required slice array from local storage safely
        try:
            stock_data = hdf5_price_store.select(f"/{ticker}", where="index >= t_zero - pd.Timedelta(days=15) and index <= t_zero + pd.Timedelta(days=20)")
            index_data = hdf5_price_store.select("/IJH", where="index >= t_zero - pd.Timedelta(days=15) and index <= t_zero + pd.Timedelta(days=20)")
            
            # Align exact sequential trading days
            t_zero_idx = stock_data.index.get_loc(t_zero, method='nearest')
            
            # Extract historical location windows
            pre_window = stock_data.iloc[t_zero_idx-5 : t_zero_idx+1]
            post_window = stock_data.iloc[t_zero_idx+1 : t_zero_idx+12]
            idx_post_window = index_data.iloc[t_zero_idx+1 : t_zero_idx+12]
            
            # Calculate Return metrics
            stock_holding_return = (post_window['close'].iloc[-1] / post_window['close'].iloc[0]) - 1
            index_holding_return = (idx_post_window['close'].iloc[-1] / idx_post_window['close'].iloc[0]) - 1
            car = stock_holding_return - index_holding_return
            
            # Calculate Volume Ratio
            rolling_vol_avg = stock_data['volume'].iloc[t_zero_idx-20 : t_zero_idx].mean()
            event_vol_avg = post_window['volume'].iloc[0:2].mean()
            volume_ratio = event_vol_avg / rolling_vol_avg
            
            # Calculate Worst Path Drawdown
            stock_pct_path = (post_window['close'] / stock_data['close'].iloc[t_zero_idx]) - 1
            idx_pct_path = (idx_post_window['close'] / index_data['close'].iloc[t_zero_idx]) - 1
            ma_drawdown_path = stock_pct_path - idx_pct_path
            max_dd = ma_drawdown_path.min()
            
            # Evaluate Verification Gates
            is_true_pead = (car > 0.03) and (volume_ratio > 2.0) and (max_dd > -0.015)
            
            verification_results.append({
                'ticker': ticker,
                'event_date': t_zero,
                'car_10d': car,
                'max_drawdown': max_dd,
                'volume_ratio': volume_ratio,
                'is_true_pead_event': is_true_pead
            })
            
        except KeyError:
            continue  # Skip if ticker data or index boundary alignment drops a row
            
    return pd.DataFrame(verification_results)
```
