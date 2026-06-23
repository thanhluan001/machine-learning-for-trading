# Data Ingestion & Micro-Regime Alignment Protocol

This script defines how to build the raw training matrix by merging the refined daily FRED macro indicators, the bulk sector ETFs, and your local stock price tables onto your primary corporate earnings events logs.

## 1. Source Reference Maps
* **FRED API Tickers:** `DFF` (Fed Funds), `T10Y2Y` (Yield Spread), `VIXCLS` (VIX alternate/backup).
* **YFinance Bulks:** Broad Market (`SPY`), Mid-Cap Benchmark (`IJH`), Sector ETFs (`XLK`, `XLF`, `XLI`, `XLY`, `XLP`, `XLV`, `XLU`, `XLE`, `XLB`, `XLRE`, `XLC`).

---

## 2. Production Data Merging Loop (Python Blueprint)

```python
import pandas as pd
import numpy as np

def align_and_merge_pipeline(earnings_events_path, sectors_h5_path, fred_csv_path):
    """
    Loads raw tables, cleans FRED data frequency, handles index matching, 
    and outputs a unified base matrix completely avoiding rate-limit loops.
    """
    # Step 1: Load your core corporate earnings event history (Your target base)
    # Must contain: Date, Ticker, actual_eps, expected_eps, actual_rev, expected_rev
    events = pd.read_csv(earnings_events_path)
    events['Date'] = pd.to_datetime(events['Date'])
    events = events.sort_values('Date')

    # Step 2: Load the bulk-downloaded Sector ETF Daily Prices
    sectors = pd.read_hdf(sectors_h5_path, key='daily_bars')
    
    # Step 3: Load and clean FRED macro table
    fred = pd.read_csv(fred_csv_path, parse_dates=['DATE'], index_col='DATE')
    fred = fred.rename(columns={
        'DFF': 'fed_funds_rate',
        'T10Y2Y': 'yield_curve_spread',
        'VIXCLS': 'vix_close_fred'
    })
    # Force numeric conversion and forward-fill any rare holiday/weekend gaps
    fred = fred.apply(pd.to_numeric, errors='coerce').ffill()

    # Step 4: Calculate 20-day returns for SPY and Sector ETFs ahead of time
    macro_trends = pd.DataFrame(index=sectors.index)
    macro_trends['spy_ret_20d'] = np.log(sectors[('SPY', 'Close')] / sectors[('SPY', 'Close')].shift(20))
    
    # Pre-calculate 20d returns for every single sector ETF
    etfs = ['XLK', 'XLF', 'XLI', 'XLY', 'XLP', 'XLV', 'XLU', 'XLE', 'XLB', 'XLRE', 'XLC']
    for etf in etfs:
        macro_trends[f'{etf}_ret_20d'] = np.log(sectors[(etf, 'Close')] / sectors[(etf, 'Close')].shift(20))
    
    # Step 5: Stitch Macro States onto Earnings Events via an As-Of Join
    # Ensures no lookahead data leakage occurs from weekend/holiday date offsets
    aligned_events = pd.merge_asof(
        events,
        fred,
        left_on='Date',
        right_index=True,
        direction='backward'
    )
    
    aligned_events = pd.merge_asof(
        aligned_events,
        macro_trends,
        left_on='Date',
        right_index=True,
        direction='backward'
    )
    
    return aligned_events