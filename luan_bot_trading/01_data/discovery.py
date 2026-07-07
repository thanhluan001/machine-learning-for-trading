#!/usr/bin/env python3
"""
Discovery Notebook / Script - PEAD Data Exploration
=====================================================
Helper functions for calculating CAR, loading data, and examining
single-stock behavior around earnings announcement dates.

Usage:
    import discovery
    # or run as script: python discovery.py
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Tuple

# ==============================================================================
# CONFIGURATION
# ==============================================================================

DB_FILE = Path(__file__).parent / "db.h5"
H5_STOCKS = "sp400"
H5_MACRO = "macros"
IJH_TICKER = "IJH"


# ==============================================================================
# LOADING HELPERS
# ==============================================================================

def load_ticker(ticker: str, start: str = None, end: str = None) -> pd.DataFrame:
    """
    Load historical OHLCV for a single ticker from the local HDF5 database.
    
    Args:
        ticker: Stock ticker (e.g., 'AAPL')
        start: Optional start date in 'YYYY-MM-DD' format
        end:   Optional end date in 'YYYY-MM-DD' format
    
    Returns:
        DataFrame with columns: Date, Open, High, Low, Close, Volume
    """
    path = f"/{H5_STOCKS}/{ticker}"
    df = pd.read_hdf(DB_FILE, path)
    
    if start:
        df = df[df["Date"] >= pd.to_datetime(start)]
    if end:
        df = df[df["Date"] <= pd.to_datetime(end)]
    
    return df.sort_values("Date").reset_index(drop=True)


def load_ijh(start: str = None, end: str = None) -> pd.DataFrame:
    """
    Load IJH (S&P Mid-Cap 400 ETF) benchmark data.
    """
    path = f"/{H5_MACRO}/{IJH_TICKER}"
    df = pd.read_hdf(DB_FILE, path)
    
    if start:
        df = df[df["Date"] >= pd.to_datetime(start)]
    if end:
        df = df[df["Date"] <= pd.to_datetime(end)]
    
    return df.sort_values("Date").reset_index(drop=True)


def load_both(ticker: str, проблемаd: list = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load both a stock and the IJH benchmark, aligned on dates.
    """
    stock_df = load_ticker(ticker, start, end)
    ijh_df = load_ijh(start, end)
    
    # Ensure same date range
    common_dates = set(stock_df["Date"]).intersection(set(ijh_df["Date"]))
    
    stock_df = stock_df[stock_df["Date"].isin(common_dates)].sort_values("Date").reset_index(drop=True)
    ijh_df = ijh_df[ijh_df["Date"].isin(common_dates)].sort_values("Date").reset_index(drop=True)
    
    return stock_df, ijh_df


# ==============================================================================
# CALCULATION HELPERS
# ==============================================================================

def calc_log_return(df: pd.DataFrame, price_col: str = "Close") -> pd.Series:
    """
    Calculate daily log returns from a price column.
    
    Formula: ln(Price_t / Price_{t-1})
    """
    prices = df[price_col]
    log_returns = np.log(prices / prices.shift(1))
    return log_returns


def calc_abnormal_return(stock_df: pd.DataFrame, ijh_df: pd.DataFrame, 
                         price_col: str = "Close") -> pd.Series:
    """
    Calculate daily abnormal returns (AR) for a stock vs IJH benchmark.
    
    Formula: AR_t = ln(Stock_t / Stock_{t-1}) - ln(IJH_t / IJH_{t-1})
    """
    stock_log_ret = calc_log_return(stock_df, price_col)
    ijh_log_ret = calc_log_return(ijh_df, price_col)
    
    ar = stock_log_ret - ijh_log_ret
    return ar


def calc_car(stock_df: pd.DataFrame, ijh_df: pd.DataFrame,
             event_date: str, horizon_days: int = 10,
             price_col: str = "Close") -> float:
    """
    Calculate Cumulative Abnormal Return (CAR) over a holding period.
    
    Uses the formula from Design.md:
        CAR_i = sum_{t=T+1}^{T+11} (R_{i,t} - R_{m,t})
    
    Args:
        stock_df: DataFrame with stock data (must include event_date)
        ijh_df:   DataFrame with IJH benchmark data
        event_date: Announcement date 'YYYY-MM-DD'
        horizon_days: Number of days to hold (default 10 -> T+1 to T+11)
        price_col: Which price column to use (default 'Close')
    
    Returns:
        CAR value as a float
    """
    # Merge on Date to align stock and benchmark
    merged = pd.merge(stock_df, ijh_df, on="Date", suffixes=('_stock', '_ijh'))
    merged = merged.sort_values("Date").reset_index(drop=True)
    
    # Calculate daily abnormal returns
    merged["ar"] = calc_abnormal_return(
        merged[["Date", f"{price_col}_stock"]].rename(columns={f"{price_col}_stock": price_col}),
        merged[["Date", f"{price_col}_ijh"]].rename(columns={f"{price_col}_ijh": price_col}),
        price_col
    )
    
    # Filter to T+1 through T+horizon_days
    event_dt = pd.to_datetime(event_date)
    merged["event_date"] = event_dt
    
    # Get trading days after the event
    post_event = merged[merged["Date"] > event_dt]
    
    # Take the first `horizon_days` trading days
    holding_period = post_event.head(horizon_days)
    
    car = holding_period["ar"].sum()
    
    return car


# ==============================================================================
# EXPLORATION HELPERS
# ==============================================================================

def get_ticker_list() -> list:
    """Return list of all tickers currently stored in the sp400 group."""
    with pd.HDFStore(DB_FILE, mode="r") as store:
        keys = store.keys()
    # Filter only sp400 tickers
    tickers = [k.split("/")[-1] for k in keys if f"/{H5_STOCKS}/" in k]
    return sorted(tickers)


def describe_ticker(ticker: str) -> Optional[pd.DataFrame]:
    """Return descriptive statistics for a single ticker's Close price history."""
    df = load_ticker(ticker)
    if df.empty:
        return None
    return df["Close"].describe()


def plot_price(ticker: str, start: str = None, end: str = None):
    """Quick plot of a stock's Close price over a date range."""
    df = load_ticker(ticker, start, end)
    
    if df.empty:
        print(f"No data for {ticker}")
        return
    
    ax = df.plot(x="Date", y="Close", figsize=(12, 5), title=f"{ticker} Close Price")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price ($)")
    ax.legend([ticker])
    return ax


# ==============================================================================
# DEMO / TEST
# ==============================================================================

if __name__ == "__main__":
    print("Discovery Notebook - PEAD Data Exploration")
    print("=" * 60)
    
    # Show available tickers
    tickers = get_ticker_list()
    print(f"\nTickers in database: {len(tickers)}")
    
    if len(tickers) >= 4:
        print(f"Sample tickers: {tickers[:4]}")
    
    if tickers:
        example = tickers[0]
        print(f"\n--- Loading {example} ---")
        df = load_ticker(example)
        print(f"  Date range: {df['Date'].min()} to {df['Date'].max()}")
        print(f"  Rows:       {len(df)}")
        print(f"  Latest Close: {df['Close'].iloc[-1]:.2f}")
        
        # Example: calculate simple daily log returns
        daily_log_ret = calc_log_return(df)
        print(f"  Daily log return (last): {daily_log_ret.iloc[-1]:.4%}")
        
        # Example: load IJH
        print("\n--- Loading IJH benchmark ---")
        ijh = load_ijh()
        print(f"  Date range: {ijh['Date'].min()} to {ijh['Date'].max()}")
        print(f"  Rows:       {len(ijh)}")
        
        print("\nFunctions available:")
        print("  load_ticker(ticker, start, end)         - Load stock data")
        print("  load_ijh(start, end)                   - Load IJH benchmark")
        print("  calc_log_return(df, price_col)         - Daily log returns")
        print("  calc_abnormal_return(stock, ijh)       - Abnormal returns")
        print("  calc_car(stock, ijh, event_date)       - Cumulative abnormal return")
        print("  get_ticker_list()                      - List all stored tickers")
