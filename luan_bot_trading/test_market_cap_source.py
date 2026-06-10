#!/usr/bin/env python3
"""
Test Market Cap Sources
=======================
Quick script to find a reliable free source for market cap data.
Tests: yfinance, pandas_datareader, manual scraping
"""

import pandas as pd


def test_yfinance():
    """Test if yfinance provides market cap."""
    try:
        import yfinance as yf
        ticker = yf.Ticker("AAPL")
        info = ticker.info
        print("\n[yfinance] Keys in info dict:")
        for k in sorted(info.keys()):
            if 'market' in k.lower() or 'cap' in k.lower():
                print(f"  {k}: {info.get(k)}")
        return info.get('marketCap')
    except ImportError:
        print("\n[yfinance] Not installed. Run: pip install yfinance")
        return None
    except Exception as e:
        print(f"\n[yfinance] Error: {e}")
        return None


def test_yahooquery():
    """Alternative: yahooquery (faster bulk)."""
    try:
        from yahooquery import Ticker
        t = Ticker("AAPL")
        data = t.summary_detail
        print("\n[yahooquery] AAPL summary_detail:")
        print(data)
        return data.get('AAPL', {}).get('marketCap')
    except ImportError:
        print("\n[yahooquery] Not installed. Run: pip install yahooquery")
        return None


def test_wikipedia():
    """Wikipedia S&P 500 table sometimes has market cap."""
    try:
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        tables = pd.read_html(url)
        df = tables[0]
        print("\n[Wikipedia] Columns in S&P 500 table:")
        print(df.columns.tolist())
        print(df.head()[['Symbol', 'Security']])
        mcap_cols = [c for c in df.columns if 'market' in c.lower()]
        print(f"  Market cap columns found: {mcap_cols}")
        return mcap_cols
    except Exception as e:
        print(f"\n[Wikipedia] Error: {e}")
        return None


if __name__ == "__main__":
    print("=" * 50)
    print("  TEST: Market Cap Data Sources")
    print("=" * 50)

    mcap = test_yfinance()
    if mcap:
        print(f"\n[SUCCESS] yfinance market cap: {mcap:,.0f}")

    mcap = test_yahooquery()
    if mcap:
        print(f"\n[SUCCESS] yahooquery market cap: {mcap:,.0f}")

    cols = test_wikipedia()
