import yfinance as yf
import pandas as pd
import time

# 1. Define your 11 Sector ETFs + Index Proxy
sector_tickers = "XLK XLF XLI XLY XLP XLV XLU XLE XLB XLRE XLC IJH SPY ^VIX"

print("Downloading broad sector and market indexes...")
# Download everything in one single call
sectors_df = yf.download(sector_tickers, start="2011-01-01", end="2026-06-20", group_by='ticker')
sectors_df.to_hdf('market_sectors_history.h5', key='daily_bars')

# 2. To download your 400 stocks safely without getting banned:
# Split your S&P 400 ticker list into chunks of 50 tickers
def bulk_download_stocks(ticker_list, chunk_size=50):
    chunks = [ticker_list[i:i + chunk_size] for i in range(0, len(ticker_list), chunk_size)]
    
    for idx, chunk in enumerate(chunks):
        chunk_string = " ".join(chunk)
        print(f"Downloading chunk {idx+1}/{len(chunks)}...")
        
        # Pull 50 stocks simultaneously
        data = yf.download(chunk_string, start="2011-01-01", end="2026-06-20", group_by='ticker')
        
        # Append locally to your HDF5 storage matrix
        data.to_hdf('midcap_stocks_history.h5', key=f'chunk_{idx}')
        
        # Polite pause to ensure Yahoo never triggers a rate limit
        time.sleep(3)

# Example execution:
# midcap_tickers = ["CRUS", "GHC", ... 398 others]
# bulk_download_stocks(midcap_tickers)