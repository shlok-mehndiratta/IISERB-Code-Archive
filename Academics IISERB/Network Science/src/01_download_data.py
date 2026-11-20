"""
01_download_data.py
-------------------
Downloads daily adjusted historical price data for the Nifty 200 universe
from Yahoo Finance using the official constituent list from the National
Stock Exchange of India (NSE).
"""

import os
import json
import pandas as pd
import numpy as np
import yfinance as yf

# Resolve paths relative to project root
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs", "tables")


def download_or_load_prices():
    """
    Reads the Nifty 200 constituent symbols and downloads historical daily
    adjusted close prices over the two-year observation window.
    """
    raw_path = os.path.join(DATA_DIR, "prices_raw.csv")
    constituents_path = os.path.join(DATA_DIR, "ind_nifty200list.csv")
    
    # Read the official Nifty 200 constituent list
    constituents = pd.read_csv(constituents_path)
    symbols = constituents["Symbol"].dropna().unique().tolist()
    
    # NSE tickers in Yahoo Finance require the '.NS' suffix
    ticker_map = {sym: f"{sym}.NS" for sym in symbols}
    tickers = list(ticker_map.values())
    
    # 2-year analysis window: 21 November 2024 to 20 November 2026
    start_date = "2024-11-20"
    end_date = "2026-11-21"
    
    if os.path.exists(raw_path):
        print(f"[Info] Loading existing raw prices from: {raw_path}")
        prices_df = pd.read_csv(raw_path, index_col=0, parse_dates=True)
    else:
        print(f"[Info] Downloading prices for {len(tickers)} tickers ({start_date} to {end_date})...")
        raw_download = yf.download(
            tickers=tickers,
            start=start_date,
            end=end_date,
            auto_adjust=True,
            progress=True,
            threads=True
        )
        
        # When auto_adjust=True, 'Close' reflects dividend- and split-adjusted prices
        if "Close" in raw_download.columns.levels[0]:
            prices_df = raw_download["Close"].copy()
        else:
            prices_df = raw_download.copy()
            
        # Clean column names by stripping the '.NS' suffix for intuitive matching
        prices_df.columns = [col.replace(".NS", "") for col in prices_df.columns]
        
        # Save raw downloaded prices to disk
        os.makedirs(DATA_DIR, exist_ok=True)
        prices_df.to_csv(raw_path)
        print(f"[Success] Saved raw prices to: {raw_path}")

    # Inspect download completeness
    total_requested = len(symbols)
    downloaded_columns = set(prices_df.columns)
    
    failed_tickers = [
        sym for sym in symbols
        if sym not in downloaded_columns or prices_df[sym].dropna().empty
    ]
    valid_tickers = [sym for sym in symbols if sym not in failed_tickers]
    
    summary = {
        "universe": "Nifty 200",
        "requested_tickers_count": total_requested,
        "downloaded_tickers_count": len(downloaded_columns),
        "valid_tickers_with_data_count": len(valid_tickers),
        "failed_tickers_count": len(failed_tickers),
        "failed_tickers": failed_tickers,
        "date_range_start": str(prices_df.index.min().date()) if not prices_df.empty else None,
        "date_range_end": str(prices_df.index.max().date()) if not prices_df.empty else None,
        "trading_days_count": len(prices_df)
    }
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(OUTPUT_DIR, "phase1_download_summary.json"), "w") as f:
        json.dump(summary, f, indent=4)
        
    print(f"[Summary] Successfully verified {len(valid_tickers)} / {total_requested} tickers across {len(prices_df)} sessions.")
    return prices_df, summary


if __name__ == "__main__":
    download_or_load_prices()
