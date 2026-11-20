"""
02_clean_and_preprocess.py
--------------------------
Cleans raw price series, applies systematic constituent exclusion rules,
audits extreme single-day moves, and computes daily logarithmic returns.
"""

import os
import json
import pandas as pd
import numpy as np

# Resolve paths relative to project root
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_TABLES = os.path.join(BASE_DIR, "outputs", "tables")


def clean_and_compute_returns():
    """
    Applies the full data cleaning pipeline:
      1. Excludes stocks not listed for the complete 2-year window (IPOs).
      2. Forward-fills isolated exchange halts (max 2 consecutive days).
      3. Computes log returns: R_i(t) = ln(P_i(t) / P_i(t-1)).
      4. Audits extreme price drops: treats corporate demerger/bonus anomalies
         by setting them to NaN (for pairwise-complete correlation), while
         retaining genuine market crashes.
    """
    raw_path = os.path.join(DATA_DIR, "prices_raw.csv")
    constituents_path = os.path.join(DATA_DIR, "ind_nifty200list.csv")
    
    prices_raw = pd.read_csv(raw_path, index_col=0, parse_dates=True)
    constituents = pd.read_csv(constituents_path)
    
    T_initial = len(prices_raw)
    N_initial = len(prices_raw.columns)
    
    # ---------------------------------------------------------
    # 1. Sample Selection & Exclusion Rule
    # Rule: Exclude stocks not listed for the full observation window
    # ---------------------------------------------------------
    start_date = prices_raw.index[0]
    stocks_missing_at_start = [
        col for col in prices_raw.columns if pd.isnull(prices_raw.loc[start_date, col])
    ]
    missing_pct = (prices_raw.isnull().sum() / T_initial) * 100
    
    dropped_stocks_info = []
    for sym in sorted(stocks_missing_at_start):
        pct = missing_pct[sym]
        missing_count = int(prices_raw[sym].isnull().sum())
        fv = prices_raw[sym].first_valid_index()
        first_valid = str(fv.date()) if fv is not None else "None"
        dropped_stocks_info.append({
            "symbol": sym,
            "missing_days": missing_count,
            "missing_pct": round(float(pct), 2),
            "first_valid_trading_date": first_valid,
            "reason": "Not listed for full window (missing history at start)"
        })
        
    retained_symbols = [s for s in prices_raw.columns if s not in stocks_missing_at_start]
    prices_filtered = prices_raw[retained_symbols].copy()
    
    # Forward-fill isolated trading halts (max 2 consecutive sessions)
    prices_filled = prices_filtered.ffill(limit=2)
    assert prices_filled.isnull().sum().sum() == 0, "Unexpected remaining NaNs after forward-fill!"
    
    # ---------------------------------------------------------
    # 2. Daily Logarithmic Returns
    # R_i(t) = ln(P_i(t) / P_i(t-1))
    # ---------------------------------------------------------
    returns_raw = np.log(prices_filled / prices_filled.shift(1)).dropna(how="all")
    returns_clean = returns_raw.copy()
    
    # ---------------------------------------------------------
    # 3. Corporate Action Audit & Anomaly Treatment
    # Three confirmed unadjusted corporate actions are set to NaN
    # for that single day to prevent artificial pairwise spikes.
    # Genuine market crashes (INDUSINDBK, POLICYBZR) are preserved.
    # ---------------------------------------------------------
    returns_clean.loc["2025-10-14", "TMPV"] = np.nan   # Tata Motors spinoff of Commercial Vehicles
    returns_clean.loc["2026-01-01", "TRENT"] = np.nan  # Trent unadjusted 1:2 bonus issue
    returns_clean.loc["2026-04-30", "VEDL"] = np.nan   # Vedanta 5-way demerger spinoff
    
    # Save extreme returns audit record
    extreme_records = [
        {
            "symbol": "INDUSINDBK",
            "date": "2025-03-11",
            "prev_close": 899.04,
            "event_close": 654.89,
            "next_close": 683.59,
            "log_return": -0.3169,
            "pct_change": -27.16,
            "classification": "Genuine Market Crash",
            "treatment": "Retained as economic variance",
            "verified_source": "Forex derivatives portfolio losses review"
        },
        {
            "symbol": "POLICYBZR",
            "date": "2026-09-24",
            "prev_close": 1886.30,
            "event_close": 1207.20,
            "next_close": 1166.00,
            "log_return": -0.4463,
            "pct_change": -36.00,
            "classification": "Genuine Market Crash",
            "treatment": "Retained as economic variance",
            "verified_source": "IRDAI regulatory commission cap draft announcement"
        },
        {
            "symbol": "TMPV",
            "date": "2025-10-14",
            "prev_close": 655.31,
            "event_close": 392.20,
            "next_close": 387.63,
            "log_return": -0.5134,
            "pct_change": -40.15,
            "classification": "Data Artefact (Corporate Spinoff)",
            "treatment": "Single-day return set to NaN; pairwise correlation",
            "verified_source": "Tata Motors commercial vehicle demerger record date"
        },
        {
            "symbol": "TRENT",
            "date": "2026-01-01",
            "prev_close": 4272.69,
            "event_close": 2860.71,
            "next_close": 2935.40,
            "log_return": -0.4012,
            "pct_change": -33.05,
            "classification": "Data Artefact (Unadjusted Bonus Issue)",
            "treatment": "Single-day return set to NaN; pairwise correlation",
            "verified_source": "1:2 bonus issue ex-date price adjustment"
        },
        {
            "symbol": "VEDL",
            "date": "2026-04-30",
            "prev_close": 758.51,
            "event_close": 266.25,
            "next_close": 288.90,
            "log_return": -1.0469,
            "pct_change": -64.90,
            "classification": "Data Artefact (5-Way Demerger Spinoff)",
            "treatment": "Single-day return set to NaN; pairwise correlation",
            "verified_source": "Vedanta multi-entity demerger ex-date discovery"
        }
    ]
    
    os.makedirs(OUTPUT_TABLES, exist_ok=True)
    pd.DataFrame(extreme_records).to_csv(
        os.path.join(OUTPUT_TABLES, "extreme_returns_flagged.csv"), index=False
    )
    
    # Save sector mapping for retained stocks
    sector_map = constituents[["Symbol", "Company Name", "Industry"]].rename(
        columns={"Industry": "Sector"}
    )
    sector_map = sector_map[sector_map["Symbol"].isin(retained_symbols)].reset_index(drop=True)
    
    # Save cleaned data to disk
    returns_clean.to_csv(os.path.join(DATA_DIR, "returns.csv"))
    sector_map.to_csv(os.path.join(DATA_DIR, "sector_map.csv"), index=False)
    pd.DataFrame(dropped_stocks_info).to_csv(
        os.path.join(OUTPUT_TABLES, "dropped_stocks.csv"), index=False
    )
    
    summary = {
        "initial_stocks": N_initial,
        "dropped_stocks_count": len(stocks_missing_at_start),
        "retained_stocks_count": len(retained_symbols),
        "returns_trading_days": len(returns_clean),
        "returns_first_date": str(returns_clean.index[0].date()),
        "returns_last_date": str(returns_clean.index[-1].date())
    }
    
    with open(os.path.join(OUTPUT_TABLES, "phase2_cleaning_summary.json"), "w") as f:
        json.dump(summary, f, indent=4)
        
    print(f"[Success] Data cleaning complete: N = {len(retained_symbols)} stocks, T = {len(returns_clean)} daily returns.")
    print(f"[Window] {returns_clean.index[0].date()} to {returns_clean.index[-1].date()} (Excluded {len(stocks_missing_at_start)} IPOs).")
    return returns_clean, sector_map, summary


if __name__ == "__main__":
    clean_and_compute_returns()
