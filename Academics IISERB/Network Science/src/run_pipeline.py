"""
run_pipeline.py
---------------
Master runner script that executes the complete Nifty 200 network science
analysis pipeline from raw data processing to figure generation.
"""

import sys
import time

from importlib import import_module


def main():
    steps = [
        ("Step 1: Download & Verify Data", "01_download_data", "download_or_load_prices"),
        ("Step 2: Clean Data & Compute Log Returns", "02_clean_and_preprocess", "clean_and_compute_returns"),
        ("Step 3: Pearson Correlation & Threshold Sweep", "03_threshold_sweep", "run_threshold_sweep"),
        ("Step 4: Degree Distribution & Centrality Analysis", "04_degree_and_centrality", "analyze_degree_and_centrality"),
        ("Step 5: Sector Homophily & Null Models", "05_sector_analysis", "run_sector_analysis"),
        ("Step 6: Market Mode Removal & Attack Simulation", "06_market_mode_and_robustness", "run_market_mode_and_robustness"),
        ("Step 7: Generate Publication-Ready Visualizations", "07_generate_visualizations", "generate_all_figures")
    ]
    
    print("=" * 70)
    print("  NIFTY 200 FINANCIAL CORRELATION NETWORK ANALYSIS PIPELINE")
    print("=" * 70)
    
    total_start = time.time()
    
    for title, module_name, func_name in steps:
        print(f"\n>>> Running: {title}...")
        step_start = time.time()
        try:
            mod = import_module(module_name)
            func = getattr(mod, func_name)
            func()
            elapsed = time.time() - step_start
            print(f"[Done] {title} finished in {elapsed:.2f}s.")
        except Exception as e:
            print(f"[Error] Failed during {title}: {e}")
            sys.exit(1)
            
    total_elapsed = time.time() - total_start
    print("\n" + "=" * 70)
    print(f"  PIPELINE COMPLETE (Total Execution Time: {total_elapsed:.2f}s)")
    print("=" * 70)


if __name__ == "__main__":
    main()
