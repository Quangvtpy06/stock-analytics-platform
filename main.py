"""
Vietnam Stock Analytics — Orchestration Pipeline
================================================
End-to-End pipeline that:
1. Crawls latest data from CafeF
2. Cleans and normalizes the data
3. Computes technical indicators (RSI) & generates trading signals
4. Starts the web dashboard

Usage:
    python main.py
"""
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

from crawl_data import run_pipeline, EXCHANGES
from signal_rsi import generate_rsi_signals, get_latest_signals

# Paths
BASE_DIR = Path(__file__).parent
OUTPUT_DIR = BASE_DIR / "outputs"
DATA_PATH = OUTPUT_DIR / "pipeline_data" / "processed" / "cafef_ohlcv.csv"
SIGNALS_PATH = OUTPUT_DIR / "rsi_signals.csv"

def step_1_crawl():
    print(f"\n{'='*60}")
    print("STEP 1: Crawling Latest Data from CafeF")
    print(f"{'='*60}")
    # run_pipeline returns (data_path, report_path)
    # By default, run_pipeline downloads EXCHANGES and saves to output_dir
    data_path, report_path = run_pipeline(EXCHANGES, OUTPUT_DIR)
    return data_path

def step_2_signals(data_path: Path):
    print(f"\n{'='*60}")
    print("STEP 2: Generating Trading Signals & Updating Screener")
    print(f"{'='*60}")
    
    print("Loading market data...")
    market_data = pd.read_csv(data_path, encoding="utf-8-sig")
    
    print("Computing RSI and extracting BUY/SELL signals...")
    all_signals = generate_rsi_signals(market_data)
    
    # Save the signals to CSV so the dashboard can load them instantly
    SIGNALS_PATH.parent.mkdir(parents=True, exist_ok=True)
    all_signals.to_csv(SIGNALS_PATH, index=False, encoding="utf-8-sig")
    
    # Generate a brief screener summary
    latest = get_latest_signals(all_signals)
    buy_count = (latest["Signal"] == "BUY").sum() if not latest.empty else 0
    sell_count = (latest["Signal"] == "SELL").sum() if not latest.empty else 0
    
    print(f"Total historical signals generated: {len(all_signals):,}")
    print(f"Latest actionable signals      : {buy_count:,} BUY | {sell_count:,} SELL")
    print(f"Signals exported to            : {SIGNALS_PATH}")

def step_3_dashboard():
    print(f"\n{'='*60}")
    print("STEP 3: Launching Web Dashboard")
    print(f"{'='*60}")
    
    # Launch dashboard in a subprocess so it doesn't block the main thread forever
    # while allowing us to catch interrupt signals.
    print("Starting dashboard server on http://localhost:8050")
    try:
        subprocess.run([sys.executable, "dashboard.py"], cwd=BASE_DIR)
    except KeyboardInterrupt:
        print("\nDashboard server stopped.")

def main():
    start_time = time.time()
    
    try:
        # Pipeline Execution
        step_1_crawl()
        step_2_signals(DATA_PATH)
        
        print(f"\nPipeline completed in {time.time() - start_time:.1f} seconds.")
        
        step_3_dashboard()
        
    except Exception as e:
        print(f"\n[ERROR] Pipeline failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
