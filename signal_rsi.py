"""
RSI Trading Signal Generator
=============================
Buy when RSI(14) < 30 (oversold), Sell when RSI(14) > 70 (overbought).

Output columns: Ticker, Exchange, Close, Signal, Indicator, Signal_Date

Usage:
    python signal_rsi.py                          # scan all exchanges
    python signal_rsi.py --exchange HOSE           # scan specific exchange
    python signal_rsi.py --latest                  # only show latest signal per ticker
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DATA_PATH = Path(__file__).parent / "outputs" / "pipeline_data" / "processed" / "cafef_ohlcv.csv"
OUTPUT_PATH = Path(__file__).parent / "outputs" / "rsi_signals.csv"

RSI_PERIOD = 14
RSI_OVERSOLD = 30
RSI_OVERBOUGHT = 70


# ---------------------------------------------------------------------------
# RSI Calculation
# ---------------------------------------------------------------------------
def compute_rsi(close: pd.Series, period: int = RSI_PERIOD) -> pd.Series:
    """Compute RSI using the standard Wilder smoothing method."""
    change = close.diff()
    gain = change.clip(lower=0).rolling(period, min_periods=period).mean()
    loss = (-change.clip(upper=0)).rolling(period, min_periods=period).mean()
    rs = gain / loss.replace(0, pd.NA)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)


# ---------------------------------------------------------------------------
# Signal Generation
# ---------------------------------------------------------------------------
def generate_rsi_signals(
    market_data: pd.DataFrame,
    exchanges: list[str] | None = None,
) -> pd.DataFrame:
    """Generate BUY/SELL/HOLD signals for all tickers based on RSI thresholds."""
    data = market_data.copy()
    data["trading_date"] = pd.to_datetime(data["trading_date"])
    data = data.sort_values(["exchange", "ticker", "trading_date"])

    if exchanges:
        data = data[data["exchange"].isin(exchanges)]

    if data.empty:
        return pd.DataFrame(columns=["Ticker", "Exchange", "Close", "Signal", "Indicator", "Signal_Date"])

    # Vectorized RSI calculation using groupby transform
    def calc_rsi(group):
        change = group.diff()
        gain = change.clip(lower=0).rolling(RSI_PERIOD, min_periods=RSI_PERIOD).mean()
        loss = (-change.clip(upper=0)).rolling(RSI_PERIOD, min_periods=RSI_PERIOD).mean()
        rs = gain / loss.replace(0, pd.NA)
        return 100 - (100 / (1 + rs))

    data["Indicator"] = data.groupby(["exchange", "ticker"])["close"].transform(calc_rsi).fillna(50).round(2)

    # Assign signals using boolean masks (vectorized)
    data["Signal"] = "HOLD"
    data.loc[data["Indicator"] < RSI_OVERSOLD, "Signal"] = "BUY"
    data.loc[data["Indicator"] > RSI_OVERBOUGHT, "Signal"] = "SELL"

    # Rename and select columns to match desired output
    data = data.rename(columns={
        "ticker": "Ticker",
        "exchange": "Exchange",
        "close": "Close",
        "trading_date": "Signal_Date"
    })
    
    result = data[["Ticker", "Exchange", "Close", "Signal", "Indicator", "Signal_Date"]]
    result = result.sort_values("Signal_Date", ascending=False).reset_index(drop=True)
    return result


def get_latest_signals(signals_df: pd.DataFrame) -> pd.DataFrame:
    """Keep only the most recent signal per ticker/exchange pair."""
    if signals_df.empty:
        return signals_df
    return (
        signals_df
        .sort_values("Signal_Date", ascending=False)
        .drop_duplicates(subset=["Ticker", "Exchange"], keep="first")
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="RSI Trading Signal Generator")
    parser.add_argument("--exchange", nargs="+", help="Filter by exchange(s)")
    parser.add_argument("--latest", action="store_true", help="Only latest signal per ticker")
    parser.add_argument("--data", type=Path, default=DATA_PATH, help="Path to OHLCV CSV")
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH, help="Output CSV path")
    args = parser.parse_args()

    print("Loading market data...")
    market_data = pd.read_csv(args.data, encoding="utf-8-sig")
    print(f"Loaded {len(market_data):,} rows")

    print(f"Generating RSI signals (BUY < {RSI_OVERSOLD}, SELL > {RSI_OVERBOUGHT})...")
    signals = generate_rsi_signals(market_data, args.exchange)

    if args.latest:
        signals = get_latest_signals(signals)

    # Save
    args.output.parent.mkdir(parents=True, exist_ok=True)
    signals.to_csv(args.output, index=False, encoding="utf-8-sig")

    # Summary
    buy_count = (signals["Signal"] == "BUY").sum()
    sell_count = (signals["Signal"] == "SELL").sum()
    unique_tickers = signals["Ticker"].nunique()

    print(f"\n{'='*60}")
    print(f"  RSI Signal Summary")
    print(f"  Total signals : {len(signals):,}")
    print(f"  BUY signals   : {buy_count:,}")
    print(f"  SELL signals  : {sell_count:,}")
    print(f"  Tickers       : {unique_tickers:,}")
    print(f"  Saved to      : {args.output}")
    print(f"{'='*60}")

    # Show top 20 latest
    if not signals.empty:
        print(f"\nLatest 20 signals:")
        display = signals.head(20).to_string(index=False)
        print(display)


if __name__ == "__main__":
    main()
