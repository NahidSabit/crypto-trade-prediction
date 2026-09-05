"""
data_loader.py

Fetches minute-level OHLCV data for a given symbol from Binance and saves it
to a CSV file for use in feature engineering / model training.
"""

import argparse
import time
from datetime import datetime

import pandas as pd

try:
    from binance.client import Client
except ImportError:
    Client = None  # allows the module to be imported even without python-binance installed


COLUMNS = [
    "open_time", "open", "high", "low", "close", "volume",
    "close_time", "quote_volume", "num_trades",
    "taker_buy_base", "taker_buy_quote", "ignore",
]


def fetch_klines(symbol: str, interval: str, start: str, end: str) -> pd.DataFrame:
    """Fetch historical klines from Binance in batches and return a tidy DataFrame."""
    if Client is None:
        raise ImportError("python-binance is required: pip install python-binance")

    client = Client()  # public endpoints do not require API keys for historical klines
    start_ts = int(pd.Timestamp(start).timestamp() * 1000)
    end_ts = int(pd.Timestamp(end).timestamp() * 1000)

    all_rows = []
    cursor = start_ts
    while cursor < end_ts:
        batch = client.get_klines(
            symbol=symbol,
            interval=interval,
            startTime=cursor,
            endTime=end_ts,
            limit=1000,
        )
        if not batch:
            break
        all_rows.extend(batch)
        cursor = batch[-1][6] + 1  # move past the last close_time
        time.sleep(0.2)  # be polite to the API

    df = pd.DataFrame(all_rows, columns=COLUMNS)
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms")
    numeric_cols = ["open", "high", "low", "close", "volume", "quote_volume", "num_trades"]
    df[numeric_cols] = df[numeric_cols].astype(float)
    return df[["open_time", "open", "high", "low", "close", "volume",
               "close_time", "quote_volume", "num_trades"]]


def main():
    parser = argparse.ArgumentParser(description="Download OHLCV data from Binance.")
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--interval", default="1m")
    parser.add_argument("--start", required=True, help="e.g. 2023-01-01")
    parser.add_argument("--end", default=datetime.utcnow().strftime("%Y-%m-%d"))
    parser.add_argument("--out", required=True, help="Output CSV path")
    args = parser.parse_args()

    df = fetch_klines(args.symbol, args.interval, args.start, args.end)
    df.to_csv(args.out, index=False)
    print(f"Saved {len(df):,} rows to {args.out}")


if __name__ == "__main__":
    main()
