"""Download real BTC-USD daily candles from the Coinbase Exchange API.

Public endpoint, no key needed. It returns at most 300 candles per call,
so this pages backwards in 300 day windows and saves one snapshot CSV.
The run uses the snapshot, so results are reproducible even if the API
changes later. Source: https://api.exchange.coinbase.com/products/BTC-USD/candles
Coinbase Exchange market data, fetched 2026-10-22 for this project.
"""
import pathlib
import time

import pandas as pd
import requests

ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
START = "2019-01-01"
END = "2026-10-22"


def main():
    DATA.mkdir(exist_ok=True)
    target = DATA / "btc_usd_daily.csv"
    if target.exists():
        print("snapshot exists:", target)
        return
    frames = []
    end = pd.Timestamp(END)
    start_floor = pd.Timestamp(START)
    while end > start_floor:
        start = max(start_floor, end - pd.Timedelta(days=299))
        r = requests.get(
            "https://api.exchange.coinbase.com/products/BTC-USD/candles",
            params={"granularity": 86400, "start": start.isoformat(), "end": end.isoformat()},
            headers={"User-Agent": "crypto-market-intelligence rebuild"},
            timeout=30,
        )
        r.raise_for_status()
        rows = r.json()
        frames.append(pd.DataFrame(rows, columns=["time", "low", "high", "open", "close", "volume"]))
        end = start - pd.Timedelta(days=1)
        time.sleep(0.35)
    df = pd.concat(frames).drop_duplicates("time").sort_values("time")
    df["date"] = pd.to_datetime(df["time"], unit="s", utc=True).dt.strftime("%Y-%m-%d")
    df = df[["date", "open", "high", "low", "close", "volume"]]
    df.to_csv(target, index=False)
    print("saved", len(df), "daily candles", df["date"].iloc[0], "to", df["date"].iloc[-1])


if __name__ == "__main__":
    main()
