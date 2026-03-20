from __future__ import annotations

import time
from datetime import datetime
from typing import Optional

import pandas as pd
import requests

BINANCE_BASE_URL = "https://api.binance.com"
INTERVAL_TO_MS = {
    "1m": 60_000,
    "3m": 180_000,
    "5m": 300_000,
    "15m": 900_000,
    "30m": 1_800_000,
    "1h": 3_600_000,
    "2h": 7_200_000,
    "4h": 14_400_000,
    "6h": 21_600_000,
    "8h": 28_800_000,
    "12h": 43_200_000,
    "1d": 86_400_000,
    "3d": 259_200_000,
    "1w": 604_800_000,
    "1M": 2_592_000_000,
}


class BinanceDataError(RuntimeError):
    pass


def _to_ms(value: Optional[datetime]) -> Optional[int]:
    if value is None:
        return None
    return int(value.timestamp() * 1000)


def fetch_klines(
    symbol: str,
    interval: str,
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None,
    limit: int = 1000,
    timeout: int = 10,
) -> pd.DataFrame:
    """拉取币安K线并转为DataFrame(index=datetime)。"""
    symbol = symbol.upper().strip()
    if interval not in INTERVAL_TO_MS:
        raise BinanceDataError(f"不支持的 interval: {interval}")

    klines = []
    current_start = _to_ms(start_time)
    end_ms = _to_ms(end_time)

    while True:
        params = {
            "symbol": symbol,
            "interval": interval,
            "limit": min(limit, 1000),
        }
        if current_start is not None:
            params["startTime"] = current_start
        if end_ms is not None:
            params["endTime"] = end_ms

        response = requests.get(
            f"{BINANCE_BASE_URL}/api/v3/klines",
            params=params,
            timeout=timeout,
        )
        if response.status_code != 200:
            raise BinanceDataError(
                f"Binance API 请求失败，status={response.status_code}, body={response.text}"
            )

        batch = response.json()
        if not batch:
            break

        klines.extend(batch)

        if len(batch) < limit:
            break

        last_open_time = int(batch[-1][0])
        next_start = last_open_time + INTERVAL_TO_MS[interval]

        if end_ms is not None and next_start >= end_ms:
            break

        current_start = next_start
        time.sleep(0.15)

    if not klines:
        raise BinanceDataError("未获取到K线数据，请检查交易对和时间范围")

    df = pd.DataFrame(
        klines,
        columns=[
            "open_time",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "close_time",
            "quote_asset_volume",
            "number_of_trades",
            "taker_buy_base_asset_volume",
            "taker_buy_quote_asset_volume",
            "ignore",
        ],
    )

    numeric_cols = ["open", "high", "low", "close", "volume"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df = df.set_index("open_time")
    df = df[["open", "high", "low", "close", "volume"]].dropna()

    return df
