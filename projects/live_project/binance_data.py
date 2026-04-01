from __future__ import annotations

import time
from datetime import datetime
from typing import Optional

import pandas as pd
import requests

BINANCE_BASE_URLS = (
    "https://api.binance.com",
    "https://api1.binance.com",
    "https://api2.binance.com",
    "https://api3.binance.com",
)
RETRYABLE_STATUS_CODES = {418, 429, 500, 502, 503, 504}
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


def _request_klines_with_retry(
    session: requests.Session,
    *,
    params: dict,
    timeout: int,
    max_retries: int,
    retry_delay: float,
) -> list:
    errors: list[str] = []
    url_path = "/api/v3/klines"
    max_retries = max(1, int(max_retries))

    for attempt in range(1, max_retries + 1):
        for base_url in BINANCE_BASE_URLS:
            try:
                response = session.get(
                    f"{base_url}{url_path}",
                    params=params,
                    timeout=timeout,
                    headers={"User-Agent": "AutoTrade/1.0"},
                )
            except requests.RequestException as exc:
                errors.append(f"{base_url} 网络异常: {exc}")
                continue

            if response.status_code == 200:
                try:
                    data = response.json()
                except Exception as exc:  # noqa: BLE001
                    errors.append(f"{base_url} JSON解析失败: {exc}")
                    continue
                if not isinstance(data, list):
                    errors.append(f"{base_url} 返回格式异常: {type(data).__name__}")
                    continue
                return data

            body = (response.text or "").replace("\n", " ")[:180]
            if response.status_code in RETRYABLE_STATUS_CODES:
                errors.append(f"{base_url} status={response.status_code}, body={body}")
                continue

            raise BinanceDataError(
                f"Binance API 请求失败（不可重试），status={response.status_code}, body={body}"
            )

        if attempt < max_retries:
            # 线性退避，避免触发限流
            time.sleep(retry_delay * attempt)

    tail = " | ".join(errors[-3:]) if errors else "未知网络错误"
    raise BinanceDataError(
        f"Binance API 网络不稳定，重试 {max_retries} 次仍失败。"
        f"请稍后重试或切换更大K线周期。最近错误：{tail}"
    )


def fetch_klines(
    symbol: str,
    interval: str,
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None,
    limit: int = 1000,
    timeout: int = 10,
    max_retries: int = 5,
    retry_delay: float = 0.4,
) -> pd.DataFrame:
    """拉取币安K线并转为DataFrame(index=datetime)。"""
    symbol = symbol.upper().strip()
    if interval not in INTERVAL_TO_MS:
        raise BinanceDataError(f"不支持的 interval: {interval}")

    klines = []
    current_start = _to_ms(start_time)
    end_ms = _to_ms(end_time)

    with requests.Session() as session:
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

            batch = _request_klines_with_retry(
                session,
                params=params,
                timeout=timeout,
                max_retries=max_retries,
                retry_delay=retry_delay,
            )
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
