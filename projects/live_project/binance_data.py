from __future__ import annotations

import os
import time
from datetime import datetime, timezone
from pathlib import Path
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


def _normalize_utc_dt(value: Optional[datetime]) -> Optional[datetime]:
    if value is None:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


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


def _cache_file_path(symbol: str, interval: str) -> Path:
    env_dir = str(os.getenv("AUTOTRADE_KLINE_CACHE_DIR", "")).strip()
    if env_dir:
        base = Path(env_dir).expanduser().resolve()
    else:
        base = Path(__file__).resolve().parent / "reports" / "kline_cache"
    base.mkdir(parents=True, exist_ok=True)
    return base / f"{symbol}_{interval}.csv"


def _load_cached_klines(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

    try:
        df = pd.read_csv(path)
    except Exception:  # noqa: BLE001
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

    required_cols = {"open_time", "open", "high", "low", "close", "volume"}
    if not required_cols.issubset(set(df.columns)):
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

    df["open_time"] = pd.to_datetime(df["open_time"], utc=True, errors="coerce")
    df = df.dropna(subset=["open_time"]).set_index("open_time")
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df[["open", "high", "low", "close", "volume"]].dropna()
    df = df[~df.index.duplicated(keep="last")].sort_index()
    return df


def _save_cached_klines(path: Path, df: pd.DataFrame) -> None:
    out = df[["open", "high", "low", "close", "volume"]].copy()
    out = out[~out.index.duplicated(keep="last")].sort_index()
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    out.to_csv(tmp_path, index=True, index_label="open_time")
    tmp_path.replace(path)


def _fetch_klines_remote(
    symbol: str,
    interval: str,
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None,
    limit: int = 1000,
    timeout: int = 10,
    max_retries: int = 5,
    retry_delay: float = 0.4,
) -> pd.DataFrame:
    """仅从 Binance API 拉取K线并转为DataFrame(index=datetime)。"""
    symbol = symbol.upper().strip()
    if interval not in INTERVAL_TO_MS:
        raise BinanceDataError(f"不支持的 interval: {interval}")

    start_time = _normalize_utc_dt(start_time)
    end_time = _normalize_utc_dt(end_time)

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


def fetch_klines(
    symbol: str,
    interval: str,
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None,
    limit: int = 1000,
    timeout: int = 10,
    max_retries: int = 5,
    retry_delay: float = 0.4,
    *,
    use_local_cache: bool = True,
) -> pd.DataFrame:
    """
    拉取币安K线并转为DataFrame(index=datetime)。
    - 当提供 start_time + end_time 时，默认启用本地缓存，减少重复请求。
    - 实时/limit 场景（无时间范围）保持直接请求。
    """
    symbol = symbol.upper().strip()
    if interval not in INTERVAL_TO_MS:
        raise BinanceDataError(f"不支持的 interval: {interval}")

    start_time = _normalize_utc_dt(start_time)
    end_time = _normalize_utc_dt(end_time)

    if not use_local_cache or start_time is None or end_time is None:
        return _fetch_klines_remote(
            symbol=symbol,
            interval=interval,
            start_time=start_time,
            end_time=end_time,
            limit=limit,
            timeout=timeout,
            max_retries=max_retries,
            retry_delay=retry_delay,
        )

    if start_time >= end_time:
        raise BinanceDataError("start_time 必须早于 end_time")

    cache_path = _cache_file_path(symbol=symbol, interval=interval)
    cached = _load_cached_klines(cache_path)
    interval_delta = pd.to_timedelta(int(INTERVAL_TO_MS[interval]), unit="ms")
    req_start = pd.Timestamp(start_time)
    req_end = pd.Timestamp(end_time)

    segments: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    if cached.empty:
        segments.append((req_start, req_end))
    else:
        cache_start = pd.Timestamp(cached.index.min())
        cache_end = pd.Timestamp(cached.index.max())
        if req_start < cache_start:
            segments.append((req_start, cache_start))
        if req_end > cache_end + interval_delta:
            segments.append((cache_end + interval_delta, req_end))

    fetched_parts: list[pd.DataFrame] = []
    for seg_start, seg_end in segments:
        if seg_start >= seg_end:
            continue
        part = _fetch_klines_remote(
            symbol=symbol,
            interval=interval,
            start_time=seg_start.to_pydatetime(),
            end_time=seg_end.to_pydatetime(),
            limit=limit,
            timeout=timeout,
            max_retries=max_retries,
            retry_delay=retry_delay,
        )
        fetched_parts.append(part)

    if fetched_parts:
        merged = pd.concat([cached] + fetched_parts) if not cached.empty else pd.concat(fetched_parts)
        merged = merged[~merged.index.duplicated(keep="last")].sort_index()
        _save_cached_klines(cache_path, merged)
        cached = merged

    out = cached[(cached.index >= req_start) & (cached.index < req_end)].copy()
    if out.empty:
        # 兜底：避免缓存异常导致空数据
        out = _fetch_klines_remote(
            symbol=symbol,
            interval=interval,
            start_time=start_time,
            end_time=end_time,
            limit=limit,
            timeout=timeout,
            max_retries=max_retries,
            retry_delay=retry_delay,
        )
        merged = pd.concat([cached, out]) if not cached.empty else out
        merged = merged[~merged.index.duplicated(keep="last")].sort_index()
        _save_cached_klines(cache_path, merged)

    return out
