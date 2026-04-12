from __future__ import annotations

import random
import sys
from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backtest_engine import build_target_series
from binance_data import fetch_klines
from live_trading_engine import infer_signal_from_strategy
from strategy_files.donchian_breakout import DonchianChannelBreakoutStrategy
from strategy_files.fast_rsi_flip import FastRsiFlipStrategy
from strategy_files.ma_trend_faber import FaberMaTrendStrategy


def _signal(pos: float) -> str:
    if pos > 0:
        return "LONG"
    if pos < 0:
        return "SHORT"
    return "FLAT"


def _check_case(name: str, strategy_cls, params: dict, interval: str, limit: int, position_percent: float) -> None:
    df = fetch_klines("BTCUSDT", interval, limit=limit)
    if isinstance(df.index, pd.DatetimeIndex) and df.index.tz is not None:
        df = df.tz_convert("UTC").tz_localize(None)

    target = build_target_series(df, strategy_cls, params, position_percent=position_percent)
    held = target.ffill().fillna(0.0)

    random.seed(42)
    idx_list = sorted(random.sample(range(80, len(df) - 1), 20))
    mismatches = []
    for i in idx_list:
        sub = df.iloc[: i + 1]
        snap = infer_signal_from_strategy(
            strategy_cls=strategy_cls,
            strategy_params=params,
            df=sub,
            position_percent=position_percent,
        )
        dt = sub.index[-1]
        exp = _signal(float(held.loc[dt]))
        if snap.signal != exp:
            mismatches.append((str(dt), exp, snap.signal, float(held.loc[dt]), snap.strategy_position))

    assert not mismatches, f"{name} parity mismatch: {mismatches[:3]}"
    print(f"[PARITY] {name}: checked={len(idx_list)} matched={len(idx_list)}")


def main() -> None:
    _check_case(
        "FaberMaTrend",
        FaberMaTrendStrategy,
        {"fast_period": 11, "slow_period": 60},
        "4h",
        800,
        95.0,
    )
    _check_case(
        "FastRsiFlip",
        FastRsiFlipStrategy,
        {"rsi_period": 8, "up": 55, "dn": 45, "ema_period": 40, "atr_period": 14, "stop_atr": 0.8, "cooldown": 2, "can_short": 1},
        "1h",
        1000,
        95.0,
    )
    _check_case(
        "Donchian",
        DonchianChannelBreakoutStrategy,
        {
            "entry_period": 55,
            "exit_period": 20,
            "atr_period": 14,
            "atr_filter_enabled": 1,
            "atr_ratio_min": 0.003,
            "atr_ratio_max": 0.05,
            "stop_atr_mult": 2.0,
            "trail_atr_mult": 2.0,
            "can_short": 1,
        },
        "1h",
        1000,
        95.0,
    )
    print("[PARITY] all parity checks passed")


if __name__ == "__main__":
    main()
