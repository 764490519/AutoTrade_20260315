from __future__ import annotations

import argparse
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


def _signal(pos: float) -> str:
    if pos > 0:
        return "LONG"
    if pos < 0:
        return "SHORT"
    return "FLAT"


def _check_case(
    *,
    symbol: str,
    checks: int,
    name: str,
    strategy_cls,
    params: dict,
    interval: str,
    limit: int,
    position_percent: float,
) -> None:
    df = fetch_klines(symbol, interval, limit=limit)
    if isinstance(df.index, pd.DatetimeIndex) and df.index.tz is not None:
        df = df.tz_convert("UTC").tz_localize(None)

    if len(df) < 90:
        raise RuntimeError(f"{name} 数据不足，至少需要 90 根K线，当前={len(df)}")

    target = build_target_series(df, strategy_cls, params, position_percent=position_percent)
    held = target.ffill().fillna(0.0)

    sample_n = max(1, min(int(checks), max(1, len(df) - 81)))
    random.seed(42)
    idx_list = sorted(random.sample(range(80, len(df) - 1), sample_n))
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
    parser = argparse.ArgumentParser(description="检查回测信号与实时信号推断一致性")
    parser.add_argument("--symbol", type=str, default="BTCUSDT")
    parser.add_argument("--position-percent", type=float, default=95.0)
    parser.add_argument("--checks", type=int, default=20)
    args = parser.parse_args()

    _check_case(
        symbol=args.symbol,
        checks=args.checks,
        name="FastRsiFlip",
        strategy_cls=FastRsiFlipStrategy,
        params={
            "rsi_period": 8,
            "up": 55,
            "dn": 45,
            "ema_period": 40,
            "atr_period": 14,
            "stop_atr": 0.8,
            "cooldown": 2,
            "can_short": 1,
        },
        interval="1h",
        limit=1000,
        position_percent=float(args.position_percent),
    )

    _check_case(
        symbol=args.symbol,
        checks=args.checks,
        name="Donchian",
        strategy_cls=DonchianChannelBreakoutStrategy,
        params={
            "entry_period": 55,
            "exit_period": 20,
            "can_short": 1,
            "block_entry_window_bj": 1,
            "adx_filter_enabled": 1,
            "adx_period": 14,
            "adx_min": 20.0,
            "trail_atr_enabled": 0,
            "trail_atr_period": 14,
            "trail_atr_mult": 3.0,
        },
        interval="1h",
        limit=1000,
        position_percent=float(args.position_percent),
    )

    print("[PARITY] all parity checks passed")


if __name__ == "__main__":
    main()

