"""
?????Fast RSI Flip?EMA + ATR?

?????
- ?? RSI ???? + EMA ?????????????
- ?????? K ??????????? K ???????????????
- ?? ATR ????/????????????????

???????
1) ????????
   - ???close[-1] > EMA[-1] ? RSI[-1] > up
   - ???close[-1] < EMA[-1] ? RSI[-1] < dn?can_short=1?
2) ????????
   - ATR ???? + ATR ???? + RSI ??????
3) ?????
   - ?? bar(i) ????????? bar(i-1) ?????????? bar(i) ? open?

???????
- rsi_period?RSI ??
- up / dn?RSI ??/??????? 0 < dn < up < 100?
- ema_period?????????
- atr_period?ATR ??
- stop_atr / take_atr???/?? ATR ??
- cooldown?????? bar ?
- can_short????????1=???0=????
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

STRATEGY_META = {
    "display_name": "Fast RSI Flip?EMA+ATR?",
    "strategy_class": "FastRsiFlipStrategy",
    "signal_func": "generate_targets",
    "params": {
        "rsi_period": {
            "type": "int",
            "default": 36,
            "min": 4,
            "max": 60,
            "step": 1,
            "desc": "RSI ?????",
        },
        "up": {
            "type": "int",
            "default": 65,
            "min": 50,
            "max": 95,
            "step": 1,
            "desc": "RSI ?????????????????????",
        },
        "dn": {
            "type": "int",
            "default": 31,
            "min": 5,
            "max": 50,
            "step": 1,
            "desc": "RSI ?????????????????????",
        },
        "ema_period": {
            "type": "int",
            "default": 186,
            "min": 20,
            "max": 500,
            "step": 1,
            "desc": "EMA ???????",
        },
        "atr_period": {
            "type": "int",
            "default": 38,
            "min": 5,
            "max": 120,
            "step": 1,
            "desc": "ATR ?????",
        },
        "stop_atr": {
            "type": "float",
            "default": 0.5,
            "min": 0.1,
            "max": 10.0,
            "step": 0.1,
            "desc": "???? ATR ???",
        },
        "take_atr": {
            "type": "float",
            "default": 5.8,
            "min": 0.1,
            "max": 15.0,
            "step": 0.1,
            "desc": "???? ATR ???",
        },
        "cooldown": {
            "type": "int",
            "default": 71,
            "min": 0,
            "max": 500,
            "step": 1,
            "desc": "?????? bar ??",
        },
        "can_short": {
            "type": "int",
            "default": 1,
            "min": 0,
            "max": 1,
            "step": 1,
            "desc": "???????1=???0=????",
        },
    },
}


def _ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False, min_periods=period).mean()


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1.0 / float(period), adjust=False, min_periods=period).mean()


def _rsi(close: pd.Series, period: int) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)

    avg_gain = gain.ewm(alpha=1.0 / float(period), adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1.0 / float(period), adjust=False, min_periods=period).mean()

    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    out = 100.0 - (100.0 / (1.0 + rs))
    out = out.where(avg_loss > 0.0, 100.0)
    both_zero = (avg_gain <= 0.0) & (avg_loss <= 0.0)
    return out.where(~both_zero, 50.0)


def generate_targets(df: pd.DataFrame, params: dict[str, Any]) -> np.ndarray:
    rsi_period = int(params.get("rsi_period", 36))
    up = float(params.get("up", 65))
    dn = float(params.get("dn", 31))
    ema_period = int(params.get("ema_period", 186))
    atr_period = int(params.get("atr_period", 38))
    stop_atr = float(params.get("stop_atr", 0.5))
    take_atr = float(params.get("take_atr", 5.8))
    cooldown = int(params.get("cooldown", 71))
    can_short = bool(int(params.get("can_short", 1)))

    if rsi_period < 2 or ema_period < 2 or atr_period < 2:
        raise ValueError("rsi_period / ema_period / atr_period ?? >= 2")
    if stop_atr <= 0 or take_atr <= 0:
        raise ValueError("stop_atr / take_atr ?? > 0")
    if cooldown < 0:
        raise ValueError("cooldown ?? >= 0")
    if not (0 < dn < up < 100):
        raise ValueError("?????????0 < dn < up < 100")

    close = pd.to_numeric(df["close"], errors="coerce").astype(float)
    open_ = pd.to_numeric(df["open"], errors="coerce").astype(float)
    high = pd.to_numeric(df["high"], errors="coerce").astype(float)
    low = pd.to_numeric(df["low"], errors="coerce").astype(float)

    ema = _ema(close, ema_period)
    rsi = _rsi(close, rsi_period)
    atr = _atr(high, low, close, atr_period)

    n = len(df)
    targets = np.full(n, np.nan, dtype=float)

    pos = 0
    entry_price: float | None = None
    entry_atr: float | None = None
    last_flat_bar = -10**9

    need_bars = max(rsi_period, ema_period, atr_period) + 2
    for i in range(n):
        if i < need_bars:
            continue

        close_prev = float(close.iloc[i - 1])
        open_now = float(open_.iloc[i])
        ema_prev = float(ema.iloc[i - 1])
        rsi_prev = float(rsi.iloc[i - 1])
        atr_prev = float(atr.iloc[i - 1])

        if not (
            np.isfinite(close_prev)
            and np.isfinite(open_now)
            and np.isfinite(ema_prev)
            and np.isfinite(rsi_prev)
            and np.isfinite(atr_prev)
        ):
            continue

        if pos > 0:
            atr_used = float(entry_atr) if entry_atr is not None else atr_prev
            stop_price = float(entry_price) - atr_used * stop_atr
            take_price = float(entry_price) + atr_used * take_atr
            if close_prev <= stop_price or close_prev >= take_price or rsi_prev < dn:
                targets[i] = 0.0
                pos = 0
                entry_price = None
                entry_atr = None
                last_flat_bar = i
            continue

        if pos < 0:
            atr_used = float(entry_atr) if entry_atr is not None else atr_prev
            stop_price = float(entry_price) + atr_used * stop_atr
            take_price = float(entry_price) - atr_used * take_atr
            if close_prev >= stop_price or close_prev <= take_price or rsi_prev > up:
                targets[i] = 0.0
                pos = 0
                entry_price = None
                entry_atr = None
                last_flat_bar = i
            continue

        if i - last_flat_bar <= cooldown:
            continue

        if close_prev > ema_prev and rsi_prev > up:
            targets[i] = 1.0
            pos = 1
            entry_price = open_now
            entry_atr = atr_prev
            continue

        if can_short and close_prev < ema_prev and rsi_prev < dn:
            targets[i] = -1.0
            pos = -1
            entry_price = open_now
            entry_atr = atr_prev
            continue

    return targets


class FastRsiFlipStrategy:
    """?????????????? generate_targets ????"""

    USE_GLOBAL_POSITION_PERCENT = True

    @staticmethod
    def generate_targets(df: pd.DataFrame, params: dict[str, Any]) -> np.ndarray:
        return generate_targets(df, params)
