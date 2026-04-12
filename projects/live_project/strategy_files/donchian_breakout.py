"""
策略名称：Donchian 通道突破（ADX过滤 + 北京时间周末禁开仓）

策略简介：
- 保留 Donchian 开仓通道 / 出场通道的核心进出场逻辑。
- 增加 ADX 强度过滤，减少震荡行情中的低质量突破。
- 增加北京时间禁开仓时段：每周五 22:00 至周一 08:00 禁止开新仓。

核心交易逻辑：
1) 开仓（空仓时）
   - 做多：close > entry_upper[-1] 且 ADX 过滤通过 且 非禁开仓时段
   - 做空：close < entry_lower[-1] 且 ADX 过滤通过 且 非禁开仓时段（can_short=1）
2) 平仓（持仓时）
   - 多头平仓：close < exit_lower[-1]
   - 空头平仓：close > exit_upper[-1]
3) 执行时序
   - 当根 K 线计算信号，下一根 K 线应用仓位。

禁开仓时间（北京时间，UTC+8）：
- 周五 22:00（含）~ 周一 08:00（不含）
- 周一 08:00 后恢复可开仓
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

STRATEGY_META = {
    "display_name": "Donchian 通道突破（ADX过滤 + 北京时间周末禁开仓）",
    "strategy_class": "DonchianChannelBreakoutStrategy",
    "signal_func": "generate_targets",
    "params": {
        "entry_period": {
            "type": "int",
            "default": 55,
            "min": 2,
            "max": 300,
            "step": 1,
            "desc": "开仓通道周期。价格突破该周期上/下轨时触发开仓。",
        },
        "exit_period": {
            "type": "int",
            "default": 20,
            "min": 2,
            "max": 180,
            "step": 1,
            "desc": "出场通道周期。",
        },
        "can_short": {
            "type": "int",
            "default": 1,
            "min": 0,
            "max": 1,
            "step": 1,
            "desc": "是否允许做空：1=允许，0=仅做多。",
        },
        "block_entry_window_bj": {
            "type": "int",
            "default": 1,
            "min": 0,
            "max": 1,
            "step": 1,
            "desc": "是否启用北京时间禁开仓窗口（周五22:00~周一08:00）：1=启用，0=关闭。",
        },
        "adx_filter_enabled": {
            "type": "int",
            "default": 1,
            "min": 0,
            "max": 1,
            "step": 1,
            "desc": "是否启用 ADX 强度过滤：1=启用，0=关闭。",
        },
        "adx_period": {
            "type": "int",
            "default": 14,
            "min": 2,
            "max": 100,
            "step": 1,
            "desc": "ADX 计算周期。",
        },
        "adx_min": {
            "type": "float",
            "default": 20.0,
            "min": 0.0,
            "max": 80.0,
            "step": 0.5,
            "desc": "ADX 最小阈值。低于该值不允许开新仓。",
        },
        "trail_atr_enabled": {
            "type": "int",
            "default": 0,
            "min": 0,
            "max": 1,
            "step": 1,
            "desc": "是否启用 ATR 浮动止盈：1=启用，0=关闭。",
        },
        "trail_atr_period": {
            "type": "int",
            "default": 14,
            "min": 2,
            "max": 100,
            "step": 1,
            "desc": "ATR 浮动止盈周期。",
        },
        "trail_atr_mult": {
            "type": "float",
            "default": 3.0,
            "min": 0.1,
            "max": 20.0,
            "step": 0.1,
            "desc": "ATR 浮动止盈倍数。",
        },
    },
}


def _calc_adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
    prev_close = close.shift(1)
    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = pd.Series(
        np.where((up_move > down_move) & (up_move > 0.0), up_move, 0.0),
        index=high.index,
        dtype=float,
    )
    minus_dm = pd.Series(
        np.where((down_move > up_move) & (down_move > 0.0), down_move, 0.0),
        index=high.index,
        dtype=float,
    )

    tr = pd.concat(
        [
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    alpha = 1.0 / float(period)
    atr = tr.ewm(alpha=alpha, adjust=False, min_periods=period).mean()
    plus_di = 100.0 * plus_dm.ewm(alpha=alpha, adjust=False, min_periods=period).mean() / atr.replace(0.0, np.nan)
    minus_di = 100.0 * minus_dm.ewm(alpha=alpha, adjust=False, min_periods=period).mean() / atr.replace(0.0, np.nan)
    denom = (plus_di + minus_di).replace(0.0, np.nan)
    dx = ((plus_di - minus_di).abs() / denom) * 100.0
    adx = dx.ewm(alpha=alpha, adjust=False, min_periods=period).mean()
    return adx.astype(float)


def _calc_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    alpha = 1.0 / float(period)
    atr = tr.ewm(alpha=alpha, adjust=False, min_periods=period).mean()
    return atr.astype(float)


def _to_beijing_timestamp(ts: pd.Timestamp) -> pd.Timestamp:
    if ts.tzinfo is None:
        ts_utc = ts.tz_localize("UTC")
    else:
        ts_utc = ts.tz_convert("UTC")
    return ts_utc.tz_convert("Asia/Shanghai")


def _is_blocked_entry_time_beijing(ts: pd.Timestamp) -> bool:
    """北京时间禁开仓窗口：周五22:00（含）至周一08:00（不含）。"""
    bj = _to_beijing_timestamp(pd.Timestamp(ts))
    wd = int(bj.dayofweek)  # Mon=0 ... Sun=6
    hm = int(bj.hour) * 60 + int(bj.minute)

    if wd == 4 and hm >= (22 * 60):
        return True
    if wd in (5, 6):
        return True
    if wd == 0 and hm < (8 * 60):
        return True
    return False


def generate_targets(df: pd.DataFrame, params: dict[str, Any]) -> np.ndarray:
    entry_period = int(params.get("entry_period", 55))
    exit_period = int(params.get("exit_period", 20))
    can_short = bool(int(params.get("can_short", 1)))
    block_entry_window_bj = bool(int(params.get("block_entry_window_bj", 1)))
    adx_filter_enabled = bool(int(params.get("adx_filter_enabled", 1)))
    adx_period = int(params.get("adx_period", 14))
    adx_min = float(params.get("adx_min", 20.0))
    trail_atr_enabled = bool(int(params.get("trail_atr_enabled", 0)))
    trail_atr_period = int(params.get("trail_atr_period", 14))
    trail_atr_mult = float(params.get("trail_atr_mult", 3.0))

    if entry_period < 2 or exit_period < 2 or adx_period < 2 or trail_atr_period < 2:
        raise ValueError("entry_period / exit_period / adx_period / trail_atr_period 必须 >= 2")
    if not (0.0 <= adx_min <= 100.0):
        raise ValueError("adx_min 必须在 [0, 100] 区间")
    if trail_atr_mult <= 0.0:
        raise ValueError("trail_atr_mult 必须 > 0")

    close = pd.to_numeric(df["close"], errors="coerce").astype(float)
    high = pd.to_numeric(df["high"], errors="coerce").astype(float)
    low = pd.to_numeric(df["low"], errors="coerce").astype(float)

    entry_upper = high.rolling(window=entry_period, min_periods=entry_period).max()
    entry_lower = low.rolling(window=entry_period, min_periods=entry_period).min()
    exit_upper = high.rolling(window=exit_period, min_periods=exit_period).max()
    exit_lower = low.rolling(window=exit_period, min_periods=exit_period).min()
    adx = _calc_adx(high=high, low=low, close=close, period=adx_period)
    atr_trail = _calc_atr(high=high, low=low, close=close, period=trail_atr_period)

    n = len(df)
    targets = np.full(n, np.nan, dtype=float)

    pos = 0
    pending_target: float | None = None
    highest_since_entry: float | None = None
    lowest_since_entry: float | None = None

    def _is_blocked_entry_at(idx: int) -> bool:
        if not block_entry_window_bj:
            return False
        if not isinstance(df.index, pd.DatetimeIndex):
            return False
        if idx < 0 or idx >= len(df.index):
            return False
        try:
            ts = pd.Timestamp(df.index[idx])
            return bool(_is_blocked_entry_time_beijing(ts))
        except Exception:  # noqa: BLE001
            return False

    need_bars = max(entry_period, exit_period, adx_period, trail_atr_period) + 1
    for i in range(n):
        if i < need_bars:
            continue

        close_now = float(close.iloc[i])
        high_now = float(high.iloc[i])
        low_now = float(low.iloc[i])
        eu_prev = float(entry_upper.iloc[i - 1])
        el_prev = float(entry_lower.iloc[i - 1])
        xu_prev = float(exit_upper.iloc[i - 1])
        xl_prev = float(exit_lower.iloc[i - 1])
        atr_prev = float(atr_trail.iloc[i - 1]) if i - 1 >= 0 else float("nan")

        if not (
            np.isfinite(close_now)
            and np.isfinite(eu_prev)
            and np.isfinite(el_prev)
            and np.isfinite(xu_prev)
            and np.isfinite(xl_prev)
        ):
            continue

        if pending_target is not None:
            # 仅拦截“空仓->开仓”的执行，不影响平仓
            if pos == 0 and pending_target != 0.0 and _is_blocked_entry_at(i):
                pending_target = None
            else:
                targets[i] = float(pending_target)
                if pending_target > 0:
                    pos = 1
                    highest_since_entry = high_now if np.isfinite(high_now) else close_now
                    lowest_since_entry = low_now if np.isfinite(low_now) else close_now
                elif pending_target < 0:
                    pos = -1
                    highest_since_entry = high_now if np.isfinite(high_now) else close_now
                    lowest_since_entry = low_now if np.isfinite(low_now) else close_now
                else:
                    pos = 0
                    highest_since_entry = None
                    lowest_since_entry = None
                pending_target = None

        if pos > 0:
            if np.isfinite(high_now):
                highest_since_entry = high_now if highest_since_entry is None else max(float(highest_since_entry), high_now)

            exit_trigger = close_now < xl_prev
            if trail_atr_enabled and highest_since_entry is not None and np.isfinite(atr_prev):
                trail_stop_long = float(highest_since_entry) - float(trail_atr_mult) * atr_prev
                exit_trigger = bool(exit_trigger or (close_now < trail_stop_long))

            if exit_trigger and i + 1 < n:
                pending_target = 0.0
            continue

        if pos < 0:
            if np.isfinite(low_now):
                lowest_since_entry = low_now if lowest_since_entry is None else min(float(lowest_since_entry), low_now)

            exit_trigger = close_now > xu_prev
            if trail_atr_enabled and lowest_since_entry is not None and np.isfinite(atr_prev):
                trail_stop_short = float(lowest_since_entry) + float(trail_atr_mult) * atr_prev
                exit_trigger = bool(exit_trigger or (close_now > trail_stop_short))

            if exit_trigger and i + 1 < n:
                pending_target = 0.0
            continue

        if adx_filter_enabled:
            adx_prev = float(adx.iloc[i - 1])
            if not np.isfinite(adx_prev) or adx_prev < adx_min:
                continue

        if close_now > eu_prev:
            if i + 1 < n:
                if _is_blocked_entry_at(i + 1):
                    continue
                pending_target = 1.0
            continue

        if can_short and close_now < el_prev:
            if i + 1 < n:
                if _is_blocked_entry_at(i + 1):
                    continue
                pending_target = -1.0
            continue

    return targets


class DonchianChannelBreakoutStrategy:
    """Donchian 通道突破（ADX过滤 + 北京时间周末禁开仓）策略。"""

    USE_GLOBAL_POSITION_PERCENT = True

    @staticmethod
    def generate_targets(df: pd.DataFrame, params: dict[str, Any]) -> np.ndarray:
        return generate_targets(df, params)
