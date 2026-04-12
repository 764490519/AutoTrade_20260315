from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


class SmaCrossStrategy:
    USE_GLOBAL_POSITION_PERCENT = True

    @staticmethod
    def generate_targets(df: pd.DataFrame, params: dict[str, Any]) -> np.ndarray:
        fast_period = int(params.get("fast_period", 20))
        slow_period = int(params.get("slow_period", 60))
        if fast_period < 2 or slow_period < 2:
            raise ValueError("fast_period / slow_period 必须 >= 2")
        if fast_period >= slow_period:
            raise ValueError("fast_period 必须小于 slow_period")

        close = pd.to_numeric(df["close"], errors="coerce").astype(float)
        fast = close.rolling(window=fast_period, min_periods=fast_period).mean()
        slow = close.rolling(window=slow_period, min_periods=slow_period).mean()
        n = len(df)
        targets = np.full(n, np.nan, dtype=float)
        pos = 0
        for i in range(slow_period + 1, n):
            f = float(fast.iloc[i - 1])
            s = float(slow.iloc[i - 1])
            if not np.isfinite(f) or not np.isfinite(s):
                continue
            if pos == 0 and f > s:
                targets[i] = 1.0
                pos = 1
            elif pos == 1 and f < s:
                targets[i] = 0.0
                pos = 0
        return targets


class RsiReversionStrategy:
    USE_GLOBAL_POSITION_PERCENT = True

    @staticmethod
    def generate_targets(df: pd.DataFrame, params: dict[str, Any]) -> np.ndarray:
        period = int(params.get("rsi_period", 14))
        oversold = float(params.get("oversold", 30))
        overbought = float(params.get("overbought", 70))

        close = pd.to_numeric(df["close"], errors="coerce").astype(float)
        delta = close.diff()
        gain = delta.clip(lower=0.0)
        loss = (-delta).clip(lower=0.0)
        avg_gain = gain.ewm(alpha=1.0 / float(period), adjust=False, min_periods=period).mean()
        avg_loss = loss.ewm(alpha=1.0 / float(period), adjust=False, min_periods=period).mean()
        rs = avg_gain / avg_loss.replace(0.0, np.nan)
        rsi = 100.0 - (100.0 / (1.0 + rs))
        rsi = rsi.where(avg_loss > 0.0, 100.0)

        n = len(df)
        targets = np.full(n, np.nan, dtype=float)
        pos = 0
        for i in range(period + 1, n):
            r = float(rsi.iloc[i - 1])
            if not np.isfinite(r):
                continue
            if pos == 0 and r < oversold:
                targets[i] = 1.0
                pos = 1
            elif pos == 1 and r > overbought:
                targets[i] = 0.0
                pos = 0
        return targets


class MacdCrossStrategy:
    USE_GLOBAL_POSITION_PERCENT = True

    @staticmethod
    def generate_targets(df: pd.DataFrame, params: dict[str, Any]) -> np.ndarray:
        fast = int(params.get("fast", 12))
        slow = int(params.get("slow", 26))
        signal = int(params.get("signal", 9))
        close = pd.to_numeric(df["close"], errors="coerce").astype(float)

        ema_fast = close.ewm(span=fast, adjust=False, min_periods=fast).mean()
        ema_slow = close.ewm(span=slow, adjust=False, min_periods=slow).mean()
        macd = ema_fast - ema_slow
        sig = macd.ewm(span=signal, adjust=False, min_periods=signal).mean()

        n = len(df)
        targets = np.full(n, np.nan, dtype=float)
        pos = 0
        need = max(fast, slow, signal) + 1
        for i in range(need, n):
            m = float(macd.iloc[i - 1])
            s = float(sig.iloc[i - 1])
            if not np.isfinite(m) or not np.isfinite(s):
                continue
            if pos == 0 and m > s:
                targets[i] = 1.0
                pos = 1
            elif pos == 1 and m < s:
                targets[i] = 0.0
                pos = 0
        return targets


STRATEGY_REGISTRY = {
    "SMA 双均线": {
        "class": SmaCrossStrategy,
        "params": {
            "fast_period": {"type": "int", "default": 20, "min": 2, "max": 200, "step": 1},
            "slow_period": {"type": "int", "default": 60, "min": 5, "max": 500, "step": 1},
        },
    },
    "RSI 均值回归": {
        "class": RsiReversionStrategy,
        "params": {
            "rsi_period": {"type": "int", "default": 14, "min": 2, "max": 100, "step": 1},
            "oversold": {"type": "float", "default": 30.0, "min": 1.0, "max": 50.0, "step": 0.5},
            "overbought": {"type": "float", "default": 70.0, "min": 50.0, "max": 99.0, "step": 0.5},
        },
    },
    "MACD 金叉死叉": {
        "class": MacdCrossStrategy,
        "params": {
            "fast": {"type": "int", "default": 12, "min": 2, "max": 100, "step": 1},
            "slow": {"type": "int", "default": 26, "min": 3, "max": 200, "step": 1},
            "signal": {"type": "int", "default": 9, "min": 2, "max": 100, "step": 1},
        },
    },
}
