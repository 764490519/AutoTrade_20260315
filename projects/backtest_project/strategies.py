from __future__ import annotations

import backtrader as bt


class SmaCrossStrategy(bt.Strategy):
    params = (
        ("fast_period", 20),
        ("slow_period", 60),
    )

    def __init__(self) -> None:
        fast_sma = bt.indicators.SimpleMovingAverage(self.data.close, period=self.params.fast_period)
        slow_sma = bt.indicators.SimpleMovingAverage(self.data.close, period=self.params.slow_period)
        self.crossover = bt.indicators.CrossOver(fast_sma, slow_sma)

    def next(self) -> None:
        if not self.position and self.crossover > 0:
            self.buy()
        elif self.position and self.crossover < 0:
            self.close()


class RsiReversionStrategy(bt.Strategy):
    params = (
        ("rsi_period", 14),
        ("oversold", 30),
        ("overbought", 70),
    )

    def __init__(self) -> None:
        self.rsi = bt.indicators.RSI(self.data.close, period=self.params.rsi_period)

    def next(self) -> None:
        if not self.position and self.rsi < self.params.oversold:
            self.buy()
        elif self.position and self.rsi > self.params.overbought:
            self.close()


class MacdCrossStrategy(bt.Strategy):
    params = (
        ("fast", 12),
        ("slow", 26),
        ("signal", 9),
    )

    def __init__(self) -> None:
        self.macd = bt.indicators.MACD(
            self.data.close,
            period_me1=self.params.fast,
            period_me2=self.params.slow,
            period_signal=self.params.signal,
        )
        self.crossover = bt.indicators.CrossOver(self.macd.macd, self.macd.signal)

    def next(self) -> None:
        if not self.position and self.crossover > 0:
            self.buy()
        elif self.position and self.crossover < 0:
            self.close()


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
