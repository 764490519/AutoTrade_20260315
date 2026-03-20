import backtrader as bt

STRATEGY_META = {
    "display_name": "MACD 金叉死叉",
    "strategy_class": "MacdCrossStrategy",
    "params": {
        "fast": {"type": "int", "default": 12, "min": 2, "max": 100, "step": 1},
        "slow": {"type": "int", "default": 26, "min": 3, "max": 200, "step": 1},
        "signal": {"type": "int", "default": 9, "min": 2, "max": 100, "step": 1},
    },
}


class MacdCrossStrategy(bt.Strategy):
    params = (
        ("fast", 12),
        ("slow", 26),
        ("signal", 9),
    )

    def __init__(self):
        self.macd = bt.indicators.MACD(
            self.data.close,
            period_me1=self.params.fast,
            period_me2=self.params.slow,
            period_signal=self.params.signal,
        )
        self.crossover = bt.indicators.CrossOver(self.macd.macd, self.macd.signal)

    def next(self):
        if not self.position and self.crossover > 0:
            self.buy()
        elif self.position and self.crossover < 0:
            self.close()
