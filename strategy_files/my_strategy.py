import backtrader as bt

STRATEGY_META = {
    "display_name": "我的策略",
    "strategy_class": "MyStrategy",
    "params": {
        "fast_periodq": {"type": "int", "default": 20, "min": 2, "max": 200, "step": 1},
        "slow_period": {"type": "int", "default": 60, "min": 5, "max": 500, "step": 1},
    },
}


class MyStrategy(bt.Strategy):
    params = (
        ("fast_period", 20),
        ("slow_period", 60),
    )

    def __init__(self):
        fast_sma = bt.indicators.SimpleMovingAverage(self.data.close, period=self.params.fast_period)
        slow_sma = bt.indicators.SimpleMovingAverage(self.data.close, period=self.params.slow_period)
        self.crossover = bt.indicators.CrossOver(fast_sma, slow_sma)

    def next(self):
        if not self.position and self.crossover > 0:
            self.buy()
        elif self.position and self.crossover < 0:
            self.close()
