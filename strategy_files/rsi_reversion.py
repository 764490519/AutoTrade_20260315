import backtrader as bt

STRATEGY_META = {
    "display_name": "RSI 均值回归",
    "strategy_class": "RsiReversionStrategy",
    "params": {
        "rsi_period": {"type": "int", "default": 14, "min": 2, "max": 100, "step": 1},
        "oversold": {"type": "float", "default": 30.0, "min": 1.0, "max": 50.0, "step": 0.5},
        "overbought": {"type": "float", "default": 70.0, "min": 50.0, "max": 99.0, "step": 0.5},
    },
}


class RsiReversionStrategy(bt.Strategy):
    params = (
        ("rsi_period", 14),
        ("oversold", 30),
        ("overbought", 70),
    )

    def __init__(self):
        self.rsi = bt.indicators.RSI(self.data.close, period=self.params.rsi_period)

    def next(self):
        if not self.position and self.rsi < self.params.oversold:
            self.buy()
        elif self.position and self.rsi > self.params.overbought:
            self.close()
