import backtrader as bt

STRATEGY_META = {
    "display_name": "MA 趋势过滤（Faber）",
    "strategy_class": "FaberMaTrendStrategy",
    "params": {
        "fast_period": {
            "type": "int",
            "default": 11,
            "min": 5,
            "max": 80,
            "step": 1,
            "desc": "快线周期。快线在慢线上方时持有多头。",
        },
        "slow_period": {
            "type": "int",
            "default": 220,
            "min": 40,
            "max": 240,
            "step": 2,
            "desc": "慢线周期。快线下穿慢线时平仓（仅多头，不做空）。",
        },
    },
}


class FaberMaTrendStrategy(bt.Strategy):
    """
    MA 趋势过滤（Long/Flat）：
    - fast[-1] > slow[-1]：开多
    - fast[-1] < slow[-1]：平多
    """

    params = (
        ("fast_period", 11),
        ("slow_period", 220),
    )

    def __init__(self):
        self.fast_period = int(self.params.fast_period)
        self.slow_period = int(self.params.slow_period)
        self.fast = bt.indicators.SimpleMovingAverage(self.data.close, period=self.fast_period)
        self.slow = bt.indicators.SimpleMovingAverage(self.data.close, period=self.slow_period)

    def next(self):
        if len(self.data) < self.slow_period + 1:
            return

        fast_prev = float(self.fast[-1])
        slow_prev = float(self.slow[-1])

        if not self.position and fast_prev > slow_prev:
            self.buy()
        elif self.position and fast_prev < slow_prev:
            self.close()
