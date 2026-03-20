import backtrader as bt

STRATEGY_META = {
    "display_name": "Donchian 通道突破",
    "strategy_class": "DonchianChannelBreakoutStrategy",
    "params": {
        "entry_period": {"type": "int", "default": 20, "min": 2, "max": 300, "step": 1},
        "exit_period": {"type": "int", "default": 20, "min": 2, "max": 300, "step": 1},
        "can_short": {"type": "int", "default": 0, "min": 0, "max": 1, "step": 1},
        "target_percent": {"type": "float", "default": 0.95, "min": 0.1, "max": 1.0, "step": 0.01}
    }
}


class DonchianChannelBreakoutStrategy(bt.Strategy):
    """
    Donchian 通道突破（Backtrader 版）

    关键点（防未来函数）：
    - 入场/出场信号使用上一根K线的通道值（upper[-1], lower[-1]）
    - 不使用当前K线参与构造的通道值进行当根信号判断
    - Backtrader 默认下一根K线撮合订单
    """

    params = (
        ("entry_period", 20),
        ("exit_period", 20),
        ("can_short", 0),  # 0: 仅做多；1: 允许做空
        ("target_percent", 0.95),
    )

    def __init__(self):
        self.entry_period = int(self.params.entry_period)
        self.exit_period = int(self.params.exit_period)
        self.can_short = bool(int(self.params.can_short))
        self.target_percent = float(self.params.target_percent)

        if self.entry_period < 2 or self.exit_period < 2:
            raise ValueError("entry_period 和 exit_period 必须 >= 2")
        if not (0 < self.target_percent <= 1):
            raise ValueError("target_percent 必须在 (0, 1] 区间")

        self.upper = bt.indicators.Highest(self.data.high, period=self.entry_period)
        self.lower = bt.indicators.Lowest(self.data.low, period=self.exit_period)

        self.pending_order = None

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return
        if order.status in [order.Completed, order.Canceled, order.Margin, order.Rejected]:
            self.pending_order = None

    def next(self):
        # 等待至少有 period+1 根K线，才能安全使用 [-1]
        need_bars = max(self.entry_period, self.exit_period) + 1
        if len(self.data) < need_bars:
            return

        if self.pending_order is not None:
            return

        close_now = float(self.data.close[0])
        upper_prev = float(self.upper[-1])
        lower_prev = float(self.lower[-1])

        # 与原始 QC 逻辑一致：
        # 1) 收盘价突破昨日上轨 -> 做多
        if close_now > upper_prev and self.position.size <= 0:
            # 使用目标仓位可在同一信号中完成反手（与 set_holdings 语义接近）
            self.pending_order = self.order_target_percent(target=self.target_percent)
            return

        # 2) 收盘价跌破昨日下轨 -> 做空（可选）/平多
        if close_now < lower_prev:
            if self.can_short:
                if self.position.size >= 0:
                    self.pending_order = self.order_target_percent(target=-self.target_percent)
            else:
                if self.position.size > 0:
                    self.pending_order = self.order_target_percent(target=0.0)
