import backtrader as bt

STRATEGY_META = {
    "display_name": "Fast RSI Flip（EMA+ATR）",
    "strategy_class": "FastRsiFlipStrategy",
    "params": {
        "rsi_period": {
            "type": "int",
            "default": 12,
            "min": 3,
            "max": 30,
            "step": 1,
            "desc": "RSI 周期。",
        },
        "up": {
            "type": "int",
            "default": 53,
            "min": 50,
            "max": 90,
            "step": 1,
            "desc": "做多阈值：RSI 高于该值才允许开多。",
        },
        "dn": {
            "type": "int",
            "default": 36,
            "min": 10,
            "max": 50,
            "step": 1,
            "desc": "做空阈值：RSI 低于该值才允许开空。",
        },
        "ema_period": {
            "type": "int",
            "default": 66,
            "min": 5,
            "max": 300,
            "step": 1,
            "desc": "趋势过滤 EMA 周期。",
        },
        "atr_period": {
            "type": "int",
            "default": 26,
            "min": 2,
            "max": 120,
            "step": 1,
            "desc": "ATR 周期。",
        },
        "stop_atr": {
            "type": "float",
            "default": 0.5,
            "min": 0.1,
            "max": 10.0,
            "step": 0.1,
            "desc": "固定止损 ATR 倍数。",
        },
        "cooldown": {
            "type": "int",
            "default": 12,
            "min": 0,
            "max": 240,
            "step": 1,
            "desc": "平仓后冷却 bar 数，冷却期内不再开仓。",
        },
        "can_short": {
            "type": "int",
            "default": 1,
            "min": 0,
            "max": 1,
            "step": 1,
            "desc": "是否允许做空：1=允许，0=仅做多。",
        },
    },
}


class FastRsiFlipStrategy(bt.Strategy):
    """
    快速切换策略（Long/Short）：
    1) 趋势 + 动量开仓
       - 开多：close > EMA 且 RSI > up
       - 开空：close < EMA 且 RSI < dn（can_short=1）
    2) 持仓出场
       - 多头：close < 入场价 - 入场ATR*stop_atr 或 RSI < dn
       - 空头：close > 入场价 + 入场ATR*stop_atr 或 RSI > up
    3) 冷却机制
       - 平仓后 cooldown 根K线内不再开仓

    为避免未来函数，信号使用上一根指标值（[-1]）。
    """

    params = (
        ("rsi_period", 12),
        ("up", 53),
        ("dn", 36),
        ("ema_period", 66),
        ("atr_period", 26),
        ("stop_atr", 0.5),
        ("cooldown", 12),
        ("can_short", 1),
    )

    def __init__(self):
        self.rsi_period = int(self.p.rsi_period)
        self.up = float(self.p.up)
        self.dn = float(self.p.dn)
        self.ema_period = int(self.p.ema_period)
        self.atr_period = int(self.p.atr_period)
        self.stop_atr = float(self.p.stop_atr)
        self.cooldown = int(self.p.cooldown)
        self.can_short = bool(int(self.p.can_short))

        if self.rsi_period < 2 or self.ema_period < 2 or self.atr_period < 2:
            raise ValueError("rsi_period / ema_period / atr_period 必须 >= 2")
        if self.stop_atr <= 0:
            raise ValueError("stop_atr 必须 > 0")
        if self.cooldown < 0:
            raise ValueError("cooldown 必须 >= 0")
        if not (0 < self.dn < self.up < 100):
            raise ValueError("参数约束必须满足：0 < dn < up < 100")

        # 使用 RSI_Safe 避免在单边行情中出现除零异常
        self.rsi = bt.indicators.RSI_Safe(self.data.close, period=self.rsi_period)
        self.ema = bt.indicators.EMA(self.data.close, period=self.ema_period)
        self.atr = bt.indicators.ATR(self.data, period=self.atr_period)

        self.pending_order = None
        self.entry_price = None
        self.entry_atr = None
        self.last_flat_bar = -10**9

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return
        if order.status in [order.Completed, order.Canceled, order.Margin, order.Rejected]:
            self.pending_order = None

        if order.status == order.Completed:
            if self.position.size != 0:
                self.entry_price = float(self.position.price)
                try:
                    self.entry_atr = float(self.atr[-1])
                except Exception:  # noqa: BLE001
                    self.entry_atr = float(self.atr[0])
            else:
                self.entry_price = None
                self.entry_atr = None
                self.last_flat_bar = len(self.data)

    def next(self):
        need_bars = max(self.rsi_period, self.ema_period, self.atr_period) + 1
        if len(self.data) < need_bars:
            return
        if self.pending_order is not None:
            return

        close_now = float(self.data.close[0])
        ema_prev = float(self.ema[-1])
        rsi_prev = float(self.rsi[-1])

        # 持仓先处理平仓
        if self.position.size > 0:
            atr_used = float(self.entry_atr) if self.entry_atr is not None else float(self.atr[-1])
            stop_price = float(self.entry_price) - atr_used * self.stop_atr
            if close_now < stop_price or rsi_prev < self.dn:
                self.pending_order = self.close()
            return

        if self.position.size < 0:
            atr_used = float(self.entry_atr) if self.entry_atr is not None else float(self.atr[-1])
            stop_price = float(self.entry_price) + atr_used * self.stop_atr
            if close_now > stop_price or rsi_prev > self.up:
                self.pending_order = self.close()
            return

        # 空仓冷却期
        if len(self.data) - int(self.last_flat_bar) <= self.cooldown:
            return

        # 空仓开仓
        if close_now > ema_prev and rsi_prev > self.up:
            self.pending_order = self.buy()
            return

        if self.can_short and close_now < ema_prev and rsi_prev < self.dn:
            self.pending_order = self.sell()
            return
