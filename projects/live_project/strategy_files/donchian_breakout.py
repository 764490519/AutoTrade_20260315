import backtrader as bt

STRATEGY_META = {
    "display_name": "Donchian 通道突破（固定仓位 + ATR过滤可开关）",
    "strategy_class": "DonchianChannelBreakoutStrategy",
    "params": {
        "entry_period": {
            "type": "int",
            "default": 55,
            "min": 2,
            "max": 120,
            "step": 1,
            "desc": "开仓通道周期。价格突破该周期上/下轨时触发开仓信号。",
        },
        "exit_period": {
            "type": "int",
            "default": 20,
            "min": 2,
            "max": 60,
            "step": 1,
            "desc": "平仓通道周期。用于 Donchian 出场边界计算。",
        },
        "atr_period": {
            "type": "int",
            "default": 14,
            "min": 2,
            "max": 120,
            "step": 1,
            "desc": "ATR 波动率计算周期。",
        },
        "atr_filter_enabled": {
            "type": "int",
            "default": 1,
            "min": 0,
            "max": 1,
            "step": 1,
            "desc": "ATR 开仓过滤开关：1=启用，0=关闭。",
        },
        "atr_ratio_min": {
            "type": "float",
            "default": 0.003,
            "min": 0.0001,
            "max": 0.05,
            "step": 0.0001,
            "desc": "最小波动阈值（ATR/价格）。低于该值视为震荡，禁止开仓（开关启用时生效）。",
        },
        "atr_ratio_max": {
            "type": "float",
            "default": 0.05,
            "min": 0.001,
            "max": 0.2,
            "step": 0.001,
            "desc": "最大波动阈值（ATR/价格）。高于该值视为高风险，禁止开仓（开关启用时生效）。",
        },
        "stop_atr_mult": {
            "type": "float",
            "default": 2.0,
            "min": 0.1,
            "max": 20.0,
            "step": 0.1,
            "desc": "固定止损 ATR 倍数。多头：入场价-ATR*倍数；空头：入场价+ATR*倍数。",
        },
        "trail_atr_mult": {
            "type": "float",
            "default": 2.0,
            "min": 0.1,
            "max": 20.0,
            "step": 0.1,
            "desc": "追踪止盈 ATR 倍数。基于入场时锁定 ATR 计算 trailing stop。",
        },
        "can_short": {
            "type": "int",
            "default": 1,
            "min": 0,
            "max": 1,
            "step": 1,
            "desc": "是否允许做空：1=允许，0=仅做多。",
        },
        "target_percent": {
            "type": "float",
            "default": 0.95,
            "min": 0.1,
            "max": 1.0,
            "step": 0.01,
            "desc": "固定目标仓位比例（多头为正、空头为负）。",
        },
    },
}


class DonchianChannelBreakoutStrategy(bt.Strategy):
    """
    Donchian 通道突破（固定仓位 + ATR过滤可开关 + 固定止损 + 追踪止盈）

    规则：
    1) ATR/价格过滤（仅影响开仓）：
       - 当 atr_filter_enabled=1 时：
         - ATR/价格 < atr_ratio_min：不开仓
         - ATR/价格 > atr_ratio_max：不开仓
       - 当 atr_filter_enabled=0 时：忽略 ATR 过滤
    2) ATR止损（开仓后生效，使用入场锁定 ATR）：
       - 多头：close < 入场价 - 入场ATR * stop_atr_mult -> 平多
       - 空头：close > 入场价 + 入场ATR * stop_atr_mult -> 平空
    3) 追踪止盈（开仓后生效，使用入场锁定 ATR）：
       - 多头：trailing_stop = 持仓后最高价 - trail_atr_mult * 入场ATR
         当 close < max(Donchian下轨(exit), trailing_stop) 时平多
       - 空头：trailing_stop = 持仓后最低价 + trail_atr_mult * 入场ATR
         当 close > min(Donchian上轨(exit), trailing_stop) 时平空
    4) 固定仓位（开仓时）：
       - 多头：target = +target_percent
       - 空头：target = -target_percent
    5) 开仓方向规则：
       - 开多：突破上轨（entry_period）
       - 开空：跌破下轨（entry_period，且 can_short=1）
       - 平仓触发后当根K线不反手

    为避免未来函数，信号判断使用上一根指标值（[-1]）。
    """

    params = (
        ("entry_period", 55),
        ("exit_period", 20),
        ("atr_period", 14),
        ("atr_filter_enabled", 1),
        ("atr_ratio_min", 0.003),
        ("atr_ratio_max", 0.05),
        ("stop_atr_mult", 2.0),
        ("trail_atr_mult", 2.0),
        ("can_short", 1),
        ("target_percent", 0.95),
    )

    def __init__(self):
        self.entry_period = int(self.params.entry_period)
        self.exit_period = int(self.params.exit_period)
        self.atr_period = int(self.params.atr_period)
        self.atr_filter_enabled = bool(int(self.params.atr_filter_enabled))
        self.atr_ratio_min = float(self.params.atr_ratio_min)
        self.atr_ratio_max = float(self.params.atr_ratio_max)
        self.stop_atr_mult = float(self.params.stop_atr_mult)
        self.trail_atr_mult = float(self.params.trail_atr_mult)
        self.can_short = bool(int(self.params.can_short))
        self.target_percent = float(self.params.target_percent)

        if self.entry_period < 2 or self.exit_period < 2 or self.atr_period < 2:
            raise ValueError("entry_period / exit_period / atr_period 必须 >= 2")
        if not (0 < self.target_percent <= 1):
            raise ValueError("target_percent 必须在 (0, 1] 区间")
        if self.atr_ratio_min <= 0 or self.atr_ratio_max <= 0:
            raise ValueError("atr_ratio_min / atr_ratio_max 必须 > 0")
        if self.atr_ratio_min >= self.atr_ratio_max:
            raise ValueError("atr_ratio_min 必须小于 atr_ratio_max")
        if self.stop_atr_mult <= 0:
            raise ValueError("stop_atr_mult 必须 > 0")
        if self.trail_atr_mult <= 0:
            raise ValueError("trail_atr_mult 必须 > 0")

        # 开仓通道（entry）
        self.entry_upper = bt.indicators.Highest(self.data.high, period=self.entry_period)
        self.entry_lower = bt.indicators.Lowest(self.data.low, period=self.entry_period)
        # 平仓通道（exit）
        self.exit_upper = bt.indicators.Highest(self.data.high, period=self.exit_period)
        self.exit_lower = bt.indicators.Lowest(self.data.low, period=self.exit_period)
        # ATR
        self.atr = bt.indicators.ATR(self.data, period=self.atr_period)

        self.pending_order = None
        self.last_entry_price = None
        self.last_entry_atr = None
        self.highest_since_entry = None
        self.lowest_since_entry = None

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return
        if order.status in [order.Completed, order.Canceled, order.Margin, order.Rejected]:
            self.pending_order = None

        if order.status == order.Completed:
            if self.position.size != 0:
                self.last_entry_price = float(self.position.price)
                try:
                    self.last_entry_atr = float(self.atr[-1])
                except Exception:  # noqa: BLE001
                    self.last_entry_atr = float(self.atr[0])
                self.highest_since_entry = self.last_entry_price
                self.lowest_since_entry = self.last_entry_price
            else:
                self.last_entry_price = None
                self.last_entry_atr = None
                self.highest_since_entry = None
                self.lowest_since_entry = None

    def next(self):
        need_bars = max(self.entry_period, self.exit_period, self.atr_period) + 1
        if len(self.data) < need_bars:
            return
        if self.pending_order is not None:
            return

        close_now = float(self.data.close[0])
        high_now = float(self.data.high[0])
        low_now = float(self.data.low[0])
        atr_prev = float(self.atr[-1])

        entry_upper_prev = float(self.entry_upper[-1])
        entry_lower_prev = float(self.entry_lower[-1])
        exit_upper_prev = float(self.exit_upper[-1])
        exit_lower_prev = float(self.exit_lower[-1])

        volatility_ok = True
        if self.atr_filter_enabled:
            price_denom = max(abs(close_now), 1e-12)
            atr_ratio = atr_prev / price_denom
            volatility_ok = self.atr_ratio_min <= atr_ratio <= self.atr_ratio_max

        # 先处理持仓平仓逻辑（不做同K线反手）
        if self.position.size > 0:
            self.highest_since_entry = (
                high_now if self.highest_since_entry is None else max(float(self.highest_since_entry), high_now)
            )

            if self.last_entry_price is not None and self.last_entry_atr is not None:
                long_stop = self.last_entry_price - self.last_entry_atr * self.stop_atr_mult
                if close_now < long_stop:
                    self.pending_order = self.order_target_percent(target=0.0)
                    return

            trail_atr_long = float(self.last_entry_atr) if self.last_entry_atr is not None else atr_prev
            trailing_stop_long = float(self.highest_since_entry) - self.trail_atr_mult * trail_atr_long
            long_exit_line = max(exit_lower_prev, trailing_stop_long)
            if close_now < long_exit_line:
                self.pending_order = self.order_target_percent(target=0.0)
            return

        if self.position.size < 0:
            self.lowest_since_entry = (
                low_now if self.lowest_since_entry is None else min(float(self.lowest_since_entry), low_now)
            )

            if self.last_entry_price is not None and self.last_entry_atr is not None:
                short_stop = self.last_entry_price + self.last_entry_atr * self.stop_atr_mult
                if close_now > short_stop:
                    self.pending_order = self.order_target_percent(target=0.0)
                    return

            trail_atr_short = float(self.last_entry_atr) if self.last_entry_atr is not None else atr_prev
            trailing_stop_short = float(self.lowest_since_entry) + self.trail_atr_mult * trail_atr_short
            short_exit_line = min(exit_upper_prev, trailing_stop_short)
            if close_now > short_exit_line:
                self.pending_order = self.order_target_percent(target=0.0)
            return

        # 空仓时开仓：可选 ATR 过滤
        if not volatility_ok:
            return

        if close_now > entry_upper_prev:
            self.pending_order = self.order_target_percent(target=self.target_percent)
            return

        if self.can_short and close_now < entry_lower_prev:
            self.pending_order = self.order_target_percent(target=-self.target_percent)
            return
