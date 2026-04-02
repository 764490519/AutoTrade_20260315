from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Callable

import backtrader as bt
import numpy as np
import pandas as pd
import vectorbt as vbt


class LeverageCryptoCommissionInfo(bt.CommInfoBase):
    """
    Backtrader 兼容手续费模型（兜底路径）。
    """

    params = (
        ("commission", 0.001),
        ("commtype", bt.CommInfoBase.COMM_PERC),
        ("percabs", True),
        ("stocklike", True),
        ("leverage", 1.0),
    )

    def getsize(self, price, cash):
        if price <= 0:
            return 0.0
        cash = float(cash)
        fee_factor = 1.0 + float(self.p.commission or 0.0)
        if fee_factor <= 0:
            fee_factor = 1.0
        return float(self.p.leverage) * cash / (float(price) * fee_factor)


class TargetPercentLeverageSizer(bt.Sizer):
    """
    Backtrader 兼容仓位Sizer（兜底路径）。
    """

    params = (
        ("percents", 95.0),
        ("leverage", 1.0),
    )

    def _getsizing(self, comminfo, cash, data, isbuy):  # noqa: ARG002
        price = float(data.close[0] or 0.0)
        if price <= 0:
            return 0.0

        perc = max(0.0, min(100.0, float(self.p.percents or 0.0)))
        lev = max(1e-12, float(self.p.leverage or 1.0))
        fee_factor = 1.0 + float(getattr(comminfo.p, "commission", 0.0) or 0.0)
        if fee_factor <= 0:
            fee_factor = 1.0

        notional = float(cash) * (perc / 100.0) * lev
        size = notional / (price * fee_factor)
        return max(0.0, float(size))


class EquityCurveAnalyzer(bt.Analyzer):
    def start(self) -> None:
        self.values = []

    def next(self) -> None:
        dt = self.strategy.datas[0].datetime.datetime(0)
        value = self.strategy.broker.getvalue()
        self.values.append((dt, value))

    def get_analysis(self):
        return self.values


class TradeDetailAnalyzer(bt.Analyzer):
    def start(self) -> None:
        self.trades = []
        self.local_trade_id = 0

    def notify_trade(self, trade) -> None:
        if not trade.isclosed:
            return
        self.local_trade_id += 1

        history = list(getattr(trade, "history", []) or [])
        entry = history[0] if history else None
        exit_ = history[-1] if history else None

        entry_status = entry.status if entry else None
        exit_status = exit_.status if exit_ else None
        entry_event = entry.event if entry else None
        exit_event = exit_.event if exit_ else None

        entry_dt = bt.num2date(entry_status.dt) if entry_status is not None else bt.num2date(trade.dtopen)
        exit_dt = bt.num2date(exit_status.dt) if exit_status is not None else bt.num2date(trade.dtclose)

        entry_price = float(entry_event.price) if entry_event is not None else float(getattr(trade, "price", 0.0))
        exit_price = float(exit_event.price) if exit_event is not None else float(getattr(trade, "price", 0.0))

        commission = float(getattr(trade, "pnl", 0.0) - getattr(trade, "pnlcomm", 0.0))

        self.trades.append(
            {
                "交易ID": int(self.local_trade_id),
                "方向": "LONG" if bool(getattr(trade, "long", True)) else "SHORT",
                "开仓时间": entry_dt,
                "平仓时间": exit_dt,
                "开仓价格": round(entry_price, 8),
                "平仓价格": round(exit_price, 8),
                "毛收益": round(float(getattr(trade, "pnl", 0.0)), 8),
                "净收益": round(float(getattr(trade, "pnlcomm", 0.0)), 8),
                "手续费": round(commission, 8),
                "交易后总资金": round(float(self.strategy.broker.getvalue()), 8),
            }
        )

    def get_analysis(self):
        return self.trades


@dataclass
class BacktestResult:
    metrics: dict[str, Any]
    equity_curve: pd.DataFrame
    trade_details: pd.DataFrame


class BinancePandasData(bt.feeds.PandasData):
    params = (
        ("datetime", None),
        ("open", "open"),
        ("high", "high"),
        ("low", "low"),
        ("close", "close"),
        ("volume", "volume"),
        ("openinterest", -1),
    )


def _coerce_input_df(df: pd.DataFrame) -> pd.DataFrame:
    data_df = df.copy()
    if isinstance(data_df.index, pd.DatetimeIndex) and data_df.index.tz is not None:
        data_df.index = data_df.index.tz_convert("UTC").tz_localize(None)
    return data_df


def _infer_periods_per_year(index: pd.Index) -> float:
    if len(index) < 3:
        return 365.0
    if not isinstance(index, pd.DatetimeIndex):
        return 365.0
    sec = pd.Series(index).diff().dt.total_seconds().dropna()
    sec = sec[sec > 0]
    if sec.empty:
        return 365.0
    median_sec = float(sec.median())
    if median_sec <= 0:
        return 365.0
    return max(1.0, 31_536_000.0 / median_sec)


def _max_drawdown_pct(equity: pd.Series) -> float:
    if equity.empty:
        return 0.0
    roll_max = equity.cummax()
    dd = equity / roll_max - 1.0
    return abs(float(dd.min() or 0.0)) * 100.0


def _annual_return_pct(initial_cash: float, final_value: float, index: pd.Index, periods_per_year: float) -> float:
    if initial_cash <= 0:
        return 0.0
    if final_value <= 0:
        return -100.0
    if len(index) < 2 or not isinstance(index, pd.DatetimeIndex):
        return (final_value / initial_cash - 1.0) * 100.0
    elapsed_sec = max(1.0, float((index[-1] - index[0]).total_seconds()))
    years = elapsed_sec / 31_536_000.0
    years = max(years, 1.0 / max(periods_per_year, 1.0))
    return (pow(final_value / initial_cash, 1.0 / years) - 1.0) * 100.0


def _sharpe_ratio(equity: pd.Series, periods_per_year: float) -> float | None:
    if equity.empty:
        return None
    ret = equity.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    if len(ret) < 2:
        return None
    std = float(ret.std(ddof=1))
    if std <= 1e-12:
        return None
    mean = float(ret.mean())
    return (mean / std) * float(np.sqrt(max(periods_per_year, 1.0)))


def _to_numeric_series(df: pd.DataFrame, col: str) -> pd.Series:
    return pd.to_numeric(df[col], errors="coerce").astype(float)


def _sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(window=period, min_periods=period).mean()


def _ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False, min_periods=period).mean()


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1.0 / float(period), adjust=False, min_periods=period).mean()


def _rsi_safe(close: pd.Series, period: int) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)

    avg_gain = gain.ewm(alpha=1.0 / float(period), adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1.0 / float(period), adjust=False, min_periods=period).mean()

    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    rsi = rsi.where(avg_loss > 0.0, 100.0)
    both_zero = (avg_gain <= 0.0) & (avg_loss <= 0.0)
    rsi = rsi.where(~both_zero, 50.0)
    return rsi


def _targets_ma_faber(df: pd.DataFrame, params: dict[str, Any]) -> np.ndarray:
    fast_period = int(params.get("fast_period", 11))
    slow_period = int(params.get("slow_period", 220))
    if fast_period < 2 or slow_period < 2:
        raise ValueError("fast_period / slow_period 必须 >= 2")

    close = _to_numeric_series(df, "close")
    fast = _sma(close, fast_period)
    slow = _sma(close, slow_period)

    n = len(df)
    targets = np.full(n, np.nan, dtype=float)
    pos = 0
    need_bars = slow_period + 1
    for i in range(n):
        if i < need_bars:
            continue
        fast_prev = float(fast.iloc[i - 1])
        slow_prev = float(slow.iloc[i - 1])
        if not np.isfinite(fast_prev) or not np.isfinite(slow_prev):
            continue
        if pos == 0 and fast_prev > slow_prev:
            targets[i] = 1.0
            pos = 1
        elif pos == 1 and fast_prev < slow_prev:
            targets[i] = 0.0
            pos = 0
    return targets


def _targets_fast_rsi_flip(df: pd.DataFrame, params: dict[str, Any]) -> np.ndarray:
    rsi_period = int(params.get("rsi_period", 12))
    up = float(params.get("up", 53))
    dn = float(params.get("dn", 36))
    ema_period = int(params.get("ema_period", 66))
    atr_period = int(params.get("atr_period", 26))
    stop_atr = float(params.get("stop_atr", 0.5))
    cooldown = int(params.get("cooldown", 12))
    can_short = bool(int(params.get("can_short", 1)))

    if rsi_period < 2 or ema_period < 2 or atr_period < 2:
        raise ValueError("rsi_period / ema_period / atr_period 必须 >= 2")
    if stop_atr <= 0:
        raise ValueError("stop_atr 必须 > 0")
    if cooldown < 0:
        raise ValueError("cooldown 必须 >= 0")
    if not (0 < dn < up < 100):
        raise ValueError("参数约束必须满足：0 < dn < up < 100")

    close = _to_numeric_series(df, "close")
    ema = _ema(close, ema_period)
    rsi = _rsi_safe(close, rsi_period)
    atr = _atr(_to_numeric_series(df, "high"), _to_numeric_series(df, "low"), close, atr_period)

    n = len(df)
    targets = np.full(n, np.nan, dtype=float)
    pos = 0
    entry_price: float | None = None
    entry_atr: float | None = None
    last_flat_bar = -10**9
    need_bars = max(rsi_period, ema_period, atr_period) + 1

    for i in range(n):
        if i < need_bars:
            continue
        close_now = float(close.iloc[i])
        ema_prev = float(ema.iloc[i - 1])
        rsi_prev = float(rsi.iloc[i - 1])
        atr_prev = float(atr.iloc[i - 1])
        if not np.isfinite(close_now) or not np.isfinite(ema_prev) or not np.isfinite(rsi_prev) or not np.isfinite(atr_prev):
            continue

        if pos > 0:
            atr_used = float(entry_atr) if entry_atr is not None else atr_prev
            stop_price = float(entry_price) - atr_used * stop_atr
            if close_now < stop_price or rsi_prev < dn:
                targets[i] = 0.0
                pos = 0
                entry_price = None
                entry_atr = None
                last_flat_bar = i
            continue

        if pos < 0:
            atr_used = float(entry_atr) if entry_atr is not None else atr_prev
            stop_price = float(entry_price) + atr_used * stop_atr
            if close_now > stop_price or rsi_prev > up:
                targets[i] = 0.0
                pos = 0
                entry_price = None
                entry_atr = None
                last_flat_bar = i
            continue

        if i - last_flat_bar <= cooldown:
            continue

        if close_now > ema_prev and rsi_prev > up:
            targets[i] = 1.0
            pos = 1
            entry_price = close_now
            entry_atr = atr_prev
            continue

        if can_short and close_now < ema_prev and rsi_prev < dn:
            targets[i] = -1.0
            pos = -1
            entry_price = close_now
            entry_atr = atr_prev
            continue

    return targets


def _targets_donchian_breakout(df: pd.DataFrame, params: dict[str, Any]) -> np.ndarray:
    entry_period = int(params.get("entry_period", 55))
    exit_period = int(params.get("exit_period", 20))
    atr_period = int(params.get("atr_period", 14))
    atr_filter_enabled = bool(int(params.get("atr_filter_enabled", 1)))
    atr_ratio_min = float(params.get("atr_ratio_min", 0.003))
    atr_ratio_max = float(params.get("atr_ratio_max", 0.05))
    stop_atr_mult = float(params.get("stop_atr_mult", 2.0))
    trail_atr_mult = float(params.get("trail_atr_mult", 2.0))
    can_short = bool(int(params.get("can_short", 1)))
    target_percent = float(params.get("target_percent", 0.95))

    if entry_period < 2 or exit_period < 2 or atr_period < 2:
        raise ValueError("entry_period / exit_period / atr_period 必须 >= 2")
    if not (0 < target_percent <= 1):
        raise ValueError("target_percent 必须在 (0, 1] 区间")
    if atr_ratio_min <= 0 or atr_ratio_max <= 0:
        raise ValueError("atr_ratio_min / atr_ratio_max 必须 > 0")
    if atr_ratio_min >= atr_ratio_max:
        raise ValueError("atr_ratio_min 必须小于 atr_ratio_max")
    if stop_atr_mult <= 0:
        raise ValueError("stop_atr_mult 必须 > 0")
    if trail_atr_mult <= 0:
        raise ValueError("trail_atr_mult 必须 > 0")

    close = _to_numeric_series(df, "close")
    high = _to_numeric_series(df, "high")
    low = _to_numeric_series(df, "low")

    entry_upper_prev = high.rolling(window=entry_period, min_periods=entry_period).max().shift(1)
    entry_lower_prev = low.rolling(window=entry_period, min_periods=entry_period).min().shift(1)
    exit_upper_prev = high.rolling(window=exit_period, min_periods=exit_period).max().shift(1)
    exit_lower_prev = low.rolling(window=exit_period, min_periods=exit_period).min().shift(1)
    atr_prev_ser = _atr(high, low, close, atr_period).shift(1)

    n = len(df)
    targets = np.full(n, np.nan, dtype=float)

    pos = 0
    entry_price: float | None = None
    entry_atr: float | None = None
    highest_since_entry: float | None = None
    lowest_since_entry: float | None = None
    need_bars = max(entry_period, exit_period, atr_period) + 1

    for i in range(n):
        if i < need_bars:
            continue

        close_now = float(close.iloc[i])
        high_now = float(high.iloc[i])
        low_now = float(low.iloc[i])
        atr_prev = float(atr_prev_ser.iloc[i])
        eu_prev = float(entry_upper_prev.iloc[i])
        el_prev = float(entry_lower_prev.iloc[i])
        xu_prev = float(exit_upper_prev.iloc[i])
        xl_prev = float(exit_lower_prev.iloc[i])

        values = (close_now, high_now, low_now, atr_prev, eu_prev, el_prev, xu_prev, xl_prev)
        if any(not np.isfinite(v) for v in values):
            continue

        volatility_ok = True
        if atr_filter_enabled:
            atr_ratio = atr_prev / max(abs(close_now), 1e-12)
            volatility_ok = atr_ratio_min <= atr_ratio <= atr_ratio_max

        if pos > 0:
            highest_since_entry = high_now if highest_since_entry is None else max(float(highest_since_entry), high_now)
            if entry_price is not None and entry_atr is not None:
                long_stop = float(entry_price) - float(entry_atr) * stop_atr_mult
                if close_now < long_stop:
                    targets[i] = 0.0
                    pos = 0
                    entry_price = None
                    entry_atr = None
                    highest_since_entry = None
                    lowest_since_entry = None
                    continue

            trail_atr = float(entry_atr) if entry_atr is not None else atr_prev
            trailing_stop_long = float(highest_since_entry) - trail_atr_mult * trail_atr
            long_exit_line = max(xl_prev, trailing_stop_long)
            if close_now < long_exit_line:
                targets[i] = 0.0
                pos = 0
                entry_price = None
                entry_atr = None
                highest_since_entry = None
                lowest_since_entry = None
            continue

        if pos < 0:
            lowest_since_entry = low_now if lowest_since_entry is None else min(float(lowest_since_entry), low_now)
            if entry_price is not None and entry_atr is not None:
                short_stop = float(entry_price) + float(entry_atr) * stop_atr_mult
                if close_now > short_stop:
                    targets[i] = 0.0
                    pos = 0
                    entry_price = None
                    entry_atr = None
                    highest_since_entry = None
                    lowest_since_entry = None
                    continue

            trail_atr = float(entry_atr) if entry_atr is not None else atr_prev
            trailing_stop_short = float(lowest_since_entry) + trail_atr_mult * trail_atr
            short_exit_line = min(xu_prev, trailing_stop_short)
            if close_now > short_exit_line:
                targets[i] = 0.0
                pos = 0
                entry_price = None
                entry_atr = None
                highest_since_entry = None
                lowest_since_entry = None
            continue

        if not volatility_ok:
            continue

        if close_now > eu_prev:
            targets[i] = float(target_percent)
            pos = 1
            entry_price = close_now
            entry_atr = atr_prev
            highest_since_entry = close_now
            lowest_since_entry = close_now
            continue

        if can_short and close_now < el_prev:
            targets[i] = -float(target_percent)
            pos = -1
            entry_price = close_now
            entry_atr = atr_prev
            highest_since_entry = close_now
            lowest_since_entry = close_now
            continue

    return targets


_VECTORBT_STRATEGY_RUNNERS: dict[str, Callable[[pd.DataFrame, dict[str, Any]], np.ndarray]] = {
    "FaberMaTrendStrategy": _targets_ma_faber,
    "FastRsiFlipStrategy": _targets_fast_rsi_flip,
    "DonchianChannelBreakoutStrategy": _targets_donchian_breakout,
}


def _lookup_equity_at(equity: pd.Series, ts: Any) -> float:
    if equity.empty:
        return 0.0
    if ts in equity.index:
        return float(equity.loc[ts])

    if isinstance(ts, (int, np.integer)):
        idx = int(ts)
        if 0 <= idx < len(equity):
            return float(equity.iloc[idx])

    try:
        ts_dt = pd.to_datetime(ts)
        if ts_dt in equity.index:
            return float(equity.loc[ts_dt])
    except Exception:  # noqa: BLE001
        pass
    return float(equity.iloc[-1])


def _vectorbt_trade_details(closed: pd.DataFrame, equity: pd.Series) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if closed.empty:
        return pd.DataFrame()

    for i, (_, row) in enumerate(closed.iterrows(), start=1):
        direction = "LONG" if str(row.get("Direction", "")).strip().lower() == "long" else "SHORT"
        entry_ts = row.get("Entry Timestamp")
        exit_ts = row.get("Exit Timestamp")
        entry_price = float(row.get("Avg Entry Price", 0.0) or 0.0)
        exit_price = float(row.get("Avg Exit Price", 0.0) or 0.0)
        entry_fees = float(row.get("Entry Fees", 0.0) or 0.0)
        exit_fees = float(row.get("Exit Fees", 0.0) or 0.0)
        net = float(row.get("PnL", 0.0) or 0.0)
        fee = entry_fees + exit_fees
        gross = net + fee

        rows.append(
            {
                "交易ID": int(i),
                "方向": direction,
                "开仓时间": pd.to_datetime(entry_ts, errors="coerce"),
                "平仓时间": pd.to_datetime(exit_ts, errors="coerce"),
                "开仓价格": round(entry_price, 8),
                "平仓价格": round(exit_price, 8),
                "毛收益": round(gross, 8),
                "净收益": round(net, 8),
                "手续费": round(fee, 8),
                "交易后总资金": round(_lookup_equity_at(equity, exit_ts), 8),
            }
        )

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(by="开仓时间").reset_index(drop=True)
    return out


def _run_backtest_vectorbt(
    df: pd.DataFrame,
    strategy_cls,
    strategy_params: dict[str, Any],
    initial_cash: float,
    commission: float,
    position_percent: float,
    leverage: float,
    include_details: bool,
) -> BacktestResult:
    data_df = _coerce_input_df(df)
    strategy_name = str(getattr(strategy_cls, "__name__", "") or "")
    runner = _VECTORBT_STRATEGY_RUNNERS.get(strategy_name)
    if runner is None:
        raise ValueError(f"策略 {strategy_name} 暂未接入 vectorbt 引擎")

    close = _to_numeric_series(data_df, "close")
    if close.empty:
        raise ValueError("输入K线为空")

    target = pd.Series(runner(data_df, strategy_params), index=data_df.index, dtype=float)
    if strategy_name != "DonchianChannelBreakoutStrategy":
        target = target * (float(position_percent) / 100.0)

    target = target.clip(lower=-1.0, upper=1.0)
    sim_init_cash = float(initial_cash) * float(leverage)
    if sim_init_cash <= 0:
        raise ValueError("initial_cash * leverage 必须大于 0")

    pf = vbt.Portfolio.from_orders(
        close=close,
        size=target,
        size_type="targetpercent",
        fees=float(commission),
        init_cash=sim_init_cash,
        freq=None,
    )

    equity_sim = pf.value()
    equity = float(initial_cash) + (equity_sim - sim_init_cash)
    equity = equity.astype(float)

    final_value = float(equity.iloc[-1]) if not equity.empty else float(initial_cash)
    total_return = (final_value / float(initial_cash) - 1.0) * 100.0
    periods_per_year = _infer_periods_per_year(data_df.index)
    sharpe = _sharpe_ratio(equity, periods_per_year)
    annual_ret = _annual_return_pct(float(initial_cash), final_value, data_df.index, periods_per_year)
    max_dd = _max_drawdown_pct(equity)

    trades = pf.trades.records_readable
    closed = trades.copy()
    if "Status" in closed.columns:
        closed = closed[closed["Status"].astype(str).str.lower() == "closed"]

    total_trades = int(len(closed))
    wins = int((pd.to_numeric(closed.get("PnL"), errors="coerce").fillna(0.0) > 0.0).sum()) if total_trades > 0 else 0
    losses = int((pd.to_numeric(closed.get("PnL"), errors="coerce").fillna(0.0) <= 0.0).sum()) if total_trades > 0 else 0
    win_rate = (wins / total_trades * 100.0) if total_trades > 0 else 0.0

    if include_details:
        equity_curve = pd.DataFrame({"datetime": data_df.index, "equity": equity.values})
        trade_details = _vectorbt_trade_details(closed=closed, equity=equity)
    else:
        equity_curve = pd.DataFrame(columns=["datetime", "equity"])
        trade_details = pd.DataFrame()

    metrics = {
        "初始资金": round(float(initial_cash), 2),
        "最终资金": round(final_value, 2),
        "总收益率(%)": round(total_return, 2),
        "年化收益率(%)": round(float(annual_ret), 2),
        "Sharpe": None if sharpe is None else round(float(sharpe), 4),
        "最大回撤(%)": round(float(max_dd), 2),
        "总交易次数": total_trades,
        "盈利次数": wins,
        "亏损次数": losses,
        "胜率(%)": round(win_rate, 2),
    }
    return BacktestResult(metrics=metrics, equity_curve=equity_curve, trade_details=trade_details)


def _run_backtest_backtrader(
    df: pd.DataFrame,
    strategy_cls,
    strategy_params: dict[str, Any],
    initial_cash: float = 10_000,
    commission: float = 0.001,
    position_percent: float = 95,
    leverage: float = 1.0,
    include_details: bool = True,
) -> BacktestResult:
    data_df = _coerce_input_df(df)
    include_details = bool(include_details)
    cerebro = bt.Cerebro(stdstats=False, tradehistory=include_details)
    data = BinancePandasData(dataname=data_df)

    cerebro.adddata(data)
    cerebro.addstrategy(strategy_cls, **strategy_params)

    cerebro.broker.setcash(initial_cash)
    comminfo = LeverageCryptoCommissionInfo(commission=commission, leverage=leverage)
    cerebro.broker.addcommissioninfo(comminfo)
    cerebro.addsizer(TargetPercentLeverageSizer, percents=position_percent, leverage=leverage)

    cerebro.addanalyzer(
        bt.analyzers.SharpeRatio_A,
        _name="sharpe",
        timeframe=bt.TimeFrame.Days,
        compression=1,
        factor=365,
        riskfreerate=0.0,
        convertrate=True,
        annualize=True,
    )
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name="drawdown")
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="trades")
    cerebro.addanalyzer(
        bt.analyzers.Returns,
        _name="returns",
        timeframe=bt.TimeFrame.Days,
        compression=1,
        tann=365,
    )
    if include_details:
        cerebro.addanalyzer(EquityCurveAnalyzer, _name="equity")
        cerebro.addanalyzer(TradeDetailAnalyzer, _name="trade_details")

    result = cerebro.run()[0]

    final_value = cerebro.broker.getvalue()
    total_return = (final_value / initial_cash - 1) * 100

    sharpe_data = result.analyzers.sharpe.get_analysis()
    drawdown_data = result.analyzers.drawdown.get_analysis()
    trade_data = result.analyzers.trades.get_analysis()
    returns_data = result.analyzers.returns.get_analysis()

    total_trades = int(trade_data.get("total", {}).get("closed", 0) or 0)
    won_trades = int(trade_data.get("won", {}).get("total", 0) or 0)
    lost_trades = int(trade_data.get("lost", {}).get("total", 0) or 0)
    win_rate = (won_trades / total_trades * 100) if total_trades > 0 else 0.0

    if include_details:
        equity_curve = pd.DataFrame(result.analyzers.equity.get_analysis(), columns=["datetime", "equity"])
        if not equity_curve.empty:
            equity_curve["datetime"] = pd.to_datetime(equity_curve["datetime"])

        trade_details = pd.DataFrame(result.analyzers.trade_details.get_analysis())
        if not trade_details.empty:
            trade_details["开仓时间"] = pd.to_datetime(trade_details["开仓时间"])
            trade_details["平仓时间"] = pd.to_datetime(trade_details["平仓时间"])
            trade_details = trade_details.sort_values(by="开仓时间").reset_index(drop=True)
    else:
        equity_curve = pd.DataFrame(columns=["datetime", "equity"])
        trade_details = pd.DataFrame()

    metrics = {
        "初始资金": round(initial_cash, 2),
        "最终资金": round(final_value, 2),
        "总收益率(%)": round(total_return, 2),
        "年化收益率(%)": round(float(returns_data.get("rnorm100", 0.0) or 0.0), 2),
        "Sharpe": None if sharpe_data.get("sharperatio") is None else round(float(sharpe_data["sharperatio"]), 4),
        "最大回撤(%)": round(float(drawdown_data.get("max", {}).get("drawdown", 0.0) or 0.0), 2),
        "总交易次数": total_trades,
        "盈利次数": won_trades,
        "亏损次数": lost_trades,
        "胜率(%)": round(win_rate, 2),
    }

    return BacktestResult(metrics=metrics, equity_curve=equity_curve, trade_details=trade_details)


def run_backtest(
    df: pd.DataFrame,
    strategy_cls,
    strategy_params: dict[str, Any],
    initial_cash: float = 10_000,
    commission: float = 0.001,
    position_percent: float = 95,
    leverage: float = 1.0,
    include_details: bool = True,
) -> BacktestResult:
    leverage = float(leverage)
    if leverage <= 0:
        raise ValueError("leverage 必须大于 0")

    position_percent = float(position_percent)
    if position_percent <= 0:
        raise ValueError("position_percent 必须大于 0")

    engine_pref = str(os.getenv("AUTOTRADE_BACKTEST_ENGINE", "auto")).strip().lower()
    if engine_pref not in {"vectorbt", "backtrader", "auto"}:
        engine_pref = "auto"

    strategy_name = str(getattr(strategy_cls, "__name__", "") or "")
    has_vectorbt_runner = strategy_name in _VECTORBT_STRATEGY_RUNNERS

    if engine_pref == "backtrader":
        return _run_backtest_backtrader(
            df=df,
            strategy_cls=strategy_cls,
            strategy_params=strategy_params,
            initial_cash=initial_cash,
            commission=commission,
            position_percent=position_percent,
            leverage=leverage,
            include_details=include_details,
        )

    if has_vectorbt_runner:
        try:
            return _run_backtest_vectorbt(
                df=df,
                strategy_cls=strategy_cls,
                strategy_params=strategy_params,
                initial_cash=initial_cash,
                commission=commission,
                position_percent=position_percent,
                leverage=leverage,
                include_details=include_details,
            )
        except Exception:
            if engine_pref == "vectorbt":
                raise

    if engine_pref == "vectorbt" and not has_vectorbt_runner:
        raise ValueError(f"策略 {strategy_name} 暂未接入 vectorbt 引擎，请切换 AUTOTRADE_BACKTEST_ENGINE=auto/backtrader")

    return _run_backtest_backtrader(
        df=df,
        strategy_cls=strategy_cls,
        strategy_params=strategy_params,
        initial_cash=initial_cash,
        commission=commission,
        position_percent=position_percent,
        leverage=leverage,
        include_details=include_details,
    )
