from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import backtrader as bt
import pandas as pd


class FractionalCryptoCommissionInfo(bt.CommInfoBase):
    """支持数字货币小数下单的手续费模型。"""

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
        price = float(price)

        # 为买入预留手续费，避免 100% 目标仓位因佣金导致 Margin 拒单
        fee_factor = 1.0 + float(self.p.commission or 0.0)
        if fee_factor <= 0:
            fee_factor = 1.0

        return float(self.p.leverage) * cash / (price * fee_factor)


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
        # 每次回测从 1 开始重新编号，避免使用 Backtrader 全局 ref 导致跨回测递增
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

    data_df = df.copy()
    if isinstance(data_df.index, pd.DatetimeIndex) and data_df.index.tz is not None:
        data_df.index = data_df.index.tz_convert("UTC").tz_localize(None)

    # 仅在需要逐笔明细时启用 tradehistory，可显著降低参数优化阶段开销
    include_details = bool(include_details)
    cerebro = bt.Cerebro(stdstats=False, tradehistory=include_details)
    data = BinancePandasData(dataname=data_df)

    cerebro.adddata(data)
    cerebro.addstrategy(strategy_cls, **strategy_params)

    cerebro.broker.setcash(initial_cash)
    comminfo = FractionalCryptoCommissionInfo(commission=commission, leverage=leverage)
    cerebro.broker.addcommissioninfo(comminfo)
    cerebro.addsizer(bt.sizers.PercentSizer, percents=position_percent)

    # 使用日收益计算 Sharpe，避免默认 Years 仅少量样本导致异常放大
    # Binance 为 7x24 市场，年化因子按 365
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
