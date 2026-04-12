from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import pandas as pd
import vectorbt as vbt


POSITION_SIZING_PERCENT_EQUITY = "percent_equity"
POSITION_SIZING_FIXED_AMOUNT = "fixed_amount"


@dataclass
class BacktestResult:
    metrics: dict[str, Any]
    equity_curve: pd.DataFrame
    trade_details: pd.DataFrame


def _coerce_input_df(df: pd.DataFrame) -> pd.DataFrame:
    data_df = df.copy()
    if isinstance(data_df.index, pd.DatetimeIndex) and data_df.index.tz is not None:
        data_df.index = data_df.index.tz_convert("UTC").tz_localize(None)
    return data_df


def _infer_periods_per_year(index: pd.Index) -> float:
    if len(index) < 3 or not isinstance(index, pd.DatetimeIndex):
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


def _resolve_strategy_callable(strategy_cls) -> Callable[[pd.DataFrame, dict[str, Any]], np.ndarray | pd.Series]:
    if strategy_cls is None:
        raise ValueError("strategy_cls 不能为空")

    maybe = getattr(strategy_cls, "generate_targets", None)
    if callable(maybe):
        return maybe

    if callable(strategy_cls):
        return strategy_cls

    raise ValueError("策略对象必须可调用，或提供 generate_targets(df, params) 方法")


def _strategy_name(strategy_cls) -> str:
    return str(getattr(strategy_cls, "__name__", strategy_cls.__class__.__name__))


def _normalize_position_sizing_mode(value: Any) -> str:
    text = str(value or POSITION_SIZING_PERCENT_EQUITY).strip().lower()
    if text in {
        "percent",
        "percent_equity",
        "pct",
        "equity_percent",
        "account_percent",
        "账户百分比",
        "百分比",
    }:
        return POSITION_SIZING_PERCENT_EQUITY
    if text in {
        "fixed",
        "fixed_amount",
        "amount",
        "固定金额",
        "固定入场金额",
    }:
        return POSITION_SIZING_FIXED_AMOUNT
    raise ValueError("position_sizing_mode 仅支持 'percent_equity' 或 'fixed_amount'")


def _build_raw_target_series(df: pd.DataFrame, strategy_cls, strategy_params: dict[str, Any]) -> pd.Series:
    fn = _resolve_strategy_callable(strategy_cls)
    raw = fn(df, dict(strategy_params or {}))
    target = pd.Series(raw, index=df.index, dtype=float)
    if len(target) != len(df):
        raise ValueError("策略返回的目标序列长度与输入K线长度不一致")
    return target


def _apply_position_percent(
    raw_target: pd.Series,
    strategy_cls,
    position_percent: float,
    *,
    leverage: float = 1.0,
    apply_leverage_to_percent: bool = False,
) -> pd.Series:
    use_global = getattr(strategy_cls, "USE_GLOBAL_POSITION_PERCENT", True)
    out = raw_target.astype(float).clip(lower=-1.0, upper=1.0)
    if bool(use_global):
        out = out * (float(position_percent) / 100.0)
    if bool(apply_leverage_to_percent):
        out = out * float(leverage)
        return out
    return out.clip(lower=-1.0, upper=1.0)


def build_target_series(
    df: pd.DataFrame,
    strategy_cls,
    strategy_params: dict[str, Any],
    position_percent: float = 95,
    position_sizing_mode: str = POSITION_SIZING_PERCENT_EQUITY,
    fixed_trade_amount: float | None = None,
    leverage: float = 1.0,
    apply_leverage_to_percent: bool = False,
) -> pd.Series:
    """
    生成传给 vectorbt 的 size 序列：
    - percent_equity: size_type=targetpercent，对应值范围通常在 [-1, 1]
    - fixed_amount:   size_type=targetvalue，对应值为目标名义金额（USDT）
    """
    data_df = _coerce_input_df(df)
    raw_target = _build_raw_target_series(data_df, strategy_cls, strategy_params)

    mode = _normalize_position_sizing_mode(position_sizing_mode)
    if mode == POSITION_SIZING_PERCENT_EQUITY:
        position_percent = float(position_percent)
        if position_percent <= 0:
            raise ValueError("position_percent 必须大于 0")
        return _apply_position_percent(
            raw_target,
            strategy_cls,
            position_percent,
            leverage=leverage,
            apply_leverage_to_percent=apply_leverage_to_percent,
        )

    amount = float(fixed_trade_amount if fixed_trade_amount is not None else 0.0)
    if amount <= 0:
        raise ValueError("fixed_trade_amount 必须大于 0")

    lev = float(leverage)
    if lev <= 0:
        raise ValueError("leverage 必须大于 0")

    # fixed_amount 视为“保证金金额”，实际目标名义金额 = fixed_trade_amount * leverage
    target_value = raw_target.astype(float) * amount * lev
    return target_value


def infer_latest_signal(
    df: pd.DataFrame,
    strategy_cls,
    strategy_params: dict[str, Any],
    position_percent: float = 95,
    position_sizing_mode: str = POSITION_SIZING_PERCENT_EQUITY,
    fixed_trade_amount: float | None = None,
    leverage: float = 1.0,
) -> dict[str, Any]:
    data_df = _coerce_input_df(df)
    if data_df.empty:
        raise ValueError("输入K线为空")

    target = build_target_series(
        data_df,
        strategy_cls,
        strategy_params,
        position_percent=position_percent,
        position_sizing_mode=position_sizing_mode,
        fixed_trade_amount=fixed_trade_amount,
        leverage=leverage,
        apply_leverage_to_percent=False,
    )
    held = target.ffill().fillna(0.0)
    latest_pos = float(held.iloc[-1])
    signal = "FLAT"
    if latest_pos > 0:
        signal = "LONG"
    elif latest_pos < 0:
        signal = "SHORT"

    close = pd.to_numeric(data_df["close"], errors="coerce").astype(float)
    return {
        "signal": signal,
        "strategy_position": latest_pos,
        "latest_bar_time": data_df.index[-1].to_pydatetime() if isinstance(data_df.index, pd.DatetimeIndex) else data_df.index[-1],
        "latest_close": float(close.iloc[-1]),
        "bars": int(len(data_df)),
        "position_sizing_mode": _normalize_position_sizing_mode(position_sizing_mode),
    }


def _lookup_equity_at(equity: pd.Series, ts: Any) -> float:
    if equity.empty:
        return 0.0
    if ts in equity.index:
        return float(equity.loc[ts])
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


def _build_portfolio(
    *,
    close: pd.Series,
    size: pd.Series,
    size_type: str,
    open_: pd.Series,
    commission: float,
    sim_init_cash: float,
):
    return vbt.Portfolio.from_orders(
        close=close,
        size=size,
        size_type=size_type,
        price=open_,
        fees=float(commission),
        init_cash=sim_init_cash,
        freq=None,
    )


def _estimate_virtual_init_cash_for_fixed_mode(size: pd.Series, initial_cash: float) -> float:
    """
    fixed_amount 回测需要保持“每笔固定名义金额”稳定，不受账户权益下滑影响。
    因此给 vectorbt 一个更大的“虚拟资金池”，避免因现金不足出现缩单/拒单。
    """
    s = pd.to_numeric(size, errors="coerce").astype(float)
    finite = s.replace([np.inf, -np.inf], np.nan).dropna()
    if finite.empty:
        return max(float(initial_cash), 1.0)

    max_notional = float(np.abs(finite).max())
    if not np.isfinite(max_notional) or max_notional <= 0.0:
        return max(float(initial_cash), 1.0)

    non_zero_cmd = finite[np.abs(finite) > 1e-12]
    cmd_count = int(len(non_zero_cmd))
    reserve_units = int(max(10, min(200_000, cmd_count + 5)))
    virtual_cash = max(float(initial_cash), max_notional * float(reserve_units))
    return float(virtual_cash)


def _run_with_liquidation_guard(
    *,
    close: pd.Series,
    open_: pd.Series,
    size: pd.Series,
    size_type: str,
    position_sizing_mode: str,
    leverage: float,
    commission: float,
    initial_cash: float,
    sim_init_cash: float,
) -> tuple[Any, pd.Series, pd.Series, int | None]:
    """
    防止“权益<=0 后仍继续交易”：
    - 若检测到权益<=0，则从该 bar 起强制目标仓位为 0
    - 若仍<=0，则从爆仓 bar 起将权益固定为 0
    """
    mode = _normalize_position_sizing_mode(position_sizing_mode)
    lev = float(leverage)
    if lev <= 0:
        raise ValueError("leverage 必须大于 0")

    size_used = size.copy()
    liquidation_idx: int | None = None

    def _equity_from_pf_value_cross(value: pd.Series) -> pd.Series:
        value = value.astype(float)
        return (float(initial_cash) + (value - sim_init_cash)).astype(float)

    def _equity_from_pf_value_isolated_percent(value: pd.Series, asset_value: pd.Series) -> tuple[pd.Series, int | None]:
        value = value.astype(float)
        asset_value = asset_value.astype(float)
        prev = value.shift(1)
        if len(prev) > 0:
            prev.iloc[0] = float(initial_cash)
        prev = prev.replace(0.0, np.nan)
        base_ret = (value / prev - 1.0).replace([np.inf, -np.inf], np.nan).fillna(0.0)

        cmd = size_used.astype(float)
        pos = cmd.ffill().fillna(0.0)
        lev_ret = pd.Series(0.0, index=value.index, dtype=float)
        liq_idx_local: int | None = None
        liquidated_active = False

        for i in range(len(value)):
            if i == 0:
                continue
            cmd_i = cmd.iloc[i]
            if np.isfinite(cmd_i):
                liquidated_active = False

            p_alloc = abs(float(pos.iloc[i]))
            if p_alloc <= 1e-12:
                lev_ret.iloc[i] = 0.0
                liquidated_active = False
                continue

            raw = float(base_ret.iloc[i]) * lev
            floor = -p_alloc
            if liquidated_active:
                lev_ret.iloc[i] = 0.0
                continue

            if raw < floor:
                lev_ret.iloc[i] = floor
                liquidated_active = True
                if liq_idx_local is None:
                    liq_idx_local = int(i)
            else:
                lev_ret.iloc[i] = raw

        equity = float(initial_cash) * (1.0 + lev_ret).cumprod()
        return equity.astype(float), liq_idx_local

    if mode == POSITION_SIZING_PERCENT_EQUITY:
        pf = _build_portfolio(
            close=close,
            size=size_used,
            size_type=size_type,
            open_=open_,
            commission=commission,
            sim_init_cash=sim_init_cash,
        )
        equity, liquidation_idx = _equity_from_pf_value_isolated_percent(pf.value(), pf.asset_value())
        # isolated 模式下仅允许单笔爆仓，不应导致后续全局停摆；仅当总权益<=0时归零并停止
        bad_idx = np.where(np.asarray(equity.values, dtype=float) <= 0.0)[0]
        if bad_idx.size > 0:
            idx = int(bad_idx[0])
            liquidation_idx = idx if liquidation_idx is None else min(liquidation_idx, idx)
            equity.iloc[idx:] = 0.0
        return pf, equity, size_used, liquidation_idx

    if mode == POSITION_SIZING_FIXED_AMOUNT:
        # 固定金额模式用于评估“固定下注规模”的长期能力：
        # 不因账户权益下降而停止后续交易，因此不启用全局清算停摆逻辑。
        pf = _build_portfolio(
            close=close,
            size=size_used,
            size_type=size_type,
            open_=open_,
            commission=commission,
            sim_init_cash=sim_init_cash,
        )
        equity = _equity_from_pf_value_cross(pf.value())
        return pf, equity, size_used, None

    for _ in range(3):
        pf = _build_portfolio(
            close=close,
            size=size_used,
            size_type=size_type,
            open_=open_,
            commission=commission,
            sim_init_cash=sim_init_cash,
        )
        equity = _equity_from_pf_value_cross(pf.value())

        bad_idx = np.where(np.asarray(equity.values, dtype=float) <= 0.0)[0]
        if bad_idx.size == 0:
            return pf, equity, size_used, liquidation_idx

        idx = int(bad_idx[0])
        liquidation_idx = idx if liquidation_idx is None else min(liquidation_idx, idx)
        size_used.iloc[idx:] = 0.0

    # 多轮后仍<=0，按爆仓处理
    pf = _build_portfolio(
        close=close,
        size=size_used,
        size_type=size_type,
        open_=open_,
        commission=commission,
        sim_init_cash=sim_init_cash,
    )
    equity = _equity_from_pf_value_cross(pf.value())
    bad_idx = np.where(np.asarray(equity.values, dtype=float) <= 0.0)[0]
    if bad_idx.size > 0:
        idx = int(bad_idx[0])
        liquidation_idx = idx if liquidation_idx is None else min(liquidation_idx, idx)
        equity.iloc[liquidation_idx:] = 0.0

    return pf, equity, size_used, liquidation_idx


def run_backtest(
    df: pd.DataFrame,
    strategy_cls,
    strategy_params: dict[str, Any],
    initial_cash: float = 10_000,
    commission: float = 0.001,
    position_percent: float = 95,
    leverage: float = 1.0,
    include_details: bool = True,
    position_sizing_mode: str = POSITION_SIZING_PERCENT_EQUITY,
    fixed_trade_amount: float | None = None,
    evaluation_start: Any | None = None,
) -> BacktestResult:
    leverage = float(leverage)
    if leverage <= 0:
        raise ValueError("leverage 必须大于 0")

    mode = _normalize_position_sizing_mode(position_sizing_mode)
    effective_fixed_trade_amount: float | None = None
    if mode == POSITION_SIZING_PERCENT_EQUITY:
        position_percent = float(position_percent)
        if position_percent <= 0:
            raise ValueError("position_percent 必须大于 0")
    else:
        if fixed_trade_amount is None:
            effective_fixed_trade_amount = float(initial_cash)
        else:
            effective_fixed_trade_amount = float(fixed_trade_amount)
        if effective_fixed_trade_amount <= 0:
            raise ValueError("fixed_trade_amount（固定模式下默认等于初始资金）必须大于 0")

    data_df = _coerce_input_df(df)
    close = pd.to_numeric(data_df["close"], errors="coerce").astype(float)
    open_ = pd.to_numeric(data_df["open"], errors="coerce").astype(float)
    if close.empty:
        raise ValueError("输入K线为空")

    eval_start_ts: pd.Timestamp | None = None
    if evaluation_start is not None and isinstance(data_df.index, pd.DatetimeIndex):
        eval_start_ts_raw = pd.to_datetime(evaluation_start, errors="coerce", utc=True)
        if pd.isna(eval_start_ts_raw):
            raise ValueError(f"evaluation_start 无法解析: {evaluation_start}")
        eval_start_ts = pd.Timestamp(eval_start_ts_raw).tz_convert("UTC").tz_localize(None)

    size = build_target_series(
        data_df,
        strategy_cls,
        strategy_params,
        position_percent=position_percent,
        position_sizing_mode=mode,
        fixed_trade_amount=effective_fixed_trade_amount,
        leverage=leverage,
        apply_leverage_to_percent=False,
    )
    if eval_start_ts is not None:
        size = size.copy()
        size.loc[size.index < eval_start_ts] = 0.0

    size_type = "targetpercent" if mode == POSITION_SIZING_PERCENT_EQUITY else "targetvalue"

    if mode == POSITION_SIZING_PERCENT_EQUITY:
        sim_init_cash = float(initial_cash)
    else:
        # fixed_amount 模式：使用“虚拟资金池”保证每笔固定金额不会因权益回撤被缩单
        sim_init_cash = _estimate_virtual_init_cash_for_fixed_mode(size=size, initial_cash=float(initial_cash))
    if sim_init_cash <= 0:
        raise ValueError("sim_init_cash 必须大于 0")

    pf, equity, size_used, liquidation_idx = _run_with_liquidation_guard(
        close=close,
        open_=open_,
        size=size,
        size_type=size_type,
        position_sizing_mode=mode,
        leverage=leverage,
        commission=float(commission),
        initial_cash=float(initial_cash),
        sim_init_cash=sim_init_cash,
    )
    if eval_start_ts is not None:
        eval_mask = data_df.index >= eval_start_ts
        if not bool(np.asarray(eval_mask).any()):
            raise ValueError("evaluation_start 晚于数据末尾，无法评估")
        equity_eval = equity.loc[eval_mask]
        data_df_eval = data_df.loc[eval_mask]
    else:
        equity_eval = equity
        data_df_eval = data_df

    initial_cash_eval = float(equity_eval.iloc[0]) if not equity_eval.empty else float(initial_cash)
    if initial_cash_eval <= 0:
        initial_cash_eval = float(initial_cash)
    final_value = max(0.0, float(equity_eval.iloc[-1])) if not equity_eval.empty else float(initial_cash_eval)
    total_return = (final_value / float(initial_cash_eval) - 1.0) * 100.0
    periods_per_year = _infer_periods_per_year(data_df_eval.index)
    sharpe = _sharpe_ratio(equity_eval, periods_per_year)
    annual_ret = _annual_return_pct(float(initial_cash_eval), final_value, data_df_eval.index, periods_per_year)
    max_dd = _max_drawdown_pct(equity_eval)

    trades = pf.trades.records_readable
    closed = trades.copy()
    if "Status" in closed.columns:
        closed = closed[closed["Status"].astype(str).str.lower() == "closed"]
    if eval_start_ts is not None and not closed.empty and "Entry Timestamp" in closed.columns:
        entry_ts = pd.to_datetime(closed["Entry Timestamp"], errors="coerce", utc=True).dt.tz_localize(None)
        closed = closed[entry_ts >= eval_start_ts]

    total_trades = int(len(closed))
    wins = int((pd.to_numeric(closed.get("PnL"), errors="coerce").fillna(0.0) > 0.0).sum()) if total_trades > 0 else 0
    losses = int((pd.to_numeric(closed.get("PnL"), errors="coerce").fillna(0.0) <= 0.0).sum()) if total_trades > 0 else 0
    win_rate = (wins / total_trades * 100.0) if total_trades > 0 else 0.0

    if include_details:
        equity_curve = pd.DataFrame({"datetime": data_df_eval.index, "equity": equity_eval.values})
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
        "仓位模式": "每笔账户百分比" if mode == POSITION_SIZING_PERCENT_EQUITY else "每笔固定金额",
        "单次仓位(%)": round(float(position_percent), 4) if mode == POSITION_SIZING_PERCENT_EQUITY else None,
        "固定入场金额(USDT)": (
            round(float(effective_fixed_trade_amount), 4) if mode == POSITION_SIZING_FIXED_AMOUNT else None
        ),
        "触发清算保护": bool(liquidation_idx is not None),
        "清算保护时间": None if liquidation_idx is None else str(data_df.index[int(liquidation_idx)]),
    }
    return BacktestResult(metrics=metrics, equity_curve=equity_curve, trade_details=trade_details)
