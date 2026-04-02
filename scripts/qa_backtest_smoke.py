from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import backtrader as bt

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backtest_engine import run_backtest
from optimization_engine import optimize_parameters, run_walk_forward


def _build_trend_df(*, n: int = 800, up: bool = True) -> pd.DataFrame:
    idx = pd.date_range("2025-01-01", periods=n, freq="h", tz="UTC")
    base = np.linspace(100.0, 260.0, n) if up else np.linspace(260.0, 100.0, n)
    noise = np.sin(np.linspace(0, 40, n)) * 0.8
    close = base + noise
    open_ = np.r_[close[0], close[:-1]]
    high = np.maximum(open_, close) * 1.001
    low = np.minimum(open_, close) * 0.999
    vol = np.full(n, 100.0)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": vol},
        index=idx,
    )


class BuyHold(bt.Strategy):
    def next(self):
        if not self.position:
            self.buy()


class ShortHold(bt.Strategy):
    def next(self):
        if not self.position:
            self.sell()


class SmaCross(bt.Strategy):
    params = (("fast", 10), ("slow", 30))

    def __init__(self):
        self.fast = bt.ind.SMA(self.data.close, period=int(self.p.fast))
        self.slow = bt.ind.SMA(self.data.close, period=int(self.p.slow))

    def next(self):
        if len(self.data) < int(self.p.slow) + 1:
            return
        if not self.position and float(self.fast[-1]) > float(self.slow[-1]):
            self.buy()
        elif self.position and float(self.fast[-1]) < float(self.slow[-1]):
            self.close()


def _metric_return(result) -> float:
    return float(result.metrics.get("总收益率(%)", 0.0))


def main() -> None:
    print("[QA] start smoke tests")

    up_df = _build_trend_df(up=True)
    dn_df = _build_trend_df(up=False)

    # 1) leverage sensitivity (long)
    r1 = _metric_return(
        run_backtest(up_df, BuyHold, {}, initial_cash=10_000, commission=0.0, position_percent=95, leverage=1.0, include_details=False)
    )
    r2 = _metric_return(
        run_backtest(up_df, BuyHold, {}, initial_cash=10_000, commission=0.0, position_percent=95, leverage=2.0, include_details=False)
    )
    r3 = _metric_return(
        run_backtest(up_df, BuyHold, {}, initial_cash=10_000, commission=0.0, position_percent=95, leverage=3.0, include_details=False)
    )
    assert r1 < r2 < r3, f"leverage-long monotonic failed: {r1}, {r2}, {r3}"
    print(f"[QA] leverage-long ok: {r1:.2f} < {r2:.2f} < {r3:.2f}")

    # 2) leverage sensitivity (short)
    s1 = _metric_return(
        run_backtest(dn_df, ShortHold, {}, initial_cash=10_000, commission=0.0, position_percent=95, leverage=1.0, include_details=False)
    )
    s2 = _metric_return(
        run_backtest(dn_df, ShortHold, {}, initial_cash=10_000, commission=0.0, position_percent=95, leverage=2.0, include_details=False)
    )
    s3 = _metric_return(
        run_backtest(dn_df, ShortHold, {}, initial_cash=10_000, commission=0.0, position_percent=95, leverage=3.0, include_details=False)
    )
    assert s1 < s2 < s3, f"leverage-short monotonic failed: {s1}, {s2}, {s3}"
    print(f"[QA] leverage-short ok: {s1:.2f} < {s2:.2f} < {s3:.2f}")

    # 3) include_details
    detail = run_backtest(
        up_df,
        SmaCross,
        {"fast": 12, "slow": 50},
        initial_cash=10_000,
        commission=0.001,
        position_percent=95,
        leverage=2.0,
        include_details=True,
    )
    assert not detail.equity_curve.empty, "equity_curve empty"
    assert {"datetime", "equity"}.issubset(set(detail.equity_curve.columns)), "equity columns missing"
    print(f"[QA] include_details ok: equity_rows={len(detail.equity_curve)} trade_rows={len(detail.trade_details)}")

    # 4) optimization grid
    grid = optimize_parameters(
        up_df,
        SmaCross,
        param_grid={"fast": [8, 12, 16], "slow": [30, 50, 70]},
        initial_cash=10_000,
        commission=0.001,
        position_percent=95,
        leverage=2.0,
        objective="Sharpe",
        max_combinations=9,
        n_jobs=2,
    )
    assert not grid.ranking.empty, "grid ranking empty"
    assert isinstance(grid.best_params, dict), "grid best_params invalid"
    print(f"[QA] grid optimize ok: rows={len(grid.ranking)} best={grid.best_params}")

    # 5) walk-forward basic
    wf_df, wf_summary = run_walk_forward(
        up_df,
        SmaCross,
        param_grid={"fast": [8, 12], "slow": [30, 50]},
        initial_cash=10_000,
        commission=0.001,
        position_percent=95,
        leverage=2.0,
        objective="Sharpe",
        folds=2,
        max_combinations=4,
        n_jobs=2,
    )
    assert len(wf_df) > 0, "walk-forward empty"
    assert int(wf_summary.get("folds", 0)) > 0, "walk-forward summary invalid"
    print(f"[QA] walk-forward ok: folds={wf_summary.get('folds')} rows={len(wf_df)}")

    # 6) process-pool optimization path (strategy_code enabled)
    strategy_code = """
import backtrader as bt

class SmaCrossFromCode(bt.Strategy):
    params = (("fast", 10), ("slow", 30))

    def __init__(self):
        self.fast = bt.ind.SMA(self.data.close, period=int(self.p.fast))
        self.slow = bt.ind.SMA(self.data.close, period=int(self.p.slow))

    def next(self):
        if len(self.data) < int(self.p.slow) + 1:
            return
        if not self.position and float(self.fast[-1]) > float(self.slow[-1]):
            self.buy()
        elif self.position and float(self.fast[-1]) < float(self.slow[-1]):
            self.close()
"""
    mp_grid = optimize_parameters(
        up_df,
        SmaCross,  # 实际会被 strategy_code 覆盖，保留占位
        param_grid={"fast": [8, 12], "slow": [30, 50]},
        initial_cash=10_000,
        commission=0.001,
        position_percent=95,
        leverage=2.0,
        objective="Sharpe",
        max_combinations=4,
        n_jobs=2,
        strategy_code=strategy_code,
        strategy_class_name="SmaCrossFromCode",
    )
    assert not mp_grid.ranking.empty, "process-pool grid ranking empty"
    if "错误" in mp_grid.ranking.columns:
        ok_rows = int(mp_grid.ranking["错误"].isna().sum())
        assert ok_rows > 0, "process-pool grid all rows failed"
    print(f"[QA] process-pool optimize ok: rows={len(mp_grid.ranking)} best={mp_grid.best_params}")

    print("[QA] all smoke tests passed")


if __name__ == "__main__":
    main()
