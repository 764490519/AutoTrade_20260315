from __future__ import annotations

import os
import sys
from pathlib import Path

import backtrader as bt
import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backtest_engine import run_backtest
from optimization_engine import (
    optimize_parameters,
    optimize_parameters_optuna,
    run_walk_forward,
    run_walk_forward_optuna,
)
from strategy_files.donchian_breakout import DonchianChannelBreakoutStrategy
from strategy_files.fast_rsi_flip import FastRsiFlipStrategy
from strategy_files.ma_trend_faber import FaberMaTrendStrategy


def _build_df(n: int = 2000, trend: float = 0.03) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
    drift = np.linspace(100.0, 100.0 * (1.0 + trend), n)
    wave = np.sin(np.linspace(0, 70, n)) * 1.2
    close = drift + wave
    open_ = np.r_[close[0], close[:-1]]
    high = np.maximum(open_, close) * 1.001
    low = np.minimum(open_, close) * 0.999
    volume = np.full(n, 100.0)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=idx,
    )


def _ret(m: dict) -> float:
    return float(m.get("总收益率(%)", 0.0) or 0.0)


def _run_vectorbt_smoke() -> None:
    print("[VBT-QA] vectorbt migration smoke start")
    os.environ["AUTOTRADE_BACKTEST_ENGINE"] = "vectorbt"
    df = _build_df()

    # 1) 三个已接入策略应可正常运行
    res_faber = run_backtest(
        df,
        FaberMaTrendStrategy,
        {"fast_period": 10, "slow_period": 80},
        initial_cash=10_000,
        commission=0.0004,
        position_percent=30,
        leverage=1.0,
        include_details=True,
    )
    assert not res_faber.equity_curve.empty, "faber equity empty"

    res_fast = run_backtest(
        df,
        FastRsiFlipStrategy,
        {"rsi_period": 12, "up": 53, "dn": 36, "ema_period": 66, "atr_period": 26, "stop_atr": 0.5, "cooldown": 8, "can_short": 1},
        initial_cash=10_000,
        commission=0.0004,
        position_percent=20,
        leverage=1.0,
        include_details=True,
    )
    assert not res_fast.equity_curve.empty, "fast_rsi equity empty"

    res_don = run_backtest(
        df,
        DonchianChannelBreakoutStrategy,
        {
            "entry_period": 30,
            "exit_period": 12,
            "atr_period": 14,
            "atr_filter_enabled": 0,
            "atr_ratio_min": 0.001,
            "atr_ratio_max": 0.08,
            "stop_atr_mult": 2.0,
            "trail_atr_mult": 2.0,
            "can_short": 1,
            "target_percent": 0.6,
        },
        initial_cash=10_000,
        commission=0.0004,
        position_percent=95,  # donchian 中由 target_percent 控制
        leverage=1.0,
        include_details=True,
    )
    assert not res_don.equity_curve.empty, "donchian equity empty"

    print("[VBT-QA] strategies run ok")

    # 2) 杠杆应生效（fast_rsi）
    base_params = {
        "rsi_period": 12,
        "up": 53,
        "dn": 36,
        "ema_period": 66,
        "atr_period": 26,
        "stop_atr": 0.5,
        "cooldown": 8,
        "can_short": 1,
    }
    r1 = _ret(
        run_backtest(
            df,
            FastRsiFlipStrategy,
            base_params,
            initial_cash=10_000,
            commission=0.0004,
            position_percent=20,
            leverage=1.0,
            include_details=False,
        ).metrics
    )
    r2 = _ret(
        run_backtest(
            df,
            FastRsiFlipStrategy,
            base_params,
            initial_cash=10_000,
            commission=0.0004,
            position_percent=20,
            leverage=2.0,
            include_details=False,
        ).metrics
    )
    r3 = _ret(
        run_backtest(
            df,
            FastRsiFlipStrategy,
            base_params,
            initial_cash=10_000,
            commission=0.0004,
            position_percent=20,
            leverage=3.0,
            include_details=False,
        ).metrics
    )
    assert r1 < r2 < r3, f"vectorbt leverage monotonic failed: {r1}, {r2}, {r3}"
    print(f"[VBT-QA] leverage ok: {r1:.2f} < {r2:.2f} < {r3:.2f}")

    # 3) 网格优化 + walk-forward
    grid = optimize_parameters(
        df=df,
        strategy_cls=FastRsiFlipStrategy,
        param_grid={"rsi_period": [8, 12], "ema_period": [40, 66], "cooldown": [4, 8]},
        initial_cash=10_000,
        commission=0.0004,
        position_percent=20,
        leverage=2.0,
        objective="Sharpe",
        max_combinations=8,
        n_jobs=2,
    )
    assert not grid.ranking.empty and grid.best_params, "vectorbt grid optimize failed"
    print(f"[VBT-QA] grid optimize ok: rows={len(grid.ranking)}")

    wf_df, wf_summary = run_walk_forward(
        df=df,
        strategy_cls=FastRsiFlipStrategy,
        param_grid={"rsi_period": [8, 12], "ema_period": [40, 66]},
        initial_cash=10_000,
        commission=0.0004,
        position_percent=20,
        leverage=2.0,
        objective="Sharpe",
        folds=3,
        max_combinations=4,
        n_jobs=2,
    )
    assert len(wf_df) > 0 and int(wf_summary.get("folds", 0)) > 0, "vectorbt walk-forward failed"
    print(f"[VBT-QA] walk-forward ok: folds={wf_summary.get('folds')}")

    # 4) Optuna + WalkForward Optuna
    schema = {
        "rsi_period": {"type": "int", "min": 6, "max": 16, "step": 1},
        "up": {"type": "int", "min": 52, "max": 60, "step": 1},
        "dn": {"type": "int", "min": 30, "max": 42, "step": 1},
        "ema_period": {"type": "int", "min": 20, "max": 100, "step": 2},
        "atr_period": {"type": "int", "min": 8, "max": 30, "step": 1},
        "stop_atr": {"type": "float", "min": 0.3, "max": 2.0, "step": 0.1},
        "cooldown": {"type": "int", "min": 0, "max": 12, "step": 1},
        "can_short": {"type": "int", "min": 1, "max": 1, "step": 1},
    }
    opt = optimize_parameters_optuna(
        df=df,
        strategy_cls=FastRsiFlipStrategy,
        param_schema=schema,
        initial_cash=10_000,
        commission=0.0004,
        position_percent=20,
        leverage=2.0,
        objective="Sharpe",
        n_trials=20,
        sampler_name="TPE",
        seed=42,
        n_jobs=2,
    )
    assert opt.best_params and not opt.ranking.empty, "vectorbt optuna failed"
    print(f"[VBT-QA] optuna ok: best_score={opt.best_score}")

    wf_opt_df, wf_opt_summary = run_walk_forward_optuna(
        df=df,
        strategy_cls=FastRsiFlipStrategy,
        param_schema=schema,
        initial_cash=10_000,
        commission=0.0004,
        position_percent=20,
        leverage=2.0,
        objective="Sharpe",
        folds=3,
        n_trials=12,
        sampler_name="TPE",
        seed=42,
        n_jobs=2,
    )
    assert len(wf_opt_df) > 0 and int(wf_opt_summary.get("folds", 0)) > 0, "vectorbt wf-optuna failed"
    print(f"[VBT-QA] walk-forward optuna ok: folds={wf_opt_summary.get('folds')}")

    # 5) 不支持的策略在 auto 模式应回退 backtrader
    class BuyHold(bt.Strategy):
        def next(self):
            if not self.position:
                self.buy()

    os.environ["AUTOTRADE_BACKTEST_ENGINE"] = "auto"
    fallback_res = run_backtest(
        df=df,
        strategy_cls=BuyHold,
        strategy_params={},
        initial_cash=10_000,
        commission=0.0,
        position_percent=95,
        leverage=2.0,
        include_details=False,
    )
    assert "总收益率(%)" in fallback_res.metrics, "fallback backtest failed"
    print("[VBT-QA] auto fallback to backtrader ok")
    print("[VBT-QA] all tests passed")


if __name__ == "__main__":
    _run_vectorbt_smoke()
