# VectorBT + Binance 回测与自动交易软件（带 UI）

一个基于 **vectorbt** 的回测与参数优化工具，配合 Binance K 线数据与 Streamlit 前端；并提供 OKX 实盘自动执行模块。

## 功能

- Binance 历史 K 线拉取（公共 API）
- vectorbt 回测执行（单次回测 / 网格优化 / Optuna / Walk-Forward）
- 策略文件本地管理（`strategy_files/*.py`）
- 参数面板由 `STRATEGY_META.params` 自动渲染
- 资金曲线、指标统计、逐笔交易明细
- OKX 自动执行（按策略信号开平仓）
- 实盘信号与回测信号共用同一套策略逻辑（保证一致性）

## 策略文件规范（仅核心进出场逻辑）

```python
from __future__ import annotations
from typing import Any
import numpy as np
import pandas as pd

STRATEGY_META = {
    "display_name": "我的策略",
    "strategy_class": "MyStrategy",
    "signal_func": "generate_targets",
    "params": {
        "fast_period": {"type": "int", "default": 20, "min": 2, "max": 200, "step": 1},
        "slow_period": {"type": "int", "default": 60, "min": 5, "max": 500, "step": 1},
    },
}

def generate_targets(df: pd.DataFrame, params: dict[str, Any]) -> np.ndarray:
    # 返回长度与 df 相同的 target 序列：
    # np.nan = 不变仓, 1.0 = 做多, -1.0 = 做空, 0.0 = 平仓
    ...

class MyStrategy:
    USE_GLOBAL_POSITION_PERCENT = True

    @staticmethod
    def generate_targets(df: pd.DataFrame, params: dict[str, Any]) -> np.ndarray:
        return generate_targets(df, params)
```

## 启动

```bash
python -m venv .venv
.venv\Scripts\activate   # Windows
pip install -r requirements.txt
streamlit run app.py
```

## 目录

- `app.py`：UI
- `backtest_engine.py`：vectorbt 回测与统计
- `optimization_engine.py`：网格/Optuna/Walk-Forward
- `live_trading_engine.py`：实时信号执行
- `strategy_files/`：策略逻辑
- `reports/`：回测与执行报告

## 说明

- 本项目用于策略研究与工程验证，不构成投资建议。
- 请先使用模拟盘验证策略行为，再投入真实资金。
