# Backtrader + Binance 回测软件（带 UI）

一个基于 **Backtrader** 回测引擎、**Binance Kline API** 数据源、**Streamlit** 图形界面的轻量回测工具。

## 功能

- Binance 现货历史 K 线拉取（公共 API）
- Backtrader 回测执行
- 所有策略均以本地文件形式管理（`strategy_files/*.py`）
- UI 内直接编辑策略代码、保存、读取、检查
- 从策略文件 `STRATEGY_META.params` 自动渲染参数输入控件
- 指标展示（收益率、最大回撤、Sharpe、胜率等）
- K 线图 + 资金曲线
- 回测结果包含逐笔交易明细（可下载 CSV）
- 一键参数优化（网格搜索）+ Walk-Forward 报告（可下载 CSV）
- 支持 Optuna 贝叶斯优化（TPE/CMA-ES/随机）+ Walk-Forward 报告
- 每次回测自动记录配置与结果（周期、币种、策略、参数、资金参数、核心指标）

## 策略文件规范

策略文件示例：

```python
import backtrader as bt

STRATEGY_META = {
    "display_name": "我的策略",
    "strategy_class": "MyStrategy",
    "params": {
        "fast_period": {"type": "int", "default": 20, "min": 2, "max": 200, "step": 1}
    },
}

class MyStrategy(bt.Strategy):
    params = (("fast_period", 20),)
```

说明：

- `display_name`：界面展示名称（可选）
- `strategy_class`：指定回测使用的类名（可选）
- `params`：参数配置（可选）。若不提供，将使用 JSON 参数输入。

## 环境安装

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
# source .venv/bin/activate

pip install -r requirements.txt
```

## 启动 UI

```bash
streamlit run app.py
```

浏览器打开后，在左侧选择/编辑策略文件，点击“开始回测”。

参数优化入口：

- 左侧「参数优化」区域点击 `一键参数优化 + Walk-Forward`
- 可选优化方法：网格搜索 / Optuna(贝叶斯)
- 可选优化目标：Sharpe / 总收益率 / 年化收益率
- 可选自动网格点数、最大组合数、Walk-Forward 窗口数
- 支持填写自定义网格 JSON

## 项目结构

- `app.py`：Streamlit UI
- `binance_data.py`：币安历史 K 线获取
- `backtest_engine.py`：回测执行与指标统计
- `strategy_files/`：策略文件目录
- `reports/backtest_run_history.csv`：回测历史汇总日志
- `reports/backtest_runs/*.json`：单次回测详细记录

## 说明

- 当前使用 Binance 公共接口，不需要 API Key。
- 若请求频率过高可能触发限流，建议缩小时间范围或稍后重试。
- 策略代码通过 `exec` 动态加载，请仅运行可信本地代码。
- 本项目用于策略研究演示，不构成投资建议。
