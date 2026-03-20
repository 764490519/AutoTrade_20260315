from __future__ import annotations

import inspect
import json
import traceback
import hashlib
import csv
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import backtrader as bt
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from backtest_engine import run_backtest
from binance_data import BinanceDataError, fetch_klines
from optimization_engine import (
    build_auto_param_grid,
    optimize_parameters,
    optimize_parameters_optuna,
    run_walk_forward,
    run_walk_forward_optuna,
)


st.set_page_config(page_title="Backtrader + Binance 回测系统", layout="wide")
st.title("📈 Backtrader + Binance 回测软件")
st.caption("数据源：Binance Spot Kline API（公共接口）")

STRATEGY_DIR = Path(__file__).resolve().parent / "strategy_files"
REPORT_DIR = Path(__file__).resolve().parent / "reports"
BACKTEST_RUN_DIR = REPORT_DIR / "backtest_runs"
BACKTEST_LOG_CSV = REPORT_DIR / "backtest_run_history.csv"
UI_STATE_FILE = REPORT_DIR / "ui_state.json"
NEW_STRATEGY_LABEL = "（新建策略文件）"

PERSIST_STATE_KEYS = {
    "symbol",
    "interval",
    "start_date",
    "end_date",
    "selected_strategy_file",
    "strategy_file_name",
    "strategy_code",
    "strategy_params_json",
    "initial_cash",
    "commission",
    "position_percent",
    "leverage",
    "opt_method",
    "opt_objective",
    "opt_folds",
    "opt_points",
    "opt_max_combinations",
    "opt_grid_json",
    "opt_trials",
    "opt_sampler",
    "opt_seed",
    "last_optimized_params",
    "last_optimized_strategy_file",
}

NEW_STRATEGY_TEMPLATE = """import backtrader as bt

STRATEGY_META = {
    "display_name": "我的策略",
    "strategy_class": "MyStrategy",
    "params": {
        "fast_period": {"type": "int", "default": 20, "min": 2, "max": 200, "step": 1},
        "slow_period": {"type": "int", "default": 60, "min": 5, "max": 500, "step": 1},
    },
}


class MyStrategy(bt.Strategy):
    params = (
        ("fast_period", 20),
        ("slow_period", 60),
    )

    def __init__(self):
        fast_sma = bt.indicators.SimpleMovingAverage(self.data.close, period=self.params.fast_period)
        slow_sma = bt.indicators.SimpleMovingAverage(self.data.close, period=self.params.slow_period)
        self.crossover = bt.indicators.CrossOver(fast_sma, slow_sma)

    def next(self):
        if not self.position and self.crossover > 0:
            self.buy()
        elif self.position and self.crossover < 0:
            self.close()
"""

DEFAULT_STRATEGY_FILES: dict[str, str] = {
    "sma_dual.py": """import backtrader as bt

STRATEGY_META = {
    "display_name": "SMA 双均线",
    "strategy_class": "SmaCrossStrategy",
    "params": {
        "fast_period": {"type": "int", "default": 20, "min": 2, "max": 200, "step": 1},
        "slow_period": {"type": "int", "default": 60, "min": 5, "max": 500, "step": 1},
    },
}


class SmaCrossStrategy(bt.Strategy):
    params = (
        ("fast_period", 20),
        ("slow_period", 60),
    )

    def __init__(self):
        fast_sma = bt.indicators.SimpleMovingAverage(self.data.close, period=self.params.fast_period)
        slow_sma = bt.indicators.SimpleMovingAverage(self.data.close, period=self.params.slow_period)
        self.crossover = bt.indicators.CrossOver(fast_sma, slow_sma)

    def next(self):
        if not self.position and self.crossover > 0:
            self.buy()
        elif self.position and self.crossover < 0:
            self.close()
""",
    "rsi_reversion.py": """import backtrader as bt

STRATEGY_META = {
    "display_name": "RSI 均值回归",
    "strategy_class": "RsiReversionStrategy",
    "params": {
        "rsi_period": {"type": "int", "default": 14, "min": 2, "max": 100, "step": 1},
        "oversold": {"type": "float", "default": 30.0, "min": 1.0, "max": 50.0, "step": 0.5},
        "overbought": {"type": "float", "default": 70.0, "min": 50.0, "max": 99.0, "step": 0.5},
    },
}


class RsiReversionStrategy(bt.Strategy):
    params = (
        ("rsi_period", 14),
        ("oversold", 30),
        ("overbought", 70),
    )

    def __init__(self):
        self.rsi = bt.indicators.RSI(self.data.close, period=self.params.rsi_period)

    def next(self):
        if not self.position and self.rsi < self.params.oversold:
            self.buy()
        elif self.position and self.rsi > self.params.overbought:
            self.close()
""",
    "macd_cross.py": """import backtrader as bt

STRATEGY_META = {
    "display_name": "MACD 金叉死叉",
    "strategy_class": "MacdCrossStrategy",
    "params": {
        "fast": {"type": "int", "default": 12, "min": 2, "max": 100, "step": 1},
        "slow": {"type": "int", "default": 26, "min": 3, "max": 200, "step": 1},
        "signal": {"type": "int", "default": 9, "min": 2, "max": 100, "step": 1},
    },
}


class MacdCrossStrategy(bt.Strategy):
    params = (
        ("fast", 12),
        ("slow", 26),
        ("signal", 9),
    )

    def __init__(self):
        self.macd = bt.indicators.MACD(
            self.data.close,
            period_me1=self.params.fast,
            period_me2=self.params.slow,
            period_signal=self.params.signal,
        )
        self.crossover = bt.indicators.CrossOver(self.macd.macd, self.macd.signal)

    def next(self):
        if not self.position and self.crossover > 0:
            self.buy()
        elif self.position and self.crossover < 0:
            self.close()
""",
}


@st.cache_data(ttl=60 * 30, show_spinner=False)
def load_binance_data(symbol: str, interval: str, start_dt: datetime, end_dt: datetime) -> pd.DataFrame:
    return fetch_klines(symbol=symbol, interval=interval, start_time=start_dt, end_time=end_dt)


def _to_utc_start(d: date) -> datetime:
    return datetime.combine(d, time.min).replace(tzinfo=timezone.utc)


def _to_utc_end(d: date) -> datetime:
    return datetime.combine(d + timedelta(days=1), time.min).replace(tzinfo=timezone.utc)


def _sanitize_name(name: str) -> str:
    cleaned = "".join(ch for ch in name.strip() if ch.isalnum() or ch in ("_", "-"))
    if not cleaned:
        raise ValueError("文件名不能为空，且仅允许字母、数字、_、-")
    return cleaned


def _ensure_strategy_dir() -> None:
    STRATEGY_DIR.mkdir(parents=True, exist_ok=True)
    for file_name, content in DEFAULT_STRATEGY_FILES.items():
        path = STRATEGY_DIR / file_name
        if not path.exists():
            path.write_text(content, encoding="utf-8")


def _list_strategy_names() -> list[str]:
    _ensure_strategy_dir()
    return sorted(p.stem for p in STRATEGY_DIR.glob("*.py"))


def _load_strategy_code(file_stem: str) -> str:
    safe_name = _sanitize_name(file_stem)
    file_path = STRATEGY_DIR / f"{safe_name}.py"
    if not file_path.exists():
        raise FileNotFoundError(f"策略文件不存在: {file_path.name}")
    # 兼容带 BOM 的 UTF-8 文件（例如某些编辑器/PowerShell 写入的文件）
    return file_path.read_text(encoding="utf-8-sig")


def _save_strategy_code(file_stem: str, code: str) -> Path:
    safe_name = _sanitize_name(file_stem)
    if not code.strip():
        raise ValueError("策略代码不能为空")

    _ensure_strategy_dir()
    file_path = STRATEGY_DIR / f"{safe_name}.py"
    file_path.write_text(code, encoding="utf-8")
    return file_path


def _compile_strategy(code: str):
    # 防止代码字符串开头含 BOM 导致 exec 报错: U+FEFF
    code = code.lstrip("\ufeff")
    namespace: dict[str, Any] = {"bt": bt}
    exec(code, namespace, namespace)

    strategy_classes = [
        obj
        for obj in namespace.values()
        if inspect.isclass(obj) and issubclass(obj, bt.Strategy) and obj is not bt.Strategy
    ]
    if not strategy_classes:
        raise ValueError("代码中未找到 bt.Strategy 子类")

    meta = namespace.get("STRATEGY_META", {})
    if not isinstance(meta, dict):
        meta = {}

    selected_cls = strategy_classes[0]
    class_name = meta.get("strategy_class")
    if isinstance(class_name, str) and class_name in namespace:
        maybe_cls = namespace[class_name]
        if inspect.isclass(maybe_cls) and issubclass(maybe_cls, bt.Strategy):
            selected_cls = maybe_cls

    display_name = str(meta.get("display_name", selected_cls.__name__))
    params_schema = meta.get("params", {})
    if not isinstance(params_schema, dict):
        params_schema = {}

    return selected_cls, display_name, params_schema


def _normalize_params_schema(raw_schema: dict[str, Any]) -> dict[str, dict[str, Any]]:
    schema: dict[str, dict[str, Any]] = {}
    for name, cfg in raw_schema.items():
        if not isinstance(name, str) or not isinstance(cfg, dict):
            continue

        default = cfg.get("default", 0)
        p_type = cfg.get("type")
        if p_type not in ("int", "float"):
            if isinstance(default, int) and not isinstance(default, bool):
                p_type = "int"
            else:
                p_type = "float"

        if p_type == "int":
            normalized = {
                "type": "int",
                "default": int(default),
                "min": int(cfg.get("min", -1_000_000)),
                "max": int(cfg.get("max", 1_000_000)),
                "step": max(1, int(cfg.get("step", 1))),
            }
        else:
            normalized = {
                "type": "float",
                "default": float(default),
                "min": float(cfg.get("min", -1_000_000.0)),
                "max": float(cfg.get("max", 1_000_000.0)),
                "step": max(0.0001, float(cfg.get("step", 0.1))),
            }

        if normalized["min"] > normalized["max"]:
            normalized["min"], normalized["max"] = normalized["max"], normalized["min"]
        schema[name] = normalized

    return schema


def _parse_params_json(raw_text: str) -> dict[str, Any]:
    text = raw_text.strip()
    if not text:
        return {}

    data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError("策略参数必须是 JSON 对象")
    return data


def _to_serializable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (datetime, date, pd.Timestamp)):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(k): _to_serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_to_serializable(v) for v in value]
    try:
        return float(value)
    except Exception:  # noqa: BLE001
        return str(value)


def _load_persisted_ui_state() -> dict[str, Any]:
    if not UI_STATE_FILE.exists():
        return {}
    try:
        data = json.loads(UI_STATE_FILE.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:  # noqa: BLE001
        return {}


def _restore_persisted_ui_state() -> None:
    if st.session_state.get("_ui_state_restored", False):
        return

    raw = _load_persisted_ui_state()
    for key, value in raw.items():
        if key in st.session_state:
            continue
        if key in {"start_date", "end_date"} and isinstance(value, str):
            try:
                value = date.fromisoformat(value)
            except Exception:  # noqa: BLE001
                continue
        st.session_state[key] = value

    st.session_state["_ui_state_restored"] = True


def _persist_ui_state() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    state: dict[str, Any] = {}

    for key in PERSIST_STATE_KEYS:
        if key in st.session_state:
            state[key] = _to_serializable(st.session_state[key])

    # 动态策略参数（param_xxx）也持久化，刷新后仍保留
    for key, value in st.session_state.items():
        if key.startswith("param_"):
            state[key] = _to_serializable(value)

    UI_STATE_FILE.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def _save_backtest_record(
    *,
    symbol: str,
    interval: str,
    strategy_display_name: str,
    strategy_params: dict[str, Any],
    initial_cash: float,
    commission: float,
    position_percent: float,
    leverage: float,
    selected_start_date: date,
    selected_end_date: date,
    data_df: pd.DataFrame,
    result_metrics: dict[str, Any],
) -> tuple[str, Path, Path]:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    BACKTEST_RUN_DIR.mkdir(parents=True, exist_ok=True)

    now_utc = datetime.now(timezone.utc)
    run_id = f"{now_utc.strftime('%Y%m%dT%H%M%SZ')}_{uuid4().hex[:8]}"

    strategy_code = str(st.session_state.get("strategy_code", ""))
    strategy_file_name = str(st.session_state.get("strategy_file_name", "")).strip()
    strategy_code_sha256 = hashlib.sha256(strategy_code.encode("utf-8")).hexdigest() if strategy_code else None

    actual_start = data_df.index.min() if not data_df.empty else None
    actual_end = data_df.index.max() if not data_df.empty else None

    detail_record = {
        "run_id": run_id,
        "run_time_utc": now_utc,
        "symbol": symbol,
        "interval": interval,
        "selected_start_date": selected_start_date,
        "selected_end_date": selected_end_date,
        "data_start": actual_start,
        "data_end": actual_end,
        "bars": int(len(data_df)),
        "strategy_display_name": strategy_display_name,
        "strategy_file_name": strategy_file_name,
        "strategy_code_sha256": strategy_code_sha256,
        "strategy_params": strategy_params,
        "initial_cash": float(initial_cash),
        "commission": float(commission),
        "position_percent": float(position_percent),
        "leverage": float(leverage),
        "metrics": result_metrics,
    }
    detail_record = _to_serializable(detail_record)

    detail_path = BACKTEST_RUN_DIR / f"{run_id}.json"
    detail_path.write_text(json.dumps(detail_record, ensure_ascii=False, indent=2), encoding="utf-8")

    flat_row = {
        "run_id": run_id,
        "run_time_utc": detail_record["run_time_utc"],
        "symbol": symbol,
        "interval": interval,
        "selected_start_date": detail_record["selected_start_date"],
        "selected_end_date": detail_record["selected_end_date"],
        "data_start": detail_record["data_start"],
        "data_end": detail_record["data_end"],
        "bars": detail_record["bars"],
        "strategy_display_name": strategy_display_name,
        "strategy_file_name": strategy_file_name,
        "strategy_params_json": json.dumps(detail_record["strategy_params"], ensure_ascii=False),
        "initial_cash": detail_record["initial_cash"],
        "commission": detail_record["commission"],
        "position_percent": detail_record["position_percent"],
        "leverage": detail_record["leverage"],
        "total_return_pct": detail_record["metrics"].get("总收益率(%)"),
        "annual_return_pct": detail_record["metrics"].get("年化收益率(%)"),
        "sharpe": detail_record["metrics"].get("Sharpe"),
        "max_drawdown_pct": detail_record["metrics"].get("最大回撤(%)"),
        "total_trades": detail_record["metrics"].get("总交易次数"),
        "detail_json_file": str(detail_path),
    }

    write_header = not BACKTEST_LOG_CSV.exists()
    with BACKTEST_LOG_CSV.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(flat_row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(flat_row)

    return run_id, detail_path, BACKTEST_LOG_CSV


def _parse_optimization_grid(raw_text: str, param_schema: dict[str, dict[str, Any]]) -> dict[str, list[Any]]:
    text = raw_text.strip()
    if not text:
        return {}

    data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError("优化网格必须是 JSON 对象，例如 {\"entry_period\":[10,20,30]}")

    grid: dict[str, list[Any]] = {}
    for key, values in data.items():
        if not isinstance(key, str) or not isinstance(values, list) or not values:
            raise ValueError(f"网格参数 {key} 格式错误，必须是非空数组")

        if key in param_schema:
            p_type = param_schema[key]["type"]
            if p_type == "int":
                cast_values = sorted(set(int(round(float(v))) for v in values))
            else:
                cast_values = sorted(set(round(float(v), 10) for v in values))
        else:
            cast_values = list(values)

        grid[key] = cast_values

    return grid


def _count_combinations(param_grid: dict[str, list[Any]]) -> int:
    total = 1
    for values in param_grid.values():
        total *= max(1, len(values))
    return total


def _apply_params_to_ui(
    best_params: dict[str, Any],
    param_schema: dict[str, dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """将最优参数写回侧边栏输入控件。"""
    applied: dict[str, Any] = {}
    untouched: dict[str, Any] = {}

    if not isinstance(best_params, dict) or not best_params:
        return applied, untouched

    if not param_schema:
        st.session_state["strategy_params_json"] = json.dumps(best_params, ensure_ascii=False)
        return {}, best_params

    for name, raw_value in best_params.items():
        if name not in param_schema:
            untouched[name] = raw_value
            continue

        cfg = param_schema[name]
        key = f"param_{name}"
        try:
            if cfg["type"] == "int":
                value = int(round(float(raw_value)))
                value = max(int(cfg["min"]), min(int(cfg["max"]), value))
            else:
                value = float(raw_value)
                value = max(float(cfg["min"]), min(float(cfg["max"]), value))
            st.session_state[key] = value
            applied[name] = value
        except Exception:  # noqa: BLE001
            untouched[name] = raw_value

    if applied:
        st.session_state["strategy_params_json"] = json.dumps(applied, ensure_ascii=False)

    return applied, untouched


_ensure_strategy_dir()
_restore_persisted_ui_state()

if "selected_strategy_file" not in st.session_state:
    st.session_state["selected_strategy_file"] = NEW_STRATEGY_LABEL
if "strategy_file_name" not in st.session_state:
    st.session_state["strategy_file_name"] = "my_strategy"
if "strategy_code" not in st.session_state:
    st.session_state["strategy_code"] = NEW_STRATEGY_TEMPLATE
if "strategy_params_json" not in st.session_state:
    st.session_state["strategy_params_json"] = "{}"
if "opt_grid_json" not in st.session_state:
    st.session_state["opt_grid_json"] = ""
if "last_optimized_params" not in st.session_state:
    st.session_state["last_optimized_params"] = {}
if "last_optimized_strategy_file" not in st.session_state:
    st.session_state["last_optimized_strategy_file"] = ""
if "_loaded_strategy_file" not in st.session_state:
    st.session_state["_loaded_strategy_file"] = st.session_state.get("selected_strategy_file", "")
if "symbol" not in st.session_state:
    st.session_state["symbol"] = "BTCUSDT"
if "interval" not in st.session_state:
    st.session_state["interval"] = "1d"
default_end = date.today()
default_start = default_end - timedelta(days=180)
if "start_date" not in st.session_state:
    st.session_state["start_date"] = default_start
if "end_date" not in st.session_state:
    st.session_state["end_date"] = default_end
if "initial_cash" not in st.session_state:
    st.session_state["initial_cash"] = 10_000.0
if "commission" not in st.session_state:
    st.session_state["commission"] = 0.001
if "position_percent" not in st.session_state:
    st.session_state["position_percent"] = 95.0
if "leverage" not in st.session_state:
    st.session_state["leverage"] = 1.0
if "opt_method" not in st.session_state:
    st.session_state["opt_method"] = "Optuna(贝叶斯)"
if "opt_objective" not in st.session_state:
    st.session_state["opt_objective"] = "Sharpe"
if "opt_folds" not in st.session_state:
    st.session_state["opt_folds"] = 4
if "opt_points" not in st.session_state:
    st.session_state["opt_points"] = 3
if "opt_max_combinations" not in st.session_state:
    st.session_state["opt_max_combinations"] = 120
if "opt_trials" not in st.session_state:
    st.session_state["opt_trials"] = 120
if "opt_sampler" not in st.session_state:
    st.session_state["opt_sampler"] = "TPE"
if "opt_seed" not in st.session_state:
    st.session_state["opt_seed"] = 42

with st.sidebar:
    st.header("参数设置")

    interval_options = ["1m", "5m", "15m", "1h", "4h", "1d"]
    if st.session_state.get("interval") not in interval_options:
        st.session_state["interval"] = "1d"

    st.text_input("交易对", key="symbol")
    symbol = str(st.session_state.get("symbol", "BTCUSDT")).upper().strip()

    interval = st.selectbox("K线周期", interval_options, key="interval")
    start_date = st.date_input("开始日期", key="start_date")
    end_date = st.date_input("结束日期", key="end_date")

    st.subheader("策略文件")
    strategy_file_options = [NEW_STRATEGY_LABEL] + _list_strategy_names()
    if st.session_state.get("selected_strategy_file") not in strategy_file_options:
        st.session_state["selected_strategy_file"] = NEW_STRATEGY_LABEL
    selected_file = st.selectbox("已保存策略", strategy_file_options, key="selected_strategy_file")

    if selected_file != st.session_state.get("_loaded_strategy_file"):
        if selected_file == NEW_STRATEGY_LABEL:
            st.session_state["strategy_file_name"] = "my_strategy"
            st.session_state["strategy_code"] = NEW_STRATEGY_TEMPLATE
        else:
            try:
                st.session_state["strategy_file_name"] = selected_file
                st.session_state["strategy_code"] = _load_strategy_code(selected_file)
            except Exception as exc:  # noqa: BLE001
                st.error(f"加载策略失败：{exc}")

        st.session_state["_loaded_strategy_file"] = selected_file

    st.text_input("策略文件名", key="strategy_file_name", help="保存为 strategy_files/<文件名>.py")
    st.text_area("策略代码", key="strategy_code", height=360)

    col_save, col_load, col_check = st.columns(3)

    if col_save.button("保存", use_container_width=True):
        try:
            saved_path = _save_strategy_code(st.session_state["strategy_file_name"], st.session_state["strategy_code"])
            st.success(f"已保存：{saved_path.name}")
            st.session_state["selected_strategy_file"] = saved_path.stem
            st.session_state["_loaded_strategy_file"] = saved_path.stem
        except Exception as exc:  # noqa: BLE001
            st.error(f"保存失败：{exc}")

    if col_load.button("读取", use_container_width=True):
        try:
            code = _load_strategy_code(st.session_state["strategy_file_name"])
            st.session_state["strategy_code"] = code
            st.session_state["selected_strategy_file"] = _sanitize_name(st.session_state["strategy_file_name"])
            st.session_state["_loaded_strategy_file"] = st.session_state["selected_strategy_file"]
            st.success("读取成功")
        except Exception as exc:  # noqa: BLE001
            st.error(f"读取失败：{exc}")

    compile_error = None
    strategy_display_name = ""
    param_schema: dict[str, dict[str, Any]] = {}

    try:
        _cls, strategy_display_name, raw_schema = _compile_strategy(st.session_state["strategy_code"])
        param_schema = _normalize_params_schema(raw_schema)
    except Exception as exc:  # noqa: BLE001
        compile_error = str(exc)

    if col_check.button("检查", use_container_width=True):
        if compile_error:
            st.error(f"检查失败：{compile_error}")
        else:
            st.success(f"编译通过：{strategy_display_name}")

    st.subheader("策略参数")
    strategy_params: dict[str, Any] = {}

    if compile_error:
        st.warning("策略编译失败，请先修复代码")
    elif param_schema:
        for param_name, cfg in param_schema.items():
            key = f"param_{param_name}"
            if key not in st.session_state:
                st.session_state[key] = cfg["default"]
            else:
                try:
                    if cfg["type"] == "int":
                        value = int(round(float(st.session_state[key])))
                        value = max(int(cfg["min"]), min(int(cfg["max"]), value))
                    else:
                        value = float(st.session_state[key])
                        value = max(float(cfg["min"]), min(float(cfg["max"]), value))
                    st.session_state[key] = value
                except Exception:  # noqa: BLE001
                    st.session_state[key] = cfg["default"]

            if cfg["type"] == "int":
                strategy_params[param_name] = st.number_input(
                    param_name,
                    min_value=int(cfg["min"]),
                    max_value=int(cfg["max"]),
                    step=int(cfg["step"]),
                    key=key,
                )
            else:
                strategy_params[param_name] = st.number_input(
                    param_name,
                    min_value=float(cfg["min"]),
                    max_value=float(cfg["max"]),
                    step=float(cfg["step"]),
                    key=key,
                )
    else:
        st.caption("未检测到 STRATEGY_META.params，使用 JSON 参数输入")
        st.text_input("策略参数(JSON)", key="strategy_params_json")

    st.subheader("资金参数")
    initial_cash = st.number_input("初始资金(USDT)", min_value=100.0, step=100.0, key="initial_cash")
    commission = st.number_input("手续费率", min_value=0.0, max_value=0.01, step=0.0001, format="%.4f", key="commission")
    position_percent = st.number_input("单次仓位(%)", min_value=1.0, max_value=100.0, step=1.0, key="position_percent")
    leverage = st.number_input("杠杆倍数", min_value=1.0, max_value=125.0, step=0.5, key="leverage")

    run_btn = st.button("开始回测", type="primary", use_container_width=True)
    st.divider()
    st.subheader("参数优化")
    opt_method_options = ["网格搜索", "Optuna(贝叶斯)"]
    if st.session_state.get("opt_method") not in opt_method_options:
        st.session_state["opt_method"] = "Optuna(贝叶斯)"
    opt_method = st.selectbox("优化方法", opt_method_options, key="opt_method")

    opt_objective_options = ["Sharpe", "总收益率(%)", "年化收益率(%)"]
    if st.session_state.get("opt_objective") not in opt_objective_options:
        st.session_state["opt_objective"] = "Sharpe"
    opt_objective = st.selectbox("优化目标", opt_objective_options, key="opt_objective")
    opt_folds = st.slider("Walk-Forward 窗口数", min_value=2, max_value=8, step=1, key="opt_folds")
    if opt_method == "网格搜索":
        opt_points = st.slider("自动网格每参数点数", min_value=2, max_value=5, step=1, key="opt_points")
        opt_max_combinations = st.number_input("最大回测组合数", min_value=10, max_value=1000, step=10, key="opt_max_combinations")
        st.text_area(
            "自定义优化网格(JSON，可选)",
            key="opt_grid_json",
            height=100,
            placeholder='例如: {"entry_period":[10,20,30], "exit_period":[10,20], "can_short":[0,1]}',
        )
        opt_trials = 0
        opt_sampler = "TPE"
        opt_seed = 42
    else:
        opt_trials = st.number_input("Optuna 试验次数(n_trials)", min_value=10, max_value=2000, step=10, key="opt_trials")
        opt_sampler_options = ["TPE", "CMA-ES", "随机"]
        if st.session_state.get("opt_sampler") not in opt_sampler_options:
            st.session_state["opt_sampler"] = "TPE"
        opt_sampler = st.selectbox("Optuna Sampler", opt_sampler_options, key="opt_sampler")
        opt_seed = st.number_input("随机种子(seed)", min_value=0, max_value=10_000_000, step=1, key="opt_seed")
        opt_points = 0
        opt_max_combinations = 0
    optimize_btn = st.button("一键参数优化 + Walk-Forward", use_container_width=True)

    last_best = st.session_state.get("last_optimized_params", {})
    if isinstance(last_best, dict) and last_best:
        last_best_file = str(st.session_state.get("last_optimized_strategy_file", ""))
        current_file = str(st.session_state.get("strategy_file_name", ""))
        st.caption("已存在最近一次优化得到的最优参数")
        if last_best_file and current_file and last_best_file != current_file:
            st.caption(f"提示：该参数来自策略文件 {last_best_file}，当前是 {current_file}")
        if st.button("一键应用最近最优参数", use_container_width=True, key="apply_last_best_sidebar"):
            applied_params, skipped_params = _apply_params_to_ui(last_best, param_schema)
            if applied_params:
                st.success(f"已应用参数：{json.dumps(applied_params, ensure_ascii=False)}")
            if skipped_params:
                st.warning(f"以下参数未应用：{json.dumps(skipped_params, ensure_ascii=False)}")
            st.rerun()

    _persist_ui_state()

if run_btn or optimize_btn:
    if start_date >= end_date:
        st.error("开始日期必须早于结束日期")
        st.stop()

    try:
        strategy_cls, strategy_display_name, raw_schema = _compile_strategy(st.session_state["strategy_code"])
        param_schema = _normalize_params_schema(raw_schema)
    except Exception as exc:  # noqa: BLE001
        st.error(f"策略编译失败：{exc}")
        with st.expander("错误详情"):
            st.code(traceback.format_exc())
        st.stop()

    if not param_schema:
        try:
            strategy_params = _parse_params_json(st.session_state["strategy_params_json"])
        except Exception as exc:  # noqa: BLE001
            st.error(f"参数解析失败：{exc}")
            st.stop()

    if (
        {"fast_period", "slow_period"}.issubset(set(strategy_params.keys()))
        and strategy_params["fast_period"] >= strategy_params["slow_period"]
    ):
        st.warning("检测到 fast_period >= slow_period，请确认这是否符合你的策略设计")

    with st.spinner("正在拉取币安历史数据..."):
        try:
            start_dt = _to_utc_start(start_date)
            end_dt = _to_utc_end(end_date)
            df = load_binance_data(symbol, interval, start_dt, end_dt)
        except BinanceDataError as exc:
            st.error(f"数据获取失败：{exc}")
            st.stop()
        except Exception as exc:  # noqa: BLE001
            st.error(f"未知错误：{exc}")
            st.stop()

    if df.empty:
        st.warning("没有可用数据，请调整参数")
        st.stop()

    if run_btn:
        with st.spinner("回测中，请稍候..."):
            result = run_backtest(
                df=df,
                strategy_cls=strategy_cls,
                strategy_params=strategy_params,
                initial_cash=float(initial_cash),
                commission=float(commission),
                position_percent=float(position_percent),
                leverage=float(leverage),
            )

        st.success("回测完成")
        st.caption(f"当前策略：{strategy_display_name}")
        try:
            run_id, detail_file, log_file = _save_backtest_record(
                symbol=symbol,
                interval=interval,
                strategy_display_name=strategy_display_name,
                strategy_params=strategy_params,
                initial_cash=float(initial_cash),
                commission=float(commission),
                position_percent=float(position_percent),
                leverage=float(leverage),
                selected_start_date=start_date,
                selected_end_date=end_date,
                data_df=df,
                result_metrics=result.metrics,
            )
            st.caption(f"已保存回测记录：{run_id}")
            st.caption(f"日志：{log_file}")
            st.caption(f"详情：{detail_file}")
        except Exception as exc:  # noqa: BLE001
            st.warning(f"回测记录保存失败：{exc}")

        st.subheader("回测指标")
        col1, col2, col3, col4 = st.columns(4)

        col1.metric("总收益率", f"{result.metrics['总收益率(%)']}%")
        col2.metric("最大回撤", f"{result.metrics['最大回撤(%)']}%")
        col3.metric("胜率", f"{result.metrics['胜率(%)']}%")
        sharpe_text = "N/A" if result.metrics["Sharpe"] is None else str(result.metrics["Sharpe"])
        col4.metric("Sharpe", sharpe_text)

        with st.expander("查看全部指标", expanded=True):
            metrics_df = pd.DataFrame([result.metrics]).T
            metrics_df.columns = ["值"]
            st.dataframe(metrics_df, use_container_width=True)

        st.subheader("K线图")
        df_plot = df.reset_index().rename(columns={"open_time": "datetime"})
        if "datetime" not in df_plot.columns:
            df_plot = df.reset_index().rename(columns={df.reset_index().columns[0]: "datetime"})

        candle_fig = go.Figure(
            data=[
                go.Candlestick(
                    x=df_plot["datetime"],
                    open=df_plot["open"],
                    high=df_plot["high"],
                    low=df_plot["low"],
                    close=df_plot["close"],
                    name=symbol,
                )
            ]
        )
        candle_fig.update_layout(height=420, margin=dict(l=10, r=10, t=20, b=10), xaxis_rangeslider_visible=False)
        st.plotly_chart(candle_fig, use_container_width=True)

        st.subheader("资金曲线")
        if result.equity_curve.empty:
            st.info("没有可绘制的资金曲线")
        else:
            equity_fig = go.Figure()
            equity_fig.add_trace(
                go.Scatter(
                    x=result.equity_curve["datetime"],
                    y=result.equity_curve["equity"],
                    mode="lines",
                    name="Equity",
                    hovertemplate="时间: %{x|%Y-%m-%d %H:%M:%S}<br>资金量: %{y:,.2f} USDT<extra></extra>",
                )
            )
            equity_fig.update_layout(height=360, margin=dict(l=10, r=10, t=20, b=10), hovermode="x unified")
            st.plotly_chart(equity_fig, use_container_width=True)

        st.subheader("逐笔交易明细")
        if result.trade_details.empty:
            st.info("本次回测没有产生已平仓交易")
        else:
            st.dataframe(result.trade_details, use_container_width=True, hide_index=True)
            csv_bytes = result.trade_details.to_csv(index=False).encode("utf-8-sig")
            st.download_button(
                "下载交易明细 CSV",
                data=csv_bytes,
                file_name=f"{symbol}_{strategy_display_name}_trade_details.csv",
                mime="text/csv",
            )

    if optimize_btn:
        st.subheader("参数优化结果")
        if opt_method == "网格搜索":
            try:
                custom_grid = _parse_optimization_grid(st.session_state.get("opt_grid_json", ""), param_schema)
            except Exception as exc:  # noqa: BLE001
                st.error(f"自定义优化网格解析失败：{exc}")
                st.stop()

            if custom_grid:
                param_grid = custom_grid
                st.caption("使用自定义优化网格")
            else:
                if not param_schema:
                    st.error("当前策略没有参数 Schema，且未提供自定义优化网格 JSON")
                    st.stop()
                param_grid = build_auto_param_grid(param_schema, strategy_params, points_per_param=int(opt_points))
                st.caption("使用自动网格（基于当前参数）")

            combo_count = _count_combinations(param_grid)
            st.caption(f"组合数：{combo_count}（最大执行：{int(opt_max_combinations)}）")

            with st.spinner("正在执行网格搜索..."):
                try:
                    gs = optimize_parameters(
                        df=df,
                        strategy_cls=strategy_cls,
                        param_grid=param_grid,
                        initial_cash=float(initial_cash),
                        commission=float(commission),
                        position_percent=float(position_percent),
                        leverage=float(leverage),
                        objective=opt_objective,
                        max_combinations=int(opt_max_combinations),
                    )
                except Exception as exc:  # noqa: BLE001
                    st.error(f"网格搜索失败：{exc}")
                    st.stop()

            if gs.ranking.empty or not gs.best_params:
                st.warning("未找到有效参数组合")
                st.stop()

            st.success("网格搜索完成")
            st.write("最优参数：")
            st.json(gs.best_params)
            st.session_state["last_optimized_params"] = dict(gs.best_params)
            st.session_state["last_optimized_strategy_file"] = str(st.session_state.get("strategy_file_name", ""))

            if st.button("一键应用该最优参数", key="apply_best_grid_main"):
                applied_params, skipped_params = _apply_params_to_ui(gs.best_params, param_schema)
                if applied_params:
                    st.success(f"已应用参数：{json.dumps(applied_params, ensure_ascii=False)}")
                if skipped_params:
                    st.warning(f"以下参数未应用：{json.dumps(skipped_params, ensure_ascii=False)}")
                st.rerun()

            topn = min(30, len(gs.ranking))
            ranking_show = gs.ranking.head(topn).copy()
            ranking_show["参数"] = ranking_show["参数"].apply(lambda x: json.dumps(x, ensure_ascii=False))
            st.dataframe(ranking_show, use_container_width=True, hide_index=True)

            with st.spinner("正在执行 Walk-Forward..."):
                try:
                    wf_df, wf_summary = run_walk_forward(
                        df=df,
                        strategy_cls=strategy_cls,
                        param_grid=param_grid,
                        initial_cash=float(initial_cash),
                        commission=float(commission),
                        position_percent=float(position_percent),
                        leverage=float(leverage),
                        objective=opt_objective,
                        folds=int(opt_folds),
                        max_combinations=int(opt_max_combinations),
                    )
                except Exception as exc:  # noqa: BLE001
                    st.error(f"Walk-Forward 失败：{exc}")
                    st.stop()
        else:
            if not param_schema:
                st.error("当前策略没有参数 Schema，无法进行 Optuna 优化")
                st.stop()

            with st.spinner("正在执行 Optuna 贝叶斯优化..."):
                try:
                    opt_res = optimize_parameters_optuna(
                        df=df,
                        strategy_cls=strategy_cls,
                        param_schema=param_schema,
                        initial_cash=float(initial_cash),
                        commission=float(commission),
                        position_percent=float(position_percent),
                        leverage=float(leverage),
                        objective=opt_objective,
                        n_trials=int(opt_trials),
                        sampler_name=str(opt_sampler),
                        seed=int(opt_seed),
                    )
                except Exception as exc:  # noqa: BLE001
                    st.error(f"Optuna 优化失败：{exc}")
                    st.stop()

            if opt_res.ranking.empty or not opt_res.best_params:
                st.warning("Optuna 未找到有效参数")
                st.stop()

            st.success("Optuna 优化完成")
            st.write("最优参数：")
            st.json(opt_res.best_params)
            st.session_state["last_optimized_params"] = dict(opt_res.best_params)
            st.session_state["last_optimized_strategy_file"] = str(st.session_state.get("strategy_file_name", ""))

            if st.button("一键应用该最优参数", key="apply_best_optuna_main"):
                applied_params, skipped_params = _apply_params_to_ui(opt_res.best_params, param_schema)
                if applied_params:
                    st.success(f"已应用参数：{json.dumps(applied_params, ensure_ascii=False)}")
                if skipped_params:
                    st.warning(f"以下参数未应用：{json.dumps(skipped_params, ensure_ascii=False)}")
                st.rerun()

            st.caption(f"最优评分：{opt_res.best_score}")

            topn = min(30, len(opt_res.ranking))
            ranking_show = opt_res.ranking.head(topn).copy()
            ranking_show["参数"] = ranking_show["参数"].apply(lambda x: json.dumps(x, ensure_ascii=False))
            st.dataframe(ranking_show, use_container_width=True, hide_index=True)

            with st.spinner("正在执行 Walk-Forward（Optuna）..."):
                try:
                    wf_df, wf_summary = run_walk_forward_optuna(
                        df=df,
                        strategy_cls=strategy_cls,
                        param_schema=param_schema,
                        initial_cash=float(initial_cash),
                        commission=float(commission),
                        position_percent=float(position_percent),
                        leverage=float(leverage),
                        objective=opt_objective,
                        folds=int(opt_folds),
                        n_trials=int(opt_trials),
                        sampler_name=str(opt_sampler),
                        seed=int(opt_seed),
                    )
                except Exception as exc:  # noqa: BLE001
                    st.error(f"Walk-Forward（Optuna）失败：{exc}")
                    st.stop()

        st.subheader("Walk-Forward 报告")
        c1, c2, c3 = st.columns(3)
        c1.metric("窗口数", wf_summary.get("folds"))
        avg_oos = wf_summary.get("平均样本外收益率(%)")
        c2.metric("平均样本外收益率", "N/A" if avg_oos is None else f"{avg_oos}%")
        c3.metric("样本外正收益窗口", f"{wf_summary.get('样本外正收益窗口数')}/{wf_summary.get('样本外总窗口数')}")

        if wf_df.empty:
            st.warning("Walk-Forward 结果为空，请扩大时间范围或减少窗口数")
        else:
            st.dataframe(wf_df, use_container_width=True, hide_index=True)
            wf_csv = wf_df.to_csv(index=False).encode("utf-8-sig")
            st.download_button(
                "下载 Walk-Forward 报告 CSV",
                data=wf_csv,
                file_name=f"{symbol}_{strategy_display_name}_walk_forward.csv",
                mime="text/csv",
            )
else:
    st.info("请在左侧选择或编辑策略文件，然后点击“开始回测”或“一键参数优化 + Walk-Forward”。")
