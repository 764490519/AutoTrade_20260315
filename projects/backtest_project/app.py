from __future__ import annotations

import json
import traceback
import hashlib
import csv
import math
import os
import re
from collections.abc import Mapping
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from backtest_engine import run_backtest
from binance_data import BinanceDataError, INTERVAL_TO_MS, fetch_klines
from optimization_engine import (
    optimize_parameters,
    optimize_parameters_optuna,
    run_walk_forward,
    run_walk_forward_optuna,
)
from api_config import API_CONFIG_FILE, get_api_config_section
from email_notifier import EmailConfig, EmailNotifier, EmailNotifyError
from live_trading_engine import (
    LiveTradingConfig,
    execute_signal_once,
    get_live_worker_state,
    start_live_worker,
    stop_live_worker,
)
from okx_trading import OKXClient, OKXConfig
from strategy_loader import compile_strategy_runtime_from_code


MODE_LOCK = str(os.getenv("AUTOTRADE_MODE_LOCK", "")).strip().lower()
if MODE_LOCK not in {"backtest", "live"}:
    MODE_LOCK = ""

st.set_page_config(page_title="VectorBT + Binance 交易系统", layout="wide")
if MODE_LOCK == "live":
    st.title("🟢 策略自动交易软件")
    st.caption("交易接口：OKX API（需配置 apis.toml）")
else:
    st.title("📈 VectorBT + Binance 回测软件")
    st.caption("数据源：Binance Spot Kline API（公共接口）")

APP_ROOT_DIR = Path(__file__).resolve().parent
_env_strategy_dir = os.getenv("AUTOTRADE_STRATEGY_DIR", "").strip()
_env_report_dir = os.getenv("AUTOTRADE_REPORT_DIR", "").strip()
if _env_strategy_dir:
    STRATEGY_DIR = Path(_env_strategy_dir).expanduser().resolve()
else:
    STRATEGY_DIR = APP_ROOT_DIR / "strategy_files"
if _env_report_dir:
    REPORT_DIR = Path(_env_report_dir).expanduser().resolve()
else:
    REPORT_DIR = APP_ROOT_DIR / "reports"
BACKTEST_RUN_DIR = REPORT_DIR / "backtest_runs"
BACKTEST_LOG_CSV = REPORT_DIR / "backtest_run_history.csv"
UI_STATE_FILE = REPORT_DIR / "ui_state.json"
OP_LOG_CSV = REPORT_DIR / "operation_logs.csv"
OP_LOG_DETAIL_DIR = REPORT_DIR / "operation_log_details"
LIVE_WORKER_KEY = "default_live_worker"
NEW_STRATEGY_LABEL = "（新建策略文件）"

PERSIST_STATE_KEYS = {
    "product_mode",
    "symbol",
    "symbol_select",
    "symbol_custom",
    "interval",
    "start_date",
    "end_date",
    "selected_strategy_file",
    "strategy_file_name",
    "strategy_params_json",
    "initial_cash",
    "commission",
    "position_sizing_mode",
    "position_percent",
    "fixed_trade_amount",
    "leverage",
    "opt_method",
    "opt_objective",
    "opt_folds",
    "opt_run_wf",
    "opt_points",
    "opt_grid_mode",
    "opt_max_combinations",
    "opt_trials",
    "opt_sampler",
    "opt_seed",
    "opt_parallel",
    "opt_workers",
    "opt_selected_params",
    "last_optimized_params",
    "last_optimized_strategy_file",
    "okx_inst_type",
    "okx_inst_id",
    "okx_td_mode",
    "okx_pos_side",
    "okx_ccy",
    "okx_show_only_nonzero",
    "notify_trade_email",
    "live_market_symbol",
    "live_symbol_select",
    "live_symbol_custom",
    "live_interval",
    "live_okx_inst_type",
    "live_td_mode",
    "live_leverage",
    "live_pos_side",
    "live_order_size",
    "live_poll_seconds",
    "live_lookback_bars",
    "live_only_new_bar",
    "live_close_on_flat",
    "live_allow_short",
    "live_confirm_trade",
}

SYMBOL_CUSTOM_OPTION = "__CUSTOM__"
COMMON_MARKET_SYMBOLS = [
    "BTCUSDT",
    "ETHUSDT",
    "SOLUSDT",
    "BNBUSDT",
    "XRPUSDT",
    "DOGEUSDT",
    "ADAUSDT",
    "TRXUSDT",
    "LTCUSDT",
    "AVAXUSDT",
]
OKX_SYMBOL_QUOTE_SUFFIXES = (
    "USDT",
    "USDC",
    "BUSD",
    "FDUSD",
    "TUSD",
    "DAI",
    "BTC",
    "ETH",
    "USD",
    "EUR",
)

NEW_STRATEGY_TEMPLATE = """from __future__ import annotations

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
    fast_period = int(params.get("fast_period", 20))
    slow_period = int(params.get("slow_period", 60))
    if fast_period < 2 or slow_period < 2 or fast_period >= slow_period:
        raise ValueError("参数不合法")

    close = pd.to_numeric(df["close"], errors="coerce").astype(float)
    fast = close.rolling(window=fast_period, min_periods=fast_period).mean()
    slow = close.rolling(window=slow_period, min_periods=slow_period).mean()

    n = len(df)
    targets = np.full(n, np.nan, dtype=float)
    pos = 0
    for i in range(slow_period + 1, n):
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


class MyStrategy:
    USE_GLOBAL_POSITION_PERCENT = True

    @staticmethod
    def generate_targets(df: pd.DataFrame, params: dict[str, Any]) -> np.ndarray:
        return generate_targets(df, params)
"""

DEFAULT_STRATEGY_FILES: dict[str, str] = {}



@st.cache_data(ttl=60 * 30, show_spinner=False)
def load_binance_data(symbol: str, interval: str, start_dt: datetime, end_dt: datetime) -> pd.DataFrame:
    return fetch_klines(symbol=symbol, interval=interval, start_time=start_dt, end_time=end_dt)


def _to_utc_start(d: date) -> datetime:
    return datetime.combine(d, time.min).replace(tzinfo=timezone.utc)


def _to_utc_end(d: date) -> datetime:
    return datetime.combine(d + timedelta(days=1), time.min).replace(tzinfo=timezone.utc)


def _infer_auto_warmup_bars(
    strategy_params: dict[str, Any],
    param_schema: dict[str, dict[str, Any]] | None = None,
) -> int:
    """
    自动预热K线数量（无需前端设置）：
    - 优先根据参数名里的周期类字段推断（period/window/lookback/length/span/cooldown）
    - 兜底至少 32 根，最多 2000 根
    """
    schema = param_schema or {}
    candidates: list[int] = []
    hint_tokens = ("period", "window", "lookback", "length", "span", "cooldown")

    for name, value in (strategy_params or {}).items():
        key = str(name).lower()
        if not any(token in key for token in hint_tokens):
            continue
        cfg = schema.get(name, {})
        if str(cfg.get("type", "")).lower() not in {"int", "float"}:
            continue
        try:
            v = float(value)
        except Exception:  # noqa: BLE001
            continue
        if not math.isfinite(v) or v <= 0:
            continue
        candidates.append(int(math.ceil(v)))

    base = max(candidates) if candidates else 30
    warmup_bars = base + 2
    return max(32, min(2000, int(warmup_bars)))


def _calc_warmup_start_dt(start_dt: datetime, interval: str, warmup_bars: int) -> datetime:
    interval_ms = INTERVAL_TO_MS.get(str(interval))
    if interval_ms is None or warmup_bars <= 0:
        return start_dt
    return start_dt - timedelta(milliseconds=int(interval_ms) * int(warmup_bars))


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
    runtime = compile_strategy_runtime_from_code(code)
    return runtime.strategy_obj, runtime.display_name, runtime.params_schema


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

        desc = cfg.get("desc", cfg.get("description", cfg.get("help", "")))
        if desc is not None:
            desc_text = str(desc).strip()
            if desc_text:
                normalized["desc"] = desc_text

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


def _normalize_market_symbol(value: Any) -> str:
    text = str(value or "").upper().strip()
    # 兼容 BTC/USDT、BTC-USDT、BTC_USDT 等写法
    text = re.sub(r"[^A-Z0-9]", "", text)
    return text


def _split_market_symbol_base_quote(value: Any) -> tuple[str, str] | None:
    symbol = _normalize_market_symbol(value)
    if not symbol:
        return None
    for quote in OKX_SYMBOL_QUOTE_SUFFIXES:
        if symbol.endswith(quote) and len(symbol) > len(quote):
            base = symbol[: -len(quote)]
            if base:
                return base, quote
    return None


def _build_okx_inst_id_from_symbol(value: Any, inst_type: Any) -> str | None:
    parsed = _split_market_symbol_base_quote(value)
    if parsed is None:
        return None
    base, quote = parsed
    inst_type_text = str(inst_type or "").upper().strip()
    if inst_type_text == "SPOT":
        return f"{base}-{quote}"
    if inst_type_text == "SWAP":
        return f"{base}-{quote}-SWAP"
    return None


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


def _normalize_persisted_choice(key: str, value: Any) -> Any:
    text = str(value or "").strip()
    low = text.lower()

    if key == "product_mode":
        if text in {"backtest", "回测与优化"} or "back" in low or "回测" in text:
            return "backtest"
        if text in {"live", "策略自动交易"} or "live" in low or "交易" in text:
            return "live"
        return "backtest"

    if key == "position_sizing_mode":
        if low in {"percent_equity", "percent", "pct"} or "百分比" in text:
            return "percent_equity"
        if low in {"fixed_amount", "fixed", "amount"} or "固定" in text:
            return "fixed_amount"
        return "percent_equity"

    if key == "opt_method":
        if "optuna" in low or "贝叶斯" in text:
            return "Optuna(贝叶斯)"
        if "网格" in text or "grid" in low:
            return "网格搜索"
        return "Optuna(贝叶斯)"

    if key == "opt_grid_mode":
        if "step" in low or "全展开" in text:
            return "按step全展开"
        if "采样" in text or "点数" in text or "sample" in low:
            return "按点数采样"
        return "按点数采样"

    if key == "opt_objective":
        if "sharpe" in low:
            return "Sharpe"
        if "总收益" in text:
            return "总收益率(%)"
        if "年化" in text:
            return "年化收益率(%)"
        if "回撤" in text:
            return "最大回撤(%)"
        return "Sharpe"

    if key == "opt_sampler":
        if "cma" in low:
            return "CMA-ES"
        if "随机" in text or "random" in low:
            return "随机"
        return "TPE"

    return value


def _normalize_persisted_ui_state(raw: dict[str, Any]) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    enum_keys = {"product_mode", "position_sizing_mode", "opt_method", "opt_grid_mode", "opt_objective", "opt_sampler"}
    # 避免历史 UI_STATE 中 strategy_code 过大/乱码影响启动，统一从策略文件加载
    ignored_keys = {"strategy_code"}

    for key, value in raw.items():
        if key in ignored_keys:
            continue
        if key in enum_keys:
            normalized[key] = _normalize_persisted_choice(key, value)
        else:
            normalized[key] = value

    return normalized


def _restore_persisted_ui_state() -> None:
    if st.session_state.get("_ui_state_restored", False):
        return

    raw = _normalize_persisted_ui_state(_load_persisted_ui_state())
    for key, value in raw.items():
        if key in st.session_state:
            continue
        if key in {"start_date", "end_date"} and isinstance(value, str):
            try:
                value = date.fromisoformat(value)
            except Exception:  # noqa: BLE001
                continue
        st.session_state[key] = value

    selected_file = str(st.session_state.get("selected_strategy_file", "")).strip()
    if selected_file and selected_file != NEW_STRATEGY_LABEL:
        try:
            st.session_state["strategy_file_name"] = selected_file
            st.session_state["strategy_code"] = _load_strategy_code(selected_file)
            st.session_state["_loaded_strategy_file"] = selected_file
        except Exception:  # noqa: BLE001
            pass

    st.session_state["_ui_state_restored"] = True


def _persist_ui_state() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    state: dict[str, Any] = {}

    for key in PERSIST_STATE_KEYS:
        if key in st.session_state:
            state[key] = _to_serializable(st.session_state[key])

    # 动态策略参数（param_xxx）也持久化，刷新后仍保留
    for key, value in st.session_state.items():
        if (
            key.startswith("param_")
            or key.startswith("opt_pick_")
            or key.startswith("opt_lb_")
            or key.startswith("opt_ub_")
            or key.startswith("opt_range_")
        ):
            state[key] = _to_serializable(value)

    UI_STATE_FILE.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def _to_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    return default


def _load_okx_config_from_env_or_secrets() -> OKXConfig | None:
    secret_bucket: dict[str, Any] = {}
    file_bucket: dict[str, Any] = get_api_config_section("okx")
    try:
        if "okx" in st.secrets:
            maybe = st.secrets["okx"]
            if isinstance(maybe, Mapping):
                secret_bucket = dict(maybe)
    except Exception:  # noqa: BLE001
        secret_bucket = {}

    file_aliases: dict[str, tuple[str, ...]] = {
        "OKX_API_KEY": ("api_key", "OKX_API_KEY"),
        "OKX_API_SECRET": ("api_secret", "OKX_API_SECRET"),
        "OKX_API_PASSPHRASE": ("api_passphrase", "passphrase", "OKX_API_PASSPHRASE"),
        "OKX_BASE_URL": ("base_url", "OKX_BASE_URL"),
        "OKX_DEMO_TRADING": ("demo_trading", "OKX_DEMO_TRADING"),
        "OKX_FLAG": ("flag", "OKX_FLAG"),
        "OKX_TIMEOUT": ("timeout", "OKX_TIMEOUT"),
    }

    def pick(key: str, default: Any = None):
        if key in secret_bucket:
            return secret_bucket.get(key)
        try:
            if key in st.secrets:
                return st.secrets.get(key)
        except Exception:  # noqa: BLE001
            pass
        env_value = os.getenv(key)
        if env_value is not None:
            return env_value
        for alias in file_aliases.get(key, ()):
            if alias in file_bucket:
                return file_bucket.get(alias)
        return default

    api_key = pick("OKX_API_KEY")
    api_secret = pick("OKX_API_SECRET")
    passphrase = pick("OKX_API_PASSPHRASE")
    if not api_key or not api_secret or not passphrase:
        return None

    base_url = str(pick("OKX_BASE_URL", "https://www.okx.com")).strip() or "https://www.okx.com"
    demo_flag = _to_bool(pick("OKX_DEMO_TRADING", pick("OKX_FLAG", "0")))
    timeout = int(float(pick("OKX_TIMEOUT", 10)))
    timeout = max(3, min(120, timeout))

    return OKXConfig(
        api_key=str(api_key).strip(),
        api_secret=str(api_secret).strip(),
        passphrase=str(passphrase).strip(),
        base_url=base_url,
        demo_trading=demo_flag,
        timeout=timeout,
    )


def _build_okx_client() -> OKXClient | None:
    cfg = _load_okx_config_from_env_or_secrets()
    if cfg is None:
        return None
    return OKXClient(cfg)


def _parse_recipients(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        items = value
    else:
        text = str(value).replace(";", ",")
        items = [part.strip() for part in text.split(",")]
    out: list[str] = []
    for item in items:
        s = str(item).strip()
        if s and s not in out:
            out.append(s)
    return out


def _load_email_config() -> EmailConfig | None:
    secret_bucket: dict[str, Any] = {}
    file_bucket: dict[str, Any] = get_api_config_section("email")
    if not file_bucket:
        file_bucket = get_api_config_section("notify.email")
    try:
        if "email" in st.secrets and isinstance(st.secrets["email"], Mapping):
            secret_bucket = dict(st.secrets["email"])
    except Exception:  # noqa: BLE001
        secret_bucket = {}

    aliases: dict[str, tuple[str, ...]] = {
        "EMAIL_NOTIFY_ENABLED": ("enabled", "EMAIL_NOTIFY_ENABLED"),
        "SMTP_HOST": ("smtp_host", "host", "SMTP_HOST"),
        "SMTP_PORT": ("smtp_port", "port", "SMTP_PORT"),
        "SMTP_USER": ("smtp_user", "user", "SMTP_USER"),
        "SMTP_PASSWORD": ("smtp_password", "password", "SMTP_PASSWORD"),
        "SMTP_SENDER": ("sender", "from", "SMTP_SENDER"),
        "SMTP_RECIPIENTS": ("recipients", "to", "SMTP_RECIPIENTS"),
        "SMTP_USE_SSL": ("use_ssl", "SMTP_USE_SSL"),
        "SMTP_USE_STARTTLS": ("use_starttls", "SMTP_USE_STARTTLS"),
        "EMAIL_SUBJECT_PREFIX": ("subject_prefix", "EMAIL_SUBJECT_PREFIX"),
    }

    def pick(key: str, default: Any = None):
        if key in secret_bucket:
            return secret_bucket.get(key)
        try:
            if key in st.secrets:
                return st.secrets.get(key)
        except Exception:  # noqa: BLE001
            pass
        env_value = os.getenv(key)
        if env_value is not None:
            return env_value
        for alias in aliases.get(key, ()):
            if alias in file_bucket:
                return file_bucket.get(alias)
        return default

    enabled = _to_bool(pick("EMAIL_NOTIFY_ENABLED", True), default=True)
    if not enabled:
        return None

    smtp_host = str(pick("SMTP_HOST", "")).strip()
    smtp_port = int(float(pick("SMTP_PORT", 465)))
    smtp_user = str(pick("SMTP_USER", "")).strip()
    smtp_password = str(pick("SMTP_PASSWORD", "")).strip()
    sender = str(pick("SMTP_SENDER", "")).strip()
    recipients = _parse_recipients(pick("SMTP_RECIPIENTS", []))
    use_ssl = _to_bool(pick("SMTP_USE_SSL", True), default=True)
    use_starttls = _to_bool(pick("SMTP_USE_STARTTLS", not use_ssl), default=not use_ssl)
    subject_prefix = str(pick("EMAIL_SUBJECT_PREFIX", "[AutoTrade]")).strip() or "[AutoTrade]"

    if not recipients and sender:
        recipients = [sender]

    if not smtp_host or smtp_port <= 0 or not sender or not recipients:
        return None
    if smtp_user and not smtp_password:
        return None

    return EmailConfig(
        smtp_host=smtp_host,
        smtp_port=smtp_port,
        smtp_user=smtp_user,
        smtp_password=smtp_password,
        sender=sender,
        recipients=recipients,
        use_ssl=use_ssl,
        use_starttls=use_starttls,
        subject_prefix=subject_prefix,
    )


def _build_email_notifier() -> EmailNotifier | None:
    cfg = _load_email_config()
    if cfg is None:
        return None
    return EmailNotifier(cfg)


def _send_trade_email_notification(
    notifier: EmailNotifier | None,
    *,
    action: str,
    status: str,
    request_payload: dict[str, Any] | None = None,
    response_payload: Any = None,
    error_message: str | None = None,
) -> tuple[bool, str]:
    if notifier is None:
        return False, "未配置邮件通知"

    action_map = {
        "place_open_order": "开仓下单",
        "close_position": "平仓",
        "close_position_spot": "现货平仓",
        "close_position_derivative": "合约平仓",
        "live_strategy_run_once": "策略实时执行",
    }
    status_map = {
        "success": "成功",
        "failed": "失败",
        "rejected": "已拒绝",
        "skipped": "已跳过",
    }

    req = _to_serializable(request_payload or {})
    resp = _to_serializable(response_payload or {})
    action_cn = action_map.get(action, action)
    status_cn = status_map.get(status, status)
    inst_id = str(req.get("instId", "") or "-")
    subject = f"{action_cn} | {status_cn} | {inst_id}"

    if isinstance(resp, dict):
        ord_id = resp.get("ordId")
        cl_ord_id = resp.get("clOrdId")
        s_code = resp.get("sCode")
        s_msg = resp.get("sMsg")
    else:
        ord_id = cl_ord_id = s_code = s_msg = None

    body = (
        "【AutoTrade 交易通知】\n\n"
        f"时间(UTC): {datetime.now(timezone.utc).isoformat()}\n"
        f"动作: {action_cn}\n"
        f"状态: {status_cn}\n"
        f"交易环境: {'模拟盘' if req.get('demo_trading') else '实盘'}\n\n"
        "【交易参数】\n"
        f"instId: {inst_id}\n"
        f"instType: {req.get('instType', '-')}\n"
        f"side: {req.get('side', '-')}\n"
        f"tdMode: {req.get('tdMode', '-')}\n"
        f"ordType: {req.get('ordType', '-')}\n"
        f"sz: {req.get('sz', '-')}\n"
        f"px: {req.get('px', '-')}\n"
        f"posSide: {req.get('posSide', '-')}\n"
        f"ccy: {req.get('ccy', '-')}\n"
        f"autoCxl: {req.get('autoCxl', '-')}\n\n"
        "【交易结果】\n"
        f"ordId: {ord_id or '-'}\n"
        f"clOrdId: {cl_ord_id or '-'}\n"
        f"sCode: {s_code or '-'}\n"
        f"sMsg: {s_msg or '-'}\n"
    )
    if error_message:
        body += f"\n错误信息:\n{error_message}\n"
    body += (
        "\n【完整请求】\n"
        f"{json.dumps(req, ensure_ascii=False, indent=2)}\n"
        "\n【完整响应】\n"
        f"{json.dumps(resp, ensure_ascii=False, indent=2)}\n"
    )

    try:
        notifier.send(subject=subject, body=body)
        return True, "邮件已发送"
    except EmailNotifyError as exc:
        return False, str(exc)
    except Exception as exc:  # noqa: BLE001
        return False, f"邮件发送失败: {exc}"


def _safe_dataframe(rows: list[dict[str, Any]]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    try:
        return pd.DataFrame(rows)
    except Exception:  # noqa: BLE001
        return pd.DataFrame([{"raw": json.dumps(rows, ensure_ascii=False)}])


def _append_operation_log(
    *,
    module: str,
    action: str,
    status: str,
    request_payload: dict[str, Any] | None = None,
    response_payload: Any = None,
    error_message: str | None = None,
) -> tuple[str, Path]:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    OP_LOG_DETAIL_DIR.mkdir(parents=True, exist_ok=True)

    now_utc = datetime.now(timezone.utc)
    log_id = f"{now_utc.strftime('%Y%m%dT%H%M%SZ')}_{uuid4().hex[:8]}"

    detail = {
        "log_id": log_id,
        "time_utc": now_utc,
        "module": module,
        "action": action,
        "status": status,
        "request": request_payload or {},
        "response": response_payload,
        "error": error_message,
    }
    detail = _to_serializable(detail)

    detail_path = OP_LOG_DETAIL_DIR / f"{log_id}.json"
    detail_path.write_text(json.dumps(detail, ensure_ascii=False, indent=2), encoding="utf-8")

    row = {
        "log_id": log_id,
        "time_utc": detail["time_utc"],
        "module": module,
        "action": action,
        "status": status,
        "error": error_message,
        "request_json": json.dumps(detail.get("request", {}), ensure_ascii=False),
        "detail_json_file": str(detail_path),
    }

    write_header = not OP_LOG_CSV.exists()
    with OP_LOG_CSV.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)

    return log_id, detail_path


def _load_recent_operation_logs(limit: int = 50) -> pd.DataFrame:
    if not OP_LOG_CSV.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(OP_LOG_CSV)
        if df.empty:
            return df
        if "time_utc" in df.columns:
            df = df.sort_values(by="time_utc", ascending=False)
        return df.head(max(1, int(limit))).reset_index(drop=True)
    except Exception:  # noqa: BLE001
        return pd.DataFrame()


def _get_current_strategy_runtime_for_live() -> tuple[Any, str, dict[str, Any], dict[str, dict[str, Any]]]:
    code = str(st.session_state.get("strategy_code", ""))
    strategy_cls, strategy_display_name, raw_schema = _compile_strategy(code)
    param_schema = _normalize_params_schema(raw_schema)

    params: dict[str, Any] = {}
    if param_schema:
        for name, cfg in param_schema.items():
            key = f"param_{name}"
            raw = st.session_state.get(key, cfg["default"])
            if cfg["type"] == "int":
                params[name] = int(round(float(raw)))
            else:
                params[name] = float(raw)
    else:
        params = _parse_params_json(str(st.session_state.get("strategy_params_json", "{}")))

    return strategy_cls, strategy_display_name, params, param_schema


def _render_okx_trading_panel() -> None:
    st.divider()
    st.subheader("🟢 实盘交易模块（OKX API）")
    st.caption("支持：基于策略信号自动开平仓（请先配置 OKX API）")

    client = _build_okx_client()
    if client is None:
        st.info(
            "未检测到 OKX API 配置。请在 `config/apis.toml` 或环境变量/`.streamlit/secrets.toml` 配置："
            "OKX_API_KEY / OKX_API_SECRET / OKX_API_PASSPHRASE（可选 OKX_DEMO_TRADING=1）。"
        )
        st.caption(f"配置文件路径：{API_CONFIG_FILE}")
        return

    mode_text = "模拟盘" if client.config.demo_trading else "实盘"
    st.success(f"已加载 OKX API 配置，当前模式：{mode_text}")
    if not client.config.demo_trading:
        st.warning("当前为实盘模式，操作将真实成交。请务必先用小仓位测试。")

    email_notifier = _build_email_notifier()
    if "notify_trade_email" not in st.session_state:
        st.session_state["notify_trade_email"] = True
    st.checkbox("开/平仓发送邮件通知", key="notify_trade_email")
    if email_notifier is None:
        st.caption("邮件通知状态：未配置（需在 config/apis.toml 配置 [email]）")
    else:
        masked_to = ", ".join(email_notifier.config.recipients[:3])
        if len(email_notifier.config.recipients) > 3:
            masked_to += "..."
        st.caption(f"邮件通知状态：已启用（收件人：{masked_to}）")

    if "okx_inst_type" not in st.session_state:
        st.session_state["okx_inst_type"] = "SWAP"
    if "okx_inst_id" not in st.session_state:
        st.session_state["okx_inst_id"] = ""
    if "okx_td_mode" not in st.session_state:
        st.session_state["okx_td_mode"] = "cross"
    if "okx_pos_side" not in st.session_state:
        st.session_state["okx_pos_side"] = "net"
    if "okx_ccy" not in st.session_state:
        st.session_state["okx_ccy"] = ""
    if "okx_show_only_nonzero" not in st.session_state:
        st.session_state["okx_show_only_nonzero"] = True
    if "live_market_symbol" not in st.session_state:
        st.session_state["live_market_symbol"] = "BTCUSDT"
    if "live_symbol_select" not in st.session_state:
        st.session_state["live_symbol_select"] = (
            st.session_state["live_market_symbol"]
            if st.session_state["live_market_symbol"] in COMMON_MARKET_SYMBOLS
            else SYMBOL_CUSTOM_OPTION
        )
    if "live_symbol_custom" not in st.session_state:
        st.session_state["live_symbol_custom"] = st.session_state["live_market_symbol"]
    if "live_interval" not in st.session_state:
        st.session_state["live_interval"] = "1m"
    if "live_okx_inst_type" not in st.session_state:
        st.session_state["live_okx_inst_type"] = "SWAP"
    if str(st.session_state.get("live_okx_inst_type", "SWAP")).upper().strip() not in {"SPOT", "SWAP"}:
        st.session_state["live_okx_inst_type"] = "SWAP"
    if "live_td_mode" not in st.session_state:
        st.session_state["live_td_mode"] = "cross"
    if "live_leverage" not in st.session_state:
        st.session_state["live_leverage"] = "5"
    if "live_pos_side" not in st.session_state:
        st.session_state["live_pos_side"] = "net"
    if "live_order_size" not in st.session_state:
        st.session_state["live_order_size"] = "0.01"
    if "live_poll_seconds" not in st.session_state:
        st.session_state["live_poll_seconds"] = 10
    if "live_lookback_bars" not in st.session_state:
        st.session_state["live_lookback_bars"] = 500
    if "live_only_new_bar" not in st.session_state:
        st.session_state["live_only_new_bar"] = True
    if "live_close_on_flat" not in st.session_state:
        st.session_state["live_close_on_flat"] = True
    if "live_allow_short" not in st.session_state:
        st.session_state["live_allow_short"] = True
    if "live_confirm_trade" not in st.session_state:
        st.session_state["live_confirm_trade"] = False

    st.markdown("#### 交易执行方式")
    st.info("已启用“策略自动交易”模式：不再提供手动开仓/平仓入口。")

    st.markdown("#### 策略实时执行（实时价格 + 策略信号）")
    try:
        live_strategy_cls, live_strategy_name, live_strategy_params, _ = _get_current_strategy_runtime_for_live()
        st.caption(f"当前策略：{live_strategy_name}")
    except Exception as exc:  # noqa: BLE001
        live_strategy_cls = None
        live_strategy_name = ""
        live_strategy_params = {}
        st.warning(f"策略编译失败，无法启动实时执行：{exc}")

    live_col1, live_col2, live_col3 = st.columns(3)
    with live_col1:
        live_symbol_options = COMMON_MARKET_SYMBOLS + [SYMBOL_CUSTOM_OPTION]
        if st.session_state.get("live_symbol_select") not in live_symbol_options:
            recovered_live_symbol = _normalize_market_symbol(st.session_state.get("live_market_symbol", "BTCUSDT")) or "BTCUSDT"
            if recovered_live_symbol in COMMON_MARKET_SYMBOLS:
                st.session_state["live_symbol_select"] = recovered_live_symbol
                st.session_state["live_symbol_custom"] = recovered_live_symbol
            else:
                st.session_state["live_symbol_select"] = SYMBOL_CUSTOM_OPTION
                st.session_state["live_symbol_custom"] = recovered_live_symbol

        live_symbol_choice = st.selectbox(
            "实时行情交易对(Binance)",
            options=live_symbol_options,
            key="live_symbol_select",
            format_func=lambda x: "自定义..." if x == SYMBOL_CUSTOM_OPTION else x,
        )
        if live_symbol_choice == SYMBOL_CUSTOM_OPTION:
            st.text_input("自定义实时行情交易对", key="live_symbol_custom", placeholder="例如 BTCUSDT / BTC-USDT / BTC/USDT")
            resolved_live_symbol = _normalize_market_symbol(st.session_state.get("live_symbol_custom", ""))
            st.session_state["live_symbol_custom"] = st.session_state.get("live_symbol_custom", "").strip()
            st.session_state["live_market_symbol"] = resolved_live_symbol or "BTCUSDT"
        else:
            st.session_state["live_market_symbol"] = live_symbol_choice

        st.selectbox("实时K线周期", ["1m", "5m", "15m", "1h", "4h", "1d"], key="live_interval")
        st.number_input("历史K线窗口", min_value=60, max_value=1000, step=10, key="live_lookback_bars")
    with live_col2:
        st.selectbox(
            "实盘类型(instType)",
            ["SPOT", "SWAP"],
            key="live_okx_inst_type",
            format_func=lambda x: "现货 (SPOT)" if x == "SPOT" else "合约 (SWAP)",
        )
        if str(st.session_state.get("live_okx_inst_type", "SWAP")).upper().strip() == "SPOT":
            st.session_state["live_td_mode"] = "cash"
            st.selectbox("实盘交易模式(tdMode)", ["cash"], key="live_td_mode")
            st.caption("现货模式自动使用 cash，不执行做空。")
        else:
            if str(st.session_state.get("live_td_mode", "cross")).strip() not in {"cross", "isolated"}:
                st.session_state["live_td_mode"] = "cross"
            st.selectbox("实盘交易模式(tdMode)", ["cross", "isolated"], key="live_td_mode")
            st.text_input("合约杠杆倍数(lever)", key="live_leverage", placeholder="例如 3 / 5 / 10")
    with live_col3:
        st.text_input("实盘下单数量(sz)", key="live_order_size")
        st.selectbox(
            "实盘持仓方向(posSide)",
            ["net", "long", "short", ""],
            key="live_pos_side",
            format_func=lambda x: "默认" if x == "" else x,
            disabled=str(st.session_state.get("live_okx_inst_type", "SWAP")).upper().strip() == "SPOT",
            help="现货模式下不使用该参数。",
        )
        st.number_input("轮询秒数", min_value=2, max_value=3600, step=1, key="live_poll_seconds")

    st.checkbox("仅在新K线出现时执行", key="live_only_new_bar")
    st.checkbox("信号为FLAT时自动平仓", key="live_close_on_flat")
    live_is_spot = str(st.session_state.get("live_okx_inst_type", "SWAP")).upper().strip() == "SPOT"
    if live_is_spot:
        st.session_state["live_allow_short"] = False
    st.checkbox("允许做空信号执行", key="live_allow_short", disabled=live_is_spot, help="现货模式下自动禁用")
    st.checkbox("我确认启用策略自动交易", key="live_confirm_trade")

    once_btn, start_btn, stop_btn = st.columns(3)
    run_once_live = once_btn.button("执行一次策略信号", use_container_width=True, key="live_run_once")
    start_live = start_btn.button("启动自动执行", type="primary", use_container_width=True, key="live_start")
    stop_live = stop_btn.button("停止自动执行", use_container_width=True, key="live_stop")

    live_cfg = None
    live_market_symbol = _normalize_market_symbol(st.session_state.get("live_market_symbol", "BTCUSDT")) or "BTCUSDT"
    live_inst_type = str(st.session_state.get("live_okx_inst_type", "SWAP")).upper().strip()
    derived_live_okx_inst_id = _build_okx_inst_id_from_symbol(live_market_symbol, live_inst_type)
    live_leverage_text = str(st.session_state.get("live_leverage", "")).strip()
    live_leverage_value: str | None = None
    if live_inst_type == "SWAP":
        if not live_leverage_text:
            st.error("合约模式请填写杠杆倍数。")
        else:
            try:
                leverage_float = float(live_leverage_text)
            except Exception:  # noqa: BLE001
                st.error("杠杆倍数格式错误，请输入数字。")
            else:
                if leverage_float <= 0:
                    st.error("杠杆倍数必须大于 0。")
                else:
                    live_leverage_value = live_leverage_text
    if derived_live_okx_inst_id:
        st.caption(f"自动映射的 OKX 标的：`{derived_live_okx_inst_id}`")
    else:
        st.error("当前交易对无法自动映射为 OKX 标的，请检查交易对格式（例如 BTCUSDT）。")

    if live_strategy_cls is not None:
        live_cfg = LiveTradingConfig(
            market_symbol=live_market_symbol,
            interval=str(st.session_state.get("live_interval", "1m")).strip(),
            lookback_bars=int(st.session_state.get("live_lookback_bars", 500)),
            poll_seconds=int(st.session_state.get("live_poll_seconds", 10)),
            only_new_bar=bool(st.session_state.get("live_only_new_bar", True)),
            include_unclosed_last_bar=True,
            close_on_flat=bool(st.session_state.get("live_close_on_flat", True)),
            allow_short=(False if live_is_spot else bool(st.session_state.get("live_allow_short", True))),
            okx_inst_id=derived_live_okx_inst_id or "",
            okx_inst_type=live_inst_type,
            okx_td_mode=("cash" if live_is_spot else str(st.session_state.get("live_td_mode", "cross")).strip()),
            okx_pos_side=("" if live_is_spot else (str(st.session_state.get("live_pos_side", "net")).strip() or "net")),
            okx_leverage=(None if live_is_spot else live_leverage_value),
            okx_ccy=(str(st.session_state.get("okx_ccy", "")).strip() or None),
            order_size=str(st.session_state.get("live_order_size", "0.01")).strip(),
            strategy_params=live_strategy_params,
            strategy_name=live_strategy_name,
            position_percent=float(st.session_state.get("position_percent", 95.0)),
        )

    if run_once_live:
        if live_strategy_cls is None or live_cfg is None:
            st.error("策略不可用，无法执行")
        elif not bool(st.session_state.get("live_confirm_trade", False)):
            st.error("请先勾选“我确认启用策略自动交易”")
        elif not live_cfg.okx_inst_id:
            st.error("交易对无法映射为 OKX 标的，请检查交易对格式（例如 BTCUSDT）。")
        elif live_inst_type == "SWAP" and not live_cfg.okx_leverage:
            st.error("合约模式下杠杆倍数无效，请检查输入。")
        else:
            live_result = execute_signal_once(
                client=client,
                strategy_cls=live_strategy_cls,
                config=live_cfg,
            )
            req_payload = {
                "instId": live_cfg.okx_inst_id,
                "instType": live_cfg.okx_inst_type,
                "tdMode": live_cfg.okx_td_mode,
                "posSide": live_cfg.okx_pos_side,
                "lever": live_cfg.okx_leverage,
                "sz": live_cfg.order_size,
                "market_symbol": live_cfg.market_symbol,
                "interval": live_cfg.interval,
                "strategy": live_cfg.strategy_name,
                "strategy_params": live_cfg.strategy_params,
                "demo_trading": bool(client.config.demo_trading),
            }
            resp_payload = {
                "action": live_result.action,
                "signal": live_result.signal,
                "message": live_result.message,
                "latest_bar_time": None if live_result.latest_bar_time is None else live_result.latest_bar_time.isoformat(),
                "latest_close": live_result.latest_close,
                "current_pos_before": live_result.current_pos_before,
                "current_pos_after": live_result.current_pos_after,
                "order_response": live_result.order_response,
                "close_response": live_result.close_response,
            }

            if live_result.status == "failed":
                _append_operation_log(
                    module="live_strategy",
                    action="run_once",
                    status="failed",
                    request_payload=req_payload,
                    response_payload=resp_payload,
                    error_message=live_result.error or live_result.message,
                )
                st.error(f"实时执行失败：{live_result.message}")
            else:
                _append_operation_log(
                    module="live_strategy",
                    action="run_once",
                    status=live_result.status,
                    request_payload=req_payload,
                    response_payload=resp_payload,
                )
                st.success(f"实时执行完成：{live_result.action} | signal={live_result.signal}")
                st.json(resp_payload)

            if st.session_state.get("notify_trade_email", True):
                mail_status = "success" if live_result.status in {"success", "skipped"} else "failed"
                ok, msg = _send_trade_email_notification(
                    email_notifier,
                    action="live_strategy_run_once",
                    status=mail_status,
                    request_payload=req_payload,
                    response_payload=resp_payload,
                    error_message=(None if live_result.status != "failed" else (live_result.error or live_result.message)),
                )
                if ok:
                    st.caption("邮件通知：已发送")
                else:
                    st.warning(f"邮件通知失败：{msg}")

    if start_live:
        if live_strategy_cls is None or live_cfg is None:
            st.error("策略不可用，无法启动")
        elif not bool(st.session_state.get("live_confirm_trade", False)):
            st.error("请先勾选“我确认启用策略自动交易”")
        elif not live_cfg.okx_inst_id:
            st.error("交易对无法映射为 OKX 标的，请检查交易对格式（例如 BTCUSDT）。")
        elif live_inst_type == "SWAP" and not live_cfg.okx_leverage:
            st.error("合约模式下杠杆倍数无效，请检查输入。")
        else:
            start_live_worker(
                key=LIVE_WORKER_KEY,
                okx_config=client.config,
                strategy_cls=live_strategy_cls,
                config=live_cfg,
            )
            st.success("已启动自动执行任务")

    if stop_live:
        stop_live_worker(LIVE_WORKER_KEY)
        st.success("已停止自动执行任务")

    worker_state = get_live_worker_state(LIVE_WORKER_KEY)
    st.caption("自动执行状态")
    ws1, ws2, ws3, ws4 = st.columns(4)
    ws1.metric("运行中", "是" if worker_state.get("running") else "否")
    ws2.metric("循环次数", worker_state.get("loops", 0))
    ws3.metric("执行次数", worker_state.get("executions", 0))
    ws4.metric("最新信号", worker_state.get("last_signal") or "-")
    with st.expander("查看自动执行详情", expanded=False):
        st.json(worker_state)

    st.markdown("#### 最近操作日志")
    log_df = _load_recent_operation_logs(limit=100)
    if log_df.empty:
        st.caption("暂无操作日志")
    else:
        st.caption(f"日志文件：{OP_LOG_CSV}")
        st.dataframe(log_df, use_container_width=True, hide_index=True)
        st.download_button(
            "下载操作日志 CSV",
            data=log_df.to_csv(index=False).encode("utf-8-sig"),
            file_name="operation_logs_recent.csv",
            mime="text/csv",
        )


def _save_backtest_record(
    *,
    symbol: str,
    interval: str,
    strategy_display_name: str,
    strategy_params: dict[str, Any],
    initial_cash: float,
    commission: float,
    position_sizing_mode: str,
    position_percent: float,
    fixed_trade_amount: float,
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
        "position_sizing_mode": str(position_sizing_mode),
        "position_percent": float(position_percent),
        "fixed_trade_amount": float(fixed_trade_amount),
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
        "position_sizing_mode": detail_record["position_sizing_mode"],
        "position_percent": detail_record["position_percent"],
        "fixed_trade_amount": detail_record["fixed_trade_amount"],
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


def _format_duration(seconds: float) -> str:
    seconds = max(0, int(round(float(seconds))))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def _float_format_from_step(step: float, fallback_decimals: int = 6) -> str:
    """根据 step 生成 number_input 的显示格式，避免小数参数看起来“无法修改”."""
    try:
        s = abs(float(step))
    except Exception:  # noqa: BLE001
        s = 0.0
    if s <= 0:
        return f"%.{fallback_decimals}f"

    text = f"{s:.12f}".rstrip("0")
    if "." in text:
        decimals = len(text.split(".")[1])
    else:
        decimals = 0
    decimals = max(1, min(10, decimals))
    return f"%.{decimals}f"


def _flatten_optimization_ranking(ranking_df: pd.DataFrame) -> pd.DataFrame:
    if ranking_df.empty or "参数" not in ranking_df.columns:
        return pd.DataFrame()

    metric_cols = ["评分", "总收益率(%)", "Sharpe", "最大回撤(%)", "总交易次数"]
    rows: list[dict[str, Any]] = []
    for _, row in ranking_df.iterrows():
        params = row.get("参数")
        if not isinstance(params, dict):
            continue

        flat_row: dict[str, Any] = {}
        for mc in metric_cols:
            if mc in ranking_df.columns:
                flat_row[mc] = row.get(mc)

        for name, value in params.items():
            if isinstance(value, bool):
                continue
            try:
                flat_row[f"param::{name}"] = float(value)
            except Exception:  # noqa: BLE001
                continue
        rows.append(flat_row)

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def _render_optimization_heatmap(
    ranking_df: pd.DataFrame,
    *,
    key_prefix: str,
    selected_params: list[str] | None = None,
) -> None:
    flat_df = _flatten_optimization_ranking(ranking_df)
    if flat_df.empty:
        return

    candidate_params: list[str] = []
    if isinstance(selected_params, list) and selected_params:
        candidate_params = [p for p in selected_params if f"param::{p}" in flat_df.columns]
    if not candidate_params:
        candidate_params = [c.replace("param::", "") for c in flat_df.columns if c.startswith("param::")]

    numeric_params: list[str] = []
    for p in candidate_params:
        col = f"param::{p}"
        s = pd.to_numeric(flat_df[col], errors="coerce")
        if s.notna().sum() <= 1:
            continue
        if s.nunique(dropna=True) <= 1:
            continue
        numeric_params.append(p)

    if len(numeric_params) < 2:
        return

    metric_options = [c for c in ["评分", "总收益率(%)", "Sharpe", "最大回撤(%)", "总交易次数"] if c in flat_df.columns]
    if not metric_options:
        return

    with st.expander("参数分布热力图", expanded=False):
        c1, c2, c3 = st.columns(3)
        x_param = c1.selectbox("X轴参数", options=numeric_params, key=f"{key_prefix}_heat_x")
        y_options = [p for p in numeric_params if p != x_param] or numeric_params
        y_param = c2.selectbox("Y轴参数", options=y_options, key=f"{key_prefix}_heat_y")
        value_col = c3.selectbox("热力值", options=metric_options, key=f"{key_prefix}_heat_metric")

        agg_col, _ = st.columns([1, 3])
        agg_name = agg_col.selectbox("聚合方式", options=["mean", "max", "min"], key=f"{key_prefix}_heat_agg")

        df2 = flat_df.copy()
        df2[value_col] = pd.to_numeric(df2[value_col], errors="coerce")
        df2 = df2.dropna(subset=[value_col, f"param::{x_param}", f"param::{y_param}"])
        if df2.empty:
            st.info("当前筛选条件下没有可绘制的数据。")
            return

        grouped = (
            df2.groupby([f"param::{y_param}", f"param::{x_param}"], as_index=False)[value_col]
            .agg(agg_name)
            .sort_values(by=[f"param::{y_param}", f"param::{x_param}"])
        )
        pivot = grouped.pivot(index=f"param::{y_param}", columns=f"param::{x_param}", values=value_col)
        if pivot.empty:
            st.info("当前参数组合不足以生成热力图。")
            return

        fig = go.Figure(
            data=go.Heatmap(
                z=pivot.values,
                x=[str(v) for v in pivot.columns.tolist()],
                y=[str(v) for v in pivot.index.tolist()],
                colorscale="Viridis",
                colorbar=dict(title=value_col),
                hovertemplate=(
                    f"{x_param}: %{{x}}<br>"
                    f"{y_param}: %{{y}}<br>"
                    f"{value_col}: %{{z:.6g}}<extra></extra>"
                ),
            )
        )
        fig.update_layout(
            height=420,
            margin=dict(l=10, r=10, t=30, b=10),
            xaxis_title=x_param,
            yaxis_title=y_param,
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption("提示：热力值为同一参数网格下的聚合结果（mean/max/min）。")


def _coerce_param_value_with_schema(raw_value: Any, cfg: dict[str, Any]) -> Any:
    """按参数 Schema 将值转换并裁剪到合法范围。"""
    if cfg.get("type") == "int":
        value = int(round(float(raw_value)))
        return max(int(cfg["min"]), min(int(cfg["max"]), value))
    value = float(raw_value)
    return max(float(cfg["min"]), min(float(cfg["max"]), value))


def _build_grid_values_from_bounds(
    cfg: dict[str, Any],
    lower: Any,
    upper: Any,
    points_per_param: int,
) -> list[Any]:
    """基于上下限与点数生成单参数网格。"""
    points = max(2, int(points_per_param))
    p_type = cfg.get("type", "float")

    if p_type == "int":
        step = max(1, int(cfg.get("step", 1)))
        lo = _coerce_param_value_with_schema(lower, cfg)
        hi = _coerce_param_value_with_schema(upper, cfg)
        lo, hi = int(lo), int(hi)
        if lo > hi:
            lo, hi = hi, lo
        if lo == hi:
            return [lo]

        values: list[int] = []
        for i in range(points):
            raw = lo + (hi - lo) * (i / (points - 1))
            v = int(round(raw / step)) * step
            v = max(int(cfg["min"]), min(int(cfg["max"]), v))
            if v not in values:
                values.append(v)
        if lo not in values:
            values.insert(0, lo)
        if hi not in values:
            values.append(hi)
        return sorted(set(values))

    step_f = max(1e-8, float(cfg.get("step", 0.1)))
    lo_f = float(_coerce_param_value_with_schema(lower, cfg))
    hi_f = float(_coerce_param_value_with_schema(upper, cfg))
    if lo_f > hi_f:
        lo_f, hi_f = hi_f, lo_f
    if abs(lo_f - hi_f) <= 1e-12:
        return [round(lo_f, 10)]

    values_f: list[float] = []
    for i in range(points):
        raw = lo_f + (hi_f - lo_f) * (i / (points - 1))
        v = round(round(raw / step_f) * step_f, 10)
        v = max(float(cfg["min"]), min(float(cfg["max"]), v))
        if all(abs(v - e) > 1e-12 for e in values_f):
            values_f.append(v)
    if all(abs(lo_f - e) > 1e-12 for e in values_f):
        values_f.insert(0, round(lo_f, 10))
    if all(abs(hi_f - e) > 1e-12 for e in values_f):
        values_f.append(round(hi_f, 10))
    values_f = sorted(values_f)
    return values_f


def _build_grid_values_by_step(
    cfg: dict[str, Any],
    lower: Any,
    upper: Any,
) -> list[Any]:
    """按 step 在上下限区间内全展开参数网格。"""
    p_type = cfg.get("type", "float")
    if p_type == "int":
        step = max(1, int(cfg.get("step", 1)))
        lo = int(_coerce_param_value_with_schema(lower, cfg))
        hi = int(_coerce_param_value_with_schema(upper, cfg))
        if lo > hi:
            lo, hi = hi, lo
        values = list(range(lo, hi + 1, step))
        if not values:
            values = [lo]
        if values[-1] != hi:
            values.append(hi)
        return sorted(set(values))

    step_f = max(1e-8, float(cfg.get("step", 0.1)))
    lo_f = float(_coerce_param_value_with_schema(lower, cfg))
    hi_f = float(_coerce_param_value_with_schema(upper, cfg))
    if lo_f > hi_f:
        lo_f, hi_f = hi_f, lo_f

    values_f: list[float] = []
    cur = lo_f
    # 防止浮点累积误差，使用计数上限保护
    max_iter = 2_000_000
    iters = 0
    while cur <= hi_f + 1e-12 and iters < max_iter:
        values_f.append(round(cur, 10))
        cur += step_f
        iters += 1

    if not values_f:
        values_f = [round(lo_f, 10)]
    if abs(values_f[-1] - hi_f) > 1e-10:
        values_f.append(round(hi_f, 10))
    return sorted(set(values_f))


def _format_bound_value(value: Any, cfg: dict[str, Any]) -> str:
    if cfg.get("type") == "int":
        return str(int(round(float(value))))
    return f"{float(value):g}"


def _parse_param_range_text(text: str, cfg: dict[str, Any]) -> tuple[Any, Any, str | None]:
    """解析“下限,上限”文本。返回(下限,上限,错误信息)。"""
    raw = str(text or "").strip().replace("，", ",")
    if not raw:
        return cfg["min"], cfg["max"], None

    parts = [p for p in re.split(r"[,\s]+", raw) if p]
    if len(parts) != 2:
        return cfg["min"], cfg["max"], "格式应为：下限,上限"

    try:
        lo = _coerce_param_value_with_schema(parts[0], cfg)
        hi = _coerce_param_value_with_schema(parts[1], cfg)
    except Exception:  # noqa: BLE001
        return cfg["min"], cfg["max"], "范围值解析失败，请输入数字"

    if lo > hi:
        lo, hi = hi, lo
    return lo, hi, None


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


def _queue_apply_params(best_params: dict[str, Any]) -> None:
    """将参数应用请求延后到下一次 rerun（在控件渲染前生效）。"""
    if isinstance(best_params, dict) and best_params:
        st.session_state["_pending_apply_params"] = dict(best_params)


def _queue_widget_updates(updates: dict[str, Any]) -> None:
    """延后更新 widget 对应的 session_state，避免“控件已实例化后不可修改”错误。"""
    if not isinstance(updates, dict) or not updates:
        return
    queued = st.session_state.get("_pending_widget_updates")
    if not isinstance(queued, dict):
        queued = {}
    queued.update(updates)
    st.session_state["_pending_widget_updates"] = queued


def _apply_pending_widget_updates() -> None:
    queued = st.session_state.pop("_pending_widget_updates", None)
    if not isinstance(queued, dict):
        return
    for key, value in queued.items():
        st.session_state[key] = value


_ensure_strategy_dir()
_restore_persisted_ui_state()
_apply_pending_widget_updates()

if "selected_strategy_file" not in st.session_state:
    st.session_state["selected_strategy_file"] = NEW_STRATEGY_LABEL
if "product_mode" not in st.session_state:
    st.session_state["product_mode"] = MODE_LOCK or "backtest"
if "strategy_file_name" not in st.session_state:
    st.session_state["strategy_file_name"] = "my_strategy"
if "strategy_code" not in st.session_state:
    st.session_state["strategy_code"] = NEW_STRATEGY_TEMPLATE
if "strategy_params_json" not in st.session_state:
    st.session_state["strategy_params_json"] = "{}"
if "last_optimized_params" not in st.session_state:
    st.session_state["last_optimized_params"] = {}
if "last_optimized_strategy_file" not in st.session_state:
    st.session_state["last_optimized_strategy_file"] = ""
if "_loaded_strategy_file" not in st.session_state:
    st.session_state["_loaded_strategy_file"] = st.session_state.get("selected_strategy_file", "")
if "symbol" not in st.session_state:
    st.session_state["symbol"] = "BTCUSDT"
if "symbol_select" not in st.session_state:
    st.session_state["symbol_select"] = (
        st.session_state["symbol"] if st.session_state["symbol"] in COMMON_MARKET_SYMBOLS else SYMBOL_CUSTOM_OPTION
    )
if "symbol_custom" not in st.session_state:
    st.session_state["symbol_custom"] = st.session_state["symbol"]
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
if "position_sizing_mode" not in st.session_state:
    st.session_state["position_sizing_mode"] = "percent_equity"
if "position_percent" not in st.session_state:
    st.session_state["position_percent"] = 95.0
if "fixed_trade_amount" not in st.session_state:
    st.session_state["fixed_trade_amount"] = 1_000.0
if "leverage" not in st.session_state:
    st.session_state["leverage"] = 1.0
if "opt_method" not in st.session_state:
    st.session_state["opt_method"] = "Optuna(贝叶斯)"
if "opt_objective" not in st.session_state:
    st.session_state["opt_objective"] = "Sharpe"
if "opt_folds" not in st.session_state:
    st.session_state["opt_folds"] = 4
if "opt_run_wf" not in st.session_state:
    st.session_state["opt_run_wf"] = False
if "opt_points" not in st.session_state:
    st.session_state["opt_points"] = 3
if "opt_grid_mode" not in st.session_state:
    st.session_state["opt_grid_mode"] = "按点数采样"
if "opt_max_combinations" not in st.session_state:
    st.session_state["opt_max_combinations"] = 120
if "opt_trials" not in st.session_state:
    st.session_state["opt_trials"] = 120
if "opt_sampler" not in st.session_state:
    st.session_state["opt_sampler"] = "TPE"
if "opt_seed" not in st.session_state:
    st.session_state["opt_seed"] = 42
if "opt_parallel" not in st.session_state:
    st.session_state["opt_parallel"] = False
if "opt_workers" not in st.session_state:
    st.session_state["opt_workers"] = max(1, min(4, os.cpu_count() or 1))
if "opt_selected_params" not in st.session_state:
    st.session_state["opt_selected_params"] = []

with st.sidebar:
    st.header("参数设置")
    pending_flash = st.session_state.pop("_pending_flash", None)
    if isinstance(pending_flash, dict):
        msg = str(pending_flash.get("message", "")).strip()
        level = str(pending_flash.get("level", "info")).strip().lower()
        if msg:
            if level == "success":
                st.success(msg)
            elif level == "warning":
                st.warning(msg)
            elif level == "error":
                st.error(msg)
            else:
                st.info(msg)

    mode_options = ["backtest", "live"]
    mode_label_map = {"backtest": "回测与优化", "live": "策略自动交易"}
    legacy_mode_map = {"回测与优化": "backtest", "策略自动交易": "live"}
    raw_mode = MODE_LOCK or st.session_state.get("product_mode", "backtest")
    if raw_mode in legacy_mode_map:
        raw_mode = legacy_mode_map[raw_mode]
        st.session_state["product_mode"] = raw_mode
    if raw_mode not in mode_options:
        st.session_state["product_mode"] = "backtest"
    if MODE_LOCK:
        product_mode = MODE_LOCK
        st.session_state["product_mode"] = product_mode
        st.caption(f"工作台已锁定：{mode_label_map.get(product_mode, product_mode)}")
    else:
        product_mode = st.radio(
            "工作台",
            mode_options,
            key="product_mode",
            format_func=lambda x: mode_label_map.get(x, str(x)),
        )
    is_backtest_mode = product_mode == "backtest"
    is_live_mode = product_mode == "live"

    run_btn = False
    optimize_btn = False

    interval_options = ["1m", "5m", "15m", "1h", "4h", "1d"]
    if st.session_state.get("interval") not in interval_options:
        st.session_state["interval"] = "1d"

    symbol = _normalize_market_symbol(st.session_state.get("symbol", "BTCUSDT"))
    interval = str(st.session_state.get("interval", "1d"))
    start_date = st.session_state.get("start_date", default_start)
    end_date = st.session_state.get("end_date", default_end)

    if is_backtest_mode:
        symbol_options = COMMON_MARKET_SYMBOLS + [SYMBOL_CUSTOM_OPTION]
        if st.session_state.get("symbol_select") not in symbol_options:
            recovered_symbol = _normalize_market_symbol(st.session_state.get("symbol", "BTCUSDT")) or "BTCUSDT"
            if recovered_symbol in COMMON_MARKET_SYMBOLS:
                st.session_state["symbol_select"] = recovered_symbol
                st.session_state["symbol_custom"] = recovered_symbol
            else:
                st.session_state["symbol_select"] = SYMBOL_CUSTOM_OPTION
                st.session_state["symbol_custom"] = recovered_symbol

        symbol_choice = st.selectbox(
            "交易对",
            symbol_options,
            key="symbol_select",
            format_func=lambda x: "自定义..." if x == SYMBOL_CUSTOM_OPTION else x,
        )
        if symbol_choice == SYMBOL_CUSTOM_OPTION:
            st.text_input("自定义交易对", key="symbol_custom", placeholder="例如 BTCUSDT")
            raw_symbol = st.session_state.get("symbol_custom", "")
        else:
            raw_symbol = symbol_choice
            st.session_state["symbol_custom"] = raw_symbol

        symbol = _normalize_market_symbol(raw_symbol)
        st.session_state["symbol"] = symbol
        if raw_symbol and str(raw_symbol).strip() and symbol != str(raw_symbol).upper().strip():
            st.caption(f"交易对已规范化为：{symbol}")
        interval = st.selectbox("K线周期", interval_options, key="interval")
        start_date = st.date_input("开始日期", key="start_date")
        end_date = st.date_input("结束日期", key="end_date")
    else:
        st.caption("当前模式：策略自动交易（回测输入已折叠）")

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
            _queue_widget_updates(
                {
                    "strategy_file_name": saved_path.stem,
                    "selected_strategy_file": saved_path.stem,
                    "_loaded_strategy_file": saved_path.stem,
                }
            )
            st.session_state["_pending_flash"] = {"level": "success", "message": f"已保存：{saved_path.name}"}
            st.rerun()
        except Exception as exc:  # noqa: BLE001
            st.error(f"保存失败：{exc}")

    if col_load.button("读取", use_container_width=True):
        try:
            safe_name = _sanitize_name(st.session_state["strategy_file_name"])
            code = _load_strategy_code(safe_name)
            _queue_widget_updates(
                {
                    "strategy_file_name": safe_name,
                    "strategy_code": code,
                    "selected_strategy_file": safe_name,
                    "_loaded_strategy_file": safe_name,
                }
            )
            st.session_state["_pending_flash"] = {"level": "success", "message": "读取成功"}
            st.rerun()
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

    pending_apply = st.session_state.pop("_pending_apply_params", None)
    if isinstance(pending_apply, dict) and pending_apply:
        applied_now, skipped_now = _apply_params_to_ui(pending_apply, param_schema)
        st.session_state["_pending_apply_result"] = {
            "applied": applied_now,
            "skipped": skipped_now,
        }

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
            desc_text = str(cfg.get("desc", "")).strip() if isinstance(cfg, dict) else ""
            if key not in st.session_state:
                st.session_state[key] = cfg["default"]
            else:
                try:
                    if cfg["type"] == "int":
                        value = int(round(float(st.session_state[key])))
                        value = max(int(cfg["min"]), min(int(cfg["max"]), value))
                    else:
                        value = float(st.session_state[key])
                        if not math.isfinite(value):
                            raise ValueError("float value is not finite")
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
                    help=(desc_text or None),
                    key=key,
                )
            else:
                float_step = float(cfg["step"])
                float_format = _float_format_from_step(float_step)
                strategy_params[param_name] = st.number_input(
                    param_name,
                    min_value=float(cfg["min"]),
                    max_value=float(cfg["max"]),
                    step=float_step,
                    format=float_format,
                    help=(desc_text or None),
                    key=key,
                )
    else:
        st.caption("未检测到 STRATEGY_META.params，使用 JSON 参数输入")
        st.text_input("策略参数(JSON)", key="strategy_params_json")

    all_param_names = list(param_schema.keys()) if param_schema else []
    raw_opt_selected = st.session_state.get("opt_selected_params", [])
    if not isinstance(raw_opt_selected, list):
        raw_opt_selected = []
    opt_selected_params: list[str] = []
    if param_schema:
        for name in all_param_names:
            pick_key = f"opt_pick_{name}"
            if pick_key not in st.session_state:
                st.session_state[pick_key] = (name in raw_opt_selected) if raw_opt_selected else True
            if bool(st.session_state.get(pick_key, False)):
                opt_selected_params.append(name)
    st.session_state["opt_selected_params"] = opt_selected_params

    pending_result = st.session_state.pop("_pending_apply_result", None)
    if isinstance(pending_result, dict):
        applied_now = pending_result.get("applied", {})
        skipped_now = pending_result.get("skipped", {})
        if isinstance(applied_now, dict) and applied_now:
            st.success(f"已应用参数：{json.dumps(applied_now, ensure_ascii=False)}")
        if isinstance(skipped_now, dict) and skipped_now:
            st.warning(f"以下参数未应用：{json.dumps(skipped_now, ensure_ascii=False)}")

    initial_cash = float(st.session_state.get("initial_cash", 10_000.0))
    commission = float(st.session_state.get("commission", 0.001))
    position_sizing_mode = str(st.session_state.get("position_sizing_mode", "percent_equity"))
    position_percent = float(st.session_state.get("position_percent", 95.0))
    fixed_trade_amount = float(st.session_state.get("fixed_trade_amount", 1_000.0))
    leverage = float(st.session_state.get("leverage", 1.0))
    opt_method = str(st.session_state.get("opt_method", "Optuna(贝叶斯)"))
    opt_objective = str(st.session_state.get("opt_objective", "Sharpe"))
    opt_folds = int(st.session_state.get("opt_folds", 4))
    opt_run_wf = bool(st.session_state.get("opt_run_wf", False))
    opt_points = int(st.session_state.get("opt_points", 3))
    opt_grid_mode = str(st.session_state.get("opt_grid_mode", "按点数采样"))
    opt_max_combinations = int(st.session_state.get("opt_max_combinations", 120))
    opt_trials = int(st.session_state.get("opt_trials", 120))
    opt_sampler = str(st.session_state.get("opt_sampler", "TPE"))
    opt_seed = int(st.session_state.get("opt_seed", 42))
    opt_parallel = bool(st.session_state.get("opt_parallel", False))
    opt_workers = int(st.session_state.get("opt_workers", max(1, min(4, os.cpu_count() or 1))))
    effective_workers = int(opt_workers) if opt_parallel else 1
    opt_bounds_runtime: dict[str, tuple[Any, Any]] = {}

    if is_backtest_mode:
        st.subheader("资金参数")
        initial_cash = st.number_input("初始资金(USDT)", min_value=100.0, step=100.0, key="initial_cash")
        commission = st.number_input("手续费率", min_value=0.0, max_value=0.01, step=0.0001, format="%.4f", key="commission")
        sizing_mode_options = ["percent_equity", "fixed_amount"]
        sizing_mode_label = {
            "percent_equity": "每笔按账户百分比",
            "fixed_amount": "每笔固定入场金额",
        }
        if st.session_state.get("position_sizing_mode") not in sizing_mode_options:
            st.session_state["position_sizing_mode"] = "percent_equity"
        position_sizing_mode = st.selectbox(
            "仓位模式",
            options=sizing_mode_options,
            key="position_sizing_mode",
            format_func=lambda x: sizing_mode_label.get(str(x), str(x)),
        )
        if position_sizing_mode == "percent_equity":
            position_percent = st.number_input("单次仓位(%)", min_value=1.0, max_value=100.0, step=1.0, key="position_percent")
            st.caption("每笔目标仓位 = 当前账户权益 × 单次仓位(%) × 杠杆倍数")
        else:
            fixed_trade_amount = float(initial_cash)
            st.caption("固定金额模式下：每笔固定入场金额 = 初始资金（无需额外设置）")
            st.caption("每笔目标名义仓位 = 初始资金 × 杠杆倍数")
        leverage = st.number_input("杠杆倍数", min_value=1.0, max_value=125.0, step=0.5, key="leverage")

        run_btn = st.button("开始回测", type="primary", use_container_width=True)
        st.divider()
        st.subheader("参数优化")
        opt_method_options = ["网格搜索", "Optuna(贝叶斯)"]
        if st.session_state.get("opt_method") not in opt_method_options:
            st.session_state["opt_method"] = "Optuna(贝叶斯)"
        opt_method = st.selectbox("优化方法", opt_method_options, key="opt_method")

        if param_schema:
            st.markdown("**参与优化的参数（勾选）**")
            pick_cols = st.columns(2)
            opt_selected_params = []
            for idx, name in enumerate(all_param_names):
                pick_key = f"opt_pick_{name}"
                with pick_cols[idx % 2]:
                    st.checkbox(name, key=pick_key)
                is_picked = bool(st.session_state.get(pick_key, False))
                if is_picked:
                    opt_selected_params.append(name)

                    if opt_method == "网格搜索":
                        cfg = param_schema[name]
                        range_key = f"opt_range_{name}"
                        if range_key not in st.session_state:
                            st.session_state[range_key] = f"{_format_bound_value(cfg['min'], cfg)},{_format_bound_value(cfg['max'], cfg)}"

                        with pick_cols[idx % 2]:
                            st.text_input(
                                f"{name} 范围(下限,上限)",
                                key=range_key,
                                help="示例：10,80",
                                placeholder="下限,上限",
                            )
                            lb_c, ub_c, err = _parse_param_range_text(st.session_state.get(range_key, ""), cfg)
                            opt_bounds_runtime[name] = (lb_c, ub_c)
                            if err:
                                st.warning(f"{name} 范围格式错误，已使用默认区间")
                            st.caption(f"范围: [{lb_c}, {ub_c}]")
            st.session_state["opt_selected_params"] = opt_selected_params

            if not opt_selected_params:
                st.warning("请至少勾选 1 个参数用于优化")
            else:
                st.caption(f"当前优化参数：{', '.join(opt_selected_params)}")
        else:
            st.caption("当前策略未提供参数 Schema，无法选择部分参数优化")

        opt_objective_options = ["Sharpe", "总收益率(%)", "年化收益率(%)", "最大回撤(%)"]
        if st.session_state.get("opt_objective") not in opt_objective_options:
            st.session_state["opt_objective"] = "Sharpe"
        opt_objective = st.selectbox("优化目标", opt_objective_options, key="opt_objective")
        if opt_objective == "最大回撤(%)":
            st.caption("说明：最大回撤为越小越好，系统会按最小回撤寻找参数。")
        opt_folds = st.slider("Walk-Forward 窗口数", min_value=2, max_value=8, step=1, key="opt_folds")
        opt_run_wf = st.checkbox("优化后执行 Walk-Forward", key="opt_run_wf")
        if not opt_run_wf:
            st.caption("已关闭 Walk-Forward，可明显缩短优化耗时。")
        if opt_method == "网格搜索":
            grid_mode_options = ["按点数采样", "按step全展开"]
            if st.session_state.get("opt_grid_mode") not in grid_mode_options:
                st.session_state["opt_grid_mode"] = "按点数采样"
            opt_grid_mode = st.selectbox(
                "网格生成方式",
                options=grid_mode_options,
                key="opt_grid_mode",
                help="按点数采样：每个参数按“每参数点数”等距取值；按step全展开：按参数step遍历区间内所有值。",
            )
            opt_points = st.slider("自动网格每参数点数", min_value=2, max_value=5, step=1, key="opt_points")
            if opt_grid_mode == "按step全展开":
                st.caption("当前模式为按step全展开，“每参数点数”将被忽略。")
            opt_max_combinations = st.number_input(
                "最大回测组合数",
                min_value=10,
                max_value=200000,
                step=100,
                key="opt_max_combinations",
                help="用于限制本次网格搜索实际执行的组合数量上限。",
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

        st.checkbox("启用并行优化", key="opt_parallel", help="开启后会使用线程并发执行参数评估。")
        max_workers_ui = max(1, (os.cpu_count() or 1) * 4)
        opt_workers = st.number_input("并发数(workers)", min_value=1, max_value=max_workers_ui, step=1, key="opt_workers")
        effective_workers = int(opt_workers) if bool(st.session_state.get("opt_parallel", False)) else 1
        st.caption(f"当前优化并发数：{effective_workers}")

        optimize_btn = st.button(
            "一键参数优化 + Walk-Forward",
            use_container_width=True,
            disabled=bool(param_schema) and not bool(opt_selected_params),
        )

        last_best = st.session_state.get("last_optimized_params", {})
        if isinstance(last_best, dict) and last_best:
            last_best_file = str(st.session_state.get("last_optimized_strategy_file", ""))
            current_file = str(st.session_state.get("strategy_file_name", ""))
            st.caption("已存在最近一次优化得到的最优参数")
            if last_best_file and current_file and last_best_file != current_file:
                st.caption(f"提示：该参数来自策略文件 {last_best_file}，当前是 {current_file}")
            if st.button("一键应用最近最优参数", use_container_width=True, key="apply_last_best_sidebar"):
                _queue_apply_params(last_best)
                st.rerun()
    else:
        st.caption("在“策略自动交易”模式下，回测与优化入口已隐藏。")

    _persist_ui_state()

if product_mode == "live":
    _render_okx_trading_panel()
else:
    if MODE_LOCK == "backtest":
        st.info("当前为“回测与优化”独立项目。")
    else:
        st.info("当前为“回测与优化”模式。切换侧边栏工作台可进入“策略自动交易”。")

if product_mode == "backtest" and (run_btn or optimize_btn):
    symbol = _normalize_market_symbol(symbol)
    if not symbol:
        symbol = "BTCUSDT"
        st.warning("交易对为空，已自动使用默认值 BTCUSDT")
    effective_fixed_trade_amount = (
        float(initial_cash) if str(position_sizing_mode) == "fixed_amount" else 0.0
    )

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
            warmup_bars = _infer_auto_warmup_bars(strategy_params=strategy_params, param_schema=param_schema)
            warmup_start_dt = _calc_warmup_start_dt(start_dt=start_dt, interval=interval, warmup_bars=warmup_bars)
            df_full = load_binance_data(symbol, interval, warmup_start_dt, end_dt)
            df = df_full
            if isinstance(df_full.index, pd.DatetimeIndex):
                start_ts = pd.Timestamp(start_dt)
                df = df_full[df_full.index >= start_ts]
            if df.empty:
                st.warning("选定区间内没有可用数据，请调整时间范围")
                st.stop()
            st.caption(f"已自动使用历史预热数据：{warmup_bars} 根K线")
        except BinanceDataError as exc:
            st.error(f"数据获取失败：{exc}")
            st.stop()
        except Exception as exc:  # noqa: BLE001
            st.error(f"未知错误：{exc}")
            st.stop()

    if run_btn:
        with st.spinner("回测中，请稍候..."):
            result = run_backtest(
                df=df_full,
                strategy_cls=strategy_cls,
                strategy_params=strategy_params,
                initial_cash=float(initial_cash),
                commission=float(commission),
                position_percent=float(position_percent),
                leverage=float(leverage),
                position_sizing_mode=str(position_sizing_mode),
                fixed_trade_amount=float(effective_fixed_trade_amount),
                evaluation_start=start_dt,
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
                position_sizing_mode=str(position_sizing_mode),
                position_percent=float(position_percent),
                fixed_trade_amount=float(effective_fixed_trade_amount),
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
        wf_df = pd.DataFrame()
        wf_summary: dict[str, Any] = {
            "folds": 0,
            "平均样本外收益率(%)": None,
            "样本外正收益窗口数": 0,
            "样本外总窗口数": 0,
        }
        selected_opt_params_runtime: list[str] = []
        selected_opt_param_set: set[str] = set()
        if param_schema:
            raw_selected = st.session_state.get("opt_selected_params", [])
            if isinstance(raw_selected, list):
                selected_opt_params_runtime = [p for p in raw_selected if p in param_schema]
            if not selected_opt_params_runtime:
                st.error("请先在侧边栏选择至少 1 个参与优化的参数")
                st.stop()
            selected_opt_param_set = set(selected_opt_params_runtime)
            st.caption(f"本次参与优化参数：{', '.join(selected_opt_params_runtime)}（其余参数固定为当前值）")

        if opt_method == "网格搜索":
            if not param_schema:
                st.error("当前策略没有参数 Schema，无法进行网格搜索")
                st.stop()
            selected_schema = {k: v for k, v in param_schema.items() if k in selected_opt_param_set}
            varying_grid: dict[str, list[Any]] = {}
            for name, cfg in selected_schema.items():
                lower, upper = opt_bounds_runtime.get(name, (cfg["min"], cfg["max"]))
                if opt_grid_mode == "按step全展开":
                    varying_grid[name] = _build_grid_values_by_step(
                        cfg=cfg,
                        lower=lower,
                        upper=upper,
                    )
                else:
                    varying_grid[name] = _build_grid_values_from_bounds(
                        cfg=cfg,
                        lower=lower,
                        upper=upper,
                        points_per_param=int(opt_points),
                    )

            param_grid = {}
            for name, cfg in param_schema.items():
                base_value = _coerce_param_value_with_schema(strategy_params.get(name, cfg["default"]), cfg)
                param_grid[name] = [base_value]
            for name, values in varying_grid.items():
                param_grid[name] = values
            if opt_grid_mode == "按step全展开":
                st.caption("使用参数范围自动网格（按step全展开，仅对勾选参数搜索）")
            else:
                st.caption("使用参数范围自动网格（按点数采样，仅对勾选参数搜索）")

            combo_count = _count_combinations(param_grid)
            st.caption(f"组合数：{combo_count}（最大执行：{int(opt_max_combinations)}）")
            effective_total = max(1, min(int(combo_count), int(opt_max_combinations)))
            if bool(opt_run_wf):
                estimated_runs = int(effective_total) * (1 + int(opt_folds)) + int(opt_folds)
            else:
                estimated_runs = int(effective_total)
            estimated_bar_evals = estimated_runs * max(1, len(df))
            st.caption(
                f"预计总回测次数（{'含' if bool(opt_run_wf) else '不含'} Walk-Forward）：约 {estimated_runs} 次，"
                f"数据规模：{len(df)} 根K线/次"
            )
            if estimated_bar_evals >= 20_000_000:
                st.warning(
                    "当前任务较重：网格组合数较多 + 数据量较大，运行数分钟到数十分钟都可能是正常现象。"
                    "可尝试缩小参数范围、减少组合数或使用更大K线周期。"
                )
            if str(interval) == "1m" and len(df) >= 60_000:
                st.warning("1m 长周期数据回测本身计算量很大，优化会明显变慢。")

            grid_progress = st.progress(0.0)
            grid_status = st.empty()
            grid_status.caption(f"网格进度：0/{effective_total}")
            grid_started_at = datetime.now(timezone.utc)

            def _on_grid_progress(done: int, total: int) -> None:
                denom = max(1, int(total))
                progress = min(1.0, float(done) / float(denom))
                grid_progress.progress(progress)
                elapsed_sec = (datetime.now(timezone.utc) - grid_started_at).total_seconds()
                eta_sec = 0.0
                if int(done) > 0:
                    eta_sec = max(0.0, elapsed_sec * (int(total) - int(done)) / int(done))
                grid_status.caption(
                    f"网格进度：{int(done)}/{int(total)}"
                    f"｜已耗时：{_format_duration(elapsed_sec)}"
                    f"｜预计剩余：{_format_duration(eta_sec)}"
                )

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
                        position_sizing_mode=str(position_sizing_mode),
                        fixed_trade_amount=float(effective_fixed_trade_amount),
                        objective=opt_objective,
                        max_combinations=int(opt_max_combinations),
                        n_jobs=int(effective_workers),
                        strategy_code=str(st.session_state.get("strategy_code", "")),
                        strategy_class_name=getattr(strategy_cls, "__name__", None),
                        progress_callback=_on_grid_progress,
                    )
                except Exception as exc:  # noqa: BLE001
                    st.error(f"网格搜索失败：{exc}")
                    st.stop()
            grid_progress.progress(1.0)
            total_elapsed = (datetime.now(timezone.utc) - grid_started_at).total_seconds()
            grid_status.caption(
                f"网格进度：{effective_total}/{effective_total}"
                f"｜总耗时：{_format_duration(total_elapsed)}"
            )

            if gs.ranking.empty or not gs.best_params:
                st.warning("未找到有效参数组合")
                st.stop()

            st.success("网格搜索完成")
            st.write("最优参数：")
            st.json(gs.best_params)
            st.session_state["last_optimized_params"] = dict(gs.best_params)
            st.session_state["last_optimized_strategy_file"] = str(st.session_state.get("strategy_file_name", ""))
            st.session_state["_last_opt_ranking"] = gs.ranking
            st.session_state["_last_opt_method"] = "网格搜索"
            st.session_state["_last_opt_selected_params"] = list(selected_opt_params_runtime)
            st.session_state["_last_opt_strategy_file"] = str(st.session_state.get("strategy_file_name", ""))

            topn = min(30, len(gs.ranking))
            ranking_show = gs.ranking.head(topn).copy()
            ranking_show["参数"] = ranking_show["参数"].apply(lambda x: json.dumps(x, ensure_ascii=False))
            st.dataframe(ranking_show, use_container_width=True, hide_index=True)
            _render_optimization_heatmap(
                gs.ranking,
                key_prefix="grid",
                selected_params=selected_opt_params_runtime,
            )

            if bool(opt_run_wf):
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
                            position_sizing_mode=str(position_sizing_mode),
                            fixed_trade_amount=float(effective_fixed_trade_amount),
                            objective=opt_objective,
                            folds=int(opt_folds),
                            max_combinations=int(opt_max_combinations),
                            n_jobs=int(effective_workers),
                            strategy_code=str(st.session_state.get("strategy_code", "")),
                            strategy_class_name=getattr(strategy_cls, "__name__", None),
                        )
                    except Exception as exc:  # noqa: BLE001
                        st.error(f"Walk-Forward 失败：{exc}")
                        st.stop()
            else:
                st.info("已按设置跳过 Walk-Forward，仅执行参数优化。")
        else:
            if not param_schema:
                st.error("当前策略没有参数 Schema，无法进行 Optuna 优化")
                st.stop()

            optuna_param_schema: dict[str, dict[str, Any]] = {}
            for name, cfg in param_schema.items():
                cfg_copy = dict(cfg)
                if name not in selected_opt_param_set:
                    fixed_value = _coerce_param_value_with_schema(strategy_params.get(name, cfg["default"]), cfg)
                    cfg_copy["min"] = fixed_value
                    cfg_copy["max"] = fixed_value
                optuna_param_schema[name] = cfg_copy

            with st.spinner("正在执行 Optuna 贝叶斯优化..."):
                try:
                    opt_res = optimize_parameters_optuna(
                        df=df,
                        strategy_cls=strategy_cls,
                        param_schema=optuna_param_schema,
                        initial_cash=float(initial_cash),
                        commission=float(commission),
                        position_percent=float(position_percent),
                        leverage=float(leverage),
                        position_sizing_mode=str(position_sizing_mode),
                        fixed_trade_amount=float(effective_fixed_trade_amount),
                        objective=opt_objective,
                        n_trials=int(opt_trials),
                        sampler_name=str(opt_sampler),
                        seed=int(opt_seed),
                        n_jobs=int(effective_workers),
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
            st.session_state["_last_opt_ranking"] = opt_res.ranking
            st.session_state["_last_opt_method"] = "Optuna(贝叶斯)"
            st.session_state["_last_opt_selected_params"] = list(selected_opt_params_runtime)
            st.session_state["_last_opt_strategy_file"] = str(st.session_state.get("strategy_file_name", ""))

            st.caption(f"最优评分：{opt_res.best_score}")

            topn = min(30, len(opt_res.ranking))
            ranking_show = opt_res.ranking.head(topn).copy()
            ranking_show["参数"] = ranking_show["参数"].apply(lambda x: json.dumps(x, ensure_ascii=False))
            st.dataframe(ranking_show, use_container_width=True, hide_index=True)
            _render_optimization_heatmap(
                opt_res.ranking,
                key_prefix="optuna",
                selected_params=selected_opt_params_runtime,
            )

            if bool(opt_run_wf):
                with st.spinner("正在执行 Walk-Forward（Optuna）..."):
                    try:
                        wf_df, wf_summary = run_walk_forward_optuna(
                            df=df,
                            strategy_cls=strategy_cls,
                            param_schema=optuna_param_schema,
                            initial_cash=float(initial_cash),
                            commission=float(commission),
                            position_percent=float(position_percent),
                            leverage=float(leverage),
                            position_sizing_mode=str(position_sizing_mode),
                            fixed_trade_amount=float(effective_fixed_trade_amount),
                            objective=opt_objective,
                            folds=int(opt_folds),
                            n_trials=int(opt_trials),
                            sampler_name=str(opt_sampler),
                            seed=int(opt_seed),
                            n_jobs=int(effective_workers),
                        )
                    except Exception as exc:  # noqa: BLE001
                        st.error(f"Walk-Forward（Optuna）失败：{exc}")
                        st.stop()
            else:
                st.info("已按设置跳过 Walk-Forward，仅执行参数优化。")

        if bool(opt_run_wf):
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
    if product_mode == "backtest":
        st.info("请在左侧选择或编辑策略文件，然后点击“开始回测”或“一键参数优化 + Walk-Forward”。")

if product_mode == "backtest" and not (run_btn or optimize_btn):
    cached_ranking = st.session_state.get("_last_opt_ranking")
    if isinstance(cached_ranking, pd.DataFrame) and not cached_ranking.empty:
        st.divider()
        st.subheader("最近一次参数优化结果")
        cached_method = str(st.session_state.get("_last_opt_method", ""))
        if cached_method:
            st.caption(f"优化方法：{cached_method}")
        cached_file = str(st.session_state.get("_last_opt_strategy_file", "")).strip()
        current_file = str(st.session_state.get("strategy_file_name", "")).strip()
        if cached_file and current_file and cached_file != current_file:
            st.caption(f"提示：结果来自策略文件 {cached_file}，当前为 {current_file}")

        topn_cached = min(30, len(cached_ranking))
        cached_show = cached_ranking.head(topn_cached).copy()
        if "参数" in cached_show.columns:
            cached_show["参数"] = cached_show["参数"].apply(lambda x: json.dumps(x, ensure_ascii=False))
        st.dataframe(cached_show, use_container_width=True, hide_index=True)

        cached_selected = st.session_state.get("_last_opt_selected_params", [])
        if not isinstance(cached_selected, list):
            cached_selected = []
        _render_optimization_heatmap(
            cached_ranking,
            key_prefix="cached",
            selected_params=cached_selected,
        )

if product_mode == "backtest":
    last_best_main = st.session_state.get("last_optimized_params", {})
    if isinstance(last_best_main, dict) and last_best_main:
        st.divider()
        st.subheader("最近一次优化参数")
        st.json(last_best_main)
        if st.button("一键应用该最优参数", key="apply_last_best_main"):
            _queue_apply_params(last_best_main)
            st.rerun()
