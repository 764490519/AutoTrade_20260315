from __future__ import annotations

import json
import os
import threading
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation, ROUND_DOWN
from pathlib import Path
from typing import Any
from uuid import uuid4

import pandas as pd

from backtest_engine import infer_latest_signal
from binance_data import INTERVAL_TO_MS, fetch_klines
from okx_trading import OKXClient, OKXConfig

APP_ROOT_DIR = Path(__file__).resolve().parent
_env_report_dir = str(os.getenv("AUTOTRADE_REPORT_DIR", "")).strip()
if _env_report_dir:
    REPORT_DIR = Path(_env_report_dir).expanduser().resolve()
else:
    REPORT_DIR = APP_ROOT_DIR / "reports"
LIVE_STRATEGY_RUN_DIR = REPORT_DIR / "live_strategy_runs"

_AUTO_TRADE_ACTIONS = {
    "OPEN_LONG",
    "OPEN_SHORT",
    "CLOSE_TO_FLAT",
    "SPOT_CLOSE_TO_FLAT",
    "CLOSE_SHORT_THEN_OPEN_LONG",
    "CLOSE_LONG_THEN_OPEN_SHORT",
}


class LiveTradingError(RuntimeError):
    pass


@dataclass
class LiveSignalSnapshot:
    signal: str  # LONG/SHORT/FLAT
    latest_bar_time: datetime
    latest_close: float
    strategy_position: float
    bars: int


@dataclass
class LiveExecutionResult:
    status: str
    action: str
    message: str
    signal: str
    latest_bar_time: datetime | None = None
    latest_close: float | None = None
    current_pos_before: float | None = None
    current_pos_after: float | None = None
    order_response: dict[str, Any] | None = None
    close_response: dict[str, Any] | None = None
    error: str | None = None


@dataclass
class LiveTradingConfig:
    market_symbol: str  # Binance symbol, e.g. BTCUSDT
    interval: str  # e.g. 1m/5m/1h/4h
    lookback_bars: int = 500
    poll_seconds: int = 5
    only_new_bar: bool = True
    include_unclosed_last_bar: bool = True
    allow_short: bool = True

    okx_inst_id: str = "BTC-USDT-SWAP"
    okx_inst_type: str = "SWAP"
    okx_td_mode: str = "cross"
    okx_pos_side: str = "net"
    okx_leverage: str | None = None
    okx_ccy: str | None = None
    order_size: str = "10"  # 统一按 USDT 金额输入

    strategy_params: dict[str, Any] = field(default_factory=dict)
    strategy_name: str = "Strategy"
    position_percent: float = 95.0


def _drop_unclosed_last_bar(df: pd.DataFrame, interval: str) -> pd.DataFrame:
    if df.empty:
        return df
    interval_ms = INTERVAL_TO_MS.get(interval)
    if interval_ms is None:
        return df
    now_utc = datetime.now(timezone.utc)
    last_open = df.index[-1]
    if isinstance(last_open, pd.Timestamp):
        last_open = last_open.to_pydatetime()
    if last_open.tzinfo is None:
        last_open = last_open.replace(tzinfo=timezone.utc)
    if now_utc < last_open + timedelta(milliseconds=int(interval_ms)):
        return df.iloc[:-1]
    return df


def infer_signal_from_strategy(
    *,
    strategy_cls,
    strategy_params: dict[str, Any],
    df: pd.DataFrame,
    position_percent: float = 95.0,
) -> LiveSignalSnapshot:
    if df.empty or len(df) < 30:
        raise LiveTradingError("K线数据不足，无法推断策略信号")

    data_df = df.copy()
    if isinstance(data_df.index, pd.DatetimeIndex) and data_df.index.tz is not None:
        data_df.index = data_df.index.tz_convert("UTC").tz_localize(None)

    try:
        info = infer_latest_signal(
            data_df,
            strategy_cls=strategy_cls,
            strategy_params=strategy_params,
            position_percent=position_percent,
        )
    except Exception as exc:  # noqa: BLE001
        raise LiveTradingError(f"策略信号计算失败: {exc}") from exc

    dt = info.get("latest_bar_time")
    if dt is None:
        raise LiveTradingError("策略未产生有效信号")

    return LiveSignalSnapshot(
        signal=str(info.get("signal", "FLAT")),
        latest_bar_time=dt if isinstance(dt, datetime) else pd.to_datetime(dt).to_pydatetime(),
        latest_close=float(info.get("latest_close", 0.0) or 0.0),
        strategy_position=float(info.get("strategy_position", 0.0) or 0.0),
        bars=int(info.get("bars", 0) or 0),
    )


def fetch_realtime_klines(
    symbol: str,
    interval: str,
    lookback_bars: int = 500,
    *,
    include_unclosed_last_bar: bool = False,
) -> pd.DataFrame:
    bars = max(60, min(1000, int(lookback_bars)))
    df = fetch_klines(symbol=symbol, interval=interval, limit=bars)
    if include_unclosed_last_bar:
        return df
    return _drop_unclosed_last_bar(df, interval=interval)


def _get_okx_position(client: OKXClient, inst_id: str, inst_type: str | None = None) -> float:
    rows = client.get_positions(inst_id=inst_id, inst_type=inst_type)
    total = 0.0
    for row in rows:
        try:
            total += float(row.get("pos", 0.0) or 0.0)
        except Exception:  # noqa: BLE001
            continue
    return total


def _to_decimal(value: Any) -> Decimal | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return Decimal(text)
    except (InvalidOperation, ValueError):
        return None


def _to_decimal_positive(value: Any, field_name: str) -> Decimal:
    dec = _to_decimal(value)
    if dec is None:
        raise LiveTradingError(f"{field_name} 不是有效数字: {value}")
    if dec <= 0:
        raise LiveTradingError(f"{field_name} 必须大于 0: {value}")
    return dec


def _decimal_floor_to_step(value: Decimal, step: Decimal) -> Decimal:
    if step <= 0:
        return value
    return (value / step).to_integral_value(rounding=ROUND_DOWN) * step


def _decimal_to_plain_str(value: Decimal) -> str:
    text = format(value, "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text or "0"


def _apply_lot_constraints(value: Decimal, lot_sz: Decimal | None, min_sz: Decimal | None) -> Decimal:
    out = value
    if lot_sz is not None and lot_sz > 0:
        out = _decimal_floor_to_step(out, lot_sz)
    if min_sz is not None and min_sz > 0 and out < min_sz:
        return Decimal("0")
    return out


def _latest_price_decimal(client: OKXClient, inst_id: str) -> Decimal:
    ticker = client.get_ticker(inst_id)
    for key in ("last", "lastPx", "askPx", "bidPx", "idxPx", "markPx"):
        dec = _to_decimal(ticker.get(key))
        if dec is not None and dec > 0:
            return dec
    raise LiveTradingError(f"无法获取最新价格: instId={inst_id}, ticker={ticker}")


def _resolve_order_request_from_usdt(
    *,
    client: OKXClient,
    config: LiveTradingConfig,
    side: str,
) -> dict[str, Any]:
    order_usdt = _to_decimal_positive(config.order_size, "实盘下单金额(USDT)")
    inst_type = str(config.okx_inst_type or "").upper().strip()
    inst_id = str(config.okx_inst_id or "").strip()
    if not inst_id:
        raise LiveTradingError("okx_inst_id 不能为空")

    inst = client.get_instrument(inst_id=inst_id, inst_type=inst_type)
    lot_sz = _to_decimal(inst.get("lotSz"))
    min_sz = _to_decimal(inst.get("minSz"))
    ref_price = _latest_price_decimal(client, inst_id)

    if inst_type == "SPOT":
        quote_ccy = str(inst.get("quoteCcy") or "").upper().strip()
        base_ccy = str(inst.get("baseCcy") or "").upper().strip()
        if quote_ccy and quote_ccy != "USDT":
            raise LiveTradingError(f"当前现货标的计价币不是 USDT，无法按 USDT 金额下单: {inst_id}")

        if str(side).lower() == "buy":
            # 现货市价买单：直接按 quote_ccy（USDT）下单，金额就是输入的 USDT
            return {
                "sz": _decimal_to_plain_str(order_usdt),
                "tgt_ccy": "quote_ccy",
                "meta": {
                    "order_notional_usdt": float(order_usdt),
                    "resolved_sz": _decimal_to_plain_str(order_usdt),
                    "resolved_sz_mode": "spot_quote_usdt",
                    "ref_price": float(ref_price),
                    "base_ccy": base_ccy,
                    "quote_ccy": quote_ccy,
                },
            }

        base_qty = order_usdt / ref_price
        base_qty = _apply_lot_constraints(base_qty, lot_sz=lot_sz, min_sz=min_sz)
        if base_qty <= 0:
            raise LiveTradingError(
                f"USDT 金额过小，按当前价格与最小下单限制换算后数量为 0: "
                f"amount={order_usdt}, price={ref_price}, lotSz={lot_sz}, minSz={min_sz}"
            )
        return {
            "sz": _decimal_to_plain_str(base_qty),
            "tgt_ccy": "base_ccy",
            "meta": {
                "order_notional_usdt": float(order_usdt),
                "resolved_sz": _decimal_to_plain_str(base_qty),
                "resolved_sz_mode": "spot_base_qty",
                "ref_price": float(ref_price),
                "base_ccy": base_ccy,
                "quote_ccy": quote_ccy,
            },
        }

    if inst_type == "SWAP":
        ct_val = _to_decimal(inst.get("ctVal"))
        if ct_val is None or ct_val <= 0:
            raise LiveTradingError(f"无法读取合约面值 ctVal: instId={inst_id}")
        ct_val_ccy = str(inst.get("ctValCcy") or "").upper().strip()
        quote_ccy = str(inst.get("settleCcy") or inst.get("quoteCcy") or "").upper().strip()
        if quote_ccy and quote_ccy != "USDT":
            raise LiveTradingError(f"当前合约结算币不是 USDT，无法按 USDT 金额下单: {inst_id}")

        if ct_val_ccy in {"USDT", "USD"}:
            usdt_per_contract = ct_val
        else:
            usdt_per_contract = ct_val * ref_price
        if usdt_per_contract <= 0:
            raise LiveTradingError(f"合约每张 USDT 名义价值无效: {usdt_per_contract}")

        contracts = order_usdt / usdt_per_contract
        contracts = _apply_lot_constraints(contracts, lot_sz=lot_sz, min_sz=min_sz)
        if contracts <= 0:
            raise LiveTradingError(
                f"USDT 金额过小，按当前价格与合约面值换算后数量为 0: "
                f"amount={order_usdt}, usdt_per_contract={usdt_per_contract}, lotSz={lot_sz}, minSz={min_sz}"
            )
        return {
            "sz": _decimal_to_plain_str(contracts),
            "tgt_ccy": None,
            "meta": {
                "order_notional_usdt": float(order_usdt),
                "resolved_sz": _decimal_to_plain_str(contracts),
                "resolved_sz_mode": "swap_contracts",
                "ref_price": float(ref_price),
                "ct_val": float(ct_val),
                "ct_val_ccy": ct_val_ccy,
                "quote_ccy": quote_ccy,
            },
        }

    raise LiveTradingError(f"暂不支持的 instType: {inst_type}")


def estimate_min_order_notional_usdt(client: OKXClient, config: LiveTradingConfig) -> dict[str, Any]:
    """
    估算当前标的的最小可下单 USDT 金额（用于前置校验）。
    返回示例：
    {
      "min_notional_usdt": 7.53,
      "min_sz": "0.01",
      "lot_sz": "0.01",
      "ref_price": 75363.4,
      "inst_type": "SWAP",
    }
    """
    inst_type = str(config.okx_inst_type or "").upper().strip()
    inst_id = str(config.okx_inst_id or "").strip()
    if not inst_id:
        raise LiveTradingError("okx_inst_id 不能为空")

    inst = client.get_instrument(inst_id=inst_id, inst_type=inst_type)
    lot_sz = _to_decimal(inst.get("lotSz"))
    min_sz = _to_decimal(inst.get("minSz"))
    ref_price = _latest_price_decimal(client, inst_id)

    min_trade_sz = None
    if min_sz is not None and min_sz > 0:
        min_trade_sz = min_sz
    elif lot_sz is not None and lot_sz > 0:
        min_trade_sz = lot_sz
    else:
        min_trade_sz = Decimal("0")

    if inst_type == "SPOT":
        quote_ccy = str(inst.get("quoteCcy") or "").upper().strip()
        if quote_ccy and quote_ccy != "USDT":
            raise LiveTradingError(f"当前现货标的计价币不是 USDT，无法按 USDT 金额下单: {inst_id}")
        min_notional = min_trade_sz * ref_price if min_trade_sz > 0 else Decimal("0")
        return {
            "min_notional_usdt": float(min_notional),
            "min_sz": _decimal_to_plain_str(min_trade_sz),
            "lot_sz": _decimal_to_plain_str(lot_sz or Decimal("0")),
            "ref_price": float(ref_price),
            "inst_type": inst_type,
        }

    if inst_type == "SWAP":
        ct_val = _to_decimal(inst.get("ctVal"))
        if ct_val is None or ct_val <= 0:
            raise LiveTradingError(f"无法读取合约面值 ctVal: instId={inst_id}")
        ct_val_ccy = str(inst.get("ctValCcy") or "").upper().strip()
        quote_ccy = str(inst.get("settleCcy") or inst.get("quoteCcy") or "").upper().strip()
        if quote_ccy and quote_ccy != "USDT":
            raise LiveTradingError(f"当前合约结算币不是 USDT，无法按 USDT 金额下单: {inst_id}")

        if ct_val_ccy in {"USDT", "USD"}:
            usdt_per_contract = ct_val
        else:
            usdt_per_contract = ct_val * ref_price

        min_notional = min_trade_sz * usdt_per_contract if min_trade_sz > 0 else Decimal("0")
        return {
            "min_notional_usdt": float(min_notional),
            "min_sz": _decimal_to_plain_str(min_trade_sz),
            "lot_sz": _decimal_to_plain_str(lot_sz or Decimal("0")),
            "ref_price": float(ref_price),
            "usdt_per_contract": float(usdt_per_contract),
            "inst_type": inst_type,
        }

    raise LiveTradingError(f"暂不支持的 instType: {inst_type}")


def _safe_json_dumps(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    except Exception:  # noqa: BLE001
        return "{}"


def _safe_json_loads(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    text = str(value or "").strip()
    if not text:
        return {}
    try:
        obj = json.loads(text)
    except Exception:  # noqa: BLE001
        return {}
    return obj if isinstance(obj, dict) else {}


def _to_float_or_none(value: Any) -> float | None:
    try:
        out = float(value)
    except Exception:  # noqa: BLE001
        return None
    if out != out:  # NaN
        return None
    if out in {float("inf"), float("-inf")}:
        return None
    return out


def _to_utc_minute_str(value: Any) -> str | None:
    ts = pd.to_datetime(value, utc=True, errors="coerce")
    if pd.isna(ts):
        return None
    return ts.strftime("%Y-%m-%d %H:%M")


def _build_operation_record(
    *,
    worker_key: str,
    run_id: str,
    result: LiveExecutionResult,
    config: LiveTradingConfig,
    operation: str,
    side: str,
    response: dict[str, Any] | None,
) -> dict[str, Any]:
    resp = dict(response or {})
    return {
        "time_utc": datetime.now(timezone.utc).isoformat(),
        "worker_key": str(worker_key),
        "run_id": str(run_id),
        "strategy_name": str(config.strategy_name),
        "market_symbol": str(config.market_symbol),
        "inst_id": str(config.okx_inst_id),
        "inst_type": str(config.okx_inst_type),
        "signal": str(result.signal),
        "action": str(result.action),
        "operation": str(operation),
        "side": str(side),
        "latest_bar_time": None if result.latest_bar_time is None else result.latest_bar_time.isoformat(),
        "latest_close": result.latest_close,
        "current_pos_before": result.current_pos_before,
        "current_pos_after": result.current_pos_after,
        "order_notional_usdt": resp.get("order_notional_usdt"),
        "resolved_sz": resp.get("resolved_sz"),
        "resolved_sz_mode": resp.get("resolved_sz_mode"),
        "ref_price": resp.get("ref_price"),
        "ord_id": resp.get("ordId"),
        "cl_ord_id": resp.get("clOrdId"),
        "s_code": resp.get("sCode"),
        "s_msg": resp.get("sMsg"),
        "response_json": _safe_json_dumps(resp),
    }


def _extract_operation_records(
    *,
    worker_key: str,
    run_id: str,
    result: LiveExecutionResult,
    config: LiveTradingConfig,
) -> list[dict[str, Any]]:
    if result.status != "success":
        return []
    action = str(result.action or "").strip().upper()
    if action not in _AUTO_TRADE_ACTIONS:
        return []

    records: list[dict[str, Any]] = []
    if action == "OPEN_LONG":
        records.append(
            _build_operation_record(
                worker_key=worker_key,
                run_id=run_id,
                result=result,
                config=config,
                operation="OPEN_LONG",
                side="buy",
                response=result.order_response,
            )
        )
    elif action == "OPEN_SHORT":
        records.append(
            _build_operation_record(
                worker_key=worker_key,
                run_id=run_id,
                result=result,
                config=config,
                operation="OPEN_SHORT",
                side="sell",
                response=result.order_response,
            )
        )
    elif action == "CLOSE_TO_FLAT":
        records.append(
            _build_operation_record(
                worker_key=worker_key,
                run_id=run_id,
                result=result,
                config=config,
                operation="CLOSE_TO_FLAT",
                side="close_position",
                response=result.close_response,
            )
        )
    elif action == "SPOT_CLOSE_TO_FLAT":
        spot_side = "sell" if float(result.current_pos_before or 0.0) > 0 else "buy"
        records.append(
            _build_operation_record(
                worker_key=worker_key,
                run_id=run_id,
                result=result,
                config=config,
                operation="CLOSE_TO_FLAT",
                side=spot_side,
                response=result.order_response,
            )
        )
    elif action == "CLOSE_SHORT_THEN_OPEN_LONG":
        records.append(
            _build_operation_record(
                worker_key=worker_key,
                run_id=run_id,
                result=result,
                config=config,
                operation="CLOSE_SHORT",
                side="close_position",
                response=result.close_response,
            )
        )
        records.append(
            _build_operation_record(
                worker_key=worker_key,
                run_id=run_id,
                result=result,
                config=config,
                operation="OPEN_LONG",
                side="buy",
                response=result.order_response,
            )
        )
    elif action == "CLOSE_LONG_THEN_OPEN_SHORT":
        records.append(
            _build_operation_record(
                worker_key=worker_key,
                run_id=run_id,
                result=result,
                config=config,
                operation="CLOSE_LONG",
                side="close_position",
                response=result.close_response,
            )
        )
        records.append(
            _build_operation_record(
                worker_key=worker_key,
                run_id=run_id,
                result=result,
                config=config,
                operation="OPEN_SHORT",
                side="sell",
                response=result.order_response,
            )
        )
    return records


def execute_signal_once(
    *,
    client: OKXClient,
    strategy_cls,
    config: LiveTradingConfig,
    last_processed_bar: datetime | None = None,
) -> LiveExecutionResult:
    try:
        df = fetch_realtime_klines(
            symbol=config.market_symbol,
            interval=config.interval,
            lookback_bars=config.lookback_bars,
            include_unclosed_last_bar=bool(config.include_unclosed_last_bar),
        )
        snapshot = infer_signal_from_strategy(
            strategy_cls=strategy_cls,
            strategy_params=config.strategy_params,
            df=df,
            position_percent=float(config.position_percent),
        )
        if config.only_new_bar and last_processed_bar is not None and snapshot.latest_bar_time <= last_processed_bar:
            return LiveExecutionResult(
                status="skipped",
                action="SKIP_SAME_BAR",
                message="同一根K线，跳过执行",
                signal=snapshot.signal,
                latest_bar_time=snapshot.latest_bar_time,
                latest_close=snapshot.latest_close,
            )
        signal = snapshot.signal
        inst_type_upper = (config.okx_inst_type or "").upper()
        allow_short = bool(config.allow_short) and inst_type_upper != "SPOT"
        leverage_text = str(config.okx_leverage or "").strip()
        current_pos = _get_okx_position(
            client=client,
            inst_id=config.okx_inst_id,
            inst_type=config.okx_inst_type or None,
        )

        action = "HOLD"
        order_resp = None
        close_resp = None

        def _maybe_set_swap_leverage() -> None:
            if inst_type_upper != "SWAP" or not leverage_text:
                return
            try:
                leverage_val = float(leverage_text)
            except Exception as exc:  # noqa: BLE001
                raise LiveTradingError(f"杠杆倍数格式错误: {leverage_text}") from exc
            if leverage_val <= 0:
                raise LiveTradingError(f"杠杆倍数必须大于 0: {leverage_text}")
            client.set_leverage(
                inst_id=config.okx_inst_id,
                lever=str(leverage_text),
                mgn_mode=config.okx_td_mode if config.okx_td_mode in {"cross", "isolated"} else "cross",
                pos_side=(config.okx_pos_side or None),
                ccy=config.okx_ccy,
            )

        def _place_market_order_with_usdt(side: str, *, reduce_only: bool | None = None) -> dict[str, Any]:
            req = _resolve_order_request_from_usdt(client=client, config=config, side=side)
            resp = client.place_order(
                inst_id=config.okx_inst_id,
                td_mode=config.okx_td_mode,
                side=side,
                ord_type="market",
                sz=str(req["sz"]),
                pos_side=config.okx_pos_side or None,
                ccy=config.okx_ccy,
                reduce_only=reduce_only,
                tgt_ccy=req.get("tgt_ccy"),
            )
            out = dict(resp or {})
            out.update(req.get("meta", {}))
            return out

        if signal == "LONG":
            if current_pos < 0:
                close_resp = client.close_position(
                    inst_id=config.okx_inst_id,
                    mgn_mode=config.okx_td_mode if config.okx_td_mode in {"cross", "isolated"} else "cross",
                    pos_side=config.okx_pos_side or "net",
                    ccy=config.okx_ccy,
                    auto_cxl=True,
                )
                _maybe_set_swap_leverage()
                order_resp = _place_market_order_with_usdt("buy")
                action = "CLOSE_SHORT_THEN_OPEN_LONG"
            elif current_pos == 0:
                _maybe_set_swap_leverage()
                order_resp = _place_market_order_with_usdt("buy")
                action = "OPEN_LONG"
            else:
                action = "HOLD_LONG"

        elif signal == "SHORT":
            if not allow_short:
                action = "HOLD_NO_SHORT"
            elif current_pos > 0:
                close_resp = client.close_position(
                    inst_id=config.okx_inst_id,
                    mgn_mode=config.okx_td_mode if config.okx_td_mode in {"cross", "isolated"} else "cross",
                    pos_side=config.okx_pos_side or "net",
                    ccy=config.okx_ccy,
                    auto_cxl=True,
                )
                _maybe_set_swap_leverage()
                order_resp = _place_market_order_with_usdt("sell")
                action = "CLOSE_LONG_THEN_OPEN_SHORT"
            elif current_pos == 0:
                _maybe_set_swap_leverage()
                order_resp = _place_market_order_with_usdt("sell")
                action = "OPEN_SHORT"
            else:
                action = "HOLD_SHORT"

        else:  # FLAT
            if abs(current_pos) > 1e-12:
                if inst_type_upper == "SPOT":
                    side = "sell" if current_pos > 0 else "buy"
                    order_resp = _place_market_order_with_usdt(side, reduce_only=True)
                    action = "SPOT_CLOSE_TO_FLAT"
                else:
                    close_resp = client.close_position(
                        inst_id=config.okx_inst_id,
                        mgn_mode=config.okx_td_mode if config.okx_td_mode in {"cross", "isolated"} else "cross",
                        pos_side=config.okx_pos_side or "net",
                        ccy=config.okx_ccy,
                        auto_cxl=True,
                    )
                    action = "CLOSE_TO_FLAT"
            else:
                action = "HOLD_FLAT"

        current_pos_after = _get_okx_position(
            client=client,
            inst_id=config.okx_inst_id,
            inst_type=config.okx_inst_type or None,
        )
        return LiveExecutionResult(
            status="success",
            action=action,
            message="ok",
            signal=signal,
            latest_bar_time=snapshot.latest_bar_time,
            latest_close=snapshot.latest_close,
            current_pos_before=current_pos,
            current_pos_after=current_pos_after,
            order_response=order_resp,
            close_response=close_resp,
        )
    except Exception as exc:  # noqa: BLE001
        return LiveExecutionResult(
            status="failed",
            action="ERROR",
            message=str(exc),
            signal="UNKNOWN",
            error=str(exc),
        )


@dataclass
class LiveWorkerState:
    running: bool = False
    run_id: str | None = None
    started_at_utc: str | None = None
    ended_at_utc: str | None = None
    last_heartbeat_utc: str | None = None
    last_error: str | None = None
    last_signal: str | None = None
    last_action: str | None = None
    last_message: str | None = None
    last_bar_time: str | None = None
    current_pos_before: float | None = None
    current_pos_after: float | None = None
    executions: int = 0
    operation_records: int = 0
    last_saved_json_file: str | None = None
    last_saved_csv_file: str | None = None
    last_saved_time_utc: str | None = None
    last_save_error: str | None = None
    run_summary: dict[str, Any] | None = None


class LiveStrategyWorker:
    def __init__(
        self,
        *,
        key: str,
        okx_config: OKXConfig,
        strategy_cls,
        config: LiveTradingConfig,
    ):
        self.key = key
        self.okx_config = okx_config
        self.strategy_cls = strategy_cls
        self.config = config
        self._run_id = f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{uuid4().hex[:8]}"
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True, name=f"live-worker-{key}")
        self._state = LiveWorkerState(running=False)
        self._last_processed_bar: datetime | None = None
        self._operation_records: list[dict[str, Any]] = []
        self._open_trade_lots: list[dict[str, Any]] = []

    def start(self) -> None:
        if self._thread.is_alive():
            return
        with self._lock:
            self._state.running = True
            self._state.run_id = self._run_id
            self._state.started_at_utc = datetime.now(timezone.utc).isoformat()
            self._state.ended_at_utc = None
            self._state.last_saved_json_file = None
            self._state.last_saved_csv_file = None
            self._state.last_saved_time_utc = None
            self._state.last_save_error = None
            self._state.run_summary = None
            self._operation_records = []
            self._open_trade_lots = []
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread.is_alive() and threading.current_thread() is not self._thread:
            self._thread.join(timeout=max(1.0, float(self.config.poll_seconds) + 5.0))
        with self._lock:
            self._state.running = False

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            out = asdict(self._state)
            out["recent_operation_records"] = list(self._operation_records[-200:])
            return out

    def _infer_open_qty(self, rec: dict[str, Any]) -> float:
        qty = _to_float_or_none(rec.get("resolved_sz"))
        if qty is not None and qty > 0:
            return qty
        pos_before = _to_float_or_none(rec.get("current_pos_before"))
        pos_after = _to_float_or_none(rec.get("current_pos_after"))
        if pos_before is not None and pos_after is not None:
            diff = abs(pos_after - pos_before)
            if diff > 0:
                return diff
        if pos_after is not None:
            return abs(pos_after)
        return 0.0

    def _register_open_lot(self, rec: dict[str, Any]) -> None:
        operation = str(rec.get("operation") or "").upper()
        if operation not in {"OPEN_LONG", "OPEN_SHORT"}:
            return

        qty = self._infer_open_qty(rec)
        if qty <= 0:
            return

        side = "LONG" if operation == "OPEN_LONG" else "SHORT"
        entry_price = _to_float_or_none(rec.get("ref_price"))
        if entry_price is None:
            entry_price = _to_float_or_none(rec.get("latest_close"))
        if entry_price is None or entry_price <= 0:
            return

        resp = _safe_json_loads(rec.get("response_json"))
        ct_val = _to_float_or_none(resp.get("ct_val"))
        ct_val_ccy = str(resp.get("ct_val_ccy") or "").upper().strip()

        entry_notional = _to_float_or_none(rec.get("order_notional_usdt"))
        if entry_notional is None or entry_notional <= 0:
            if ct_val is not None and ct_val > 0:
                if ct_val_ccy in {"USDT", "USD"}:
                    base_per_contract = ct_val / entry_price
                else:
                    base_per_contract = ct_val
            else:
                base_per_contract = 1.0
            entry_notional = float(qty * entry_price * base_per_contract)

        self._open_trade_lots.append(
            {
                "side": side,
                "remaining_qty": float(qty),
                "entry_price": entry_price,
                "entry_time_utc": rec.get("time_utc"),
                "entry_bar_time_utc": rec.get("latest_bar_time"),
                "entry_signal": rec.get("signal"),
                "entry_ord_id": rec.get("ord_id"),
                "entry_cl_ord_id": rec.get("cl_ord_id"),
                "entry_s_code": rec.get("s_code"),
                "entry_s_msg": rec.get("s_msg"),
                "entry_notional_usdt": entry_notional,
                "ct_val": ct_val,
                "ct_val_ccy": ct_val_ccy,
            }
        )

    def _determine_close_side(self, rec: dict[str, Any]) -> str | None:
        operation = str(rec.get("operation") or "").upper()
        if operation == "CLOSE_LONG":
            return "LONG"
        if operation == "CLOSE_SHORT":
            return "SHORT"
        if operation == "CLOSE_TO_FLAT":
            pos_before = _to_float_or_none(rec.get("current_pos_before"))
            if pos_before is not None:
                if pos_before > 0:
                    return "LONG"
                if pos_before < 0:
                    return "SHORT"
            side = str(rec.get("side") or "").lower()
            if side == "sell":
                return "LONG"
            if side == "buy":
                return "SHORT"
        return None

    def _infer_close_qty(self, rec: dict[str, Any]) -> float:
        pos_before = _to_float_or_none(rec.get("current_pos_before"))
        if pos_before is not None and abs(pos_before) > 0:
            return abs(pos_before)
        pos_after = _to_float_or_none(rec.get("current_pos_after"))
        if pos_before is not None and pos_after is not None:
            diff = abs(pos_before - pos_after)
            if diff > 0:
                return diff
        qty = _to_float_or_none(rec.get("resolved_sz"))
        if qty is not None and qty > 0:
            return qty
        return 0.0

    def _build_trade_record(
        self,
        *,
        lot: dict[str, Any],
        close_rec: dict[str, Any],
        close_side: str,
        used_qty: float,
        exit_price: float,
    ) -> dict[str, Any]:
        entry_price = _to_float_or_none(lot.get("entry_price"))
        ct_val = _to_float_or_none(lot.get("ct_val"))
        ct_val_ccy = str(lot.get("ct_val_ccy") or "").upper().strip()

        if entry_price is not None and entry_price > 0 and ct_val is not None and ct_val > 0:
            if ct_val_ccy in {"USDT", "USD"}:
                base_per_contract = ct_val / entry_price
            else:
                base_per_contract = ct_val
        else:
            base_per_contract = 1.0

        entry_notional = None
        if entry_price is not None and entry_price > 0:
            entry_notional = float(used_qty * entry_price * base_per_contract)
        exit_notional = float(used_qty * exit_price * base_per_contract)

        pnl_usdt = None
        if entry_price is not None and entry_price > 0:
            if close_side == "LONG":
                pnl_usdt = float((exit_price - entry_price) * used_qty * base_per_contract)
            else:
                pnl_usdt = float((entry_price - exit_price) * used_qty * base_per_contract)

        pnl_pct = None
        if pnl_usdt is not None and entry_notional is not None and entry_notional > 0:
            pnl_pct = float(pnl_usdt / entry_notional * 100.0)

        open_ts = pd.to_datetime(lot.get("entry_time_utc"), utc=True, errors="coerce")
        close_ts = pd.to_datetime(close_rec.get("time_utc"), utc=True, errors="coerce")
        hold_minutes = None
        if not pd.isna(open_ts) and not pd.isna(close_ts):
            hold_minutes = max(0.0, float((close_ts - open_ts).total_seconds() / 60.0))

        return {
            "open_time_utc": _to_utc_minute_str(lot.get("entry_time_utc")),
            "close_time_utc": _to_utc_minute_str(close_rec.get("time_utc")),
            "trade_side": close_side,
            "qty": float(used_qty),
            "entry_price": entry_price,
            "exit_price": float(exit_price),
            "pnl_usdt": pnl_usdt,
            "pnl_pct": pnl_pct,
            "hold_minutes": hold_minutes,
            "entry_notional_usdt": entry_notional,
            "exit_notional_usdt": exit_notional,
            "open_signal": lot.get("entry_signal"),
            "close_signal": close_rec.get("signal"),
            "open_bar_time_utc": _to_utc_minute_str(lot.get("entry_bar_time_utc")),
            "close_bar_time_utc": _to_utc_minute_str(close_rec.get("latest_bar_time")),
            "open_ord_id": lot.get("entry_ord_id"),
            "close_ord_id": close_rec.get("ord_id"),
            "open_cl_ord_id": lot.get("entry_cl_ord_id"),
            "close_cl_ord_id": close_rec.get("cl_ord_id"),
            "open_s_code": lot.get("entry_s_code"),
            "close_s_code": close_rec.get("s_code"),
            "open_s_msg": lot.get("entry_s_msg"),
            "close_s_msg": close_rec.get("s_msg"),
            "worker_key": self.key,
            "run_id": self._run_id,
            "strategy_name": str(self.config.strategy_name),
            "market_symbol": str(self.config.market_symbol),
            "inst_id": str(self.config.okx_inst_id),
            "inst_type": str(self.config.okx_inst_type),
        }

    def _close_to_trade_records(self, rec: dict[str, Any]) -> list[dict[str, Any]]:
        close_side = self._determine_close_side(rec)
        if not close_side:
            return []

        exit_price = _to_float_or_none(rec.get("ref_price"))
        if exit_price is None:
            exit_price = _to_float_or_none(rec.get("latest_close"))
        if exit_price is None or exit_price <= 0:
            return []

        close_qty = self._infer_close_qty(rec)
        if close_qty <= 0:
            close_qty = sum(
                float(x.get("remaining_qty") or 0.0)
                for x in self._open_trade_lots
                if str(x.get("side")) == close_side
            )
        if close_qty <= 0:
            return []

        remaining = float(close_qty)
        trade_rows: list[dict[str, Any]] = []
        for lot in self._open_trade_lots:
            if remaining <= 1e-12:
                break
            if str(lot.get("side")) != close_side:
                continue
            lot_qty = float(lot.get("remaining_qty") or 0.0)
            if lot_qty <= 0:
                continue
            used = min(remaining, lot_qty)
            lot["remaining_qty"] = lot_qty - used
            remaining -= used
            trade_rows.append(
                self._build_trade_record(
                    lot=lot,
                    close_rec=rec,
                    close_side=close_side,
                    used_qty=float(used),
                    exit_price=float(exit_price),
                )
            )

        self._open_trade_lots = [x for x in self._open_trade_lots if float(x.get("remaining_qty") or 0.0) > 1e-12]
        return trade_rows

    def _enrich_operation_records(self, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        trade_rows: list[dict[str, Any]] = []
        for rec in records:
            operation = str(rec.get("operation") or "").upper()
            if operation in {"OPEN_LONG", "OPEN_SHORT"}:
                self._register_open_lot(rec)
                continue
            if operation in {"CLOSE_TO_FLAT", "CLOSE_SHORT", "CLOSE_LONG"}:
                trade_rows.extend(self._close_to_trade_records(rec))
        return trade_rows

    def _build_run_summary(
        self,
        *,
        records: list[dict[str, Any]],
        started_at_utc: str | None,
        ended_at_utc: str,
    ) -> dict[str, Any]:
        pnl_values: list[float] = []
        entry_notionals: list[float] = []
        wins = 0
        losses = 0
        for rec in records:
            pnl = _to_float_or_none(rec.get("pnl_usdt"))
            if pnl is None:
                continue
            pnl_values.append(pnl)
            if pnl > 0:
                wins += 1
            elif pnl < 0:
                losses += 1
            en = _to_float_or_none(rec.get("entry_notional_usdt"))
            if en is not None and en > 0:
                entry_notionals.append(en)

        total_pnl = float(sum(pnl_values))
        closed_trades = len(pnl_values)
        win_rate = (wins / closed_trades * 100.0) if closed_trades > 0 else None
        total_entry_notional = float(sum(entry_notionals))
        total_return_pct = (total_pnl / total_entry_notional * 100.0) if total_entry_notional > 0 else None

        duration_seconds = None
        if started_at_utc:
            st_ts = pd.to_datetime(started_at_utc, utc=True, errors="coerce")
            ed_ts = pd.to_datetime(ended_at_utc, utc=True, errors="coerce")
            if not pd.isna(st_ts) and not pd.isna(ed_ts):
                duration_seconds = max(0.0, float((ed_ts - st_ts).total_seconds()))

        return {
            "started_at_utc": started_at_utc,
            "ended_at_utc": ended_at_utc,
            "duration_seconds": duration_seconds,
            "total_trade_records": int(len(records)),
            "closed_trades": int(closed_trades),
            "win_trades": int(wins),
            "loss_trades": int(losses),
            "win_rate_pct": win_rate,
            "total_realized_pnl_usdt": total_pnl,
            "avg_pnl_usdt": (total_pnl / closed_trades) if closed_trades > 0 else None,
            "best_trade_pnl_usdt": (max(pnl_values) if pnl_values else None),
            "worst_trade_pnl_usdt": (min(pnl_values) if pnl_values else None),
            "total_entry_notional_usdt": total_entry_notional,
            "total_return_pct": total_return_pct,
        }

    def _persist_operation_records(self) -> None:
        stopped_at_utc = datetime.now(timezone.utc).isoformat()
        with self._lock:
            records = list(self._operation_records)
            snapshot = asdict(self._state)
            started_at_utc = self._state.started_at_utc

        run_overview = self._build_run_summary(
            records=records,
            started_at_utc=started_at_utc,
            ended_at_utc=stopped_at_utc,
        )
        snapshot["ended_at_utc"] = stopped_at_utc
        snapshot["run_summary"] = run_overview

        LIVE_STRATEGY_RUN_DIR.mkdir(parents=True, exist_ok=True)
        base = f"live_{self.key}_{self._run_id}"
        json_path = LIVE_STRATEGY_RUN_DIR / f"{base}.json"
        csv_path = LIVE_STRATEGY_RUN_DIR / f"{base}_operations.csv"

        payload = {
            "worker_key": self.key,
            "run_id": self._run_id,
            "saved_at_utc": stopped_at_utc,
            "config": {
                "market_symbol": self.config.market_symbol,
                "interval": self.config.interval,
                "lookback_bars": self.config.lookback_bars,
                "poll_seconds": self.config.poll_seconds,
                "only_new_bar": self.config.only_new_bar,
                "include_unclosed_last_bar": self.config.include_unclosed_last_bar,
                "allow_short": self.config.allow_short,
                "okx_inst_id": self.config.okx_inst_id,
                "okx_inst_type": self.config.okx_inst_type,
                "okx_td_mode": self.config.okx_td_mode,
                "okx_pos_side": self.config.okx_pos_side,
                "okx_leverage": self.config.okx_leverage,
                "order_size_usdt": self.config.order_size,
                "strategy_name": self.config.strategy_name,
                "strategy_params": self.config.strategy_params,
            },
            "state": snapshot,
            "run_overview": run_overview,
            "operation_records": records,
        }

        try:
            json_path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2, default=str),
                encoding="utf-8",
            )
            cols = [
                "open_time_utc",
                "close_time_utc",
                "trade_side",
                "qty",
                "entry_price",
                "exit_price",
                "pnl_usdt",
                "pnl_pct",
                "hold_minutes",
                "entry_notional_usdt",
                "exit_notional_usdt",
                "open_signal",
                "close_signal",
                "open_bar_time_utc",
                "close_bar_time_utc",
                "open_ord_id",
                "close_ord_id",
                "open_cl_ord_id",
                "close_cl_ord_id",
                "open_s_code",
                "close_s_code",
                "open_s_msg",
                "close_s_msg",
                "worker_key",
                "run_id",
                "strategy_name",
                "market_symbol",
                "inst_id",
                "inst_type",
            ]
            pd.DataFrame(records, columns=cols).to_csv(csv_path, index=False, encoding="utf-8-sig")
            with self._lock:
                self._state.last_saved_json_file = str(json_path)
                self._state.last_saved_csv_file = str(csv_path)
                self._state.last_saved_time_utc = stopped_at_utc
                self._state.ended_at_utc = stopped_at_utc
                self._state.run_summary = run_overview
                self._state.last_save_error = None
        except Exception as exc:  # noqa: BLE001
            with self._lock:
                self._state.last_saved_time_utc = stopped_at_utc
                self._state.ended_at_utc = stopped_at_utc
                self._state.run_summary = run_overview
                self._state.last_save_error = str(exc)

    def _run(self) -> None:
        client = OKXClient(self.okx_config)
        try:
            while not self._stop_event.is_set():
                now = datetime.now(timezone.utc).isoformat()
                with self._lock:
                    self._state.last_heartbeat_utc = now

                result = execute_signal_once(
                    client=client,
                    strategy_cls=self.strategy_cls,
                    config=self.config,
                    last_processed_bar=self._last_processed_bar,
                )
                with self._lock:
                    if result.status == "failed":
                        self._state.last_error = result.error or result.message
                        self._state.last_message = result.message
                    elif result.status == "skipped":
                        self._state.last_message = result.message
                        self._state.last_signal = result.signal
                        self._state.last_action = result.action
                        self._state.last_bar_time = None if result.latest_bar_time is None else result.latest_bar_time.isoformat()
                    else:
                        if result.latest_bar_time is not None:
                            self._last_processed_bar = result.latest_bar_time
                        self._state.executions += 1
                        self._state.last_signal = result.signal
                        self._state.last_action = result.action
                        self._state.last_message = result.message
                        self._state.current_pos_before = result.current_pos_before
                        self._state.current_pos_after = result.current_pos_after
                        self._state.last_bar_time = None if result.latest_bar_time is None else result.latest_bar_time.isoformat()

                        op_records = _extract_operation_records(
                            worker_key=self.key,
                            run_id=self._run_id,
                            result=result,
                            config=self.config,
                        )
                        if op_records:
                            op_records = self._enrich_operation_records(op_records)
                            self._operation_records.extend(op_records)
                            self._state.operation_records = int(len(self._operation_records))
                            self._state.last_error = None

                sleep_s = max(1, int(self.config.poll_seconds))
                self._stop_event.wait(timeout=sleep_s)
        finally:
            with self._lock:
                self._state.running = False
            self._persist_operation_records()


_WORKERS: dict[str, LiveStrategyWorker] = {}


def start_live_worker(
    *,
    key: str,
    okx_config: OKXConfig,
    strategy_cls,
    config: LiveTradingConfig,
) -> None:
    old = _WORKERS.get(key)
    if old is not None:
        old.stop()
    worker = LiveStrategyWorker(
        key=key,
        okx_config=okx_config,
        strategy_cls=strategy_cls,
        config=config,
    )
    _WORKERS[key] = worker
    worker.start()


def stop_live_worker(key: str) -> None:
    worker = _WORKERS.get(key)
    if worker is None:
        return
    worker.stop()


def get_live_worker_state(key: str) -> dict[str, Any]:
    worker = _WORKERS.get(key)
    if worker is None:
        return {
            "running": False,
            "run_id": None,
            "started_at_utc": None,
            "ended_at_utc": None,
            "last_heartbeat_utc": None,
            "last_error": None,
            "last_signal": None,
            "last_action": None,
            "last_message": None,
            "last_bar_time": None,
            "current_pos_before": None,
            "current_pos_after": None,
            "executions": 0,
            "operation_records": 0,
            "recent_operation_records": [],
            "last_saved_json_file": None,
            "last_saved_csv_file": None,
            "last_saved_time_utc": None,
            "last_save_error": None,
            "run_summary": None,
        }
    return worker.snapshot()
