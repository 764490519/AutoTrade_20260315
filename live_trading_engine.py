from __future__ import annotations

import threading
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

import backtrader as bt
import pandas as pd

from binance_data import INTERVAL_TO_MS, fetch_klines
from okx_trading import OKXClient, OKXConfig


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
    close_on_flat: bool = True
    allow_short: bool = True

    okx_inst_id: str = "BTC-USDT-SWAP"
    okx_inst_type: str = "SWAP"
    okx_td_mode: str = "cross"
    okx_pos_side: str = "net"
    okx_leverage: str | None = None
    okx_ccy: str | None = None
    order_size: str = "0.01"

    strategy_params: dict[str, Any] = field(default_factory=dict)
    strategy_name: str = "Strategy"


class _SignalProbeAnalyzer(bt.Analyzer):
    def start(self) -> None:
        self.last_dt = None
        self.last_close = None
        self.last_pos = 0.0
        self.count = 0

    def next(self) -> None:
        self.last_dt = self.strategy.datas[0].datetime.datetime(0)
        self.last_close = float(self.strategy.datas[0].close[0])
        self.last_pos = float(getattr(self.strategy.position, "size", 0.0))
        self.count += 1

    def get_analysis(self) -> dict[str, Any]:
        return {
            "dt": self.last_dt,
            "close": self.last_close,
            "pos": self.last_pos,
            "bars": self.count,
        }


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
) -> LiveSignalSnapshot:
    if df.empty or len(df) < 30:
        raise LiveTradingError("K线数据不足，无法推断策略信号")

    data_df = df.copy()
    if isinstance(data_df.index, pd.DatetimeIndex) and data_df.index.tz is not None:
        data_df.index = data_df.index.tz_convert("UTC").tz_localize(None)

    cerebro = bt.Cerebro(stdstats=False)
    cerebro.broker.set_coc(True)  # 以收盘价成交，便于实时信号映射
    data_feed = bt.feeds.PandasData(
        dataname=data_df,
        datetime=None,
        open="open",
        high="high",
        low="low",
        close="close",
        volume="volume",
        openinterest=-1,
    )
    cerebro.adddata(data_feed)
    cerebro.addstrategy(strategy_cls, **(strategy_params or {}))
    cerebro.broker.setcash(10_000)
    cerebro.broker.setcommission(commission=0.001)
    cerebro.addsizer(bt.sizers.PercentSizer, percents=95)
    cerebro.addanalyzer(_SignalProbeAnalyzer, _name="probe")

    result = cerebro.run()[0]
    probe = result.analyzers.probe.get_analysis()
    dt = probe.get("dt")
    close = float(probe.get("close", 0.0) or 0.0)
    pos = float(probe.get("pos", 0.0) or 0.0)
    bars = int(probe.get("bars", 0) or 0)
    if dt is None:
        raise LiveTradingError("策略未产生有效信号（可能参数导致未运行 next）")

    signal = "FLAT"
    if pos > 0:
        signal = "LONG"
    elif pos < 0:
        signal = "SHORT"

    return LiveSignalSnapshot(
        signal=signal,
        latest_bar_time=dt if isinstance(dt, datetime) else pd.to_datetime(dt).to_pydatetime(),
        latest_close=close,
        strategy_position=pos,
        bars=bars,
    )


def fetch_realtime_closed_klines(symbol: str, interval: str, lookback_bars: int = 500) -> pd.DataFrame:
    bars = max(60, min(1000, int(lookback_bars)))
    df = fetch_klines(symbol=symbol, interval=interval, limit=bars)
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


def execute_signal_once(
    *,
    client: OKXClient,
    strategy_cls,
    config: LiveTradingConfig,
    last_processed_bar: datetime | None = None,
) -> LiveExecutionResult:
    try:
        df = fetch_realtime_closed_klines(
            symbol=config.market_symbol,
            interval=config.interval,
            lookback_bars=config.lookback_bars,
        )
        snapshot = infer_signal_from_strategy(
            strategy_cls=strategy_cls,
            strategy_params=config.strategy_params,
            df=df,
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

        # 信号 -> 执行动作
        action = "HOLD"
        order_resp = None
        close_resp = None

        def _maybe_set_swap_leverage() -> None:
            if inst_type_upper != "SWAP" or not leverage_text:
                return
            try:
                leverage_val = float(leverage_text)
            except Exception as exc:  # noqa: BLE001
                raise LiveTradingError(f"杠杆倍数格式错误：{leverage_text}") from exc
            if leverage_val <= 0:
                raise LiveTradingError(f"杠杆倍数必须大于 0：{leverage_text}")
            client.set_leverage(
                inst_id=config.okx_inst_id,
                lever=str(leverage_text),
                mgn_mode=config.okx_td_mode if config.okx_td_mode in {"cross", "isolated"} else "cross",
                pos_side=(config.okx_pos_side or None),
                ccy=config.okx_ccy,
            )

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
                order_resp = client.place_order(
                    inst_id=config.okx_inst_id,
                    td_mode=config.okx_td_mode,
                    side="buy",
                    ord_type="market",
                    sz=str(config.order_size),
                    pos_side=config.okx_pos_side or None,
                    ccy=config.okx_ccy,
                )
                action = "CLOSE_SHORT_THEN_OPEN_LONG"
            elif current_pos == 0:
                _maybe_set_swap_leverage()
                order_resp = client.place_order(
                    inst_id=config.okx_inst_id,
                    td_mode=config.okx_td_mode,
                    side="buy",
                    ord_type="market",
                    sz=str(config.order_size),
                    pos_side=config.okx_pos_side or None,
                    ccy=config.okx_ccy,
                )
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
                order_resp = client.place_order(
                    inst_id=config.okx_inst_id,
                    td_mode=config.okx_td_mode,
                    side="sell",
                    ord_type="market",
                    sz=str(config.order_size),
                    pos_side=config.okx_pos_side or None,
                    ccy=config.okx_ccy,
                )
                action = "CLOSE_LONG_THEN_OPEN_SHORT"
            elif current_pos == 0:
                _maybe_set_swap_leverage()
                order_resp = client.place_order(
                    inst_id=config.okx_inst_id,
                    td_mode=config.okx_td_mode,
                    side="sell",
                    ord_type="market",
                    sz=str(config.order_size),
                    pos_side=config.okx_pos_side or None,
                    ccy=config.okx_ccy,
                )
                action = "OPEN_SHORT"
            else:
                action = "HOLD_SHORT"

        else:  # FLAT
            if config.close_on_flat and abs(current_pos) > 1e-12:
                if inst_type_upper == "SPOT":
                    side = "sell" if current_pos > 0 else "buy"
                    order_resp = client.place_order(
                        inst_id=config.okx_inst_id,
                        td_mode="cash",
                        side=side,
                        ord_type="market",
                        sz=str(config.order_size),
                        ccy=config.okx_ccy,
                        reduce_only=True,
                    )
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
    started_at_utc: str | None = None
    last_heartbeat_utc: str | None = None
    last_error: str | None = None
    last_signal: str | None = None
    last_action: str | None = None
    last_message: str | None = None
    last_bar_time: str | None = None
    current_pos_before: float | None = None
    current_pos_after: float | None = None
    loops: int = 0
    executions: int = 0


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
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True, name=f"live-worker-{key}")
        self._state = LiveWorkerState(running=False)
        self._last_processed_bar: datetime | None = None

    def start(self) -> None:
        if self._thread.is_alive():
            return
        with self._lock:
            self._state.running = True
            self._state.started_at_utc = datetime.now(timezone.utc).isoformat()
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        with self._lock:
            self._state.running = False

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return asdict(self._state)

    def _run(self) -> None:
        client = OKXClient(self.okx_config)
        while not self._stop_event.is_set():
            now = datetime.now(timezone.utc).isoformat()
            with self._lock:
                self._state.last_heartbeat_utc = now
                self._state.loops += 1

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
                    self._state.last_error = None
                    self._state.last_signal = result.signal
                    self._state.last_action = result.action
                    self._state.last_message = result.message
                    self._state.current_pos_before = result.current_pos_before
                    self._state.current_pos_after = result.current_pos_after
                    self._state.last_bar_time = None if result.latest_bar_time is None else result.latest_bar_time.isoformat()

            sleep_s = max(1, int(self.config.poll_seconds))
            self._stop_event.wait(timeout=sleep_s)

        with self._lock:
            self._state.running = False


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
            "started_at_utc": None,
            "last_heartbeat_utc": None,
            "last_error": None,
            "last_signal": None,
            "last_action": None,
            "last_message": None,
            "last_bar_time": None,
            "current_pos_before": None,
            "current_pos_after": None,
            "loops": 0,
            "executions": 0,
        }
    return worker.snapshot()
