from __future__ import annotations

import io
import json
import os
import time
import zipfile
from datetime import datetime
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import requests

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backtest_engine import run_backtest
from optimization_engine import optimize_parameters
from strategy_files.fast_rsi_flip import FastRsiFlipStrategy


def _build_synth_df(n: int = 12_000) -> pd.DataFrame:
    idx = pd.date_range("2022-01-01", periods=n, freq="h", tz="UTC")
    trend = np.linspace(80.0, 160.0, n)
    wave = np.sin(np.linspace(0, 220, n)) * 2.5
    close = trend + wave
    open_ = np.r_[close[0], close[:-1]]
    high = np.maximum(open_, close) * 1.001
    low = np.minimum(open_, close) * 0.999
    vol = np.full(n, 100.0)
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": vol}, index=idx)


def _load_binance_2024(symbol: str = "XRPUSDT", interval: str = "1h") -> pd.DataFrame:
    sess = requests.Session()
    dfs = []
    for month in range(1, 13):
        ym = f"2024-{month:02d}"
        url = f"https://data.binance.vision/data/spot/monthly/klines/{symbol}/{interval}/{symbol}-{interval}-{ym}.zip"
        resp = sess.get(url, timeout=30)
        if resp.status_code != 200:
            continue
        zf = zipfile.ZipFile(io.BytesIO(resp.content))
        with zf.open(zf.namelist()[0]) as f:
            part = pd.read_csv(f, header=None, usecols=[0, 1, 2, 3, 4, 5], dtype=str)
            dfs.append(part)
    if not dfs:
        raise RuntimeError("未能从 data.binance.vision 拉取到基准数据")
    df = pd.concat(dfs, ignore_index=True)
    df.columns = ["open_time", "open", "high", "low", "close", "volume"]
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna()
    df["open_time"] = pd.to_datetime(df["open_time"].astype("int64"), unit="ms", utc=True, errors="coerce")
    df = df.dropna().drop_duplicates("open_time").sort_values("open_time").set_index("open_time")
    return df


def _timeit(fn, repeat: int = 3) -> dict[str, float]:
    times: list[float] = []
    fn()  # warmup
    for _ in range(max(1, repeat)):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return {
        "mean_sec": float(np.mean(times)),
        "median_sec": float(np.median(times)),
        "min_sec": float(np.min(times)),
        "max_sec": float(np.max(times)),
    }


def _bench_single(df: pd.DataFrame, engine: str) -> dict[str, float]:
    os.environ["AUTOTRADE_BACKTEST_ENGINE"] = engine
    params = {
        "rsi_period": 12,
        "up": 53,
        "dn": 36,
        "ema_period": 66,
        "atr_period": 26,
        "stop_atr": 0.5,
        "cooldown": 12,
        "can_short": 1,
    }

    def _run():
        run_backtest(
            df=df,
            strategy_cls=FastRsiFlipStrategy,
            strategy_params=params,
            initial_cash=10_000,
            commission=0.0004,
            position_percent=17,
            leverage=2.0,
            include_details=False,
        )

    return _timeit(_run, repeat=3)


def _bench_optimize(df: pd.DataFrame, engine: str, n_jobs: int) -> dict[str, float]:
    os.environ["AUTOTRADE_BACKTEST_ENGINE"] = engine
    strategy_code = Path("strategy_files/fast_rsi_flip.py").read_text(encoding="utf-8-sig")
    grid = {
        "rsi_period": [8, 12],
        "up": [52, 54],
        "dn": [34, 38],
        "ema_period": [40, 66],
        "cooldown": [4, 8],
        "can_short": [1],
    }  # 32 combos

    def _run():
        optimize_parameters(
            df=df,
            strategy_cls=FastRsiFlipStrategy,
            param_grid=grid,
            initial_cash=10_000,
            commission=0.0004,
            position_percent=17,
            leverage=2.0,
            objective="Sharpe",
            max_combinations=32,
            n_jobs=n_jobs,
            strategy_code=strategy_code,
            strategy_class_name="FastRsiFlipStrategy",
        )

    # 优化任务较重，repeat=1
    return _timeit(_run, repeat=1)


def main() -> None:
    out_dir = Path("reports/vectorbt_migration_20260402")
    out_dir.mkdir(parents=True, exist_ok=True)

    synth_df = _build_synth_df()
    real_df = _load_binance_2024(symbol="XRPUSDT", interval="1h")

    results: dict[str, dict] = {
        "generated_at_utc": datetime.utcnow().isoformat() + "Z",
        "datasets": {
            "synthetic": {"bars": int(len(synth_df))},
            "xrpusdt_2024_1h": {
                "bars": int(len(real_df)),
                "start": str(real_df.index.min()),
                "end": str(real_df.index.max()),
            },
        },
    }

    # 单次回测
    single = {}
    for ds_name, ds in [("synthetic", synth_df), ("xrpusdt_2024_1h", real_df)]:
        bt_stat = _bench_single(ds, engine="backtrader")
        vbt_stat = _bench_single(ds, engine="vectorbt")
        speedup = bt_stat["median_sec"] / max(vbt_stat["median_sec"], 1e-9)
        single[ds_name] = {
            "backtrader": bt_stat,
            "vectorbt": vbt_stat,
            "speedup_x_median": round(float(speedup), 4),
        }
    results["single_backtest"] = single

    # 参数优化（多进程路径）
    opt = {}
    for ds_name, ds in [("synthetic", synth_df), ("xrpusdt_2024_1h", real_df)]:
        bt_opt_1 = _bench_optimize(ds, engine="backtrader", n_jobs=1)
        vbt_opt_1 = _bench_optimize(ds, engine="vectorbt", n_jobs=1)
        bt_opt_4 = _bench_optimize(ds, engine="backtrader", n_jobs=4)
        vbt_opt_4 = _bench_optimize(ds, engine="vectorbt", n_jobs=4)
        opt[ds_name] = {
            "n_jobs_1": {
                "backtrader": bt_opt_1,
                "vectorbt": vbt_opt_1,
                "speedup_x_median": round(bt_opt_1["median_sec"] / max(vbt_opt_1["median_sec"], 1e-9), 4),
            },
            "n_jobs_4": {
                "backtrader": bt_opt_4,
                "vectorbt": vbt_opt_4,
                "speedup_x_median": round(bt_opt_4["median_sec"] / max(vbt_opt_4["median_sec"], 1e-9), 4),
            },
        }
    results["optimize_grid"] = opt

    report_json = out_dir / "report.json"
    report_json.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    md_lines = [
        "# VectorBT Migration Performance Report",
        "",
        f"- Generated at: {results['generated_at_utc']}",
        "",
        "## Single Backtest Speedup (Backtrader / VectorBT)",
    ]
    for ds_name, item in results["single_backtest"].items():
        md_lines.append(
            f"- {ds_name}: {item['speedup_x_median']}x "
            f"(BT median={item['backtrader']['median_sec']:.4f}s, VBT median={item['vectorbt']['median_sec']:.4f}s)"
        )
    md_lines.append("")
    md_lines.append("## Grid Optimization Speedup (Backtrader / VectorBT)")
    for ds_name, item in results["optimize_grid"].items():
        md_lines.append(
            f"- {ds_name} n_jobs=1: {item['n_jobs_1']['speedup_x_median']}x "
            f"(BT={item['n_jobs_1']['backtrader']['median_sec']:.4f}s, VBT={item['n_jobs_1']['vectorbt']['median_sec']:.4f}s)"
        )
        md_lines.append(
            f"- {ds_name} n_jobs=4: {item['n_jobs_4']['speedup_x_median']}x "
            f"(BT={item['n_jobs_4']['backtrader']['median_sec']:.4f}s, VBT={item['n_jobs_4']['vectorbt']['median_sec']:.4f}s)"
        )
    (out_dir / "report.md").write_text("\n".join(md_lines), encoding="utf-8")

    print(json.dumps(results, ensure_ascii=False, indent=2))
    print(f"[done] {report_json.as_posix()}")


if __name__ == "__main__":
    main()
