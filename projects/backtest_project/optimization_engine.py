from __future__ import annotations

import itertools
import json
import os
import threading
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, ThreadPoolExecutor, as_completed, wait
from dataclasses import dataclass
from typing import Any, Callable

import pandas as pd

from backtest_engine import BacktestResult, run_backtest
from strategy_loader import compile_strategy_runtime_from_code

try:
    import optuna
except Exception:  # noqa: BLE001
    optuna = None


BAD_SCORE = -1e12

_MP_DF: pd.DataFrame | None = None
_MP_STRATEGY_CLS = None


@dataclass
class GridSearchResult:
    best_params: dict[str, Any]
    ranking: pd.DataFrame


@dataclass
class OptunaSearchResult:
    best_params: dict[str, Any]
    best_score: float | None
    ranking: pd.DataFrame


@dataclass
class MultiSymbolOptimizationResult:
    """多币种优化结果（统一排序）。"""

    ranking: pd.DataFrame
    per_symbol: dict[str, dict[str, Any]]
    best_symbol: str | None
    objective: str


ScoreFunc = Callable[[dict[str, Any], str, dict[str, Any]], float]


def _normalize_n_jobs(n_jobs: int | None) -> int:
    cpu = os.cpu_count() or 1
    if n_jobs is None:
        return 1
    try:
        n = int(n_jobs)
    except Exception:  # noqa: BLE001
        return 1
    return max(1, min(n, max(1, cpu * 4)))


def _dedupe_keep_order(values: list[Any]) -> list[Any]:
    out: list[Any] = []
    for value in values:
        if value not in out:
            out.append(value)
    return out


def _estimate_effective_grid_evals(param_grid: dict[str, list[Any]], max_combinations: int) -> int:
    """估算单币种网格优化实际评估次数（受 max_combinations 限制）。"""
    cap = max(1, int(max_combinations))
    total = 1
    if not param_grid:
        return 1
    for values in param_grid.values():
        total *= max(1, len(values))
        if total >= cap:
            return cap
    return max(1, int(total))


def build_auto_param_grid(
    param_schema: dict[str, dict[str, Any]],
    base_params: dict[str, Any],
    points_per_param: int = 3,
) -> dict[str, list[Any]]:
    """基于当前参数自动构造网格。"""
    grid: dict[str, list[Any]] = {}
    points_per_param = max(2, int(points_per_param))

    for name, cfg in param_schema.items():
        p_type = cfg.get("type", "float")
        min_v = cfg.get("min")
        max_v = cfg.get("max")
        step = cfg.get("step", 1)
        base = base_params.get(name, cfg.get("default"))

        if p_type == "int":
            min_i = int(min_v)
            max_i = int(max_v)
            step_i = max(1, int(step))
            base_i = int(round(float(base)))

            if min_i == 0 and max_i == 1:
                values = [0, 1]
            else:
                center = max(min_i, min(max_i, base_i))
                values = [center]
                for i in range(1, points_per_param):
                    down = center - i * step_i
                    up = center + i * step_i
                    if down >= min_i:
                        values.insert(0, down)
                    if up <= max_i:
                        values.append(up)

                if len(values) < points_per_param:
                    all_values = list(range(min_i, max_i + 1, step_i))
                    if len(all_values) <= points_per_param:
                        values = all_values

            values = [int(v) for v in values if min_i <= int(v) <= max_i]
            values = _dedupe_keep_order(values)
            if not values:
                values = [max(min_i, min(max_i, base_i))]
            grid[name] = values

        else:
            min_f = float(min_v)
            max_f = float(max_v)
            step_f = max(1e-8, float(step))
            base_f = float(base)
            center = max(min_f, min(max_f, base_f))

            values = [center]
            for i in range(1, points_per_param):
                down = center - i * step_f
                up = center + i * step_f
                if down >= min_f:
                    values.insert(0, down)
                if up <= max_f:
                    values.append(up)

            values = [round(float(v), 10) for v in values if min_f <= float(v) <= max_f]
            values = _dedupe_keep_order(values)
            if not values:
                values = [round(center, 10)]
            grid[name] = values

    return grid


def _iter_param_combinations(param_grid: dict[str, list[Any]]):
    if not param_grid:
        yield {}
        return

    keys = list(param_grid.keys())
    value_lists = [param_grid[k] for k in keys]
    for values in itertools.product(*value_lists):
        yield dict(zip(keys, values, strict=False))


def _safe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        out = float(value)
        if pd.isna(out):
            return None
        return out
    except Exception:  # noqa: BLE001
        return None


def _return_drawdown_ratio(metrics: dict[str, Any]) -> float | None:
    ann = _safe_float(metrics.get("年化收益率(%)"))
    mdd = _safe_float(metrics.get("最大回撤(%)"))
    if ann is None or mdd is None:
        return None
    mdd_abs = abs(mdd)
    if mdd_abs <= 1e-12:
        return None
    return ann / mdd_abs


def _objective_label(objective: Any, score_func: ScoreFunc | None = None) -> str:
    if callable(score_func):
        return "自定义评分函数(score_func)"
    if callable(objective):
        return f"自定义目标({getattr(objective, '__name__', 'callable')})"
    return str(objective or "Sharpe")


def _calc_score(
    metrics: dict[str, Any],
    objective: Any,
    *,
    score_func: ScoreFunc | None = None,
    symbol: str = "",
    params: dict[str, Any] | None = None,
) -> float:
    if callable(score_func):
        try:
            return float(score_func(metrics, symbol, dict(params or {})))
        except Exception:  # noqa: BLE001
            return BAD_SCORE

    if callable(objective):
        try:
            return float(objective(metrics))
        except Exception:  # noqa: BLE001
            return BAD_SCORE

    objective_text = str(objective or "Sharpe")
    if objective_text in {"收益回撤比", "年化收益回撤比", "return_drawdown_ratio", "calmar_like"}:
        ratio = _return_drawdown_ratio(metrics)
        return BAD_SCORE if ratio is None else float(ratio)

    value = metrics.get(objective_text)
    if value is None:
        # 在较短窗口中 Sharpe 可能为 None，回退到总收益率避免全部参数并列 -inf
        if objective_text == "Sharpe":
            fallback = metrics.get("总收益率(%)")
            if fallback is None:
                return BAD_SCORE
            try:
                return float(fallback)
            except Exception:  # noqa: BLE001
                return BAD_SCORE
        return BAD_SCORE
    try:
        numeric_value = float(value)
        # 最大回撤是“越小越好”，统一转为“越大越好”的评分
        if objective_text == "最大回撤(%)":
            return -abs(numeric_value)
        return numeric_value
    except Exception:  # noqa: BLE001
        return BAD_SCORE


def _build_success_row(
    params: dict[str, Any],
    metrics: dict[str, Any],
    objective: Any,
    *,
    score_func: ScoreFunc | None = None,
    symbol: str = "",
) -> dict[str, Any]:
    return {
        "参数": params,
        "评分": _calc_score(metrics, objective, score_func=score_func, symbol=symbol, params=params),
        "总收益率(%)": metrics.get("总收益率(%)"),
        "年化收益率(%)": metrics.get("年化收益率(%)"),
        "Sharpe": metrics.get("Sharpe"),
        "最大回撤(%)": metrics.get("最大回撤(%)"),
        "胜率(%)": metrics.get("胜率(%)"),
        "收益回撤比": _return_drawdown_ratio(metrics),
        "总交易次数": metrics.get("总交易次数"),
    }


def _build_error_row(params: dict[str, Any], exc: Exception) -> dict[str, Any]:
    return {
        "参数": params,
        "评分": BAD_SCORE,
        "总收益率(%)": None,
        "Sharpe": None,
        "最大回撤(%)": None,
        "总交易次数": None,
        "错误": str(exc),
    }


def _eval_one_local(
    *,
    df: pd.DataFrame,
    strategy_cls,
    params: dict[str, Any],
    initial_cash: float,
    commission: float,
    position_percent: float,
    leverage: float,
    position_sizing_mode: str,
    fixed_trade_amount: float | None,
    objective: Any,
    score_func: ScoreFunc | None = None,
    symbol: str = "",
) -> dict[str, Any]:
    try:
        result: BacktestResult = run_backtest(
            df=df,
            strategy_cls=strategy_cls,
            strategy_params=params,
            initial_cash=initial_cash,
            commission=commission,
            position_percent=position_percent,
            leverage=leverage,
            include_details=False,
            position_sizing_mode=position_sizing_mode,
            fixed_trade_amount=fixed_trade_amount,
        )
        return _build_success_row(
            params,
            result.metrics,
            objective,
            score_func=score_func,
            symbol=symbol,
        )
    except Exception as exc:  # noqa: BLE001
        return _build_error_row(params, exc)


def _compile_strategy_from_code(strategy_code: str, strategy_class_name: str | None = None):
    runtime = compile_strategy_runtime_from_code(strategy_code)
    if strategy_class_name and getattr(runtime.strategy_obj, "__name__", None) != strategy_class_name:
        # 指定了策略名但与实际解析不同，尝试按名称再取一次（兼容 strategy_class）
        code = str(strategy_code or "").lstrip("\ufeff")
        namespace: dict[str, Any] = {}
        exec(code, namespace, namespace)
        maybe = namespace.get(strategy_class_name)
        if maybe is not None:
            return maybe
    return runtime.strategy_obj


def _mp_worker_init(df: pd.DataFrame, strategy_code: str, strategy_class_name: str | None = None) -> None:
    global _MP_DF, _MP_STRATEGY_CLS
    _MP_DF = df
    _MP_STRATEGY_CLS = _compile_strategy_from_code(strategy_code=strategy_code, strategy_class_name=strategy_class_name)


def _mp_eval_one(
    params: dict[str, Any],
    initial_cash: float,
    commission: float,
    position_percent: float,
    leverage: float,
    position_sizing_mode: str,
    fixed_trade_amount: float | None,
    objective: Any,
) -> dict[str, Any]:
    global _MP_DF, _MP_STRATEGY_CLS
    if _MP_DF is None or _MP_STRATEGY_CLS is None:
        return _build_error_row(params, RuntimeError("多进程上下文未初始化"))

    return _eval_one_local(
        df=_MP_DF,
        strategy_cls=_MP_STRATEGY_CLS,
        params=params,
        initial_cash=initial_cash,
        commission=commission,
        position_percent=position_percent,
        leverage=leverage,
        position_sizing_mode=position_sizing_mode,
        fixed_trade_amount=fixed_trade_amount,
        objective=objective,
    )


def optimize_parameters(
    df: pd.DataFrame,
    strategy_cls,
    param_grid: dict[str, list[Any]],
    initial_cash: float,
    commission: float,
    position_percent: float,
    leverage: float = 1.0,
    position_sizing_mode: str = "percent_equity",
    fixed_trade_amount: float | None = None,
    objective: Any = "Sharpe",
    score_func: ScoreFunc | None = None,
    max_combinations: int = 80,
    n_jobs: int = 1,
    progress_callback: Any | None = None,
    strategy_code: str | None = None,
    strategy_class_name: str | None = None,
) -> GridSearchResult:
    max_combinations = max(1, int(max_combinations))
    combos = list(itertools.islice(_iter_param_combinations(param_grid), max_combinations))
    rows: list[dict[str, Any]] = []

    if not combos:
        return GridSearchResult(best_params={}, ranking=pd.DataFrame())

    n_jobs = _normalize_n_jobs(n_jobs)

    has_custom_scoring = callable(score_func) or callable(objective)

    if n_jobs <= 1 or len(combos) <= 1:
        total = len(combos)
        for idx, params in enumerate(combos, start=1):
            rows.append(
                _eval_one_local(
                    df=df,
                    strategy_cls=strategy_cls,
                    params=params,
                    initial_cash=initial_cash,
                    commission=commission,
                    position_percent=position_percent,
                    leverage=leverage,
                    position_sizing_mode=position_sizing_mode,
                    fixed_trade_amount=fixed_trade_amount,
                    objective=objective,
                    score_func=score_func,
                )
            )
            if callable(progress_callback):
                try:
                    progress_callback(idx, total)
                except Exception:  # noqa: BLE001
                    pass
    else:
        total = len(combos)
        done = 0
        use_process_pool = bool(strategy_code) and (not has_custom_scoring)
        if use_process_pool:
            try:
                with ProcessPoolExecutor(
                    max_workers=n_jobs,
                    initializer=_mp_worker_init,
                    initargs=(df, str(strategy_code or ""), strategy_class_name),
                ) as executor:
                    future_map = {
                        executor.submit(
                            _mp_eval_one,
                            params,
                            float(initial_cash),
                            float(commission),
                            float(position_percent),
                            float(leverage),
                            str(position_sizing_mode),
                            None if fixed_trade_amount is None else float(fixed_trade_amount),
                            objective,
                        ): params
                        for params in combos
                    }
                    for future in as_completed(future_map):
                        params = future_map[future]
                        try:
                            rows.append(future.result())
                        except Exception as exc:  # noqa: BLE001
                            rows.append(_build_error_row(params, exc))
                        done += 1
                        if callable(progress_callback):
                            try:
                                progress_callback(done, total)
                            except Exception:  # noqa: BLE001
                                pass
            except Exception:
                # 多进程失败时回退线程，保证功能可用
                use_process_pool = False

        if not use_process_pool:
            with ThreadPoolExecutor(max_workers=n_jobs) as executor:
                future_map = {
                    executor.submit(
                        _eval_one_local,
                        df=df,
                        strategy_cls=strategy_cls,
                        params=params,
                        initial_cash=initial_cash,
                        commission=commission,
                        position_percent=position_percent,
                        leverage=leverage,
                        position_sizing_mode=position_sizing_mode,
                        fixed_trade_amount=fixed_trade_amount,
                        objective=objective,
                        score_func=score_func,
                    ): params
                    for params in combos
                }
                for future in as_completed(future_map):
                    params = future_map[future]
                    try:
                        rows.append(future.result())
                    except Exception as exc:  # noqa: BLE001
                        rows.append(_build_error_row(params, exc))
                    done += 1
                    if callable(progress_callback):
                        try:
                            progress_callback(done, total)
                        except Exception:  # noqa: BLE001
                            pass

    ranking = pd.DataFrame(rows)
    if ranking.empty:
        return GridSearchResult(best_params={}, ranking=ranking)

    ranking = ranking.sort_values(by="评分", ascending=False, na_position="last").reset_index(drop=True)
    best_params = ranking.iloc[0]["参数"] if isinstance(ranking.iloc[0]["参数"], dict) else {}
    return GridSearchResult(best_params=best_params, ranking=ranking)


def run_walk_forward(
    df: pd.DataFrame,
    strategy_cls,
    param_grid: dict[str, list[Any]],
    initial_cash: float,
    commission: float,
    position_percent: float,
    leverage: float = 1.0,
    position_sizing_mode: str = "percent_equity",
    fixed_trade_amount: float | None = None,
    objective: Any = "Sharpe",
    score_func: ScoreFunc | None = None,
    folds: int = 4,
    max_combinations: int = 80,
    n_jobs: int = 1,
    strategy_code: str | None = None,
    strategy_class_name: str | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    folds = max(2, int(folds))
    n = len(df)
    if n < folds * 20:
        raise ValueError("样本太少，无法进行 Walk-Forward（至少需要 folds*20 根K线）")

    test_size = n // (folds + 1)
    if test_size < 10:
        raise ValueError("测试窗口太短，请增大数据范围或减少 folds")

    fold_rows: list[dict[str, Any]] = []

    for i in range(1, folds + 1):
        train_end = test_size * i
        test_start = train_end
        test_end = test_size * (i + 1) if i < folds else n

        train_df = df.iloc[:train_end]
        test_df = df.iloc[test_start:test_end]

        if train_df.empty or test_df.empty:
            continue

        try:
            opt_res = optimize_parameters(
                df=train_df,
                strategy_cls=strategy_cls,
                param_grid=param_grid,
                initial_cash=initial_cash,
                commission=commission,
                position_percent=position_percent,
                leverage=leverage,
                position_sizing_mode=position_sizing_mode,
                fixed_trade_amount=fixed_trade_amount,
                objective=objective,
                score_func=score_func,
                max_combinations=max_combinations,
                n_jobs=n_jobs,
                strategy_code=strategy_code,
                strategy_class_name=strategy_class_name,
            )
        except Exception as exc:  # noqa: BLE001
            fold_rows.append(
                {
                    "Fold": i,
                    "训练区间": f"{train_df.index[0]} ~ {train_df.index[-1]}",
                    "测试区间": f"{test_df.index[0]} ~ {test_df.index[-1]}",
                    "最优参数": None,
                    "样本外收益率(%)": None,
                    "样本外Sharpe": None,
                    "样本外最大回撤(%)": None,
                    "样本外交易次数": None,
                    "错误": f"训练优化失败: {exc}",
                }
            )
            continue

        best_params = opt_res.best_params
        if not best_params:
            fold_rows.append(
                {
                    "Fold": i,
                    "训练区间": f"{train_df.index[0]} ~ {train_df.index[-1]}",
                    "测试区间": f"{test_df.index[0]} ~ {test_df.index[-1]}",
                    "最优参数": None,
                    "样本外收益率(%)": None,
                    "样本外Sharpe": None,
                    "样本外最大回撤(%)": None,
                    "样本外交易次数": None,
                    "错误": "训练阶段未找到有效参数",
                }
            )
            continue

        try:
            oos_res = run_backtest(
                df=test_df,
                strategy_cls=strategy_cls,
                strategy_params=best_params,
                initial_cash=initial_cash,
                commission=commission,
                position_percent=position_percent,
                leverage=leverage,
                include_details=False,
                position_sizing_mode=position_sizing_mode,
                fixed_trade_amount=fixed_trade_amount,
            )
            oos_return = oos_res.metrics.get("总收益率(%)")
            oos_sharpe = oos_res.metrics.get("Sharpe")
            oos_mdd = oos_res.metrics.get("最大回撤(%)")
            oos_trades = oos_res.metrics.get("总交易次数")
            err = None
        except Exception as exc:  # noqa: BLE001
            oos_return = None
            oos_sharpe = None
            oos_mdd = None
            oos_trades = None
            err = f"样本外回测失败: {exc}"

        fold_rows.append(
            {
                "Fold": i,
                "训练区间": f"{train_df.index[0]} ~ {train_df.index[-1]}",
                "测试区间": f"{test_df.index[0]} ~ {test_df.index[-1]}",
                "最优参数": json.dumps(best_params, ensure_ascii=False),
                "样本外收益率(%)": oos_return,
                "样本外Sharpe": oos_sharpe,
                "样本外最大回撤(%)": oos_mdd,
                "样本外交易次数": oos_trades,
                "错误": err,
            }
        )

    wf_df = pd.DataFrame(fold_rows)
    if wf_df.empty:
        return wf_df, {
            "folds": 0,
            "平均样本外收益率(%)": None,
            "样本外正收益窗口数": 0,
            "样本外总窗口数": 0,
        }

    oos_returns = pd.to_numeric(wf_df["样本外收益率(%)"], errors="coerce")
    valid_oos_returns = oos_returns.dropna()
    summary = {
        "folds": int(len(wf_df)),
        "平均样本外收益率(%)": round(float(valid_oos_returns.mean()), 4) if not valid_oos_returns.empty else None,
        "样本外正收益窗口数": int((valid_oos_returns > 0).sum()),
        "样本外总窗口数": int(len(wf_df)),
    }

    return wf_df, summary


def _suggest_params_from_schema(trial, param_schema: dict[str, dict[str, Any]]) -> dict[str, Any]:
    params: dict[str, Any] = {}
    for name, cfg in param_schema.items():
        p_type = cfg.get("type", "float")
        min_v = cfg.get("min")
        max_v = cfg.get("max")
        step = cfg.get("step", 1)

        if min_v is None or max_v is None:
            continue

        if p_type == "int":
            params[name] = trial.suggest_int(
                name,
                int(min_v),
                int(max_v),
                step=max(1, int(step)),
            )
        else:
            params[name] = trial.suggest_float(
                name,
                float(min_v),
                float(max_v),
                step=max(1e-8, float(step)),
            )
    return params


def _build_optuna_sampler(sampler_name: str, seed: int | None = 42):
    if optuna is None:
        raise ImportError("未安装 optuna，请先执行 pip install optuna")

    if sampler_name == "CMA-ES":
        return optuna.samplers.CmaEsSampler(seed=seed)
    if sampler_name == "随机":
        return optuna.samplers.RandomSampler(seed=seed)
    return optuna.samplers.TPESampler(seed=seed)


def _param_schema_has_discrete_space(param_schema: dict[str, dict[str, Any]]) -> bool:
    """CMA-ES 对离散/整数空间稳定性较差，存在时建议自动降级。"""
    for cfg in param_schema.values():
        p_type = cfg.get("type", "float")
        if p_type == "int":
            return True
        step = cfg.get("step")
        if step is not None:
            try:
                if float(step) > 0:
                    return True
            except Exception:  # noqa: BLE001
                return True
    return False


def optimize_parameters_optuna(
    df: pd.DataFrame,
    strategy_cls,
    param_schema: dict[str, dict[str, Any]],
    initial_cash: float,
    commission: float,
    position_percent: float,
    leverage: float = 1.0,
    position_sizing_mode: str = "percent_equity",
    fixed_trade_amount: float | None = None,
    objective: Any = "Sharpe",
    score_func: ScoreFunc | None = None,
    n_trials: int = 80,
    sampler_name: str = "TPE",
    seed: int | None = 42,
    n_jobs: int = 1,
) -> OptunaSearchResult:
    if optuna is None:
        raise ImportError("未安装 optuna，请先执行 pip install optuna")
    if not param_schema:
        raise ValueError("Optuna 优化需要策略参数 Schema")

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    effective_sampler_name = str(sampler_name)
    if effective_sampler_name == "CMA-ES" and _param_schema_has_discrete_space(param_schema):
        # 混合离散空间下自动回退，避免底层 cmaes 偶发 IndexError
        effective_sampler_name = "TPE"

    sampler = _build_optuna_sampler(sampler_name=effective_sampler_name, seed=seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)

    def objective_func(trial):
        params: dict[str, Any] = {}
        try:
            params = _suggest_params_from_schema(trial, param_schema)
            result: BacktestResult = run_backtest(
                df=df,
                strategy_cls=strategy_cls,
                strategy_params=params,
                initial_cash=initial_cash,
                commission=commission,
                position_percent=position_percent,
                leverage=leverage,
                include_details=False,
                position_sizing_mode=position_sizing_mode,
                fixed_trade_amount=fixed_trade_amount,
            )

            metrics = result.metrics
            score = _calc_score(metrics, objective, score_func=score_func, params=params)
            trial.set_user_attr("params", params)
            trial.set_user_attr("总收益率(%)", metrics.get("总收益率(%)"))
            trial.set_user_attr("Sharpe", metrics.get("Sharpe"))
            trial.set_user_attr("最大回撤(%)", metrics.get("最大回撤(%)"))
            trial.set_user_attr("总交易次数", metrics.get("总交易次数"))
            trial.set_user_attr("error", None)
            return score
        except Exception as exc:  # noqa: BLE001
            # 单次 trial 失败时不终止整个优化
            trial.set_user_attr("params", params)
            trial.set_user_attr("总收益率(%)", None)
            trial.set_user_attr("Sharpe", None)
            trial.set_user_attr("最大回撤(%)", None)
            trial.set_user_attr("总交易次数", None)
            trial.set_user_attr("error", str(exc))
            return BAD_SCORE

    n_trials = max(1, int(n_trials))
    n_jobs = _normalize_n_jobs(n_jobs)
    try:
        study.optimize(objective_func, n_trials=n_trials, catch=(Exception,), n_jobs=n_jobs)
    except Exception as exc:  # noqa: BLE001
        # sampler 层报错时兜底到 TPE，保证优化流程不中断
        if effective_sampler_name != "TPE":
            fallback_sampler = _build_optuna_sampler(sampler_name="TPE", seed=seed)
            study = optuna.create_study(direction="maximize", sampler=fallback_sampler)
            study.optimize(objective_func, n_trials=n_trials, catch=(Exception,), n_jobs=n_jobs)
        else:
            raise RuntimeError(f"Optuna 优化执行失败（sampler={effective_sampler_name}）：{exc}") from exc

    rows: list[dict[str, Any]] = []
    for t in study.trials:
        if t.value is None:
            continue
        rows.append(
            {
                "trial": t.number,
                "参数": dict(t.params),
                "评分": float(t.value),
                "总收益率(%)": t.user_attrs.get("总收益率(%)"),
                "Sharpe": t.user_attrs.get("Sharpe"),
                "最大回撤(%)": t.user_attrs.get("最大回撤(%)"),
                "总交易次数": t.user_attrs.get("总交易次数"),
                "错误": t.user_attrs.get("error"),
                "状态": str(t.state.name),
            }
        )

    ranking = pd.DataFrame(rows)
    if not ranking.empty:
        ranking = ranking.sort_values(by="评分", ascending=False).reset_index(drop=True)

    valid = ranking
    if not ranking.empty and "错误" in ranking.columns:
        valid = ranking[ranking["错误"].isna()]
    if valid is None or valid.empty:
        return OptunaSearchResult(best_params={}, best_score=None, ranking=ranking)

    best_row = valid.iloc[0]
    best_params = best_row["参数"] if isinstance(best_row["参数"], dict) else {}
    best_score = None if pd.isna(best_row["评分"]) else float(best_row["评分"])
    return OptunaSearchResult(best_params=best_params, best_score=best_score, ranking=ranking)


def run_walk_forward_optuna(
    df: pd.DataFrame,
    strategy_cls,
    param_schema: dict[str, dict[str, Any]],
    initial_cash: float,
    commission: float,
    position_percent: float,
    leverage: float = 1.0,
    position_sizing_mode: str = "percent_equity",
    fixed_trade_amount: float | None = None,
    objective: Any = "Sharpe",
    score_func: ScoreFunc | None = None,
    folds: int = 4,
    n_trials: int = 80,
    sampler_name: str = "TPE",
    seed: int | None = 42,
    n_jobs: int = 1,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    folds = max(2, int(folds))
    n = len(df)
    if n < folds * 20:
        raise ValueError("样本太少，无法进行 Walk-Forward（至少需要 folds*20 根K线）")

    test_size = n // (folds + 1)
    if test_size < 10:
        raise ValueError("测试窗口太短，请增大数据范围或减少 folds")

    fold_rows: list[dict[str, Any]] = []

    for i in range(1, folds + 1):
        train_end = test_size * i
        test_start = train_end
        test_end = test_size * (i + 1) if i < folds else n

        train_df = df.iloc[:train_end]
        test_df = df.iloc[test_start:test_end]

        if train_df.empty or test_df.empty:
            continue

        try:
            opt_res = optimize_parameters_optuna(
                df=train_df,
                strategy_cls=strategy_cls,
                param_schema=param_schema,
                initial_cash=initial_cash,
                commission=commission,
                position_percent=position_percent,
                leverage=leverage,
                position_sizing_mode=position_sizing_mode,
                fixed_trade_amount=fixed_trade_amount,
                objective=objective,
                score_func=score_func,
                n_trials=n_trials,
                sampler_name=sampler_name,
                seed=None if seed is None else int(seed) + i,
                n_jobs=n_jobs,
            )
        except Exception as exc:  # noqa: BLE001
            fold_rows.append(
                {
                    "Fold": i,
                    "训练区间": f"{train_df.index[0]} ~ {train_df.index[-1]}",
                    "测试区间": f"{test_df.index[0]} ~ {test_df.index[-1]}",
                    "最优参数": None,
                    "训练最优评分": None,
                    "样本外收益率(%)": None,
                    "样本外Sharpe": None,
                    "样本外最大回撤(%)": None,
                    "样本外交易次数": None,
                    "错误": f"训练优化失败: {exc}",
                }
            )
            continue

        if not opt_res.best_params:
            fold_rows.append(
                {
                    "Fold": i,
                    "训练区间": f"{train_df.index[0]} ~ {train_df.index[-1]}",
                    "测试区间": f"{test_df.index[0]} ~ {test_df.index[-1]}",
                    "最优参数": None,
                    "训练最优评分": opt_res.best_score,
                    "样本外收益率(%)": None,
                    "样本外Sharpe": None,
                    "样本外最大回撤(%)": None,
                    "样本外交易次数": None,
                    "错误": "训练阶段未找到有效参数",
                }
            )
            continue

        try:
            oos_res = run_backtest(
                df=test_df,
                strategy_cls=strategy_cls,
                strategy_params=opt_res.best_params,
                initial_cash=initial_cash,
                commission=commission,
                position_percent=position_percent,
                leverage=leverage,
                include_details=False,
                position_sizing_mode=position_sizing_mode,
                fixed_trade_amount=fixed_trade_amount,
            )
            oos_return = oos_res.metrics.get("总收益率(%)")
            oos_sharpe = oos_res.metrics.get("Sharpe")
            oos_mdd = oos_res.metrics.get("最大回撤(%)")
            oos_trades = oos_res.metrics.get("总交易次数")
            err = None
        except Exception as exc:  # noqa: BLE001
            oos_return = None
            oos_sharpe = None
            oos_mdd = None
            oos_trades = None
            err = f"样本外回测失败: {exc}"

        fold_rows.append(
            {
                "Fold": i,
                "训练区间": f"{train_df.index[0]} ~ {train_df.index[-1]}",
                "测试区间": f"{test_df.index[0]} ~ {test_df.index[-1]}",
                "最优参数": json.dumps(opt_res.best_params, ensure_ascii=False),
                "训练最优评分": opt_res.best_score,
                "样本外收益率(%)": oos_return,
                "样本外Sharpe": oos_sharpe,
                "样本外最大回撤(%)": oos_mdd,
                "样本外交易次数": oos_trades,
                "错误": err,
            }
        )

    wf_df = pd.DataFrame(fold_rows)
    if wf_df.empty:
        return wf_df, {
            "folds": 0,
            "平均样本外收益率(%)": None,
            "样本外正收益窗口数": 0,
            "样本外总窗口数": 0,
        }

    oos_returns = pd.to_numeric(wf_df["样本外收益率(%)"], errors="coerce")
    valid_oos_returns = oos_returns.dropna()
    summary = {
        "folds": int(len(wf_df)),
        "平均样本外收益率(%)": round(float(valid_oos_returns.mean()), 4) if not valid_oos_returns.empty else None,
        "样本外正收益窗口数": int((valid_oos_returns > 0).sum()),
        "样本外总窗口数": int(len(wf_df)),
    }

    return wf_df, summary


def _symbol_result_row(
    *,
    symbol: str,
    best_params: dict[str, Any],
    metrics: dict[str, Any] | None,
    objective: Any,
    score_func: ScoreFunc | None,
    error: str | None,
) -> dict[str, Any]:
    safe_metrics = dict(metrics or {})
    score = BAD_SCORE if error else _calc_score(
        safe_metrics,
        objective,
        score_func=score_func,
        symbol=symbol,
        params=best_params,
    )
    return {
        "币种": symbol,
        "评分": float(score),
        "优化目标": _objective_label(objective, score_func),
        "最优参数": dict(best_params or {}),
        "总收益率(%)": safe_metrics.get("总收益率(%)"),
        "年化收益率(%)": safe_metrics.get("年化收益率(%)"),
        "Sharpe": safe_metrics.get("Sharpe"),
        "最大回撤(%)": safe_metrics.get("最大回撤(%)"),
        "胜率(%)": safe_metrics.get("胜率(%)"),
        "收益回撤比": _return_drawdown_ratio(safe_metrics),
        "总交易次数": safe_metrics.get("总交易次数"),
        "错误": error,
    }


def _sort_multi_symbol_ranking(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    out["__has_error"] = out["错误"].notna()
    out["__score_num"] = pd.to_numeric(out["评分"], errors="coerce").fillna(BAD_SCORE)
    out = out.sort_values(by=["__has_error", "__score_num"], ascending=[True, False], na_position="last")
    return out.drop(columns=["__has_error", "__score_num"]).reset_index(drop=True)


def optimize_parameters_multi_symbol(
    symbol_data_map: dict[str, pd.DataFrame],
    strategy_cls,
    param_grid: dict[str, list[Any]],
    initial_cash: float,
    commission: float,
    position_percent: float,
    leverage: float = 1.0,
    position_sizing_mode: str = "percent_equity",
    fixed_trade_amount: float | None = None,
    objective: Any = "Sharpe",
    score_func: ScoreFunc | None = None,
    max_combinations: int = 80,
    n_jobs: int = 1,
    symbol_n_jobs: int = 1,
    progress_callback: Any | None = None,
    strategy_code: str | None = None,
    strategy_class_name: str | None = None,
) -> MultiSymbolOptimizationResult:
    """
    多币种批量网格优化：
    - 每个币种独立优化，得到该币种最优参数
    - 对最优结果进行统一指标汇总并按目标评分排序
    """
    if not symbol_data_map:
        return MultiSymbolOptimizationResult(
            ranking=pd.DataFrame(),
            per_symbol={},
            best_symbol=None,
            objective=_objective_label(objective, score_func),
        )

    symbol_n_jobs = _normalize_n_jobs(symbol_n_jobs)
    symbol_items = [(str(sym), df) for sym, df in symbol_data_map.items()]
    per_symbol_total_evals = _estimate_effective_grid_evals(param_grid=param_grid, max_combinations=max_combinations)
    total_evals = max(1, int(per_symbol_total_evals) * max(1, len(symbol_items)))
    _progress_lock = threading.Lock()
    _symbol_done_map: dict[str, int] = {str(sym): 0 for sym, _ in symbol_items}

    def _emit_multi_progress(symbol: str, done: int) -> None:
        try:
            done_i = int(done)
        except Exception:  # noqa: BLE001
            done_i = 0
        done_i = max(0, min(done_i, int(per_symbol_total_evals)))
        with _progress_lock:
            prev = int(_symbol_done_map.get(symbol, 0))
            if done_i < prev:
                done_i = prev
            _symbol_done_map[symbol] = done_i
    
    def _current_global_done() -> int:
        with _progress_lock:
            return int(sum(_symbol_done_map.values()))

    def _flush_progress_to_callback() -> None:
        if not callable(progress_callback):
            return
        try:
            progress_callback(_current_global_done(), total_evals)
        except Exception:  # noqa: BLE001
            pass

    if callable(progress_callback):
        try:
            progress_callback(0, total_evals)
        except Exception:  # noqa: BLE001
            pass

    def _run_one_symbol(symbol: str, df: pd.DataFrame) -> tuple[str, dict[str, Any], dict[str, Any]]:
        detail: dict[str, Any] = {
            "symbol": symbol,
            "best_params": {},
            "metrics": None,
            "score": BAD_SCORE,
            "ranking": pd.DataFrame(),
            "error": None,
        }
        def _on_one_symbol_progress(done: int, total: int) -> None:
            _emit_multi_progress(symbol, done)
        try:
            gs = optimize_parameters(
                df=df,
                strategy_cls=strategy_cls,
                param_grid=param_grid,
                initial_cash=initial_cash,
                commission=commission,
                position_percent=position_percent,
                leverage=leverage,
                position_sizing_mode=position_sizing_mode,
                fixed_trade_amount=fixed_trade_amount,
                objective=objective,
                score_func=score_func,
                max_combinations=max_combinations,
                n_jobs=n_jobs,
                progress_callback=_on_one_symbol_progress,
                strategy_code=strategy_code,
                strategy_class_name=strategy_class_name,
            )
            detail["ranking"] = gs.ranking
            best_params = gs.best_params if isinstance(gs.best_params, dict) else {}
            detail["best_params"] = dict(best_params)

            if not best_params:
                detail["error"] = "未找到有效参数"
                row = _symbol_result_row(
                    symbol=symbol,
                    best_params={},
                    metrics=None,
                    objective=objective,
                    score_func=score_func,
                    error=detail["error"],
                )
                return symbol, row, detail

            bt_res = run_backtest(
                df=df,
                strategy_cls=strategy_cls,
                strategy_params=best_params,
                initial_cash=initial_cash,
                commission=commission,
                position_percent=position_percent,
                leverage=leverage,
                include_details=False,
                position_sizing_mode=position_sizing_mode,
                fixed_trade_amount=fixed_trade_amount,
            )
            metrics = dict(bt_res.metrics or {})
            detail["metrics"] = metrics
            detail["score"] = _calc_score(metrics, objective, score_func=score_func, symbol=symbol, params=best_params)
            row = _symbol_result_row(
                symbol=symbol,
                best_params=best_params,
                metrics=metrics,
                objective=objective,
                score_func=score_func,
                error=None,
            )
            return symbol, row, detail
        except Exception as exc:  # noqa: BLE001
            detail["error"] = str(exc)
            row = _symbol_result_row(
                symbol=symbol,
                best_params=detail.get("best_params") or {},
                metrics=None,
                objective=objective,
                score_func=score_func,
                error=str(exc),
            )
            return symbol, row, detail
        finally:
            _emit_multi_progress(symbol, int(per_symbol_total_evals))

    rows: list[dict[str, Any]] = []
    per_symbol: dict[str, dict[str, Any]] = {}

    if symbol_n_jobs <= 1 or len(symbol_items) <= 1:
        for symbol, df in symbol_items:
            sym, row, detail = _run_one_symbol(symbol, df)
            rows.append(row)
            per_symbol[sym] = detail
            _flush_progress_to_callback()
    else:
        with ThreadPoolExecutor(max_workers=symbol_n_jobs) as executor:
            future_map = {executor.submit(_run_one_symbol, symbol, df): symbol for symbol, df in symbol_items}
            pending_futures = set(future_map.keys())
            while pending_futures:
                done_set, pending_futures = wait(pending_futures, timeout=0.5, return_when=FIRST_COMPLETED)
                _flush_progress_to_callback()
                for future in done_set:
                    sym, row, detail = future.result()
                    rows.append(row)
                    per_symbol[sym] = detail

    _flush_progress_to_callback()

    ranking = _sort_multi_symbol_ranking(pd.DataFrame(rows))
    best_symbol: str | None = None
    if not ranking.empty:
        valid = ranking[ranking["错误"].isna()]
        if not valid.empty:
            best_symbol = str(valid.iloc[0]["币种"])

    return MultiSymbolOptimizationResult(
        ranking=ranking,
        per_symbol=per_symbol,
        best_symbol=best_symbol,
        objective=_objective_label(objective, score_func),
    )


def optimize_parameters_optuna_multi_symbol(
    symbol_data_map: dict[str, pd.DataFrame],
    strategy_cls,
    param_schema: dict[str, dict[str, Any]],
    initial_cash: float,
    commission: float,
    position_percent: float,
    leverage: float = 1.0,
    position_sizing_mode: str = "percent_equity",
    fixed_trade_amount: float | None = None,
    objective: Any = "Sharpe",
    score_func: ScoreFunc | None = None,
    n_trials: int = 80,
    sampler_name: str = "TPE",
    seed: int | None = 42,
    n_jobs: int = 1,
    symbol_n_jobs: int = 1,
) -> MultiSymbolOptimizationResult:
    """多币种批量 Optuna 优化并统一排序。"""
    if not symbol_data_map:
        return MultiSymbolOptimizationResult(
            ranking=pd.DataFrame(),
            per_symbol={},
            best_symbol=None,
            objective=_objective_label(objective, score_func),
        )

    symbol_n_jobs = _normalize_n_jobs(symbol_n_jobs)
    symbol_items = [(str(sym), df) for sym, df in symbol_data_map.items()]

    def _run_one_symbol(symbol: str, df: pd.DataFrame, symbol_idx: int) -> tuple[str, dict[str, Any], dict[str, Any]]:
        detail: dict[str, Any] = {
            "symbol": symbol,
            "best_params": {},
            "metrics": None,
            "score": BAD_SCORE,
            "ranking": pd.DataFrame(),
            "error": None,
        }
        symbol_seed = None if seed is None else int(seed) + int(symbol_idx)
        try:
            opt_res = optimize_parameters_optuna(
                df=df,
                strategy_cls=strategy_cls,
                param_schema=param_schema,
                initial_cash=initial_cash,
                commission=commission,
                position_percent=position_percent,
                leverage=leverage,
                position_sizing_mode=position_sizing_mode,
                fixed_trade_amount=fixed_trade_amount,
                objective=objective,
                score_func=score_func,
                n_trials=n_trials,
                sampler_name=sampler_name,
                seed=symbol_seed,
                n_jobs=n_jobs,
            )
            detail["ranking"] = opt_res.ranking
            best_params = opt_res.best_params if isinstance(opt_res.best_params, dict) else {}
            detail["best_params"] = dict(best_params)

            if not best_params:
                detail["error"] = "未找到有效参数"
                row = _symbol_result_row(
                    symbol=symbol,
                    best_params={},
                    metrics=None,
                    objective=objective,
                    score_func=score_func,
                    error=detail["error"],
                )
                return symbol, row, detail

            bt_res = run_backtest(
                df=df,
                strategy_cls=strategy_cls,
                strategy_params=best_params,
                initial_cash=initial_cash,
                commission=commission,
                position_percent=position_percent,
                leverage=leverage,
                include_details=False,
                position_sizing_mode=position_sizing_mode,
                fixed_trade_amount=fixed_trade_amount,
            )
            metrics = dict(bt_res.metrics or {})
            detail["metrics"] = metrics
            detail["score"] = _calc_score(metrics, objective, score_func=score_func, symbol=symbol, params=best_params)
            row = _symbol_result_row(
                symbol=symbol,
                best_params=best_params,
                metrics=metrics,
                objective=objective,
                score_func=score_func,
                error=None,
            )
            return symbol, row, detail
        except Exception as exc:  # noqa: BLE001
            detail["error"] = str(exc)
            row = _symbol_result_row(
                symbol=symbol,
                best_params=detail.get("best_params") or {},
                metrics=None,
                objective=objective,
                score_func=score_func,
                error=str(exc),
            )
            return symbol, row, detail

    rows: list[dict[str, Any]] = []
    per_symbol: dict[str, dict[str, Any]] = {}

    if symbol_n_jobs <= 1 or len(symbol_items) <= 1:
        for idx, (symbol, df) in enumerate(symbol_items):
            sym, row, detail = _run_one_symbol(symbol, df, idx)
            rows.append(row)
            per_symbol[sym] = detail
    else:
        with ThreadPoolExecutor(max_workers=symbol_n_jobs) as executor:
            future_map = {
                executor.submit(_run_one_symbol, symbol, df, idx): symbol
                for idx, (symbol, df) in enumerate(symbol_items)
            }
            for future in as_completed(future_map):
                sym, row, detail = future.result()
                rows.append(row)
                per_symbol[sym] = detail

    ranking = _sort_multi_symbol_ranking(pd.DataFrame(rows))
    best_symbol: str | None = None
    if not ranking.empty:
        valid = ranking[ranking["错误"].isna()]
        if not valid.empty:
            best_symbol = str(valid.iloc[0]["币种"])

    return MultiSymbolOptimizationResult(
        ranking=ranking,
        per_symbol=per_symbol,
        best_symbol=best_symbol,
        objective=_objective_label(objective, score_func),
    )
