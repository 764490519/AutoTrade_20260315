from __future__ import annotations

import itertools
import json
from dataclasses import dataclass
from typing import Any

import pandas as pd

from backtest_engine import BacktestResult, run_backtest

try:
    import optuna
except Exception:  # noqa: BLE001
    optuna = None


BAD_SCORE = -1e12


@dataclass
class GridSearchResult:
    best_params: dict[str, Any]
    ranking: pd.DataFrame


@dataclass
class OptunaSearchResult:
    best_params: dict[str, Any]
    best_score: float | None
    ranking: pd.DataFrame


def _dedupe_keep_order(values: list[Any]) -> list[Any]:
    out: list[Any] = []
    for value in values:
        if value not in out:
            out.append(value)
    return out


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


def _calc_score(metrics: dict[str, Any], objective: str) -> float:
    value = metrics.get(objective)
    if value is None:
        # 在较短窗口中 Sharpe 可能为 None，回退到总收益率避免全部参数并列 -inf
        if objective == "Sharpe":
            fallback = metrics.get("总收益率(%)")
            if fallback is None:
                return BAD_SCORE
            try:
                return float(fallback)
            except Exception:  # noqa: BLE001
                return BAD_SCORE
        return BAD_SCORE
    try:
        return float(value)
    except Exception:  # noqa: BLE001
        return BAD_SCORE


def optimize_parameters(
    df: pd.DataFrame,
    strategy_cls,
    param_grid: dict[str, list[Any]],
    initial_cash: float,
    commission: float,
    position_percent: float,
    leverage: float = 1.0,
    objective: str = "Sharpe",
    max_combinations: int = 80,
) -> GridSearchResult:
    rows: list[dict[str, Any]] = []

    for idx, params in enumerate(_iter_param_combinations(param_grid), start=1):
        if idx > max_combinations:
            break

        try:
            result: BacktestResult = run_backtest(
                df=df,
                strategy_cls=strategy_cls,
                strategy_params=params,
                initial_cash=initial_cash,
                commission=commission,
                position_percent=position_percent,
                leverage=leverage,
            )
            metrics = result.metrics
            score = _calc_score(metrics, objective)
            rows.append(
                {
                    "参数": params,
                    "评分": score,
                    "总收益率(%)": metrics.get("总收益率(%)"),
                    "Sharpe": metrics.get("Sharpe"),
                    "最大回撤(%)": metrics.get("最大回撤(%)"),
                    "总交易次数": metrics.get("总交易次数"),
                }
            )
        except Exception as exc:  # noqa: BLE001
            rows.append(
                {
                    "参数": params,
                    "评分": BAD_SCORE,
                    "总收益率(%)": None,
                    "Sharpe": None,
                    "最大回撤(%)": None,
                    "总交易次数": None,
                    "错误": str(exc),
                }
            )

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
    objective: str = "Sharpe",
    folds: int = 4,
    max_combinations: int = 80,
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

        opt_res = optimize_parameters(
            df=train_df,
            strategy_cls=strategy_cls,
            param_grid=param_grid,
            initial_cash=initial_cash,
            commission=commission,
            position_percent=position_percent,
            leverage=leverage,
            objective=objective,
            max_combinations=max_combinations,
        )

        best_params = opt_res.best_params
        if not best_params:
            continue

        oos_res = run_backtest(
            df=test_df,
            strategy_cls=strategy_cls,
            strategy_params=best_params,
            initial_cash=initial_cash,
            commission=commission,
            position_percent=position_percent,
            leverage=leverage,
        )

        fold_rows.append(
            {
                "Fold": i,
                "训练区间": f"{train_df.index[0]} ~ {train_df.index[-1]}",
                "测试区间": f"{test_df.index[0]} ~ {test_df.index[-1]}",
                "最优参数": json.dumps(best_params, ensure_ascii=False),
                "样本外收益率(%)": oos_res.metrics.get("总收益率(%)"),
                "样本外Sharpe": oos_res.metrics.get("Sharpe"),
                "样本外最大回撤(%)": oos_res.metrics.get("最大回撤(%)"),
                "样本外交易次数": oos_res.metrics.get("总交易次数"),
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
    summary = {
        "folds": int(len(wf_df)),
        "平均样本外收益率(%)": round(float(oos_returns.mean()), 4) if not oos_returns.empty else None,
        "样本外正收益窗口数": int((oos_returns > 0).sum()),
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
    objective: str = "Sharpe",
    n_trials: int = 80,
    sampler_name: str = "TPE",
    seed: int | None = 42,
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
            )

            metrics = result.metrics
            score = _calc_score(metrics, objective)
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
    try:
        study.optimize(objective_func, n_trials=n_trials, catch=(Exception,))
    except Exception as exc:  # noqa: BLE001
        # sampler 层报错时兜底到 TPE，保证优化流程不中断
        if effective_sampler_name != "TPE":
            fallback_sampler = _build_optuna_sampler(sampler_name="TPE", seed=seed)
            study = optuna.create_study(direction="maximize", sampler=fallback_sampler)
            study.optimize(objective_func, n_trials=n_trials, catch=(Exception,))
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
    objective: str = "Sharpe",
    folds: int = 4,
    n_trials: int = 80,
    sampler_name: str = "TPE",
    seed: int | None = 42,
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

        opt_res = optimize_parameters_optuna(
            df=train_df,
            strategy_cls=strategy_cls,
            param_schema=param_schema,
            initial_cash=initial_cash,
            commission=commission,
            position_percent=position_percent,
            leverage=leverage,
            objective=objective,
            n_trials=n_trials,
            sampler_name=sampler_name,
            seed=None if seed is None else int(seed) + i,
        )

        if not opt_res.best_params:
            continue

        oos_res = run_backtest(
            df=test_df,
            strategy_cls=strategy_cls,
            strategy_params=opt_res.best_params,
            initial_cash=initial_cash,
            commission=commission,
            position_percent=position_percent,
            leverage=leverage,
        )

        fold_rows.append(
            {
                "Fold": i,
                "训练区间": f"{train_df.index[0]} ~ {train_df.index[-1]}",
                "测试区间": f"{test_df.index[0]} ~ {test_df.index[-1]}",
                "最优参数": json.dumps(opt_res.best_params, ensure_ascii=False),
                "训练最优评分": opt_res.best_score,
                "样本外收益率(%)": oos_res.metrics.get("总收益率(%)"),
                "样本外Sharpe": oos_res.metrics.get("Sharpe"),
                "样本外最大回撤(%)": oos_res.metrics.get("最大回撤(%)"),
                "样本外交易次数": oos_res.metrics.get("总交易次数"),
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
    summary = {
        "folds": int(len(wf_df)),
        "平均样本外收益率(%)": round(float(oos_returns.mean()), 4) if not oos_returns.empty else None,
        "样本外正收益窗口数": int((oos_returns > 0).sum()),
        "样本外总窗口数": int(len(wf_df)),
    }

    return wf_df, summary
