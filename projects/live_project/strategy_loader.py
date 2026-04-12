from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import pandas as pd


@dataclass
class StrategyRuntime:
    strategy_obj: Any
    display_name: str
    params_schema: dict[str, dict[str, Any]]
    strategy_name: str


def _resolve_strategy_obj(namespace: dict[str, Any], meta: dict[str, Any]) -> Any:
    class_name = meta.get("strategy_class")
    signal_func = meta.get("signal_func", "generate_targets")

    if isinstance(class_name, str) and class_name in namespace:
        return namespace[class_name]

    if isinstance(signal_func, str):
        maybe = namespace.get(signal_func)
        if callable(maybe):
            return maybe

    # fallback 1: class with generate_targets
    for obj in namespace.values():
        if inspect.isclass(obj) and callable(getattr(obj, "generate_targets", None)):
            return obj

    # fallback 2: plain callable generate_targets
    maybe = namespace.get("generate_targets")
    if callable(maybe):
        return maybe

    raise ValueError("策略代码中未找到可执行策略对象（需提供 strategy_class 或 generate_targets）")


def compile_strategy_runtime_from_code(code: str) -> StrategyRuntime:
    source = str(code or "").lstrip("\ufeff")
    namespace: dict[str, Any] = {"pd": pd, "np": np}
    exec(source, namespace, namespace)

    meta = namespace.get("STRATEGY_META", {})
    if not isinstance(meta, dict):
        meta = {}

    strategy_obj = _resolve_strategy_obj(namespace, meta)
    strategy_name = str(getattr(strategy_obj, "__name__", strategy_obj.__class__.__name__))
    display_name = str(meta.get("display_name", strategy_name))
    params_schema = meta.get("params", {})
    if not isinstance(params_schema, dict):
        params_schema = {}

    return StrategyRuntime(
        strategy_obj=strategy_obj,
        display_name=display_name,
        params_schema=params_schema,
        strategy_name=strategy_name,
    )
