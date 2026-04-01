from __future__ import annotations

import os
from pathlib import Path
from typing import Any

try:
    import tomllib  # py311+
except Exception:  # noqa: BLE001
    tomllib = None


BASE_DIR = Path(__file__).resolve().parent
CONFIG_DIR = BASE_DIR / "config"
_env_api_cfg = os.getenv("AUTOTRADE_API_CONFIG_FILE", "").strip()
if _env_api_cfg:
    API_CONFIG_FILE = Path(_env_api_cfg).expanduser().resolve()
else:
    API_CONFIG_FILE = CONFIG_DIR / "apis.toml"


def load_api_config_file() -> dict[str, Any]:
    if tomllib is None:
        return {}
    if not API_CONFIG_FILE.exists():
        return {}
    try:
        with API_CONFIG_FILE.open("rb") as f:
            data = tomllib.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:  # noqa: BLE001
        return {}


def get_api_config_section(section: str) -> dict[str, Any]:
    data = load_api_config_file()
    value: Any = data
    for part in str(section).split("."):
        if not isinstance(value, dict):
            return {}
        value = value.get(part, {})
    return value if isinstance(value, dict) else {}
