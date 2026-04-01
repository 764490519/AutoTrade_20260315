from __future__ import annotations

import base64
import hashlib
import hmac
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any
from urllib.parse import urlencode

import requests


class OKXAPIError(RuntimeError):
    pass


@dataclass
class OKXConfig:
    api_key: str
    api_secret: str
    passphrase: str
    base_url: str = "https://www.okx.com"
    demo_trading: bool = False
    timeout: int = 10


class OKXClient:
    def __init__(self, config: OKXConfig):
        self.config = config

    @staticmethod
    def _utc_timestamp() -> str:
        now = datetime.now(timezone.utc)
        return now.isoformat(timespec="milliseconds").replace("+00:00", "Z")

    def _sign(self, timestamp: str, method: str, request_path: str, body: str = "") -> str:
        message = f"{timestamp}{method.upper()}{request_path}{body}"
        digest = hmac.new(
            self.config.api_secret.encode("utf-8"),
            message.encode("utf-8"),
            hashlib.sha256,
        ).digest()
        return base64.b64encode(digest).decode("utf-8")

    def _request(self, method: str, path: str, params: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        method = method.upper()
        params = params or {}

        if method == "GET":
            query = urlencode({k: v for k, v in params.items() if v is not None})
            request_path = path if not query else f"{path}?{query}"
            body = ""
            url = f"{self.config.base_url}{path}"
            req_kwargs: dict[str, Any] = {"params": {k: v for k, v in params.items() if v is not None}}
        else:
            request_path = path
            body = json.dumps({k: v for k, v in params.items() if v is not None}, separators=(",", ":"), ensure_ascii=False)
            url = f"{self.config.base_url}{path}"
            req_kwargs = {"data": body}

        ts = self._utc_timestamp()
        headers = {
            "OK-ACCESS-KEY": self.config.api_key,
            "OK-ACCESS-SIGN": self._sign(ts, method, request_path, body),
            "OK-ACCESS-TIMESTAMP": ts,
            "OK-ACCESS-PASSPHRASE": self.config.passphrase,
            "Content-Type": "application/json",
        }
        if self.config.demo_trading:
            headers["x-simulated-trading"] = "1"

        response = requests.request(
            method=method,
            url=url,
            headers=headers,
            timeout=self.config.timeout,
            **req_kwargs,
        )

        if response.status_code != 200:
            raise OKXAPIError(f"OKX HTTP 错误: {response.status_code}, body={response.text}")

        try:
            payload = response.json()
        except Exception as exc:  # noqa: BLE001
            raise OKXAPIError(f"OKX 返回非 JSON 响应: {response.text}") from exc

        code = str(payload.get("code", ""))
        if code != "0":
            msg = payload.get("msg", "unknown error")
            raise OKXAPIError(f"OKX API 错误 code={code}, msg={msg}")

        data = payload.get("data")
        if isinstance(data, list):
            return data
        return [data] if data else []

    def get_positions(self, inst_id: str | None = None, inst_type: str | None = None) -> list[dict[str, Any]]:
        return self._request(
            "GET",
            "/api/v5/account/positions",
            {"instId": inst_id, "instType": inst_type},
        )

    def get_balances(self, ccy: str | None = None) -> list[dict[str, Any]]:
        return self._request("GET", "/api/v5/account/balance", {"ccy": ccy})

    def place_order(
        self,
        *,
        inst_id: str,
        td_mode: str,
        side: str,
        ord_type: str,
        sz: str,
        px: str | None = None,
        reduce_only: bool | None = None,
        pos_side: str | None = None,
        ccy: str | None = None,
    ) -> dict[str, Any]:
        data = self._request(
            "POST",
            "/api/v5/trade/order",
            {
                "instId": inst_id,
                "tdMode": td_mode,
                "side": side,
                "ordType": ord_type,
                "sz": sz,
                "px": px,
                "reduceOnly": str(reduce_only).lower() if reduce_only is not None else None,
                "posSide": pos_side,
                "ccy": ccy,
            },
        )
        return data[0] if data else {}

    def close_position(
        self,
        *,
        inst_id: str,
        mgn_mode: str = "cross",
        pos_side: str = "net",
        ccy: str | None = None,
        auto_cxl: bool | None = None,
    ) -> dict[str, Any]:
        data = self._request(
            "POST",
            "/api/v5/trade/close-position",
            {
                "instId": inst_id,
                "mgnMode": mgn_mode,
                "posSide": pos_side,
                "ccy": ccy,
                "autoCxl": str(auto_cxl).lower() if auto_cxl is not None else None,
            },
        )
        return data[0] if data else {}
