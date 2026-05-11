from __future__ import annotations

import json
import time
from typing import Any

import httpx


class HttpClient:
    def __init__(self, timeout_seconds: int = 60, max_retries: int = 3, retry_sleep_seconds: float = 2.0) -> None:
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.retry_sleep_seconds = retry_sleep_seconds

    def _request(self, method: str, url: str, **kwargs: Any) -> httpx.Response:
        last_error: Exception | None = None
        with httpx.Client(timeout=self.timeout_seconds, follow_redirects=True) as client:
            for attempt in range(1, self.max_retries + 1):
                try:
                    response = client.request(method, url, **kwargs)
                    response.raise_for_status()
                    return response
                except Exception as exc:  # pragma: no cover - network path
                    last_error = exc
                    if attempt == self.max_retries:
                        raise
                    time.sleep(self.retry_sleep_seconds * attempt)
        raise RuntimeError(f"Request failed with unknown error: {last_error}")

    def get_json(self, url: str, *, params: dict[str, Any] | None = None, headers: dict[str, str] | None = None) -> Any:
        response = self._request("GET", url, params=params, headers=headers)
        return response.json()

    def get_text(self, url: str, *, params: dict[str, Any] | None = None, headers: dict[str, str] | None = None) -> str:
        response = self._request("GET", url, params=params, headers=headers)
        return response.text

    def serialize_payload(self, payload: Any) -> str:
        if isinstance(payload, str):
            return payload
        return json.dumps(payload, ensure_ascii=True, indent=2, default=str)
