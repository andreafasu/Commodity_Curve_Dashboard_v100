from __future__ import annotations

import pandas as pd

from ..models import FetchResult, SeriesDefinition
from .base import BaseAdapter


class EnergyChartsAdapter(BaseAdapter):
    source_name = "energy_charts"

    def fetch_series(self, definition: SeriesDefinition, start: pd.Timestamp, end: pd.Timestamp) -> FetchResult:
        endpoint = definition.params.get("endpoint", "price")
        params = {
            key: value
            for key, value in definition.params.items()
            if key not in {"endpoint", "value_key", "list_key", "list_name"}
        }
        params.setdefault("start", start.date().isoformat())
        params.setdefault("end", end.date().isoformat())
        payload = self.client.get_json(f"https://api.energy-charts.info/{endpoint}", params=params)
        list_key = definition.params.get("list_key")

        if list_key:
            frame = self._extract_nested_series(
                payload=payload,
                list_key=list_key,
                list_name=definition.params["list_name"],
            )
        else:
            value_key = definition.params.get("value_key") or self._infer_value_key(payload)
            frame = pd.DataFrame(
                {
                    "timestamp_utc": pd.to_datetime(payload.get("unix_seconds", []), unit="s", utc=True),
                    "value": payload.get(value_key, []),
                }
            )
        standardized = self._standardize(definition, frame)
        return FetchResult(standardized=standardized, raw_payload=payload, raw_extension="json")

    @staticmethod
    def _infer_value_key(payload: dict[str, object]) -> str:
        excluded = {"license_info", "unix_seconds", "unit", "deprecated"}
        for key, value in payload.items():
            if key in excluded:
                continue
            if isinstance(value, list):
                return key
        raise ValueError("Could not infer value key from Energy-Charts payload.")

    @staticmethod
    def _extract_nested_series(payload: dict[str, object], *, list_key: str, list_name: str) -> pd.DataFrame:
        unix_seconds = payload.get("unix_seconds", [])
        series_list = payload.get(list_key, [])
        if not isinstance(series_list, list):
            raise ValueError(f"Energy-Charts payload field '{list_key}' is not a list.")

        target_series = next((item for item in series_list if item.get("name") == list_name), None)
        if target_series is None:
            available = [item.get("name") for item in series_list if isinstance(item, dict)]
            raise ValueError(f"Series '{list_name}' not found in '{list_key}'. Available: {available}")

        return pd.DataFrame(
            {
                "timestamp_utc": pd.to_datetime(unix_seconds, unit="s", utc=True),
                "value": target_series.get("data", []),
            }
        )
