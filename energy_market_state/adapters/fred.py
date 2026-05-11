from __future__ import annotations

import pandas as pd

from ..models import FetchResult, SeriesDefinition
from .base import BaseAdapter


class FredAdapter(BaseAdapter):
    source_name = "fred"

    def fetch_series(self, definition: SeriesDefinition, start: pd.Timestamp, end: pd.Timestamp) -> FetchResult:
        series_id = definition.params["series_id"]
        api_key = definition.params.get("api_key") or self.settings.credentials.get("fred_api_key")

        params = {
            "series_id": series_id,
            "observation_start": start.date().isoformat(),
            "observation_end": end.date().isoformat(),
            "file_type": "json",
        }
        if api_key:
            params["api_key"] = api_key

        payload = self.client.get_json(
            "https://api.stlouisfed.org/fred/series/observations",
            params=params,
        )

        records = []
        for observation in payload.get("observations", []):
            raw_value = observation.get("value")
            if raw_value in (None, ".", ""):
                continue
            records.append(
                {
                    "timestamp": observation["date"],
                    "value": float(raw_value),
                }
            )

        if not records:
            return self._empty_result()

        frame = pd.DataFrame(records)
        frame["timestamp_utc"] = self._to_utc(frame["timestamp"], definition.timezone)
        standardized = self._standardize(definition, frame[["timestamp_utc", "value"]])
        return FetchResult(standardized=standardized, raw_payload=payload, raw_extension="json")

