from __future__ import annotations

import pandas as pd

from ..models import FetchResult, SeriesDefinition
from .base import BaseAdapter


class OpenMeteoAdapter(BaseAdapter):
    source_name = "open_meteo"

    def fetch_series(self, definition: SeriesDefinition, start: pd.Timestamp, end: pd.Timestamp) -> FetchResult:
        variable = definition.params["variable"]
        params = {
            "latitude": definition.params["latitude"],
            "longitude": definition.params["longitude"],
            "start_date": start.date().isoformat(),
            "end_date": end.date().isoformat(),
            "hourly": variable,
            "timezone": "UTC",
        }
        if "wind_speed_unit" in definition.params:
            params["wind_speed_unit"] = definition.params["wind_speed_unit"]

        payload = self.client.get_json("https://archive-api.open-meteo.com/v1/archive", params=params)
        hourly_payload = payload.get("hourly", {})
        times = hourly_payload.get("time", [])
        values = hourly_payload.get(variable, [])

        frame = pd.DataFrame({"timestamp_utc": pd.to_datetime(times, utc=True), "value": values})
        standardized = self._standardize(definition, frame)
        return FetchResult(standardized=standardized, raw_payload=payload, raw_extension="json")
