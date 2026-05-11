from __future__ import annotations

import pandas as pd

from ..models import FetchResult, SeriesDefinition
from .base import BaseAdapter


class GenericJsonAdapter(BaseAdapter):
    source_name = "generic_json"

    def fetch_series(self, definition: SeriesDefinition, start: pd.Timestamp, end: pd.Timestamp) -> FetchResult:
        url = definition.params["url"]
        query = dict(definition.params.get("query", {}))
        query.setdefault("start", start.isoformat())
        query.setdefault("end", end.isoformat())

        headers = {}
        auth = definition.params.get("auth", {})
        if auth:
            credential_key = auth.get("credential_key")
            credential_value = self.settings.credentials.get(credential_key) if credential_key else None
            if credential_value:
                if auth.get("type") == "header":
                    headers[auth["name"]] = credential_value
                elif auth.get("type") == "query":
                    query[auth["name"]] = credential_value

        payload = self.client.get_json(url, params=query, headers=headers or None)

        data_key = definition.params.get("data_key")
        if data_key:
            records = payload.get(data_key, [])
        elif isinstance(payload, list):
            records = payload
        else:
            raise ValueError("Generic JSON adapter needs either a list payload or a data_key.")

        timestamp_key = definition.params["timestamp_key"]
        value_key = definition.params["value_key"]
        frame = pd.DataFrame(records)
        if frame.empty:
            return self._empty_result()

        frame["timestamp_utc"] = self._to_utc(frame[timestamp_key], definition.timezone)
        frame["value"] = frame[value_key]
        available_key = definition.params.get("available_timestamp_key")
        if available_key:
            frame["available_at_utc"] = self._to_utc(frame[available_key], definition.timezone)
            standardized = self._standardize(definition, frame, available_col="available_at_utc")
        else:
            standardized = self._standardize(definition, frame)

        return FetchResult(standardized=standardized, raw_payload=payload, raw_extension="json")

