from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import pandas as pd
import polars as pl

from ..http import HttpClient
from ..models import FetchResult, SeriesDefinition, Settings


class BaseAdapter(ABC):
    source_name = ""

    def __init__(self, client: HttpClient, settings: Settings) -> None:
        self.client = client
        self.settings = settings

    @abstractmethod
    def fetch_series(self, definition: SeriesDefinition, start: pd.Timestamp, end: pd.Timestamp) -> FetchResult:
        raise NotImplementedError

    def _empty_result(self) -> FetchResult:
        empty = pl.DataFrame(
            schema={
                "feature_name": pl.Utf8,
                "market": pl.Utf8,
                "region": pl.Utf8,
                "source": pl.Utf8,
                "timestamp_utc": pl.Datetime(time_zone="UTC"),
                "available_at_utc": pl.Datetime(time_zone="UTC"),
                "value": pl.Float64,
                "unit": pl.Utf8,
                "native_frequency": pl.Utf8,
                "quality_flag": pl.Utf8,
            }
        )
        return FetchResult(standardized=empty, raw_payload={})

    def _to_utc(self, values: Any, timezone: str | None) -> pd.Series:
        series = pd.Series(pd.to_datetime(values))
        if getattr(series.dt, "tz", None) is None:
            localized = series.dt.tz_localize(timezone or "UTC", nonexistent="shift_forward", ambiguous="NaT")
        else:
            localized = series
        return localized.dt.tz_convert("UTC")

    def _standardize(
        self,
        definition: SeriesDefinition,
        frame: pd.DataFrame,
        *,
        timestamp_col: str = "timestamp_utc",
        value_col: str = "value",
        available_col: str | None = None,
        quality_flag: str = "raw",
    ) -> pl.DataFrame:
        if frame.empty:
            return self._empty_result().standardized

        standardized = frame.copy()
        standardized[timestamp_col] = pd.to_datetime(standardized[timestamp_col], utc=True)
        if available_col is None:
            standardized["available_at_utc"] = standardized[timestamp_col] + definition.availability_timedelta
        else:
            standardized["available_at_utc"] = pd.to_datetime(standardized[available_col], utc=True)

        standardized["feature_name"] = definition.name
        standardized["market"] = definition.market
        standardized["region"] = definition.region
        standardized["source"] = definition.source
        standardized["unit"] = definition.unit
        standardized["native_frequency"] = definition.native_frequency
        standardized["quality_flag"] = quality_flag
        standardized["value"] = pd.to_numeric(standardized[value_col], errors="coerce")
        standardized = standardized.dropna(subset=["value", timestamp_col]).sort_values(timestamp_col)

        column_order = [
            "feature_name",
            "market",
            "region",
            "source",
            timestamp_col,
            "available_at_utc",
            "value",
            "unit",
            "native_frequency",
            "quality_flag",
        ]
        standardized = standardized[column_order].rename(columns={timestamp_col: "timestamp_utc"})
        return pl.from_dicts(standardized.to_dict(orient="records"))
