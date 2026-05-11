from __future__ import annotations

import pandas as pd
import yfinance as yf

from ..models import FetchResult, SeriesDefinition
from .base import BaseAdapter


class YFinanceAdapter(BaseAdapter):
    source_name = "yfinance"

    def fetch_series(self, definition: SeriesDefinition, start: pd.Timestamp, end: pd.Timestamp) -> FetchResult:
        ticker = definition.params["ticker"]
        interval = definition.params.get("interval", "1d")
        history = yf.download(
            tickers=ticker,
            start=start.date().isoformat(),
            end=(end + pd.Timedelta(days=1)).date().isoformat(),
            interval=interval,
            auto_adjust=False,
            progress=False,
            threads=False,
        )

        if history.empty:
            return self._empty_result()

        if isinstance(history.columns, pd.MultiIndex):
            history.columns = history.columns.get_level_values(0)

        value_column = definition.params.get("price_field", "Close")
        history = history.reset_index()
        frame = history[["Date" if "Date" in history.columns else "Datetime", value_column]].copy()
        timestamp_column = "Date" if "Date" in frame.columns else "Datetime"
        frame["timestamp_utc"] = self._to_utc(frame[timestamp_column], definition.timezone)
        frame["value"] = frame[value_column]
        standardized = self._standardize(definition, frame[["timestamp_utc", "value"]])
        raw_payload = history.to_dict(orient="records")
        return FetchResult(standardized=standardized, raw_payload=raw_payload, raw_extension="json")
