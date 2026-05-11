from __future__ import annotations

import re
import xml.etree.ElementTree as et

import pandas as pd

from ..models import FetchResult, SeriesDefinition
from .base import BaseAdapter

DURATION_PATTERN = re.compile(r"P(?:(?P<days>\d+)D)?(?:T(?:(?P<hours>\d+)H)?(?:(?P<minutes>\d+)M)?)?")


class EntsoeAdapter(BaseAdapter):
    source_name = "entsoe"

    def fetch_series(self, definition: SeriesDefinition, start: pd.Timestamp, end: pd.Timestamp) -> FetchResult:
        token = definition.params.get("securityToken") or self.settings.credentials.get("entsoe_token")
        if not token:
            raise ValueError("ENTSO-E token missing. Set ENTSOE_API_TOKEN or securityToken in the series definition.")

        params = {
            key: value
            for key, value in definition.params.items()
            if key != "securityToken"
        }
        params["securityToken"] = token
        params["periodStart"] = start.strftime("%Y%m%d%H%M")
        params["periodEnd"] = end.strftime("%Y%m%d%H%M")

        payload = self.client.get_text("https://web-api.tp.entsoe.eu/api", params=params)
        frame = self._parse_timeseries(payload)
        if frame.empty:
            return self._empty_result()

        standardized = self._standardize(definition, frame)
        return FetchResult(standardized=standardized, raw_payload=payload, raw_extension="xml")

    def _parse_timeseries(self, xml_text: str) -> pd.DataFrame:
        root = et.fromstring(xml_text)
        records: list[dict[str, object]] = []
        for time_series in root.iter():
            if self._local_name(time_series.tag) != "TimeSeries":
                continue
            for period in time_series.iter():
                if self._local_name(period.tag) != "Period":
                    continue
                start_text = self._find_text(period, "start")
                resolution_text = self._find_text(period, "resolution")
                if not start_text or not resolution_text:
                    continue

                start_ts = pd.Timestamp(start_text).tz_convert("UTC")
                resolution = self._parse_resolution(resolution_text)

                for point in period:
                    if self._local_name(point.tag) != "Point":
                        continue
                    position = int(self._find_text(point, "position") or "0")
                    value = self._extract_numeric_value(point)
                    if position <= 0 or value is None:
                        continue
                    records.append(
                        {
                            "timestamp_utc": start_ts + (position - 1) * resolution,
                            "value": value,
                        }
                    )

        return pd.DataFrame(records)

    @staticmethod
    def _local_name(tag: str) -> str:
        if "}" in tag:
            return tag.split("}", 1)[1]
        return tag

    def _find_text(self, element: et.Element, local_name: str) -> str | None:
        for child in element.iter():
            if self._local_name(child.tag) == local_name:
                return child.text
        return None

    def _extract_numeric_value(self, point: et.Element) -> float | None:
        for child in point:
            local_name = self._local_name(child.tag)
            if local_name == "position":
                continue
            if child.text is None:
                continue
            try:
                return float(child.text)
            except ValueError:
                continue
        return None

    @staticmethod
    def _parse_resolution(duration_text: str) -> pd.Timedelta:
        match = DURATION_PATTERN.fullmatch(duration_text)
        if not match:
            raise ValueError(f"Unsupported ENTSO-E duration: {duration_text}")
        days = int(match.group("days") or 0)
        hours = int(match.group("hours") or 0)
        minutes = int(match.group("minutes") or 0)
        return pd.Timedelta(days=days, hours=hours, minutes=minutes)

