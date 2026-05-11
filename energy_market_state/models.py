from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass
class PathsConfig:
    root_dir: Path
    bronze_dir: Path
    silver_dir: Path
    gold_dir: Path
    duckdb_path: Path

    def ensure_directories(self) -> None:
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self.bronze_dir.mkdir(parents=True, exist_ok=True)
        self.silver_dir.mkdir(parents=True, exist_ok=True)
        self.gold_dir.mkdir(parents=True, exist_ok=True)
        self.duckdb_path.parent.mkdir(parents=True, exist_ok=True)


@dataclass
class MasterDatasetConfig:
    start: str
    end: str
    target_frequency: str = "1h"
    add_calendar_features: bool = True
    default_ffill_limit_hours: int = 48

    @property
    def start_ts(self) -> pd.Timestamp:
        return pd.Timestamp(self.start).tz_convert("UTC")

    @property
    def end_ts(self) -> pd.Timestamp:
        return pd.Timestamp(self.end).tz_convert("UTC")


@dataclass
class CredentialsConfig:
    values: dict[str, Any] = field(default_factory=dict)

    def get(self, key: str, default: Any = None) -> Any:
        return self.values.get(key, default)


@dataclass
class Settings:
    paths: PathsConfig
    master_dataset: MasterDatasetConfig
    credentials: CredentialsConfig


@dataclass
class SeriesDefinition:
    name: str
    source: str
    market: str
    region: str
    unit: str
    native_frequency: str
    timezone: str
    params: dict[str, Any] = field(default_factory=dict)
    target_frequency: str = "1h"
    aggregation: str = "last"
    availability_lag: str = "0h"
    max_ffill_hours: int | None = None
    description: str = ""

    @property
    def availability_timedelta(self) -> pd.Timedelta:
        return pd.Timedelta(self.availability_lag)

    def resolved_ffill_limit_hours(self, default_hours: int) -> int:
        if self.max_ffill_hours is None:
            return default_hours
        return self.max_ffill_hours


@dataclass
class FetchResult:
    standardized: Any
    raw_payload: Any
    raw_extension: str = "json"
    metadata: dict[str, Any] = field(default_factory=dict)

