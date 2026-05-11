from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import sys

import pandas as pd

from .adapters.energy_charts import EnergyChartsAdapter
from .adapters.entsoe import EntsoeAdapter
from .adapters.fred import FredAdapter
from .adapters.generic_json import GenericJsonAdapter
from .adapters.open_meteo import OpenMeteoAdapter
from .adapters.yfinance_adapter import YFinanceAdapter
from .config import load_series_registry, load_settings
from .http import HttpClient
from .models import SeriesDefinition, Settings
from .storage import persist_feature_catalog, persist_fetch_result, persist_master_dataset, sync_duckdb
from .transform.master import build_master_datasets

ADAPTERS = {
    "energy_charts": EnergyChartsAdapter,
    "entsoe": EntsoeAdapter,
    "fred": FredAdapter,
    "generic_json": GenericJsonAdapter,
    "open_meteo": OpenMeteoAdapter,
    "yfinance": YFinanceAdapter,
}


def run_collection(settings_path: str | Path, registry_path: str | Path) -> tuple[Settings, list[SeriesDefinition]]:
    settings = load_settings(settings_path)
    definitions = load_series_registry(registry_path)
    client = HttpClient()
    completed_definitions: list[SeriesDefinition] = []
    failures: list[tuple[str, str]] = []

    for definition in definitions:
        adapter_cls = ADAPTERS.get(definition.source)
        if adapter_cls is None:
            raise ValueError(f"Unsupported source in registry: {definition.source}")

        adapter = adapter_cls(client, settings)
        try:
            result = adapter.fetch_series(
                definition,
                start=settings.master_dataset.start_ts,
                end=settings.master_dataset.end_ts,
            )
            persist_fetch_result(settings.paths, definition.name, result, client)
            completed_definitions.append(definition)
            print(f"[OK] collected {definition.name}", file=sys.stderr)
        except Exception as exc:
            failures.append((definition.name, str(exc)))
            print(f"[WARN] failed {definition.name}: {exc}", file=sys.stderr)

    if not completed_definitions:
        raise RuntimeError(f"No series were collected successfully. Failures: {failures}")

    if failures:
        print(f"[WARN] collection completed with {len(failures)} failed series.", file=sys.stderr)

    return settings, completed_definitions


def run_master_build(settings_path: str | Path, registry_path: str | Path) -> tuple[Settings, list[SeriesDefinition]]:
    settings = load_settings(settings_path)
    definitions = load_series_registry(registry_path)
    delivery, available = build_master_datasets(settings, definitions)

    persist_master_dataset(settings.paths, "master_delivery_hourly", delivery)
    persist_master_dataset(settings.paths, "master_available_hourly", available)
    feature_catalog = pd.DataFrame([asdict(definition) for definition in definitions])
    persist_feature_catalog(settings.paths, feature_catalog)
    sync_duckdb(settings.paths)
    return settings, definitions


def run_all(settings_path: str | Path, registry_path: str | Path) -> tuple[Settings, list[SeriesDefinition]]:
    settings, definitions = run_collection(settings_path, registry_path)
    delivery, available = build_master_datasets(settings, definitions)
    persist_master_dataset(settings.paths, "master_delivery_hourly", delivery)
    persist_master_dataset(settings.paths, "master_available_hourly", available)
    feature_catalog = pd.DataFrame([asdict(definition) for definition in definitions])
    persist_feature_catalog(settings.paths, feature_catalog)
    sync_duckdb(settings.paths)
    return settings, definitions
