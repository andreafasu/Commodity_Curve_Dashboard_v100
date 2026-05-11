from __future__ import annotations

from pathlib import Path

import pandas as pd
import polars as pl

from ..models import SeriesDefinition, Settings


def build_master_datasets(settings: Settings, definitions: list[SeriesDefinition]) -> tuple[pl.DataFrame, pl.DataFrame]:
    delivery = _build_master(settings, definitions, mode="delivery")
    available = _build_master(settings, definitions, mode="available")
    delivery = _derive_power_aggregates(_derive_degree_days(delivery))
    available = _derive_power_aggregates(_derive_degree_days(available))
    return delivery, available


def _build_master(settings: Settings, definitions: list[SeriesDefinition], *, mode: str) -> pl.DataFrame:
    grid = pd.DataFrame(
        {
            "timestamp_utc": pd.date_range(
                settings.master_dataset.start_ts,
                settings.master_dataset.end_ts,
                freq=settings.master_dataset.target_frequency,
            )
        }
    )
    grid["timestamp_utc"] = pd.to_datetime(grid["timestamp_utc"], utc=True)

    for definition in definitions:
        series_path = settings.paths.silver_dir / f"{definition.name}.parquet"
        if not series_path.exists():
            continue

        standardized = pd.DataFrame(pl.read_parquet(series_path).to_dicts())
        if standardized.empty:
            continue

        series = _resample_to_target(standardized, definition, settings.master_dataset.target_frequency)
        if series.empty:
            continue

        if mode == "delivery":
            merged = grid.merge(
                series[["timestamp_utc", "value"]],
                on="timestamp_utc",
                how="left",
            )
            periods = _ffill_period_limit(
                definition.resolved_ffill_limit_hours(settings.master_dataset.default_ffill_limit_hours),
                settings.master_dataset.target_frequency,
            )
            grid[definition.name] = merged["value"].ffill(limit=periods)
        elif mode == "available":
            series["available_at_utc"] = pd.to_datetime(series["available_at_utc"], utc=True)
            aligned = pd.merge_asof(
                grid.sort_values("timestamp_utc"),
                series.sort_values("available_at_utc"),
                left_on="timestamp_utc",
                right_on="available_at_utc",
                direction="backward",
            )
            grid[definition.name] = aligned["value"]
        else:
            raise ValueError(f"Unsupported master build mode: {mode}")

    if settings.master_dataset.add_calendar_features:
        grid["date_utc"] = grid["timestamp_utc"].dt.date.astype(str)
        grid["hour_utc"] = grid["timestamp_utc"].dt.hour
        grid["day_of_week"] = grid["timestamp_utc"].dt.dayofweek
        grid["month"] = grid["timestamp_utc"].dt.month

    ordered_columns = ["timestamp_utc", "date_utc", "hour_utc", "day_of_week", "month"]
    ordered_columns.extend([definition.name for definition in definitions if definition.name in grid.columns])
    ordered_columns = [column for column in ordered_columns if column in grid.columns]
    grid = grid[ordered_columns]
    return pl.from_dicts(grid.to_dict(orient="records"))


def _resample_to_target(frame: pd.DataFrame, definition: SeriesDefinition, target_frequency: str) -> pd.DataFrame:
    series = frame.copy()
    series["timestamp_utc"] = pd.to_datetime(series["timestamp_utc"], utc=True)
    series["available_at_utc"] = pd.to_datetime(series["available_at_utc"], utc=True)
    series = series.sort_values("timestamp_utc")

    aggregated = getattr(series.set_index("timestamp_utc")["value"].resample(target_frequency), definition.aggregation)()
    available = series.set_index("timestamp_utc")["available_at_utc"].resample(target_frequency).max()

    result = pd.DataFrame(
        {
            "timestamp_utc": aggregated.index,
            "available_at_utc": list(available),
            "value": aggregated.values,
        }
    )
    result["timestamp_utc"] = pd.to_datetime(result["timestamp_utc"], utc=True)
    result["available_at_utc"] = pd.to_datetime(result["available_at_utc"], utc=True)
    result = result.dropna(subset=["value"])
    return result


def _ffill_period_limit(hours_limit: int, target_frequency: str) -> int:
    frequency_delta = pd.Timedelta(target_frequency)
    return max(1, int(pd.Timedelta(hours=hours_limit) / frequency_delta))


def _derive_degree_days(frame: pl.DataFrame) -> pl.DataFrame:
    result = frame
    for column in frame.columns:
        if not column.endswith("_temp_c"):
            continue
        prefix = column[: -len("_temp_c")]
        result = result.with_columns(
            (pl.lit(18.0) - pl.col(column)).clip(lower_bound=0.0).alias(f"{prefix}_hdd_18"),
            (pl.col(column) - pl.lit(22.0)).clip(lower_bound=0.0).alias(f"{prefix}_cdd_22"),
        )
    return result


def _derive_power_aggregates(frame: pl.DataFrame) -> pl.DataFrame:
    result = frame

    if {"power_de_wind_onshore_mw", "power_de_wind_offshore_mw"}.issubset(frame.columns):
        result = result.with_columns(
            (pl.col("power_de_wind_onshore_mw") + pl.col("power_de_wind_offshore_mw")).alias("power_de_wind_total_mw")
        )

    hydro_components = [
        "power_de_hydro_run_of_river_mw",
        "power_de_hydro_reservoir_mw",
        "power_de_hydro_pumped_storage_mw",
    ]
    if set(hydro_components).issubset(frame.columns):
        result = result.with_columns(
            (
                pl.col("power_de_hydro_run_of_river_mw")
                + pl.col("power_de_hydro_reservoir_mw")
                + pl.col("power_de_hydro_pumped_storage_mw")
            ).alias("power_de_hydro_total_mw")
        )

    renewable_components = [
        "power_de_biomass_mw",
        "power_de_solar_mw",
        "power_de_wind_total_mw",
        "power_de_hydro_total_mw",
    ]
    if set(renewable_components).issubset(result.columns):
        result = result.with_columns(
            (
                pl.col("power_de_biomass_mw")
                + pl.col("power_de_solar_mw")
                + pl.col("power_de_wind_total_mw")
                + pl.col("power_de_hydro_total_mw")
            ).alias("power_de_renewables_total_mw")
        )

    return result
