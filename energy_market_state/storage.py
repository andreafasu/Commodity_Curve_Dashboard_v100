from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd
import polars as pl

from .http import HttpClient
from .models import FetchResult, PathsConfig


def persist_fetch_result(paths: PathsConfig, feature_name: str, result: FetchResult, http_client: HttpClient) -> None:
    fetched_at = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    bronze_path = paths.bronze_dir / f"{feature_name}_{fetched_at}.{result.raw_extension}"
    bronze_path.write_text(http_client.serialize_payload(result.raw_payload), encoding="utf-8")

    silver_parquet = paths.silver_dir / f"{feature_name}.parquet"
    silver_csv = paths.silver_dir / f"{feature_name}.csv"
    result.standardized.write_parquet(silver_parquet)
    result.standardized.write_csv(silver_csv)


def persist_master_dataset(paths: PathsConfig, name: str, frame: pl.DataFrame) -> None:
    frame.write_parquet(paths.gold_dir / f"{name}.parquet")
    frame.write_csv(paths.gold_dir / f"{name}.csv")


def persist_feature_catalog(paths: PathsConfig, catalog: pd.DataFrame) -> Path:
    catalog_path = paths.gold_dir / "feature_catalog.csv"
    catalog.to_csv(catalog_path, index=False)
    return catalog_path


def sync_duckdb(paths: PathsConfig) -> None:
    con = duckdb.connect(str(paths.duckdb_path))
    try:
        silver_glob = str((paths.silver_dir / "*.parquet").resolve()).replace("\\", "/")
        delivery_path = str((paths.gold_dir / "master_delivery_hourly.parquet").resolve()).replace("\\", "/")
        available_path = str((paths.gold_dir / "master_available_hourly.parquet").resolve()).replace("\\", "/")
        feature_catalog_path = str((paths.gold_dir / "feature_catalog.csv").resolve()).replace("\\", "/")

        con.execute("create schema if not exists silver")
        con.execute("create schema if not exists gold")

        if list(paths.silver_dir.glob("*.parquet")):
            con.execute(f"create or replace view silver.series_long as select * from read_parquet('{silver_glob}')")
        if Path(delivery_path).exists():
            con.execute(f"create or replace table gold.master_delivery_hourly as select * from read_parquet('{delivery_path}')")
        if Path(available_path).exists():
            con.execute(f"create or replace table gold.master_available_hourly as select * from read_parquet('{available_path}')")
        if Path(feature_catalog_path).exists():
            con.execute(f"create or replace table gold.feature_catalog as select * from read_csv_auto('{feature_catalog_path}')")
    finally:
        con.close()

