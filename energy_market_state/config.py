from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import yaml

from .models import CredentialsConfig, MasterDatasetConfig, PathsConfig, SeriesDefinition, Settings

ENV_PATTERN = re.compile(r"\$\{([^}]+)\}")


def _project_root_from_config_path(config_path: Path) -> Path:
    config_path = config_path.resolve()
    if config_path.parent.name == "config":
        return config_path.parent.parent
    return config_path.parent


def _resolve_path(base_dir: Path, candidate: str) -> Path:
    path = Path(candidate)
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def _expand_env(value: Any) -> Any:
    if isinstance(value, str):
        def replacer(match: re.Match[str]) -> str:
            env_name = match.group(1)
            return os.getenv(env_name, "")

        return ENV_PATTERN.sub(replacer, value)
    if isinstance(value, list):
        return [_expand_env(item) for item in value]
    if isinstance(value, dict):
        return {key: _expand_env(item) for key, item in value.items()}
    return value


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    return _expand_env(raw)


def load_settings(path: str | Path) -> Settings:
    config_path = Path(path)
    project_root = _project_root_from_config_path(config_path)
    raw = _load_yaml(config_path)

    paths_raw = raw.get("paths", {})
    root_dir = _resolve_path(project_root, paths_raw.get("root_dir", "./runtime"))
    bronze_dir = _resolve_path(project_root, paths_raw.get("bronze_dir", str(root_dir / "bronze")))
    silver_dir = _resolve_path(project_root, paths_raw.get("silver_dir", str(root_dir / "silver")))
    gold_dir = _resolve_path(project_root, paths_raw.get("gold_dir", str(root_dir / "gold")))
    duckdb_path = _resolve_path(project_root, paths_raw.get("duckdb_path", str(root_dir / "energy_market_state.duckdb")))

    settings = Settings(
        paths=PathsConfig(
            root_dir=root_dir,
            bronze_dir=bronze_dir,
            silver_dir=silver_dir,
            gold_dir=gold_dir,
            duckdb_path=duckdb_path,
        ),
        master_dataset=MasterDatasetConfig(**raw.get("master_dataset", {})),
        credentials=CredentialsConfig(values=raw.get("credentials", {})),
    )
    settings.paths.ensure_directories()
    return settings


def load_series_registry(path: str | Path) -> list[SeriesDefinition]:
    raw = _load_yaml(Path(path))
    return [SeriesDefinition(**item) for item in raw.get("series", [])]

