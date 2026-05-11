from __future__ import annotations

import argparse
from pathlib import Path

from .pipeline import run_all, run_collection, run_master_build


def _default_path(*parts: str) -> str:
    project_root = Path(__file__).resolve().parents[2]
    return str(project_root.joinpath(*parts))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Energy market state ETL pipeline.")
    parser.add_argument(
        "--settings",
        default=_default_path("config", "settings.example.yml"),
        help="Path to the settings YAML file.",
    )
    parser.add_argument(
        "--registry",
        default=_default_path("config", "series_registry.yml"),
        help="Path to the series registry YAML file.",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("collect", help="Download and standardize source data.")
    subparsers.add_parser("build-master", help="Build delivery-aligned and availability-safe master datasets.")
    subparsers.add_parser("run-all", help="Run collection and master build end-to-end.")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "collect":
        run_collection(args.settings, args.registry)
    elif args.command == "build-master":
        run_master_build(args.settings, args.registry)
    elif args.command == "run-all":
        run_all(args.settings, args.registry)
    else:  # pragma: no cover - argparse guards this
        parser.error(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
