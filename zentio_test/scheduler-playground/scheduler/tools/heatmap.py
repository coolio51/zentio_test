"""Simple CLI to print hotspot heatmaps from merged profiling summaries."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import List

DEFAULT_SUMMARY = Path(".profile/summary_merged.csv")


def load_rows(path: Path) -> List[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Summary file not found: {path}")

    with path.open("r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        return list(reader)


def print_heatmap(rows: List[dict], top: int) -> None:
    if not rows:
        print("No data available.")
        return

    def to_float(row: dict, key: str) -> float:
        try:
            return float(row.get(key, 0) or 0)
        except (ValueError, TypeError):
            return 0.0

    by_wall = sorted(rows, key=lambda r: to_float(r, "wall_ms_total"), reverse=True)[:top]
    by_cpu = sorted(rows, key=lambda r: to_float(r, "cpu_ms_total"), reverse=True)[:top]

    print("=== Top by wall_ms_total ===")
    for row in by_wall:
        print(
            f"{row.get('module')}::{row.get('name')} [{row.get('type')}] - "
            f"wall_ms_total={row.get('wall_ms_total')} cpu_ms_total={row.get('cpu_ms_total')}"
        )

    print("\n=== Top by cpu_ms_total ===")
    for row in by_cpu:
        print(
            f"{row.get('module')}::{row.get('name')} [{row.get('type')}] - "
            f"cpu_ms_total={row.get('cpu_ms_total')} wall_ms_total={row.get('wall_ms_total')}"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Render a simple profiling heatmap")
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--top", type=int, default=20)

    args = parser.parse_args(argv)

    rows = load_rows(args.summary)
    print_heatmap(rows, args.top)
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry
    raise SystemExit(main())
