"""Command line helper to merge profiling artifacts."""

from __future__ import annotations

import argparse
import sys

from . import merge_profile_records


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Merge per-process profiler outputs")
    parser.add_argument(
        "--by",
        dest="group_by",
        choices=["corr_id"],
        default=None,
        help="Group merged summary by the provided dimension (e.g. corr_id)",
    )

    args = parser.parse_args(argv)

    merged_path = merge_profile_records(by_corr_id=args.group_by == "corr_id")
    print(f"Merged profiling summary written to {merged_path}")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry
    raise SystemExit(main())
