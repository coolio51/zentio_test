"""Deterministic benchmarking harness for scheduler conversions and pipeline."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

# Ensure profiling is enabled before importing profiling helpers
os.environ.setdefault("ZENTIO_PROFILE", os.getenv("ZENTIO_PROFILE", "1"))

from scheduler.common.profiling import (  # noqa: E402  (import after env setup)
    Profiler,
    profile_section,
    profile_enabled,
    reset_correlation_id,
    set_correlation_id,
)
from scheduler.common.console import get_console  # noqa: E402
from scheduler.services.resource_manager import ResourceManager  # noqa: E402
from scheduler.services.scheduler import SchedulerService  # noqa: E402
from utils.data_converters import (  # noqa: E402
    convert_manufacturing_orders,
    convert_resources,
)


def _load_dataset(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _run_iteration(data: Dict[str, Any], iteration: int) -> Dict[str, Any]:
    token = set_correlation_id(f"benchmark-iter-{iteration:03d}")
    try:
        with profile_section("benchmark.convert_mos"):
            manufacturing_orders = convert_manufacturing_orders(
                data.get("manufacturing_orders") or data.get("manufacturingOrdersRequirements") or []
            )
        with profile_section("benchmark.convert_resources"):
            resources = convert_resources(
                data.get("resources") or data.get("availableResources") or []
            )

        operations: List[Any] = []
        for manufacturing_order in manufacturing_orders:
            if manufacturing_order.operations_graph and manufacturing_order.operations_graph.nodes:
                operations.extend(manufacturing_order.operations_graph.nodes)

        with profile_section("benchmark.resource_manager"):
            resource_manager = ResourceManager(resources)

        with profile_section("benchmark.schedule"):
            schedule = SchedulerService.schedule(operations, resource_manager)

        return {
            "iteration": iteration,
            "manufacturing_orders": len(manufacturing_orders),
            "operations": len(operations),
            "resources": len(resources),
            "tasks": len(schedule.tasks),
            "dropped_operations": len(schedule.dropped_operations or []),
            "makespan": str(schedule.makespan) if schedule.makespan else None,
        }
    finally:
        reset_correlation_id(token)


def run_benchmark(dataset: Path, iterations: int) -> Dict[str, Any]:
    data = _load_dataset(dataset)
    profiler = Profiler.instance()
    summary: List[Dict[str, Any]] = []

    with profile_section("benchmark.total_run"):
        for iteration in range(1, iterations + 1):
            summary.append(_run_iteration(data, iteration))

    if getattr(profiler, "enabled", False):
        profiler.flush()

    return {
        "dataset": str(dataset),
        "iterations": iterations,
        "results": summary,
        "profile_dir": str(Path(".profile").resolve()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "data" / "real_combined.json",
        help="Path to the dataset JSON file (defaults to bundled real_combined.json)",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=3,
        help="Number of benchmark iterations to execute",
    )
    args = parser.parse_args()

    console = get_console()
    result = run_benchmark(args.dataset, max(1, args.iterations))

    console.print("[bold cyan]Scheduler benchmark summary[/bold cyan]")
    console.print(json.dumps(result, indent=2))

    if not profile_enabled():
        console.print(
            "[yellow]Warning:[/] Profiling artifacts were not generated because"
            " ZENTIO_PROFILE was disabled."
        )


if __name__ == "__main__":
    main()
