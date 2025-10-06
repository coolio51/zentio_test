#!/usr/bin/env python3
"""
Test genetic algorithm with real production data.

This test uses actual manufacturing orders and resources exported from the live system
to test the genetic algorithm performance with realistic scenarios.
"""

import json
import os
import sys
from pathlib import Path
from datetime import datetime
import argparse

from rich.console import Console

# Add the project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from scheduler.services.genetic_optimizer import (
    optimize_schedule,
    GeneticAlgorithmConfig,
    OptimizationObjective,
)
from scheduler.services.resource_manager import ResourceManager
from api.zentio_api import convert_manufacturing_orders, convert_resources
from scheduler.utils.schedule_logger import ScheduleLogger


def load_real_data():
    """Load real data from JSON files."""
    data_dir = Path(__file__).parent / "data"

    # Check if data files exist
    combined_file = data_dir / "real_combined.json"
    mo_file = data_dir / "real_manufacturing_orders.json"
    resources_file = data_dir / "real_resources.json"

    if combined_file.exists():
        # Load from combined file
        with open(combined_file, "r") as f:
            data = json.load(f)
        return data["manufacturing_orders"], data["resources"], data.get("metadata", {})

    elif mo_file.exists() and resources_file.exists():
        # Load from separate files
        with open(mo_file, "r") as f:
            mo_data = json.load(f)
        with open(resources_file, "r") as f:
            resources_data = json.load(f)

        metadata_file = data_dir / "real_metadata.json"
        metadata = {}
        if metadata_file.exists():
            with open(metadata_file, "r") as f:
                metadata = json.load(f)

        return mo_data, resources_data, metadata

    else:
        raise FileNotFoundError(
            f"Real data files not found in {data_dir}. "
            "Run 'python export_real_data.py' first to export data from the live system."
        )


def test_genetic_real(
    console: Console,
    config_overrides: dict | None = None,
    show_schedule: bool | None = None,
    enable_evolution_logging: bool | None = None,
):
    """Test genetic algorithm optimization with real production data."""

    console.print(
        "[bold blue]🧬 GENETIC ALGORITHM TEST - REAL PRODUCTION DATA[/bold blue]"
    )
    console.print()

    try:
        # Load real data
        console.print("📁 Loading real production data...")
        mo_data, resources_data, metadata = load_real_data()

        if metadata:
            console.print(f"📊 Data exported: {metadata.get('exported_at', 'Unknown')}")
            console.print(
                f"🏢 Organization: {metadata.get('organization_id', 'Unknown')}"
            )
            console.print(
                f"📅 Date range: {metadata.get('days_ahead', 'Unknown')} days ahead"
            )

        # Convert data to scheduler format
        console.print("🔄 Converting data to scheduler format...")
        manufacturing_orders = convert_manufacturing_orders(mo_data)
        resources = convert_resources(resources_data)

        console.print(f"✅ Loaded {len(manufacturing_orders)} manufacturing orders")
        console.print(f"✅ Loaded {len(resources)} resources")

        if not manufacturing_orders:
            console.print("[red]❌ No manufacturing orders found in data![/red]")
            return

        if not resources:
            console.print("[red]❌ No resources found in data![/red]")
            return

        # Extract operations
        operations = []
        for mo in manufacturing_orders:
            if mo.operations_graph and mo.operations_graph.nodes:
                operations.extend(mo.operations_graph.nodes)

        console.print(f"📋 Extracted {len(operations)} operations")

        if not operations:
            console.print("[red]❌ No operations found in manufacturing orders![/red]")
            return

        # Configure genetic algorithm for real data testing
        overrides = config_overrides or {}
        config = GeneticAlgorithmConfig(
            population_size=int(overrides.get("population_size", 15)),
            max_generations=int(overrides.get("max_generations", 30)),
            crossover_rate=float(overrides.get("crossover_rate", 0.8)),
            mutation_rate=float(overrides.get("mutation_rate", 0.1)),
            elite_size=int(overrides.get("elite_size", 5)),
            max_operation_splits=int(overrides.get("max_operation_splits", 5)),
            min_split_quantity=int(overrides.get("min_split_quantity", 5)),
            optimization_objectives=[
                OptimizationObjective.MINIMIZE_MAKESPAN,
                OptimizationObjective.MINIMIZE_DROPPED_OPERATIONS,
            ],
            stagnation_generations=int(overrides.get("stagnation_generations", 20)),
            parallel_operators=(
                int(overrides["parallel_operators"])
                if "parallel_operators" in overrides
                else None
            ),
        )

        console.print(f"⚙️  Genetic Algorithm Configuration:")
        console.print(f"   • Population size: {config.population_size}")
        console.print(f"   • Max generations: {config.max_generations}")
        console.print(f"   • Elite size: {config.elite_size}")
        console.print(f"   • Max operation splits: {config.max_operation_splits}")
        console.print()

        # Create resource manager
        resource_manager = ResourceManager(resources)

        console.print(
            "\n[bold magenta]🚀 Starting Genetic Optimization with Real Data...[/bold magenta]"
        )
        console.print()

        # Run optimization
        schedule, best_chromosome, optimizer = optimize_schedule(
            operations=operations,
            resource_manager=resource_manager,
            manufacturing_orders=manufacturing_orders,
            config=config,
            show_schedule=(False if show_schedule is None else bool(show_schedule)),
            enable_evolution_logging=(
                True
                if enable_evolution_logging is None
                else bool(enable_evolution_logging)
            ),
        )

        console.print()
        console.print("[bold green]✅ Optimization Complete![/bold green]")
        console.print()

        # Display results summary
        console.print(ScheduleLogger.schedule_summary_table(schedule))

        if schedule.dropped_operations:
            console.print()
            console.print("[bold red]⚠️  Dropped Operations Analysis[/bold red]")
            dropped_table = ScheduleLogger.dropped_operations_table(schedule)
            if dropped_table:
                console.print(dropped_table)
        else:
            console.print("[bold green]🎉 No operations were dropped![/bold green]")

        # Show genetic algorithm decisions
        console.print()
        console.print("[bold magenta]🧬 Genetic Algorithm Results[/bold magenta]")
        console.print(f"   • Operation splits: {len(best_chromosome.operation_splits)}")
        console.print(f"   • Final fitness score: {schedule.fitness_score:.2f}")

        if schedule.makespan:
            console.print(f"   • Total makespan: {schedule.makespan}")

        console.print(
            f"\n[bold green]🎯 Real Data Genetic Algorithm Test Complete![/bold green]"
        )

        # Return summary for any external use
        return {
            "success": True,
            "operations_count": len(operations),
            "manufacturing_orders_count": len(manufacturing_orders),
            "resources_count": len(resources),
            "tasks_scheduled": len(schedule.tasks),
            "operations_scheduled": schedule.number_of_operations_scheduled,
            "operations_dropped": (
                len(schedule.dropped_operations) if schedule.dropped_operations else 0
            ),
            "fitness_score": schedule.fitness_score,
            "makespan": str(schedule.makespan) if schedule.makespan else "0:00:00",
        }

    except FileNotFoundError as e:
        console.print(f"[red]❌ Data files not found: {e}[/red]")
        console.print(
            "[yellow]💡 Run 'python export_real_data.py' first to export data from the live system.[/yellow]"
        )
        return {"success": False, "error": str(e)}

    except Exception as e:
        console.print(f"[red]❌ Test failed: {e}[/red]")
        import traceback

        console.print(f"[dim]{traceback.format_exc()}[/dim]")
        return {"success": False, "error": str(e)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the genetic optimizer on real production data with optional overrides.",
    )
    parser.add_argument("--population-size", type=int, dest="population_size")
    parser.add_argument("--max-generations", type=int, dest="max_generations")
    parser.add_argument("--crossover-rate", type=float, dest="crossover_rate")
    parser.add_argument("--mutation-rate", type=float, dest="mutation_rate")
    parser.add_argument("--elite-size", type=int, dest="elite_size")
    parser.add_argument("--max-operation-splits", type=int, dest="max_operation_splits")
    parser.add_argument("--min-split-quantity", type=int, dest="min_split_quantity")
    parser.add_argument(
        "--stagnation-generations", type=int, dest="stagnation_generations"
    )
    parser.add_argument("--parallel-operators", type=int, dest="parallel_operators")
    parser.add_argument(
        "--show-schedule", action="store_true", dest="show_schedule", default=False
    )
    parser.add_argument(
        "--no-evolution-logging",
        action="store_true",
        dest="no_evolution_logging",
        default=False,
        help="Disable evolution logging output.",
    )

    args = parser.parse_args()
    overrides = {
        k: v
        for k, v in vars(args).items()
        if v is not None and k not in {"show_schedule", "no_evolution_logging"}
    }
    console = Console()
    test_genetic_real(
        console,
        config_overrides=overrides,
        show_schedule=args.show_schedule,
        enable_evolution_logging=(not args.no_evolution_logging),
    )
