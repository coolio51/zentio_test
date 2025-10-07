from datetime import datetime, timedelta
from rich.console import Console
from scheduler.common.console import get_console

from scheduler.models import ManufacturingOrder, Priority
from scheduler.services.resource_manager import ResourceManager
from scheduler.services.genetic_optimizer import (
    optimize_schedule,
    GeneticAlgorithmConfig,
    OptimizationObjective,
)
from scheduler.utils.resource_logger import ResourceLogger as rr
from scheduler.utils.schedule_logger import ScheduleLogger as sr

# Import the same operations and resources from test_1
from .test_1 import operations, resources


def create_test_manufacturing_orders():
    """Create manufacturing orders for testing with priorities and due dates."""
    base_date = datetime(2025, 8, 5, 8, 0)  # Base date for due dates

    # Get unique manufacturing order IDs from operations
    mo_ids = list(
        set(op.manufacturing_order_id for op in operations if op.manufacturing_order_id)
    )

    manufacturing_orders = []
    priorities = [Priority.HIGH, Priority.URGENT, Priority.MEDIUM, Priority.LOW]

    for i, mo_id in enumerate(mo_ids):
        # Find first operation with this MO ID to get the name
        sample_op = next(op for op in operations if op.manufacturing_order_id == mo_id)

        mo = ManufacturingOrder(
            manufacturing_order_id=mo_id,
            manufacturing_order_name=sample_op.manufacturing_order_name
            or f"Order {mo_id}",
            article_id=sample_op.article_id or f"ART_{i}",
            article_name=sample_op.article_name or f"Article {i}",
            quantity=100 + i * 50,  # Varying quantities
            operations_graph=None,  # Not needed for this test
            required_by_date=base_date + timedelta(days=2 + i),  # Staggered due dates
            priority=priorities[i % len(priorities)],  # Cycle through priorities
        )
        manufacturing_orders.append(mo)

    return manufacturing_orders


def test_genetic(console: Console):
    """Test genetic algorithm optimization with the same operations as test_1."""

    console.print("[bold blue]🧬 GENETIC ALGORITHM OPTIMIZATION TEST[/bold blue]")
    console.print(f"Operations to schedule: {len(operations)}")
    console.print(f"Resources available: {len(resources)}")

    # Create manufacturing orders and resource manager
    manufacturing_orders = create_test_manufacturing_orders()
    resource_manager = ResourceManager(resources)

    console.print(
        f"\n[bold green]📋 Manufacturing Orders Created: {len(manufacturing_orders)}[/bold green]"
    )
    for mo in manufacturing_orders:
        console.print(
            f"  • {mo.manufacturing_order_name} ({mo.manufacturing_order_id}) - Priority: {mo.priority.value.upper()}"
        )

    # Display initial state
    console.print("\n[bold green]📋 Initial Resource Availability[/bold green]")
    console.print(rr.initial_availability_table(resource_manager.resources))

    # Configure genetic algorithm (adjusted for smaller operation quantities in test_1.py)
    ga_config = GeneticAlgorithmConfig(
        population_size=20,  # Smaller for faster testing
        max_generations=30,  # Fewer generations for faster testing
        crossover_rate=0.8,
        mutation_rate=0.15,
        elite_size=3,
        max_operation_splits=2,  # Limit splits since operations are small
        min_split_quantity=1,  # FIXED: Allow splits as small as 1 for test_1.py operations
        optimization_objectives=[
            OptimizationObjective.MINIMIZE_MAKESPAN,
            OptimizationObjective.MINIMIZE_DROPPED_OPERATIONS,
            OptimizationObjective.MINIMIZE_IDLE_TIME,
        ],
        stagnation_generations=5,  # Stop early if no improvement
        parallel_operators=None,  # Auto-detect CPU cores
    )

    console.print(f"\n[bold yellow]⚙️  Genetic Algorithm Configuration[/bold yellow]")
    console.print(f"Population size: {ga_config.population_size}")
    console.print(f"Max generations: {ga_config.max_generations}")
    console.print(f"Crossover rate: {ga_config.crossover_rate}")
    console.print(f"Mutation rate: {ga_config.mutation_rate}")
    console.print(f"Elite size: {ga_config.elite_size}")
    console.print(f"Max operation splits: {ga_config.max_operation_splits}")
    console.print(f"Optimization objectives: {len(ga_config.optimization_objectives)}")

    # Run genetic optimization
    console.print("\n[bold magenta]🚀 Starting Genetic Optimization...[/bold magenta]")
    start_time = datetime.now()

    schedule, best_chromosome, optimizer = optimize_schedule(
        operations, resource_manager, manufacturing_orders, ga_config
    )

    end_time = datetime.now()
    optimization_time = end_time - start_time

    # Display results
    console.print(f"\n[bold green]✅ Optimization Complete![/bold green]")
    console.print(f"Optimization time: {optimization_time.total_seconds():.2f} seconds")

    # Calculate and display fitness score
    from scheduler.services.genetic_optimizer import GeneticSchedulerOptimizer

    temp_optimizer = GeneticSchedulerOptimizer(ga_config)
    temp_optimizer.operations = operations
    temp_optimizer.manufacturing_orders = {
        mo.manufacturing_order_id: mo for mo in manufacturing_orders
    }
    fitness_score = temp_optimizer._calculate_fitness(schedule, best_chromosome)

    from scheduler.utils.genetic_algorithm_logger import GeneticAlgorithmLogger

    GeneticAlgorithmLogger.print_optimization_summary(
        fitness_score, best_chromosome, optimization_time.total_seconds(), console
    )

    # Display detailed results
    console.print("\n[bold cyan]📊 Optimized Schedule Results[/bold cyan]")
    console.print(rr.resource_bookings_table(resource_manager))
    console.print(sr.schedule_table(schedule))
    console.print(sr.schedule_summary_table(schedule))

    if schedule.dropped_operations:
        console.print(sr.dropped_operations_table(schedule))
    else:
        console.print("[bold green]🎉 No operations were dropped![/bold green]")

    # Show chromosome details using the dedicated logger
    from scheduler.utils.genetic_algorithm_logger import GeneticAlgorithmLogger

    GeneticAlgorithmLogger.print_chromosome_details(
        best_chromosome, operations, console
    )

    console.print(f"\n[bold green]🎯 Genetic Algorithm Test Complete![/bold green]")


if __name__ == "__main__":
    console = get_console()
    test_genetic(console)
