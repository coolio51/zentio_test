from datetime import datetime, timedelta
from copy import deepcopy
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
from tests.utils import generate_capabilities, generate_availabilities

# Import resources from test_1, but create our own operations
from .test_1 import resources
from scheduler.models import Resource, ResourceType
from .utils import Shift

resources.append(
    Resource(
        resource_id="machine-3",
        resource_type=ResourceType.MACHINE,
        resource_name="CNC Machine 3",
        resource_capabilities=[],
        availabilities=[
            *generate_availabilities(Shift.ALL_DAY, datetime.now().date(), 10),
        ],
    )
)


def create_flexible_operations():
    """Create operations identical to test_1.py but with flexible machine assignments."""
    from .test_1 import (
        operation_cnc_machining,
        operation_deburring,
        mo1_manufacturing_order_id,
        mo1_manufacturing_order_name,
        mo1_article_id,
        mo1_article_name,
        mo1_quantity,
        mo2_manufacturing_order_id,
        mo2_manufacturing_order_name,
        mo2_article_id,
        mo2_article_name,
        mo2_quantity,
        mo3_manufacturing_order_id,
        mo3_manufacturing_order_name,
        mo3_article_id,
        mo3_article_name,
        mo3_quantity,
        mo4_manufacturing_order_id,
        mo4_manufacturing_order_name,
        mo4_article_id,
        mo4_article_name,
        mo4_quantity,
        mo5_manufacturing_order_id,
        mo5_manufacturing_order_name,
        mo5_article_id,
        mo5_article_name,
        mo5_quantity,
        mo_debug_manufacturing_order_id,
        mo_debug_manufacturing_order_name,
        mo_debug_article_id,
        mo_debug_article_name,
        mo_debug_quantity,
    )

    # MO-00001 - Allow GA to choose machines
    mo1_cnc_machining = deepcopy(operation_cnc_machining)
    mo1_cnc_machining.operation_instance_id = "mo-00001-cnc-machining-1"
    mo1_cnc_machining.article_id = mo1_article_id
    mo1_cnc_machining.article_name = mo1_article_name
    mo1_cnc_machining.manufacturing_order_id = mo1_manufacturing_order_id
    mo1_cnc_machining.manufacturing_order_name = mo1_manufacturing_order_name
    mo1_cnc_machining.quantity = mo1_quantity
    mo1_cnc_machining.required_machine_id = None  # 🎯 Let GA choose

    mo1_deburring = deepcopy(operation_deburring)
    mo1_deburring.operation_instance_id = "mo-00001-deburring-1"
    mo1_deburring.article_id = mo1_article_id
    mo1_deburring.article_name = mo1_article_name
    mo1_deburring.manufacturing_order_id = mo1_manufacturing_order_id
    mo1_deburring.manufacturing_order_name = mo1_manufacturing_order_name
    mo1_deburring.dependencies = [mo1_cnc_machining]
    mo1_deburring.quantity = mo1_quantity
    mo1_deburring.required_machine_id = None  # 🎯 Let GA choose

    # MO-00002 - Allow GA to choose machines for parallel execution
    mo2_cnc_machining = deepcopy(operation_cnc_machining)
    mo2_cnc_machining.operation_instance_id = "mo-00002-cnc-machining-1"
    mo2_cnc_machining.article_id = mo2_article_id
    mo2_cnc_machining.article_name = mo2_article_name
    mo2_cnc_machining.manufacturing_order_id = mo2_manufacturing_order_id
    mo2_cnc_machining.manufacturing_order_name = mo2_manufacturing_order_name
    mo2_cnc_machining.quantity = mo2_quantity
    mo2_cnc_machining.required_machine_id = None  # 🎯 Was "machine-2", now flexible

    mo2_deburring_1 = deepcopy(operation_deburring)
    mo2_deburring_1.operation_instance_id = "mo-00002-deburring-1"
    mo2_deburring_1.article_id = mo2_article_id
    mo2_deburring_1.article_name = mo2_article_name
    mo2_deburring_1.manufacturing_order_id = mo2_manufacturing_order_id
    mo2_deburring_1.manufacturing_order_name = mo2_manufacturing_order_name
    mo2_deburring_1.quantity = mo2_quantity // 2
    mo2_deburring_1.required_machine_id = None  # 🎯 Was "machine-2", now flexible
    mo2_deburring_1.dependencies = [mo2_cnc_machining]

    mo2_deburring_2 = deepcopy(operation_deburring)
    mo2_deburring_2.operation_instance_id = "mo-00002-deburring-2"
    mo2_deburring_2.article_id = mo2_article_id
    mo2_deburring_2.article_name = mo2_article_name
    mo2_deburring_2.manufacturing_order_id = mo2_manufacturing_order_id
    mo2_deburring_2.manufacturing_order_name = mo2_manufacturing_order_name
    mo2_deburring_2.quantity = mo2_quantity // 2
    mo2_deburring_2.required_machine_id = None  # 🎯 Was "machine-2", now flexible
    mo2_deburring_2.dependencies = [mo2_cnc_machining]

    # MO-00003 - Allow GA to choose machines
    mo3_cnc_machining = deepcopy(operation_cnc_machining)
    mo3_cnc_machining.operation_instance_id = "mo-00003-cnc-machining-1"
    mo3_cnc_machining.article_id = mo3_article_id
    mo3_cnc_machining.article_name = mo3_article_name
    mo3_cnc_machining.manufacturing_order_id = mo3_manufacturing_order_id
    mo3_cnc_machining.manufacturing_order_name = mo3_manufacturing_order_name
    mo3_cnc_machining.quantity = mo3_quantity
    mo3_cnc_machining.required_machine_id = None  # 🎯 Was "machine-2", now flexible

    # Copy operator requirements from original
    from .test_1 import mo3_cnc_machining as original_mo3_cnc

    mo3_cnc_machining.operator_requirements = original_mo3_cnc.operator_requirements

    mo3_deburring = deepcopy(operation_deburring)
    mo3_deburring.operation_instance_id = "mo-00003-deburring-1"
    mo3_deburring.article_id = mo3_article_id
    mo3_deburring.article_name = mo3_article_name
    mo3_deburring.manufacturing_order_id = mo3_manufacturing_order_id
    mo3_deburring.manufacturing_order_name = mo3_manufacturing_order_name
    mo3_deburring.quantity = mo3_quantity
    mo3_deburring.dependencies = [mo3_cnc_machining]
    mo3_deburring.required_machine_id = None  # 🎯 Was "machine-2", now flexible

    # MO-00004 - Allow GA to choose machines
    mo4_cnc_machining = deepcopy(operation_cnc_machining)
    mo4_cnc_machining.operation_instance_id = "mo-00004-cnc-machining-1"
    mo4_cnc_machining.article_id = mo4_article_id
    mo4_cnc_machining.article_name = mo4_article_name
    mo4_cnc_machining.manufacturing_order_id = mo4_manufacturing_order_id
    mo4_cnc_machining.manufacturing_order_name = mo4_manufacturing_order_name
    mo4_cnc_machining.quantity = mo4_quantity
    mo4_cnc_machining.required_machine_id = None  # 🎯 Let GA choose

    mo4_deburring = deepcopy(operation_deburring)
    mo4_deburring.operation_instance_id = "mo-00004-deburring-1"
    mo4_deburring.article_id = mo4_article_id
    mo4_deburring.article_name = mo4_article_name
    mo4_deburring.manufacturing_order_id = mo4_manufacturing_order_id
    mo4_deburring.manufacturing_order_name = mo4_manufacturing_order_name
    mo4_deburring.dependencies = [mo4_cnc_machining]
    mo4_deburring.quantity = mo4_quantity
    mo4_deburring.required_machine_id = None  # 🎯 Let GA choose

    # MO-00005 - Allow GA to choose machines
    mo5_cnc_machining = deepcopy(operation_cnc_machining)
    mo5_cnc_machining.operation_instance_id = "mo-00005-cnc-machining-1"
    mo5_cnc_machining.article_id = mo5_article_id
    mo5_cnc_machining.article_name = mo5_article_name
    mo5_cnc_machining.manufacturing_order_id = mo5_manufacturing_order_id
    mo5_cnc_machining.manufacturing_order_name = mo5_manufacturing_order_name
    mo5_cnc_machining.quantity = mo5_quantity
    mo5_cnc_machining.required_machine_id = None  # 🎯 Let GA choose

    mo5_deburring = deepcopy(operation_deburring)
    mo5_deburring.operation_instance_id = "mo-00005-deburring-1"
    mo5_deburring.article_id = mo5_article_id
    mo5_deburring.article_name = mo5_article_name
    mo5_deburring.manufacturing_order_id = mo5_manufacturing_order_id
    mo5_deburring.manufacturing_order_name = mo5_manufacturing_order_name
    mo5_deburring.dependencies = [mo5_cnc_machining]
    mo5_deburring.quantity = mo5_quantity
    mo5_deburring.required_machine_id = None  # 🎯 Let GA choose

    # Debug MO - Keep simple, let GA choose
    mo_debug_cnc_machining = deepcopy(operation_cnc_machining)
    mo_debug_cnc_machining.operation_instance_id = "mo-debug-cnc-machining-1"
    mo_debug_cnc_machining.article_id = mo_debug_article_id
    mo_debug_cnc_machining.article_name = mo_debug_article_name
    mo_debug_cnc_machining.manufacturing_order_id = mo_debug_manufacturing_order_id
    mo_debug_cnc_machining.manufacturing_order_name = mo_debug_manufacturing_order_name
    mo_debug_cnc_machining.quantity = mo_debug_quantity
    mo_debug_cnc_machining.dependencies = []
    mo_debug_cnc_machining.required_machine_id = None  # 🎯 Let GA choose

    # Combine all operations
    operations = [
        mo_debug_cnc_machining,
        mo1_cnc_machining,
        mo1_deburring,
        mo2_cnc_machining,
        mo2_deburring_1,
        mo2_deburring_2,
        mo3_cnc_machining,
        mo3_deburring,
        mo4_cnc_machining,
        mo4_deburring,
        mo5_cnc_machining,
        mo5_deburring,
    ]

    return operations


def create_test_manufacturing_orders():
    """Create manufacturing orders for testing with priorities and due dates."""
    base_date = datetime(2025, 8, 5, 8, 0)  # Base date for due dates

    # Get unique manufacturing order IDs from operations
    operations = create_flexible_operations()
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


def test_genetic_2(console: Console):
    """Test genetic algorithm optimization with flexible machine assignments for parallel execution."""

    console.print(
        "[bold blue]🧬 GENETIC ALGORITHM OPTIMIZATION TEST 2 - FLEXIBLE MACHINES[/bold blue]"
    )

    # Create flexible operations
    operations = create_flexible_operations()

    console.print(f"Operations to schedule: {len(operations)}")
    console.print(f"Resources available: {len(resources)}")
    console.print(
        "[bold green]🎯 Key Change: All operations have flexible machine assignments (GA will choose)[/bold green]"
    )

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

    # Configure genetic algorithm
    ga_config = GeneticAlgorithmConfig(
        population_size=20,
        max_generations=20,
        crossover_rate=0.8,
        mutation_rate=0.15,
        elite_size=3,
        max_operation_splits=3,  # Slightly more splits to test parallelization
        min_split_quantity=1,
        optimization_objectives=[
            OptimizationObjective.MINIMIZE_MAKESPAN,
            OptimizationObjective.MINIMIZE_DROPPED_OPERATIONS,
            OptimizationObjective.MINIMIZE_IDLE_TIME,
        ],
        stagnation_generations=5,
        parallel_operators=None,
    )

    console.print(f"\n[bold yellow]⚙️  Genetic Algorithm Configuration[/bold yellow]")
    console.print(f"Population size: {ga_config.population_size}")
    console.print(f"Max generations: {ga_config.max_generations}")
    console.print(f"Max operation splits: {ga_config.max_operation_splits}")
    console.print(
        "[bold cyan]🔧 Machine Assignment: GA will choose optimal machines for each operation[/bold cyan]"
    )

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

    # Optionally display generation evolution history
    if len(optimizer.generation_history) > 0:
        from scheduler.utils.genetic_algorithm_logger import GeneticAlgorithmLogger

    GeneticAlgorithmLogger.print_generation_history(
        optimizer.generation_history, console
    )

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
    console.print(sr.schedule_table(schedule))
    console.print(sr.schedule_summary_table(schedule))

    if schedule.dropped_operations:
        console.print(sr.dropped_operations_table(schedule))
    else:
        console.print("[bold green]🎉 No operations were dropped![/bold green]")

    # Show chromosome details and parallel execution analysis using the dedicated logger
    GeneticAlgorithmLogger.print_chromosome_details(
        best_chromosome, operations, console
    )

    GeneticAlgorithmLogger.print_parallel_execution_analysis(
        best_chromosome, operations, console
    )

    console.print(f"\n[bold green]🎯 Genetic Algorithm Test 2 Complete![/bold green]")
    console.print(
        "[bold cyan]💡 Compare results with test_genetic.py to see the impact of flexible machine assignments![/bold cyan]"
    )


if __name__ == "__main__":
    console = get_console()
    test_genetic_2(console)
