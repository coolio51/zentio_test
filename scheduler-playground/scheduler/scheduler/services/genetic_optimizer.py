from datetime import datetime, timedelta
from typing import (
    List,
    Dict,
    Optional,
    Tuple,
    Any,
    Union,
    TYPE_CHECKING,
    Callable,
    Awaitable,
)
from dataclasses import dataclass, field
from copy import deepcopy
import random
import math
from enum import Enum
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
import os

from scheduler.models import (
    OperationNode,
    Schedule,
    ResourceType,
    TaskPhase,
    ManufacturingOrder,
    Priority,
)
from scheduler.services.resource_manager import ResourceManager
from scheduler.services.scheduler import SchedulerService
from scheduler.utils.resource_logger import ResourceLogger

if TYPE_CHECKING:
    from scheduler.utils.genetic_algorithm_logger import GeneticAlgorithmLogger


class OptimizationObjective(Enum):
    MINIMIZE_MAKESPAN = "minimize_makespan"
    MAXIMIZE_RESOURCE_UTILIZATION = "maximize_resource_utilization"
    MINIMIZE_DROPPED_OPERATIONS = "minimize_dropped_operations"
    MINIMIZE_IDLE_TIME = "minimize_idle_time"


@dataclass
class OperationSplit:
    """Represents a split of an operation into multiple smaller operations."""

    original_operation: OperationNode
    split_quantities: List[int]  # List of quantities for each split

    def __post_init__(self):
        # Validate that split quantities sum to original quantity
        if sum(self.split_quantities) != self.original_operation.quantity:
            raise ValueError(
                f"Split quantities {self.split_quantities} don't sum to original quantity {self.original_operation.quantity}"
            )


@dataclass
class SchedulingChromosome:
    """
    DNA representation for the genetic algorithm.

    Contains all the decisions needed to create a deterministic schedule:
    - Manufacturing Order sequence (IDs of MOs to prioritize)
    - Operation splits for large operations
    """

    mo_order: List[str]  # Sequence of manufacturing_order_id values
    operation_splits: Dict[
        int, List[int]
    ]  # operation_index -> list of split quantities

    def __post_init__(self):
        # Validate mo order contains unique IDs
        if len(set(self.mo_order)) != len(self.mo_order):
            raise ValueError("Manufacturing order sequence contains duplicates")


@dataclass
class GeneticAlgorithmConfig:
    """Configuration for the genetic algorithm."""

    population_size: int = 100
    max_generations: int = 500
    crossover_rate: float = 0.8
    mutation_rate: float = 0.1
    elite_size: int = 10  # Number of best individuals to preserve each generation
    tournament_size: int = 3  # Size of tournament selection
    max_operation_splits: int = 5  # Maximum number of splits per operation
    min_split_quantity: int = 1  # Minimum quantity for each split
    optimization_objectives: List[OptimizationObjective] = field(
        default_factory=lambda: [
            OptimizationObjective.MINIMIZE_MAKESPAN,
            OptimizationObjective.MINIMIZE_DROPPED_OPERATIONS,
        ]
    )
    stagnation_generations: int = 50  # Stop if no improvement for this many generations
    target_fitness: Optional[float] = None  # Stop if this fitness is reached
    parallel_operators: Optional[int] = None  # None = auto-detect CPU cores


def _evaluate_chromosome_parallel(args: Tuple) -> Tuple[Schedule, float, int]:
    """
    Standalone function for parallel chromosome evaluation.

    Args:
        args: Tuple containing (chromosome, operations, resource_manager, manufacturing_orders, config, chromosome_index)

    Returns:
        Tuple of (schedule, fitness, chromosome_index)
    """
    (
        chromosome,
        operations,
        resource_manager,
        manufacturing_orders,
        config,
        chromosome_index,
    ) = args

    # Create a temporary optimizer instance for evaluation
    optimizer = GeneticSchedulerOptimizer(config)
    optimizer.operations = operations
    optimizer.original_resource_manager = resource_manager
    optimizer.manufacturing_orders = {
        mo.manufacturing_order_id: mo for mo in manufacturing_orders
    }
    # No machine assignment cache needed; scheduler selects machines dynamically

    # Evaluate the chromosome
    schedule = optimizer._evaluate_chromosome(chromosome)
    fitness = optimizer._calculate_fitness(schedule, chromosome)

    return schedule, fitness, chromosome_index


class GeneticSchedulerOptimizer:
    """
    Genetic Algorithm optimizer for the scheduler.

    Evolves scheduling decisions to find optimal schedules by:
    1. Representing scheduling decisions as chromosomes (DNA)
    2. Evaluating fitness using the actual scheduler
    3. Evolving better solutions through selection, crossover, and mutation
    """

    def __init__(self, config: Optional[GeneticAlgorithmConfig] = None):
        self.config = config or GeneticAlgorithmConfig()
        self.operations: List[OperationNode] = []
        self.manufacturing_orders: Dict[str, ManufacturingOrder] = (
            {}
        )  # manufacturing_order_id -> ManufacturingOrder
        self.resource_manager: Optional[ResourceManager] = None
        self.original_resource_manager: Optional[ResourceManager] = None
        self.capable_machines_cache: Dict[str, List[str]] = (
            {}
        )  # operation_id -> machine_ids
        self.generation_history: List[Dict] = []  # Track evolution across generations

    def optimize(
        self,
        operations: List[OperationNode],
        resource_manager: ResourceManager,
        manufacturing_orders: List[ManufacturingOrder],
        logger: Optional["GeneticAlgorithmLogger"] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> Tuple[Schedule, SchedulingChromosome]:
        """
        Find optimal schedule using genetic algorithm.

        Args:
            operations: List of operations to schedule
            resource_manager: Resource manager with available resources
            manufacturing_orders: List of manufacturing orders with priorities and due dates
            logger: Optional logger for evolution logging and display
            progress_callback: Optional callback for progress updates (generation, total_generations)

        Returns:
            Tuple of (best_schedule, best_chromosome)
        """
        self.operations = operations
        self.original_resource_manager = resource_manager
        # Convert manufacturing orders list to dictionary for fast lookup
        self.manufacturing_orders = {
            mo.manufacturing_order_id: mo for mo in manufacturing_orders
        }
        # No machine assignment decisions in DNA; scheduler will choose machines

        # Debug: Print initial resource availability table before optimization
        try:
            from rich.console import Console

            console = Console()
            console.print("\n[bold cyan]Initial Resource Availability (GA)[/bold cyan]")
            availability_table = ResourceLogger.initial_availability_table(
                self.original_resource_manager.resources
            )
            console.print(availability_table)
        except Exception as e:
            print(f"Warning: failed to print initial resource availability: {e}")

        # Initialize population
        population = self._initialize_population()
        best_fitness = float("inf")  # Lower is better now
        best_chromosome = None
        best_schedule = None
        stagnation_counter = 0

        # Initialize generation tracking
        self.generation_history = []  # Reset for new optimization

        # Log optimization start
        if logger:
            logger.log_optimization_start(len(operations), self.config)

        for generation in range(self.config.max_generations):
            # Log generation start
            if logger:
                logger.log_generation_start(generation + 1, self.config.max_generations)

            # Call progress callback if provided
            if progress_callback:
                try:
                    progress_callback(generation + 1, self.config.max_generations)
                except Exception as e:
                    # Don't let callback errors stop optimization
                    print(f"Warning: Progress callback failed: {e}")

            # Evaluate fitness for each chromosome in parallel
            fitness_scores = []
            schedules = []

            # Determine number of operators
            max_workers = self.config.parallel_operators or mp.cpu_count()
            max_workers = min(
                max_workers, len(population)
            )  # Don't use more operators than chromosomes

            if max_workers > 1:

                # Prepare arguments for parallel processing
                eval_args = [
                    (
                        chromosome,
                        self.operations,
                        self.original_resource_manager,
                        list(self.manufacturing_orders.values()),
                        self.config,
                        i,
                    )
                    for i, chromosome in enumerate(population)
                ]

                # Use ProcessPoolExecutor for CPU-bound tasks
                with ProcessPoolExecutor(max_workers=max_workers) as executor:
                    # Submit all chromosome evaluations
                    future_to_index = {
                        executor.submit(_evaluate_chromosome_parallel, args): args[
                            5
                        ]  # args[5] is the index
                        for args in eval_args
                    }

                    # Create placeholders for results
                    fitness_scores = [0.0] * len(population)
                    schedules: List[Optional[Schedule]] = [None] * len(population)

                    # Collect results as they complete
                    for future in as_completed(future_to_index):
                        try:
                            schedule, fitness, idx = future.result()
                            # Propagate computed fitness into the schedule for downstream display/logging
                            try:
                                schedule.fitness_score = fitness
                            except Exception:
                                pass
                            fitness_scores[idx] = fitness
                            schedules[idx] = schedule

                            # Track best solution (lower is better)
                            if fitness < best_fitness:
                                old_fitness = best_fitness
                                best_fitness = fitness
                                best_chromosome = deepcopy(population[idx])
                                best_schedule = schedule
                                stagnation_counter = 0
                                # Log new best found
                                if logger:
                                    logger.log_new_best_found(
                                        generation + 1, fitness, old_fitness
                                    )

                        except Exception as e:

                            # Set default poor values for failed evaluations
                            idx = future_to_index[future]
                            fitness_scores[idx] = (
                                10000.0  # High penalty for failed evaluations
                            )
                            schedules[idx] = Schedule(
                                fitness_score=0.0,
                                tasks=[],
                                number_of_operations_scheduled=0,
                                number_of_resources_scheduled=0,
                                makespan=timedelta(0),
                            )
            else:
                # Fallback to sequential evaluation
                for i, chromosome in enumerate(population):
                    schedule = self._evaluate_chromosome(chromosome)
                    fitness = self._calculate_fitness(schedule, chromosome)
                    # Propagate computed fitness into the schedule for downstream display/logging
                    try:
                        schedule.fitness_score = fitness
                    except Exception:
                        pass
                    fitness_scores.append(fitness)
                    schedules.append(schedule)

                    # Track best solution (lower is better)
                    if fitness < best_fitness:
                        old_fitness = best_fitness
                        best_fitness = fitness
                        best_chromosome = deepcopy(chromosome)
                        best_schedule = schedule
                        stagnation_counter = 0
                        # Log new best found
                        if logger:
                            logger.log_new_best_found(
                                generation + 1, fitness, old_fitness
                            )

                    else:
                        stagnation_counter += 1

            # Check stopping criteria (lower is better)
            if (
                self.config.target_fitness
                and best_fitness <= self.config.target_fitness
            ):
                if logger:
                    logger.log_target_fitness_reached(self.config.target_fitness)
                break

            if stagnation_counter >= self.config.stagnation_generations:
                if logger:
                    logger.log_stagnation_stop(stagnation_counter)
                break

            # Create next generation
            population = self._create_next_generation(population, fitness_scores)

            # Statistics
            avg_fitness = sum(fitness_scores) / len(fitness_scores)
            worst_fitness = max(fitness_scores)
            current_best_fitness = min(fitness_scores)

            # Check if we found a new global best in this generation
            found_new_best = best_fitness < (
                self.generation_history[-1]["global_best_fitness"]
                if self.generation_history
                else float("inf")
            )

            # Log generation summary
            if logger:
                splits_count = (
                    len(best_chromosome.operation_splits) if best_chromosome else 0
                )

                # Extract operation statistics from best schedule
                scheduled_ops = (
                    best_schedule.number_of_operations_scheduled if best_schedule else 0
                )
                dropped_ops = (
                    len(best_schedule.dropped_operations)
                    if best_schedule and best_schedule.dropped_operations
                    else 0
                )
                total_ops = len(operations)
                mo_count = len(manufacturing_orders)

                logger.log_generation_summary(
                    generation + 1,
                    current_best_fitness,
                    best_fitness,
                    avg_fitness,
                    splits_count,
                    stagnation_counter,
                    found_new_best,
                    scheduled_ops,
                    total_ops,
                    dropped_ops,
                    mo_count,
                )

            # Track generation data
            generation_data = {
                "generation": generation + 1,
                "best_fitness": current_best_fitness,
                "avg_fitness": avg_fitness,
                "worst_fitness": worst_fitness,
                "global_best_fitness": best_fitness,
                "stagnation": stagnation_counter,
                "new_best": found_new_best,
                "splits_count": (
                    len(best_chromosome.operation_splits) if best_chromosome else 0
                ),
            }
            self.generation_history.append(generation_data)

        if best_schedule is None or best_chromosome is None:
            raise RuntimeError("Genetic algorithm failed to find any valid solution")

        # Log optimization completion
        if logger:
            # Calculate final statistics
            final_scheduled = (
                best_schedule.number_of_operations_scheduled if best_schedule else 0
            )
            final_dropped = (
                len(best_schedule.dropped_operations)
                if best_schedule and best_schedule.dropped_operations
                else 0
            )
            total_operations = len(operations)
            mo_count = len(manufacturing_orders)

            logger.log_optimization_complete(
                len(self.generation_history),
                best_fitness,
                (
                    self.generation_history[0]["best_fitness"]
                    if self.generation_history
                    else float("inf")
                ),
                best_chromosome,
                final_scheduled,
                total_operations,
                final_dropped,
                mo_count,
                best_schedule,
            )

        return best_schedule, best_chromosome

    def _topological_order_for_mo(self, mo_id: str) -> List[int]:
        """Generate a topological ordering of operations for a given manufacturing order."""
        # Collect indices of operations belonging to this MO
        op_indices = [
            i
            for i, op in enumerate(self.operations)
            if op.manufacturing_order_id == mo_id
        ]
        index_set = set(op_indices)

        # Build graph among these indices
        in_degree = {i: 0 for i in op_indices}
        adjacency: Dict[int, List[int]] = {i: [] for i in op_indices}

        # Map identity to index for quick lookups
        op_to_index = {id(op): i for i, op in enumerate(self.operations)}

        for i in op_indices:
            operation = self.operations[i]
            for dep in operation.dependencies:
                dep_index = op_to_index.get(id(dep))
                if dep_index is not None and dep_index in index_set:
                    adjacency[dep_index].append(i)
                    in_degree[i] += 1

        # Kahn's algorithm
        available = [i for i in op_indices if in_degree[i] == 0]
        order: List[int] = []
        while available:
            current = available.pop(0)
            order.append(current)
            for neighbor in adjacency[current]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    available.append(neighbor)

        # If cycle or missing deps, fallback to original order within MO
        if len(order) != len(op_indices):
            order = sorted(op_indices)
        return order

    def _initialize_population(self) -> List[SchedulingChromosome]:
        """Initialize random population of chromosomes."""
        population = []

        for _ in range(self.config.population_size):
            chromosome = self._create_random_chromosome()
            population.append(chromosome)

        return population

    def _create_random_chromosome(self) -> SchedulingChromosome:
        """Create a random chromosome with valid scheduling decisions."""
        # Random manufacturing order sequence
        mo_ids = list(self.manufacturing_orders.keys())
        random.shuffle(mo_ids)

        # Random operation splits
        operation_splits = {}
        for i, operation in enumerate(self.operations):
            if operation.quantity > self.config.min_split_quantity * 2:
                # Randomly decide whether to split this operation
                if random.random() < 0.3:  # 30% chance to split
                    splits = self._generate_random_splits(operation.quantity)
                    if len(splits) > 1:
                        operation_splits[i] = splits

        return SchedulingChromosome(
            mo_order=mo_ids,
            operation_splits=operation_splits,
        )

    def _generate_random_topological_order(self) -> List[int]:
        """Generate a random topological ordering of operations that respects dependencies."""
        # Build dependency graph
        in_degree = [0] * len(self.operations)
        adjacency = [[] for _ in range(len(self.operations))]

        # Map operation to index
        op_to_index = {id(op): i for i, op in enumerate(self.operations)}

        for i, operation in enumerate(self.operations):
            for dependency in operation.dependencies:
                dep_index = op_to_index[id(dependency)]
                adjacency[dep_index].append(i)
                in_degree[i] += 1

        # Kahn's algorithm with randomization
        available = [i for i in range(len(self.operations)) if in_degree[i] == 0]
        order = []

        while available:
            # Randomly select from available operations
            current = random.choice(available)
            available.remove(current)
            order.append(current)

            # Update neighbors
            for neighbor in adjacency[current]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    available.append(neighbor)

        if len(order) != len(self.operations):
            raise ValueError("Circular dependency detected in operations")

        return order

    def _generate_random_splits(self, total_quantity: int) -> List[int]:
        """Generate even splits for an operation quantity."""
        max_splits = min(
            self.config.max_operation_splits,
            total_quantity // self.config.min_split_quantity,
        )

        # Ensure we have at least 2 splits, otherwise don't split
        if max_splits < 2:
            return [total_quantity]

        num_splits = random.randint(2, max_splits)

        # Calculate even splits
        base_quantity = total_quantity // num_splits
        remainder = total_quantity % num_splits

        # Create even splits with remainder distributed
        splits = []
        for i in range(num_splits):
            # Distribute remainder among first operations
            quantity = base_quantity + (1 if i < remainder else 0)
            splits.append(quantity)

        # Ensure no split is below minimum
        if any(s < self.config.min_split_quantity for s in splits):
            # If even splits would be too small, reduce number of splits
            return self._generate_random_splits_fallback(total_quantity)

        return splits

    def _generate_random_splits_fallback(self, total_quantity: int) -> List[int]:
        """Fallback to fewer splits when even splits would be too small."""
        # Find maximum splits that meet minimum quantity requirement
        max_viable_splits = total_quantity // self.config.min_split_quantity

        # Need at least 2 splits to make sense
        if max_viable_splits < 2:
            return [total_quantity]

        num_splits = random.randint(
            2, min(max_viable_splits, self.config.max_operation_splits)
        )

        # Calculate even splits
        base_quantity = total_quantity // num_splits
        remainder = total_quantity % num_splits

        splits = []
        for i in range(num_splits):
            quantity = base_quantity + (1 if i < remainder else 0)
            splits.append(quantity)

        return splits

    def _evaluate_chromosome(self, chromosome: SchedulingChromosome) -> Schedule:
        """Evaluate a chromosome by running the scheduler with its decisions."""
        # Create a fresh copy of the resource manager
        if self.original_resource_manager is None:
            raise RuntimeError("Original resource manager not initialized")
        self.resource_manager = ResourceManager(
            self.original_resource_manager.original_resources
        )

        # Apply chromosome decisions to create modified operations
        modified_operations = self._apply_chromosome_to_operations(chromosome)

        # Run the scheduler
        schedule = SchedulerService.schedule(modified_operations, self.resource_manager)

        # Fix the operation counting: we need to count against original operations, not split instances
        original_operations_scheduled = self._count_original_operations_scheduled(
            schedule, chromosome
        )

        # Create a corrected schedule with proper operation counting
        corrected_schedule = Schedule(
            fitness_score=schedule.fitness_score,
            tasks=schedule.tasks,
            number_of_operations_scheduled=original_operations_scheduled,
            number_of_resources_scheduled=schedule.number_of_resources_scheduled,
            makespan=schedule.makespan,
            dropped_operations=schedule.dropped_operations,
            idles=schedule.idles,
        )

        return corrected_schedule

    def _apply_chromosome_to_operations(
        self, chromosome: SchedulingChromosome
    ) -> List[OperationNode]:
        """Apply chromosome decisions to create the actual operations list for scheduling."""
        modified_operations = []

        # Process operations grouped by manufacturing order sequence
        for mo_id in chromosome.mo_order:
            # Determine a valid internal order for this MO
            op_indices = self._topological_order_for_mo(mo_id)
            for op_index in op_indices:
                original_op = self.operations[op_index]

                # Check if this operation should be split
                if op_index in chromosome.operation_splits:
                    split_quantities = chromosome.operation_splits[op_index]

                    # Create multiple operations for each split
                    for i, quantity in enumerate(split_quantities):
                        split_op = deepcopy(original_op)
                        split_op.quantity = quantity
                        split_op.operation_instance_id = f"{original_op.operation_instance_id or original_op.operation_id}_split_{i+1}"

                        # Update dependencies to point to split operations
                        split_op.dependencies = self._map_dependencies_to_splits(
                            original_op.dependencies, chromosome, modified_operations
                        )

                        modified_operations.append(split_op)
                else:
                    # Single operation (no split)
                    modified_op = deepcopy(original_op)

                    # Update dependencies to point to split operations
                    modified_op.dependencies = self._map_dependencies_to_splits(
                        original_op.dependencies, chromosome, modified_operations
                    )

                    modified_operations.append(modified_op)

        return modified_operations

    def _count_original_operations_scheduled(
        self, schedule: Schedule, chromosome: SchedulingChromosome
    ) -> int:
        """Count how many original operations (before splitting) were successfully scheduled."""
        if not schedule.tasks:
            return 0

        # Track which original operation instances (per MO) have at least one scheduled task
        scheduled_original_ops: set[tuple[str, str]] = set()

        for task in schedule.tasks:
            mo_id = task.operation.manufacturing_order_id
            op_id = task.operation.operation_id
            if mo_id and op_id:
                scheduled_original_ops.add((mo_id, op_id))

        return len(scheduled_original_ops)

    def _unique_operation_keys_from_operations(self) -> set[tuple[str, str]]:
        """Return unique (manufacturing_order_id, operation_id) pairs from original operations list."""
        unique_keys: set[tuple[str, str]] = set()
        for op in self.operations:
            if op.manufacturing_order_id and op.operation_id:
                unique_keys.add((op.manufacturing_order_id, op.operation_id))
        return unique_keys

    @staticmethod
    def _unique_operation_keys_from_schedule(
        schedule: Schedule,
    ) -> set[tuple[str, str]]:
        """Return unique (manufacturing_order_id, operation_id) pairs that were scheduled."""
        unique_keys: set[tuple[str, str]] = set()
        for task in schedule.tasks:
            mo_id = task.operation.manufacturing_order_id
            op_id = task.operation.operation_id
            if mo_id and op_id:
                unique_keys.add((mo_id, op_id))
        return unique_keys

    @staticmethod
    def _unique_execution_units_scheduled(schedule: Schedule) -> int:
        """Return number of distinct scheduled execution units (unique operation_instance_id values)."""
        if not schedule.tasks:
            return 0
        instance_ids = {
            task.operation_instance_id
            for task in schedule.tasks
            if task.operation_instance_id
        }
        return len(instance_ids)

    def _map_dependencies_to_splits(
        self,
        original_dependencies: List[OperationNode],
        chromosome: SchedulingChromosome,
        processed_operations: List[OperationNode],
    ) -> List[OperationNode]:
        """Map original dependencies to their split operations in the processed list."""
        mapped_dependencies = []

        for dep in original_dependencies:
            # Find all operations in processed_operations that came from this dependency
            split_ops = [
                op
                for op in processed_operations
                if (
                    op.operation_id == dep.operation_id
                    and op.manufacturing_order_id == dep.manufacturing_order_id
                )
            ]

            if split_ops:
                # If the dependency was split, depend on the last split
                mapped_dependencies.append(split_ops[-1])
            else:
                # Dependency not processed yet - this indicates a topological order violation
                # Find the original operation in self.operations and get its index
                dep_index = None
                for i, original_op in enumerate(self.operations):
                    if (
                        original_op.operation_id == dep.operation_id
                        and original_op.manufacturing_order_id
                        == dep.manufacturing_order_id
                    ):
                        dep_index = i
                        break

                if dep_index is not None:
                    # The dependency should have been processed earlier
                    # This indicates a topological order violation - skip this dependency

                    continue
                else:
                    # Keep original dependency as fallback (shouldn't happen)
                    mapped_dependencies.append(dep)

        return mapped_dependencies

    def _calculate_fitness(
        self, schedule: Schedule, chromosome: SchedulingChromosome
    ) -> float:
        """
        Calculate single fitness score: lower is better.

        fitness = total_makespan + weighted_tardiness

        Returns:
            Fitness score (lower is better)
        """
        # If no tasks were scheduled, return extremely high penalty
        if not schedule.tasks:
            return 50000.0  # Extremely high penalty for completely failed schedules

        # Heavy penalty for dropped operations
        dropped_penalty = 0.0
        if schedule.dropped_operations:
            dropped_count = len(schedule.dropped_operations)
            dropped_penalty = (
                dropped_count * 1000.0
            )  # 1000 penalty per dropped operation

        # Group tasks by manufacturing order and find completion times
        mo_completion_times: Dict[str, datetime] = {}
        mo_makespans: Dict[str, float] = {}

        for task in schedule.tasks:
            mo_id = task.operation.manufacturing_order_id
            if mo_id is None:
                continue  # Skip tasks without manufacturing order ID
            task_end = task.datetime_end

            # Track the latest completion time for each manufacturing order
            if (
                mo_id not in mo_completion_times
                or task_end > mo_completion_times[mo_id]
            ):
                mo_completion_times[mo_id] = task_end

        # Calculate makespan for each manufacturing order (from schedule start to MO completion)
        if schedule.tasks:
            schedule_start = min(task.datetime_start for task in schedule.tasks)
            for mo_id, completion_time in mo_completion_times.items():
                makespan_hours = (
                    completion_time - schedule_start
                ).total_seconds() / 3600
                mo_makespans[mo_id] = makespan_hours

        # Calculate total makespan across all manufacturing orders
        total_makespan = sum(mo_makespans.values()) if mo_makespans else 0.0

        # Calculate weighted tardiness
        weighted_tardiness = 0.0
        priority_weights = {
            Priority.LOW: 1.0,
            Priority.MEDIUM: 2.0,
            Priority.HIGH: 4.0,
            Priority.URGENT: 8.0,
        }

        for mo_id, completion_time in mo_completion_times.items():
            if mo_id in self.manufacturing_orders:
                mo = self.manufacturing_orders[mo_id]

                # Skip if no required_by_date is set
                if mo.required_by_date is None:
                    continue

                # Calculate tardiness (0 if on time, positive if late)
                tardiness_hours = max(
                    0, (completion_time - mo.required_by_date).total_seconds() / 3600
                )

                # Weight by priority (default to LOW if None)
                priority = mo.priority or Priority.LOW
                weight = priority_weights.get(priority, 1.0)
                weighted_tardiness += tardiness_hours * weight

        # Final fitness: lower is better
        fitness = total_makespan + weighted_tardiness + dropped_penalty

        return fitness

    def _create_next_generation(
        self, population: List[SchedulingChromosome], fitness_scores: List[float]
    ) -> List[SchedulingChromosome]:
        """Create the next generation using selection, crossover, and mutation."""
        next_generation = []

        # Elite selection - preserve best individuals (lowest fitness scores)
        elite_indices = sorted(
            range(len(fitness_scores)), key=lambda i: fitness_scores[i], reverse=False
        )
        # Ensure elite size doesn't exceed population size
        actual_elite_size = min(self.config.elite_size, len(population))
        for i in range(actual_elite_size):
            next_generation.append(deepcopy(population[elite_indices[i]]))

        # Generate the rest through crossover and mutation
        while len(next_generation) < self.config.population_size:
            # Tournament selection for parents
            parent1 = self._tournament_selection(population, fitness_scores)
            parent2 = self._tournament_selection(population, fitness_scores)

            # Crossover
            if random.random() < self.config.crossover_rate:
                child1, child2 = self._crossover(parent1, parent2)
            else:
                child1, child2 = deepcopy(parent1), deepcopy(parent2)

            # Mutation
            if random.random() < self.config.mutation_rate:
                child1 = self._mutate(child1)
            if random.random() < self.config.mutation_rate:
                child2 = self._mutate(child2)

            next_generation.extend([child1, child2])

        # Trim to exact population size
        return next_generation[: self.config.population_size]

    def _tournament_selection(
        self, population: List[SchedulingChromosome], fitness_scores: List[float]
    ) -> SchedulingChromosome:
        """Select an individual using tournament selection (lower fitness is better)."""
        tournament_indices = random.sample(
            range(len(population)), self.config.tournament_size
        )
        best_index = min(tournament_indices, key=lambda i: fitness_scores[i])
        return population[best_index]

    def _crossover(
        self, parent1: SchedulingChromosome, parent2: SchedulingChromosome
    ) -> Tuple[SchedulingChromosome, SchedulingChromosome]:
        """Perform crossover between two parents to create offspring."""
        # Order crossover for MO order
        child1_mo_order = self._order_crossover(parent1.mo_order, parent2.mo_order)
        child2_mo_order = self._order_crossover(parent2.mo_order, parent1.mo_order)

        # Uniform crossover for operation splits
        child1_splits = {}
        child2_splits = {}
        for op_index in set(parent1.operation_splits.keys()) | set(
            parent2.operation_splits.keys()
        ):
            if random.random() < 0.5:
                if op_index in parent1.operation_splits:
                    child1_splits[op_index] = parent1.operation_splits[op_index].copy()
                if op_index in parent2.operation_splits:
                    child2_splits[op_index] = parent2.operation_splits[op_index].copy()
            else:
                if op_index in parent2.operation_splits:
                    child1_splits[op_index] = parent2.operation_splits[op_index].copy()
                if op_index in parent1.operation_splits:
                    child2_splits[op_index] = parent1.operation_splits[op_index].copy()

        child1 = SchedulingChromosome(child1_mo_order, child1_splits)
        child2 = SchedulingChromosome(child2_mo_order, child2_splits)

        return child1, child2

    def _order_crossover(
        self, parent1_order: List[Any], parent2_order: List[Any]
    ) -> List[Any]:
        """Perform order crossover (OX) for sequence order."""
        size = len(parent1_order)

        # Select random crossover points
        start, end = sorted(random.sample(range(size), 2))

        # Initialize child with None values
        child: List[Optional[Any]] = [None] * size

        # Copy segment from parent1
        child[start:end] = parent1_order[start:end]
        copied_elements = set(parent1_order[start:end])

        # Create list of remaining elements from parent2 in order (excluding copied ones)
        remaining = [x for x in parent2_order if x not in copied_elements]

        # Fill remaining positions with elements from parent2 in order
        remaining_idx = 0

        # Fill positions before start
        for i in range(start):
            if remaining_idx < len(remaining):
                child[i] = remaining[remaining_idx]
                remaining_idx += 1

        # Fill positions after end
        for i in range(end, size):
            if remaining_idx < len(remaining):
                child[i] = remaining[remaining_idx]
                remaining_idx += 1

        # Safety check: ensure no None values remain
        if None in child:
            # Fallback: if we have issues, just return parent1_order

            return parent1_order.copy()

        # Convert to List[int] and validate topological order
        child_seq: List[Any] = [x for x in child if x is not None]
        return child_seq

    def _ensure_topological_order(self, order: List[int]) -> List[int]:
        """Ensure the operation order respects dependencies. (Used in legacy paths)"""
        # Build dependency map using operation indices
        order_set = set(order)  # Operations that should be in the final order

        # Create mapping from operation to its dependencies (by index)
        dependency_map = {}
        for op_index in order:
            operation = self.operations[op_index]
            deps = []
            for dep in operation.dependencies:
                # Find the index of the dependency in our operations list
                for i, orig_op in enumerate(self.operations):
                    if (
                        orig_op.operation_id == dep.operation_id
                        and orig_op.manufacturing_order_id == dep.manufacturing_order_id
                        and i in order_set
                    ):
                        deps.append(i)
                        break
            dependency_map[op_index] = deps

        # Kahn's algorithm for topological sorting
        # Calculate in-degrees
        in_degree = {op_index: 0 for op_index in order}
        for op_index in order:
            for dep_index in dependency_map[op_index]:
                if dep_index in in_degree:
                    in_degree[op_index] += 1

        # Find all operations with no dependencies
        queue = [op_index for op_index in order if in_degree[op_index] == 0]
        corrected_order = []

        while queue:
            # Remove a node with no incoming edges
            current = queue.pop(0)
            corrected_order.append(current)

            # For each operation that depends on the current one
            for op_index in order:
                if current in dependency_map[op_index]:
                    in_degree[op_index] -= 1
                    if in_degree[op_index] == 0:
                        queue.append(op_index)

        # Check for cycles
        if len(corrected_order) != len(order):

            # Return operations in dependency-safe order based on original operations list
            safe_order = []
            for op_index in range(len(self.operations)):
                if op_index in order_set:
                    safe_order.append(op_index)
            return safe_order

        return corrected_order

    def _mutate(self, chromosome: SchedulingChromosome) -> SchedulingChromosome:
        """Apply mutation to a chromosome."""
        mutated = deepcopy(chromosome)

        # Mutate MO order (swap two random MOs)
        if len(mutated.mo_order) > 1 and random.random() < 0.3:
            idx1, idx2 = random.sample(range(len(mutated.mo_order)), 2)
            mutated.mo_order[idx1], mutated.mo_order[idx2] = (
                mutated.mo_order[idx2],
                mutated.mo_order[idx1],
            )

        # Mutate operation splits
        for op_index in list(mutated.operation_splits.keys()):
            if random.random() < 0.1:  # 10% chance to mutate each split
                operation = self.operations[op_index]
                new_splits = self._generate_random_splits(operation.quantity)
                mutated.operation_splits[op_index] = new_splits

        # Randomly add new splits
        for i, operation in enumerate(self.operations):
            if (
                i not in mutated.operation_splits
                and operation.quantity > self.config.min_split_quantity * 2
                and random.random() < 0.05
            ):  # 5% chance to add new split
                splits = self._generate_random_splits(operation.quantity)
                if len(splits) > 1:
                    mutated.operation_splits[i] = splits

        return mutated

    def _can_swap_operations(self, order: List[int], idx1: int, idx2: int) -> bool:
        """Check if two operations can be swapped without violating dependencies."""
        op1_index = order[idx1]
        op2_index = order[idx2]
        op1 = self.operations[op1_index]
        op2 = self.operations[op2_index]

        # Check direct dependencies using operation_id and manufacturing_order_id
        # Check if op1 depends on op2
        for dep in op1.dependencies:
            if (
                dep.operation_id == op2.operation_id
                and dep.manufacturing_order_id == op2.manufacturing_order_id
            ):
                return False

        # Check if op2 depends on op1
        for dep in op2.dependencies:
            if (
                dep.operation_id == op1.operation_id
                and dep.manufacturing_order_id == op1.manufacturing_order_id
            ):
                return False

        # Check transitive dependencies by ensuring swapping doesn't violate topological order
        test_order = order.copy()
        test_order[idx1], test_order[idx2] = test_order[idx2], test_order[idx1]
        corrected_order = self._ensure_topological_order(test_order)

        # If the corrected order is different from our test order, swapping would violate dependencies
        return corrected_order == test_order


def optimize_schedule(
    operations: List[OperationNode],
    resource_manager: ResourceManager,
    manufacturing_orders: List[ManufacturingOrder],
    config: Optional[GeneticAlgorithmConfig] = None,
    show_schedule: bool = False,
    enable_evolution_logging: bool = True,
    log_level: str = "INFO",
) -> Tuple[Schedule, SchedulingChromosome, "GeneticSchedulerOptimizer"]:
    """
    Convenience function to optimize a schedule using genetic algorithm.

    Args:
        operations: List of operations to schedule
        resource_manager: Resource manager with available resources
        manufacturing_orders: List of manufacturing orders with priorities and due dates
        config: Genetic algorithm configuration (optional)
        show_schedule: Whether to display the final schedule table (optional)
        enable_evolution_logging: Whether to enable logging to see evolution progress (optional)
        log_level: Logging level for evolution tracking ("DEBUG", "INFO", "WARNING", "ERROR")

    Returns:
        Tuple of (best_schedule, best_chromosome, optimizer)
        The optimizer contains generation_history for displaying evolution if desired.
    """
    # Set up logger for evolution logging if requested
    logger = None
    if enable_evolution_logging:
        from scheduler.utils.genetic_algorithm_logger import GeneticAlgorithmLogger

        logger = GeneticAlgorithmLogger()
        logger.configure_evolution_logging(log_level)

    optimizer = GeneticSchedulerOptimizer(config)
    best_schedule, best_chromosome = optimizer.optimize(
        operations, resource_manager, manufacturing_orders, logger
    )

    # Optionally display the schedule table
    if show_schedule:
        from scheduler.utils.schedule_logger import ScheduleLogger

        ScheduleLogger.print(best_schedule, title="Genetic Algorithm Result")

    return best_schedule, best_chromosome, optimizer
