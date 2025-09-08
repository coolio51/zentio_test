from typing import List, Dict, Tuple, Optional, TYPE_CHECKING
from rich import box, table
from rich.console import Console
from rich.panel import Panel
import logging
import multiprocessing as mp

from scheduler.models import Schedule, OperationNode, ManufacturingOrder
from scheduler.utils.schedule_logger import ScheduleLogger

if TYPE_CHECKING:
    from scheduler.services.genetic_optimizer import (
        SchedulingChromosome,
        GeneticAlgorithmConfig,
    )


class GeneticAlgorithmLogger:
    """Logger for genetic algorithm optimization results and chromosome details."""

    @staticmethod
    def print_optimization_config(config: "GeneticAlgorithmConfig", console: Console):
        """Print genetic algorithm configuration details."""
        console.print(
            f"\n[bold yellow]⚙️  Genetic Algorithm Configuration[/bold yellow]"
        )
        console.print(f"Population size: {config.population_size}")
        console.print(f"Max generations: {config.max_generations}")
        console.print(f"Crossover rate: {config.crossover_rate}")
        console.print(f"Mutation rate: {config.mutation_rate}")
        console.print(f"Elite size: {config.elite_size}")
        console.print(f"Max operation splits: {config.max_operation_splits}")
        console.print(f"Min split quantity: {config.min_split_quantity}")
        console.print(f"Optimization objectives: {len(config.optimization_objectives)}")
        for obj in config.optimization_objectives:
            console.print(f"  • {obj.value}")

    @staticmethod
    def print_optimization_summary(
        fitness_score: float,
        chromosome: "SchedulingChromosome",
        optimization_time: float,
        console: Console,
    ):
        """Print optimization summary with key metrics."""
        console.print(f"\n[bold green]✅ Optimization Complete![/bold green]")
        console.print(f"Optimization time: {optimization_time:.2f} seconds")
        console.print(
            f"[bold cyan]🎯 Final Fitness Score: {fitness_score:.2f}[/bold cyan] (lower is better)"
        )
        console.print(f"Best chromosome splits: {len(chromosome.operation_splits)}")
        # Machine assignments removed from DNA; scheduler selects machines

    @staticmethod
    def print_chromosome_details(
        chromosome: "SchedulingChromosome",
        operations: List[OperationNode],
        console: Console,
        show_operation_order: bool = True,
    ):
        """Print detailed chromosome information."""
        console.print(f"\n[bold blue]🧬 Best Chromosome Details[/bold blue]")

        if show_operation_order:
            console.print(f"MO order: {getattr(chromosome, 'mo_order', [])}")

        # Operation splits
        if chromosome.operation_splits:
            console.print(f"\n[bold magenta]✂️  Operation Splits:[/bold magenta]")
            for op_index, splits in chromosome.operation_splits.items():
                operation = operations[op_index]
                console.print(
                    f"  {operation.manufacturing_order_name} {operation.operation_id} → {splits} (total: {sum(splits)})"
                )
        else:
            console.print(
                "[bold magenta]✂️  No operation splits made by GA[/bold magenta]"
            )

    @staticmethod
    def print_operation_splits_with_verification(
        chromosome: "SchedulingChromosome",
        operations: List[OperationNode],
        console: Console,
    ):
        """Print operation splits with detailed verification."""
        if not chromosome.operation_splits:
            console.print(
                "[bold magenta]✂️  No operation splits made by GA[/bold magenta]"
            )
            return

        console.print("\n[cyan]✂️ Operation Splits:[/cyan]")
        for op_index, splits in chromosome.operation_splits.items():
            operation = operations[op_index]
            total_check = sum(splits)
            console.print(
                f"  • {operation.operation_id}: {operation.quantity} items → {len(splits)} batches {splits}"
            )
            console.print(
                f"    [dim]Verification: {' + '.join(map(str, splits))} = {total_check} ✓[/dim]"
            )

    @staticmethod
    def print_operation_execution_order(
        chromosome: "SchedulingChromosome",
        operations: List[OperationNode],
        console: Console,
    ):
        """Print the manufacturing order execution sequence."""
        console.print(f"\n[cyan]📋 Manufacturing Order Sequence:[/cyan]")
        mo_order = getattr(chromosome, "mo_order", [])
        for i, mo_id in enumerate(mo_order):
            console.print(f"  {i+1}. {mo_id}")

    @staticmethod
    def print_parallel_execution_analysis(
        chromosome: "SchedulingChromosome",
        operations: List[OperationNode],
        console: Console,
    ):
        """Analyze and print parallel execution potential."""
        console.print(f"\n[bold green]🏃‍♂️ Parallel Execution Analysis:[/bold green]")

        if len(chromosome.operation_splits) == 0:
            console.print(
                "❌ No operations with splits found - limited parallel potential"
            )
            return

        console.print(
            "✅ Operations with splits found - checking for parallel potential..."
        )

        split_ops = [
            (op_index, operations[op_index])
            for op_index in chromosome.operation_splits.keys()
        ]

        for op_index, operation in split_ops:
            splits = chromosome.operation_splits[op_index]
            console.print(
                f"  📋 [{op_index}] {operation.manufacturing_order_name} {operation.operation_id}: {len(splits)} splits (machine selection by scheduler)"
            )

    @staticmethod
    def print_comparison_table(
        standard_schedule: Schedule, ga_schedule: Schedule, console: Console
    ) -> table.Table:
        """Create a comparison table between standard and GA schedules."""
        comparison_table = table.Table(
            title="📊 Schedule Comparison: Standard vs Genetic Algorithm",
            title_style="bold cyan",
            box=box.ROUNDED,
        )

        comparison_table.add_column("Metric", style="bold")
        comparison_table.add_column("Standard Scheduler", justify="center")
        comparison_table.add_column("Genetic Algorithm", justify="center")
        comparison_table.add_column("Improvement", justify="center")

        # Makespan comparison
        if standard_schedule.makespan and ga_schedule.makespan:
            standard_hours = standard_schedule.makespan.total_seconds() / 3600
            ga_hours = ga_schedule.makespan.total_seconds() / 3600
            improvement = ((standard_hours - ga_hours) / standard_hours) * 100
            status = (
                "✅ Better"
                if improvement > 0
                else "🔄 Same" if improvement == 0 else "❌ Worse"
            )

            comparison_table.add_row(
                "Makespan",
                f"{standard_hours:.2f}h",
                f"{ga_hours:.2f}h",
                f"{improvement:+.1f}% {status}",
            )

        # Scheduled operations
        standard_scheduled = len(standard_schedule.tasks)
        ga_scheduled = len(ga_schedule.tasks)
        scheduled_diff = ga_scheduled - standard_scheduled
        scheduled_status = (
            "✅ Better"
            if scheduled_diff > 0
            else "🔄 Same" if scheduled_diff == 0 else "❌ Worse"
        )

        comparison_table.add_row(
            "Scheduled Operations",
            str(standard_scheduled),
            str(ga_scheduled),
            f"{scheduled_diff:+d} {scheduled_status}",
        )

        # Dropped operations
        standard_dropped = (
            len(standard_schedule.dropped_operations)
            if standard_schedule.dropped_operations
            else 0
        )
        ga_dropped = (
            len(ga_schedule.dropped_operations) if ga_schedule.dropped_operations else 0
        )
        dropped_diff = standard_dropped - ga_dropped  # Reduction is good
        dropped_status = (
            "✅ Better"
            if dropped_diff > 0
            else "🔄 Same" if dropped_diff == 0 else "❌ Worse"
        )

        comparison_table.add_row(
            "Dropped Operations",
            str(standard_dropped),
            str(ga_dropped),
            f"{dropped_diff:+d} {dropped_status}",
        )

        return comparison_table

    @staticmethod
    def print_genetic_decisions_summary(
        chromosome: "SchedulingChromosome",
        operations: List[OperationNode],
        console: Console,
    ):
        """Print a summary of genetic algorithm decisions."""
        if not chromosome.operation_splits:
            console.print(
                "\n[bold magenta]🧬 No genetic algorithm decisions made[/bold magenta]"
            )
            return

        console.print("\n[bold magenta]🧬 GENETIC ALGORITHM DECISIONS[/bold magenta]")

        if chromosome.operation_splits:
            GeneticAlgorithmLogger.print_operation_splits_with_verification(
                chromosome, operations, console
            )

    @staticmethod
    def print_complete_results(
        standard_schedule: Schedule,
        ga_schedule: Schedule,
        chromosome: "SchedulingChromosome",
        operations: List[OperationNode],
        fitness_score: float,
        optimization_time: float,
        console: Console,
        show_schedules: bool = True,
    ):
        """Print complete genetic algorithm optimization results."""
        # Optimization summary
        GeneticAlgorithmLogger.print_optimization_summary(
            fitness_score, chromosome, optimization_time, console
        )

        # Comparison table
        comparison_table = GeneticAlgorithmLogger.print_comparison_table(
            standard_schedule, ga_schedule, console
        )
        console.print(comparison_table)

        # Genetic decisions
        GeneticAlgorithmLogger.print_genetic_decisions_summary(
            chromosome, operations, console
        )

        # Operation execution order
        GeneticAlgorithmLogger.print_operation_execution_order(
            chromosome, operations, console
        )

        # Parallel execution analysis
        GeneticAlgorithmLogger.print_parallel_execution_analysis(
            chromosome, operations, console
        )

        # Detailed chromosome info
        GeneticAlgorithmLogger.print_chromosome_details(
            chromosome, operations, console, show_operation_order=False
        )

        if show_schedules:
            # Display the final optimized schedule table
            console.print("\n[bold green]📋 FINAL OPTIMIZED SCHEDULE[/bold green]")
            ScheduleLogger.print(
                ga_schedule, title="🧬 Genetic Algorithm Optimized Schedule"
            )

            # Also show the standard schedule for comparison
            console.print(
                "\n[bold yellow]📋 STANDARD SCHEDULE (BASELINE)[/bold yellow]"
            )
            ScheduleLogger.print(
                standard_schedule, title="📊 Standard Scheduler Baseline"
            )

    @staticmethod
    def print_generation_history(
        generation_history: List[Dict], console: Optional[Console] = None
    ) -> None:
        """Print a detailed table showing the evolution across generations."""
        if console is None:
            console = Console()

        try:
            from rich.table import Table
            from rich.text import Text

            # Create table
            table = Table(
                title="🧬 Genetic Algorithm Evolution History",
                show_header=True,
                header_style="bold magenta",
            )

            # Add columns
            table.add_column("Gen", style="cyan", width=4)
            table.add_column("Current Best", style="green", width=12)
            table.add_column("Global Best", style="bold green", width=12)
            table.add_column("Average", style="yellow", width=12)
            table.add_column("Worst", style="red", width=12)
            table.add_column("Stagnation", style="yellow", width=10)
            table.add_column("Splits", style="blue", width=7)
            # Removed machines column (machine selection is scheduler-driven)
            table.add_column("Improvement", style="bold", width=12)

            # Add rows
            for data in generation_history:
                # Format improvement indicator
                if data["new_best"]:
                    improvement = Text("🎯 NEW BEST", style="bold green")
                elif data["stagnation"] == 0:
                    improvement = Text("✅ Progress", style="green")
                else:
                    improvement = Text(f"⏳ Stagnant", style="dim")

                # Format stagnation counter with color coding
                stagnation_text = str(data["stagnation"])
                if data["stagnation"] > 10:
                    stagnation_style = "bold red"
                elif data["stagnation"] > 5:
                    stagnation_style = "red"
                elif data["stagnation"] > 0:
                    stagnation_style = "yellow"
                else:
                    stagnation_style = "green"

                table.add_row(
                    str(data["generation"]),
                    f"{data['best_fitness']:.2f}",
                    f"{data['global_best_fitness']:.2f}",
                    f"{data['avg_fitness']:.2f}",
                    f"{data['worst_fitness']:.2f}",
                    Text(stagnation_text, style=stagnation_style),
                    str(data["splits_count"]),
                    improvement,
                )

            console.print("\n")
            console.print(table)

            # Print summary statistics
            if generation_history:
                initial_best = generation_history[0]["best_fitness"]
                final_best = generation_history[-1]["global_best_fitness"]
                improvement_pct = (
                    ((initial_best - final_best) / initial_best) * 100
                    if initial_best > 0
                    else 0
                )

                console.print(f"\n[bold green]📊 Evolution Summary:[/bold green]")
                console.print(f"  Initial Best Fitness: {initial_best:.2f}")
                console.print(f"  Final Best Fitness: {final_best:.2f}")
                console.print(f"  Total Improvement: {improvement_pct:.1f}%")
                console.print(f"  Generations Run: {len(generation_history)}")

                # Count how many times we found new best
                new_best_count = sum(
                    1 for data in generation_history if data["new_best"]
                )
                console.print(f"  New Best Found: {new_best_count} times")

        except ImportError:
            # Fallback if rich is not available
            print("\n🧬 Generation History (rich not available):")
            print("Gen | Current Best | Global Best | Average | Stagnation")
            print("-" * 60)
            for data in generation_history:
                new_indicator = " 🎯" if data["new_best"] else ""
                print(
                    f"{data['generation']:3d} | {data['best_fitness']:10.2f} | {data['global_best_fitness']:9.2f} | {data['avg_fitness']:7.2f} | {data['stagnation']:8d}{new_indicator}"
                )

    def configure_evolution_logging(self, level: str = "INFO") -> None:
        """Configure logging to see genetic algorithm evolution progress.

        Args:
            level: Logging level ("DEBUG", "INFO", "WARNING", "ERROR")
        """
        logging.basicConfig(
            level=getattr(logging, level.upper()),
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[logging.StreamHandler()],
        )
        # Set the genetic optimizer logger level specifically
        self.logger = logging.getLogger("scheduler.services.genetic_optimizer")
        self.logger.setLevel(getattr(logging, level.upper()))

    def log_optimization_start(
        self, num_operations: int, config: "GeneticAlgorithmConfig"
    ) -> None:
        """Log the start of genetic algorithm optimization."""
        if not hasattr(self, "logger"):
            self.logger = logging.getLogger("scheduler.services.genetic_optimizer")

        self.logger.info(
            f"🧬 Starting genetic algorithm optimization with {num_operations} operations"
        )
        self.logger.info(
            f"   Population size: {config.population_size}, Max generations: {config.max_generations}"
        )
        self.logger.info(
            f"   Using {config.parallel_operators or mp.cpu_count()} parallel operators"
        )
        self.logger.info(
            f"   Target: Schedule all {num_operations} operations with minimal makespan and no drops"
        )

    def log_generation_start(self, generation: int, max_generations: int) -> None:
        """Log the start of a generation."""
        if not hasattr(self, "logger"):
            self.logger = logging.getLogger("scheduler.services.genetic_optimizer")

        self.logger.debug(
            f"Generation {generation}/{max_generations}: Evaluating population..."
        )

    def log_new_best_found(
        self, generation: int, fitness: float, old_fitness: float
    ) -> None:
        """Log when a new best solution is found."""
        if not hasattr(self, "logger"):
            self.logger = logging.getLogger("scheduler.services.genetic_optimizer")

        improvement = (
            ((old_fitness - fitness) / old_fitness * 100) if old_fitness > 0 else 0
        )
        self.logger.info(
            f"🎯 NEW BEST found in generation {generation}! "
            f"Fitness: {fitness:.2f} (improved by {improvement:.1f}%)"
        )

    def log_generation_summary(
        self,
        generation: int,
        current_best: float,
        global_best: float,
        avg_fitness: float,
        splits_count: int,
        stagnation_counter: int,
        found_new_best: bool,
        scheduled_ops: int = 0,
        total_ops: int = 0,
        dropped_ops: int = 0,
        mo_count: int = 0,
        execution_units_scheduled: int = 0,
    ) -> None:
        """Log generation summary statistics."""
        if not hasattr(self, "logger"):
            self.logger = logging.getLogger("scheduler.services.genetic_optimizer")

        # Log generation summary (every generation for first 10, then every 10th)
        if generation <= 10 or generation % 10 == 0 or found_new_best:
            stagnation_status = (
                "🔥 Active"
                if stagnation_counter == 0
                else f"⏳ Stagnant ({stagnation_counter})"
            )

            # Calculate success rate
            success_rate = (scheduled_ops / total_ops * 100) if total_ops > 0 else 0

            self.logger.info(
                f"📊 Gen {generation:3d}: Best={current_best:.2f} "
                f"Global={global_best:.2f} Avg={avg_fitness:.2f} "
                f"Ops={scheduled_ops}/{total_ops} ({success_rate:.1f}%) "
                f"Units={execution_units_scheduled} Dropped={dropped_ops} MOs={mo_count} "
                f"Splits={splits_count} {stagnation_status}"
            )

    def log_target_fitness_reached(self, target_fitness: float) -> None:
        """Log when target fitness is reached."""
        if not hasattr(self, "logger"):
            self.logger = logging.getLogger("scheduler.services.genetic_optimizer")

        self.logger.info(
            f"🎯 Target fitness {target_fitness} reached! Stopping optimization."
        )

    def log_stagnation_stop(self, stagnation_counter: int) -> None:
        """Log when stopping due to stagnation."""
        if not hasattr(self, "logger"):
            self.logger = logging.getLogger("scheduler.services.genetic_optimizer")

        self.logger.info(
            f"⏹️  Stopping: No improvement for {stagnation_counter} generations."
        )

    def log_optimization_complete(
        self,
        generations_run: int,
        final_fitness: float,
        initial_fitness: float,
        best_chromosome: "SchedulingChromosome",
        scheduled_ops: int = 0,
        total_ops: int = 0,
        dropped_ops: int = 0,
        mo_count: int = 0,
        best_schedule: Optional["Schedule"] = None,
    ) -> None:
        """Log optimization completion summary with detailed operation statistics."""
        if not hasattr(self, "logger"):
            self.logger = logging.getLogger("scheduler.services.genetic_optimizer")

        final_improvement = (
            ((initial_fitness - final_fitness) / initial_fitness * 100)
            if initial_fitness > 0
            else 0
        )
        success_rate = (scheduled_ops / total_ops * 100) if total_ops > 0 else 0

        self.logger.info(f"✅ Optimization complete! Ran {generations_run} generations")
        self.logger.info(
            f"   Final fitness: {final_fitness:.2f} (improved {final_improvement:.1f}% from start)"
        )
        self.logger.info(
            f"   Operations scheduled: {scheduled_ops}/{total_ops} ({success_rate:.1f}% success rate)"
        )
        if dropped_ops > 0:
            self.logger.info(
                f"   ⚠️  Operations dropped: {dropped_ops} ({dropped_ops/total_ops*100:.1f}%)"
            )
            self.logger.info(
                f"   💡 TIP: Check dropped operations for scheduling conflicts, resource constraints, or dependency issues"
            )
        else:
            self.logger.info(
                f"   🎉 No operations dropped! Perfect scheduling achieved"
            )
        self.logger.info(f"   Manufacturing orders: {mo_count} orders processed")
        self.logger.info(
            f"   Best solution has {len(best_chromosome.operation_splits)} splits and uses scheduler-driven machine selection"
        )

        # Log dropped operation breakdown by reason
        if best_schedule and best_schedule.dropped_operations:
            self.logger.info(f"   📊 Dropped Operations Breakdown:")
            drop_reasons = {}
            for dropped_op in best_schedule.dropped_operations:
                reason = dropped_op.reason.value
                drop_reasons[reason] = drop_reasons.get(reason, 0) + 1

            for reason, count in drop_reasons.items():
                percentage = (count / len(best_schedule.dropped_operations)) * 100
                self.logger.info(
                    f"      • {reason.replace('_', ' ').title()}: {count} ({percentage:.1f}%)"
                )

            # Show sample error messages for phase scheduling failures
            phase_failures = [
                op
                for op in best_schedule.dropped_operations
                if op.reason.value == "phase_scheduling_failed" and op.error_message
            ]
            if phase_failures:
                self.logger.info(f"   🔍 Sample scheduling failure reasons:")
                for i, failure in enumerate(
                    phase_failures[:3]
                ):  # Show first 3 examples
                    self.logger.info(
                        f"      • {failure.operation_id}: {failure.error_message}"
                    )
                if len(phase_failures) > 3:
                    self.logger.info(f"      • ... and {len(phase_failures) - 3} more")
