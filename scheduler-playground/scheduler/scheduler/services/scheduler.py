from datetime import datetime
from typing import Dict, List, Optional

from scheduler.models import (
    Schedule,
    Task,
    TaskPhase,
    OperationNode,
    DroppedOperation,
    DropReason,
    Idle,
)

from .resource_manager import ResourceManager
from .operation_scheduler import OperationScheduler
from rich.console import Console
from scheduler.utils.resource_logger import ResourceLogger


class SchedulerService:
    console = Console()

    @staticmethod
    def schedule(
        operations: list[OperationNode],
        resource_manager: ResourceManager,
    ) -> Schedule:
        tasks = []
        all_dropped_operations = []
        all_idles = []

        # Schedule all operations
        operation_tasks, operation_idles, dropped_operations = (
            SchedulerService._schedule_operations(operations, resource_manager)
        )
        tasks.extend(operation_tasks)
        all_idles.extend(operation_idles)
        all_dropped_operations.extend(dropped_operations)

        makespan = None
        if tasks:
            makespan = max(task.datetime_end for task in tasks) - min(
                task.datetime_start for task in tasks
            )

        # Calculate actual operations scheduled (operations that have at least one task)
        scheduled_operation_ids = set()
        for task in tasks:
            if task.operation.operation_id:
                scheduled_operation_ids.add(task.operation.operation_id)

        actual_operations_scheduled = len(scheduled_operation_ids)

        return Schedule(
            fitness_score=0.0,
            tasks=tasks,
            number_of_operations_scheduled=actual_operations_scheduled,
            number_of_resources_scheduled=len(resource_manager.resources),
            makespan=makespan,
            dropped_operations=(
                all_dropped_operations if all_dropped_operations else None
            ),
            idles=all_idles if all_idles else None,
        )

    @staticmethod
    def _schedule_operations(
        operations: list[OperationNode],
        resource_manager: ResourceManager,
    ) -> tuple[List[Task], List[Idle], List[DroppedOperation]]:
        """
        Schedule all operations, respecting dependencies.
        """
        tasks = []
        dropped_operations = []
        all_idles = []

        # Track completion times for each operation instance (unique_key -> end_time)
        operation_completion_times: Dict[str, datetime] = {}

        # Counter for generating unique operation instance IDs
        operation_instance_counter: Dict[str, int] = {}

        # Operations are provided directly

        # Keep track of which operation instances have been scheduled and dropped
        # Use object identity to distinguish between different instances of the same operation_id
        scheduled_operations = set()
        dropped_operation_ids = set()

        # Create operation scheduler instance
        operation_scheduler = OperationScheduler(resource_manager)

        # Schedule operations in dependency order
        while len(scheduled_operations) + len(dropped_operation_ids) < len(operations):
            progress_made = False

            for operation in operations:
                operation_instance_key = id(operation)
                if (
                    operation_instance_key in scheduled_operations
                    or operation_instance_key in dropped_operation_ids
                ):
                    continue

                # Check if all dependencies are scheduled (not dropped)
                dependencies_satisfied = True
                max_dependency_end_time = None
                dropped_due_to_dependency = False

                for dependency in operation.dependencies:
                    dependency_instance_key = id(dependency)
                    if dependency_instance_key in dropped_operation_ids:
                        # This operation must be dropped because its dependency was dropped
                        if (
                            operation_instance_key not in dropped_operation_ids
                        ):  # Prevent duplicate drops
                            # Removed verbose logging - use summary at the end instead
                            dropped_operations.append(
                                DroppedOperation(
                                    manufacturing_order_id=operation.manufacturing_order_id,
                                    operation_id=operation.operation_id,
                                    operation_name=operation.operation_name,
                                    reason=DropReason.DEPENDENCY_DROPPED,
                                    dependent_operation_id=dependency.operation_id,
                                )
                            )
                            dropped_operation_ids.add(operation_instance_key)
                        dropped_due_to_dependency = True
                        progress_made = True
                        break
                    elif dependency_instance_key not in scheduled_operations:
                        dependencies_satisfied = False
                        break

                    # Find the maximum end time among dependencies
                    # Use a unique key that combines operation_id and instance for completion times
                    dep_completion_key = f"{dependency.operation_id}_{id(dependency)}"
                    dep_end_time = operation_completion_times[dep_completion_key]
                    if (
                        max_dependency_end_time is None
                        or dep_end_time > max_dependency_end_time
                    ):
                        max_dependency_end_time = dep_end_time

                if dropped_due_to_dependency:
                    continue

                if dependencies_satisfied:

                    # Try to schedule this operation
                    operation_tasks, operation_idles, operation_dropped = (
                        operation_scheduler.schedule_operation(
                            operation,
                            earliest_start=max_dependency_end_time,
                            operation_instance_counter=operation_instance_counter,
                        )
                    )

                    if operation_dropped:
                        # Operation was dropped, add to dropped list
                        if (
                            operation_instance_key not in dropped_operation_ids
                        ):  # Prevent duplicate drops
                            # Removed verbose logging - use summary at the end instead
                            dropped_operations.append(operation_dropped)
                            dropped_operation_ids.add(operation_instance_key)
                        progress_made = True
                    elif operation_tasks:
                        # Operation was successfully scheduled

                        tasks.extend(operation_tasks)
                        all_idles.extend(operation_idles)
                        # Record the completion time of this operation instance
                        completion_key = f"{operation.operation_id}_{id(operation)}"
                        operation_completion_times[completion_key] = max(
                            task.datetime_end for task in operation_tasks
                        )
                        scheduled_operations.add(operation_instance_key)
                        progress_made = True

            if not progress_made:
                # This should not happen if the dependency graph is valid
                remaining_ops = [
                    op.operation_id
                    for op in operations
                    if id(op) not in scheduled_operations
                    and id(op) not in dropped_operation_ids
                ]
                SchedulerService.console.log(
                    f"Warning: Could not schedule operations due to circular dependencies or missing dependencies: {remaining_ops}"
                )
                # Drop the remaining operations with a circular dependency reason
                for op in operations:
                    if (
                        id(op) not in scheduled_operations
                        and id(op) not in dropped_operation_ids
                    ):
                        dropped_operations.append(
                            DroppedOperation(
                                manufacturing_order_id=op.manufacturing_order_id,
                                operation_id=op.operation_id,
                                operation_name=op.operation_name,
                                reason=DropReason.CIRCULAR_DEPENDENCY,
                            )
                        )
                        dropped_operation_ids.add(id(op))
                break

        return tasks, all_idles, dropped_operations
