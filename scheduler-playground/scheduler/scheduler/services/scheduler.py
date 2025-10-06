from __future__ import annotations

from datetime import datetime

from typing import Dict, List, Optional
import heapq

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
from scheduler.common.profiling import profile_function, profile_section
from scheduler.common.settings import get_scheduler_mode
from rich.console import Console
from scheduler.utils.resource_logger import ResourceLogger


class SchedulerService:
    console = Console()

    @staticmethod
    @profile_function()
    def schedule(
        operations: list[OperationNode],
        resource_manager: ResourceManager,
    ) -> Schedule:
        tasks: List[Task] = []
        all_dropped_operations: List[DroppedOperation] = []
        all_idles: List[Idle] = []

        # Schedule all operations
        operation_tasks, operation_idles, dropped_operations = (
            SchedulerService._schedule_operations(operations, resource_manager)
        )
        tasks.extend(operation_tasks)
        all_idles.extend(operation_idles)
        all_dropped_operations.extend(dropped_operations)

        makespan = None
        if tasks:
            earliest_start = min(task.datetime_start for task in tasks)
            latest_end = max(task.datetime_end for task in tasks)
            makespan = latest_end - earliest_start

        # Calculate actual operations scheduled (operations that have at least one task)
        scheduled_operation_ids = {
            task.operation.operation_id
            for task in tasks
            if task.operation.operation_id
        }

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
    @profile_function()
    def _schedule_operations(
        operations: list[OperationNode],
        resource_manager: ResourceManager,
    ) -> tuple[List[Task], List[Idle], List[DroppedOperation]]:
        mode = get_scheduler_mode()
        if mode == "naive":
            return SchedulerService._schedule_operations_naive(
                operations, resource_manager
            )
        return SchedulerService._schedule_operations_topological(
            operations, resource_manager
        )

    @staticmethod
    def _get_cached_operation_scheduler(
        resource_manager: ResourceManager,
    ) -> OperationScheduler:
        cached = getattr(resource_manager, "_operation_scheduler_cache", None)
        if cached is None or getattr(cached, "resource_manager", None) is not resource_manager:
            cached = OperationScheduler(resource_manager)
            setattr(resource_manager, "_operation_scheduler_cache", cached)
        else:
            cached.reset(resource_manager)
        return cached

    @staticmethod
    def _schedule_operations_naive(
        operations: list[OperationNode],
        resource_manager: ResourceManager,
    ) -> tuple[List[Task], List[Idle], List[DroppedOperation]]:
        """Naive O(n^2) scheduling loop retained for backward compatibility."""
        tasks: List[Task] = []
        dropped_operations: List[DroppedOperation] = []
        all_idles: List[Idle] = []

        # Track completion times for each operation instance (unique_key -> end_time)
        operation_completion_times: Dict[tuple[str, int], datetime] = {}

        # Counter for generating unique operation instance IDs
        operation_instance_counter: Dict[str, int] = {}

        # Operations are provided directly

        # Keep track of which operation instances have been scheduled and dropped
        # Use object identity to distinguish between different instances of the same operation_id
        scheduled_operations = set()
        dropped_operation_ids = set()

        dependency_cache: Dict[int, tuple[OperationNode, ...]] = {
            id(operation): tuple(operation.dependencies)
            for operation in operations
        }

        # Create operation scheduler instance
        operation_scheduler = OperationScheduler(resource_manager)

        # Schedule operations in dependency order
        while len(scheduled_operations) + len(dropped_operation_ids) < len(operations):
            progress_made = False

            with profile_section("scheduler.dependency_scan"):
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

                    for dependency in dependency_cache.get(
                        operation_instance_key, ()
                    ):
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
                        dep_completion_key = (
                            dependency.operation_id,
                            dependency_instance_key,
                        )
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
                            completion_key = (
                                operation.operation_id,
                                operation_instance_key,
                            )
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

    @staticmethod
    def _schedule_operations_topological(
        operations: list[OperationNode],
        resource_manager: ResourceManager,
    ) -> tuple[List[Task], List[Idle], List[DroppedOperation]]:
        """Topological scheduling using a ready queue (Kahn algorithm)."""
        tasks: List[Task] = []
        dropped_operations: List[DroppedOperation] = []
        all_idles: List[Idle] = []

        if not operations:
            return tasks, all_idles, dropped_operations

        operation_scheduler = SchedulerService._get_cached_operation_scheduler(
            resource_manager
        )
        operation_instance_counter: Dict[str, int] = {}

        num_operations = len(operations)
        index_by_identity = {id(op): idx for idx, op in enumerate(operations)}

        indegree: List[int] = [0] * num_operations
        successors: List[List[int]] = [[] for _ in range(num_operations)]
        ready_times: List[Optional[datetime]] = [None] * num_operations
        completion_times: List[Optional[datetime]] = [None] * num_operations
        dependency_drop_cause: List[Optional[int]] = [None] * num_operations

        # Build graph metadata once
        for idx, op in enumerate(operations):
            deps = getattr(op, "dependencies", ())
            indegree_value = 0
            for dep in deps:
                dep_idx = index_by_identity.get(id(dep))
                if dep_idx is None:
                    # Dependency is unknown – treat as an immediate drop trigger
                    dependency_drop_cause[idx] = -1
                    continue
                successors[dep_idx].append(idx)
                indegree_value += 1
            indegree[idx] = indegree_value

        heap: List[tuple[datetime, int, int]] = []
        counter = 0
        for idx, op in enumerate(operations):
            if indegree[idx] == 0:
                key_time = datetime.min
                heapq.heappush(heap, (key_time, counter, idx))
                counter += 1

        processed = [False] * num_operations

        while heap:
            _, _, op_idx = heapq.heappop(heap)
            if processed[op_idx]:
                continue
            processed[op_idx] = True

            operation = operations[op_idx]
            dependency_index = dependency_drop_cause[op_idx]

            with profile_section("scheduler.dependency_scan"):
                if dependency_index is not None:
                    dependent_operation_id = None
                    if dependency_index >= 0:
                        dependent_operation = operations[dependency_index]
                        dependent_operation_id = dependent_operation.operation_id
                    dropped_operations.append(
                        DroppedOperation(
                            manufacturing_order_id=operation.manufacturing_order_id,
                            operation_id=operation.operation_id,
                            operation_name=operation.operation_name,
                            reason=DropReason.DEPENDENCY_DROPPED,
                            dependent_operation_id=dependent_operation_id,
                        )
                    )
                    completion_times[op_idx] = None
                else:
                    earliest_start = ready_times[op_idx]
                    operation_tasks, operation_idles, operation_dropped = (
                        operation_scheduler.schedule_operation(
                            operation,
                            earliest_start=earliest_start,
                            operation_instance_counter=operation_instance_counter,
                        )
                    )

                    if operation_dropped:
                        dropped_operations.append(operation_dropped)
                        completion_times[op_idx] = None
                        dependency_index = op_idx
                    elif operation_tasks:
                        tasks.extend(operation_tasks)
                        all_idles.extend(operation_idles)
                        completion_times[op_idx] = max(
                            task.datetime_end for task in operation_tasks
                        )
                    else:
                        completion_times[op_idx] = None

                # Propagate completion/drop information to successors
                completion_time = completion_times[op_idx]
                for successor_idx in successors[op_idx]:
                    indegree[successor_idx] -= 1
                    if completion_time is None and dependency_index is not None:
                        if dependency_drop_cause[successor_idx] is None:
                            dependency_drop_cause[successor_idx] = dependency_index
                    else:
                        successor_ready = ready_times[successor_idx]
                        if completion_time and (
                            successor_ready is None or completion_time > successor_ready
                        ):
                            ready_times[successor_idx] = completion_time

                    if indegree[successor_idx] == 0 and not processed[successor_idx]:
                        ready_value = ready_times[successor_idx] or datetime.min
                        heapq.heappush(heap, (ready_value, counter, successor_idx))
                        counter += 1

        remaining = [
            op for idx, op in enumerate(operations) if not processed[idx]
        ]
        if remaining:
            for op in remaining:
                dropped_operations.append(
                    DroppedOperation(
                        manufacturing_order_id=op.manufacturing_order_id,
                        operation_id=op.operation_id,
                        operation_name=op.operation_name,
                        reason=DropReason.CIRCULAR_DEPENDENCY,
                    )
                )

        return tasks, all_idles, dropped_operations
