from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional
from weakref import WeakKeyDictionary
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
from scheduler.common.console import get_console
from scheduler.utils.resource_logger import ResourceLogger


@dataclass(frozen=True)
class OperationSchedulingProfile:
    """Lightweight cache of dependency data for an operation instance."""

    operation: OperationNode
    instance_key: int
    dependencies: tuple[OperationNode, ...]
    dependency_keys: tuple[int, ...]


class SchedulerService:
    console = get_console()
    _operation_scheduler_pool: "WeakKeyDictionary[ResourceManager, OperationScheduler]" = (
        WeakKeyDictionary()
    )

    @classmethod
    def _acquire_operation_scheduler(
        cls, resource_manager: ResourceManager
    ) -> OperationScheduler:
        """Get a cached ``OperationScheduler`` for this resource manager."""
        operation_scheduler = cls._operation_scheduler_pool.get(resource_manager)
        if operation_scheduler is None:
            operation_scheduler = OperationScheduler(resource_manager)
            cls._operation_scheduler_pool[resource_manager] = operation_scheduler
        return operation_scheduler

    @staticmethod
    @profile_function()
    def schedule(
        operations: list[OperationNode],
        resource_manager: ResourceManager,
    ) -> Schedule:
        tasks: List[Task] = []
        all_dropped_operations: List[DroppedOperation] = []
        all_idles: List[Idle] = []

        operation_scheduler = SchedulerService._acquire_operation_scheduler(
            resource_manager
        )
        operation_scheduler.reset_run_state()

        operation_tasks, operation_idles, dropped_operations = (
            SchedulerService._schedule_operations(
                operations, resource_manager, operation_scheduler
            )
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
        operation_scheduler: OperationScheduler,
    ) -> tuple[List[Task], List[Idle], List[DroppedOperation]]:
        mode = get_scheduler_mode()
        if mode == "topo":
            return SchedulerService._schedule_operations_topological(
                operations, resource_manager, operation_scheduler
            )
        if mode == "naive":
            return SchedulerService._schedule_operations_naive(
                operations, resource_manager, operation_scheduler
            )
        # Fallback to topological even for unknown modes to guarantee performance
        return SchedulerService._schedule_operations_topological(
            operations, resource_manager, operation_scheduler
        )

    @staticmethod
    def _build_operation_profiles(
        operations: list[OperationNode],
    ) -> Dict[OperationNode, OperationSchedulingProfile]:
        """Precompute dependency metadata for each operation."""

        profiles: Dict[OperationNode, OperationSchedulingProfile] = {}
        for operation in operations:
            dependencies = tuple(operation.dependencies)
            dependency_keys = tuple(id(dependency) for dependency in dependencies)
            profiles[operation] = OperationSchedulingProfile(
                operation=operation,
                instance_key=id(operation),
                dependencies=dependencies,
                dependency_keys=dependency_keys,
            )
        return profiles

    @staticmethod
    def _schedule_operations_naive(
        operations: list[OperationNode],
        resource_manager: ResourceManager,
        operation_scheduler: OperationScheduler,
    ) -> tuple[List[Task], List[Idle], List[DroppedOperation]]:
        """Naive O(n^2) scheduling loop retained for backward compatibility."""
        tasks: List[Task] = []
        dropped_operations: List[DroppedOperation] = []
        all_idles: List[Idle] = []

        # Track completion times for each operation instance (unique_key -> end_time)
        operation_completion_times: Dict[int, datetime] = {}

        # Counter for generating unique operation instance IDs
        operation_instance_counter: Dict[str, int] = {}

        # Operations are provided directly

        # Keep track of which operation instances have been scheduled and dropped
        # Use object identity to distinguish between different instances of the same operation_id
        scheduled_operations = set()
        dropped_operation_ids = set()

        profiles = SchedulerService._build_operation_profiles(operations)

        # Schedule operations in dependency order
        while len(scheduled_operations) + len(dropped_operation_ids) < len(operations):
            progress_made = False

            with profile_section("scheduler.dependency_scan"):
                for operation in operations:
                    profile = profiles[operation]
                    operation_instance_key = profile.instance_key
                    if (
                        operation_instance_key in scheduled_operations
                        or operation_instance_key in dropped_operation_ids
                    ):
                        continue

                    # Check if all dependencies are scheduled (not dropped)
                    dependencies_satisfied = True
                    max_dependency_end_time = None
                    dropped_due_to_dependency = False

                    for dependency, dependency_instance_key in zip(
                        profile.dependencies, profile.dependency_keys
                    ):
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
                        dep_end_time = operation_completion_times.get(
                            dependency_instance_key
                        )
                        if dep_end_time is None:
                            dependencies_satisfied = False
                            break
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
                            operation_completion_times[operation_instance_key] = max(
                                task.datetime_end for task in operation_tasks
                            )
                            scheduled_operations.add(operation_instance_key)
                            progress_made = True

            if not progress_made:
                # This should not happen if the dependency graph is valid
                remaining_ops = [
                    op.operation_id
                    for op in operations
                    if profiles[op].instance_key not in scheduled_operations
                    and profiles[op].instance_key not in dropped_operation_ids
                ]
                SchedulerService.console.log(
                    f"Warning: Could not schedule operations due to circular dependencies or missing dependencies: {remaining_ops}"
                )
                # Drop the remaining operations with a circular dependency reason
                for op in operations:
                    profile = profiles[op]
                    if (
                        profile.instance_key not in scheduled_operations
                        and profile.instance_key not in dropped_operation_ids
                    ):
                        dropped_operations.append(
                            DroppedOperation(
                                manufacturing_order_id=op.manufacturing_order_id,
                                operation_id=op.operation_id,
                                operation_name=op.operation_name,
                                reason=DropReason.CIRCULAR_DEPENDENCY,
                            )
                        )
                        dropped_operation_ids.add(profile.instance_key)
                break

        return tasks, all_idles, dropped_operations

    @staticmethod
    def _schedule_operations_topological(
        operations: list[OperationNode],
        resource_manager: ResourceManager,
        operation_scheduler: OperationScheduler,
    ) -> tuple[List[Task], List[Idle], List[DroppedOperation]]:
        """Topological scheduling using a ready queue (Kahn algorithm)."""
        tasks: List[Task] = []
        dropped_operations: List[DroppedOperation] = []
        all_idles: List[Idle] = []

        operation_instance_counter: Dict[str, int] = {}
        completion_times: List[Optional[datetime]] = [None] * len(operations)

        profiles = SchedulerService._build_operation_profiles(operations)
        index_by_op = {op: idx for idx, op in enumerate(operations)}
        indegree: List[int] = [len(profiles[op].dependencies) for op in operations]
        successors: List[List[int]] = [[] for _ in operations]
        ready_times: List[Optional[datetime]] = [None] * len(operations)
        dropped_due_to_dependency: Dict[int, OperationNode] = {}

        for idx, op in enumerate(operations):
            for dependency in profiles[op].dependencies:
                dep_index = index_by_op[dependency]
                successors[dep_index].append(idx)

        heap: List[tuple[float, int, int]] = []
        counter = 0
        for idx, op in enumerate(operations):
            if indegree[idx] == 0:
                ready_times[idx] = None
                key = (float("-inf"), counter, idx)
                heapq.heappush(heap, key)
                counter += 1

        processed: set[int] = set()

        while heap:
            ready_dt, _, op_index = heapq.heappop(heap)
            if op_index in processed:
                continue
            processed.add(op_index)
            op = operations[op_index]

            with profile_section("scheduler.dependency_scan"):
                # If any dependency was dropped, propagate the drop
                if op_index in dropped_due_to_dependency:
                    dependency = dropped_due_to_dependency[op_index]
                    dropped_operations.append(
                        DroppedOperation(
                            manufacturing_order_id=op.manufacturing_order_id,
                            operation_id=op.operation_id,
                            operation_name=op.operation_name,
                            reason=DropReason.DEPENDENCY_DROPPED,
                            dependent_operation_id=dependency.operation_id,
                        )
                    )
                    completion_times[op_index] = None
                else:
                    earliest_start = ready_times[op_index]
                    operation_tasks, operation_idles, operation_dropped = (
                        operation_scheduler.schedule_operation(
                            op,
                            earliest_start=earliest_start,
                            operation_instance_counter=operation_instance_counter,
                        )
                    )

                    if operation_dropped:
                        dropped_operations.append(operation_dropped)
                        completion_times[op_index] = None
                    elif operation_tasks:
                        tasks.extend(operation_tasks)
                        all_idles.extend(operation_idles)
                        completion_times[op_index] = max(
                            task.datetime_end for task in operation_tasks
                        )
                    else:
                        completion_times[op_index] = None

                # Update successors
                for successor_index in successors[op_index]:
                    indegree[successor_index] -= 1
                    successor = operations[successor_index]
                    if completion_times[op_index] is None:
                        dropped_due_to_dependency.setdefault(successor_index, op)
                    else:
                        successor_ready = ready_times[successor_index]
                        new_ready = completion_times[op_index]
                        if new_ready is not None:
                            if successor_ready is None or new_ready > successor_ready:
                                ready_times[successor_index] = new_ready
                    if (
                        indegree[successor_index] == 0
                        and successor_index not in processed
                    ):
                        ready_value = ready_times[successor_index]
                        if ready_value is None:
                            heap_key_dt = float("-inf")
                        else:
                            try:
                                heap_key_dt = ready_value.timestamp()
                            except Exception:  # pragma: no cover - defensive
                                heap_key_dt = float("-inf")
                        heapq.heappush(heap, (heap_key_dt, counter, successor_index))
                        counter += 1

        # Any operations not processed are part of a cycle or were never enqueued
        remaining = [
            operations[idx]
            for idx in range(len(operations))
            if idx not in processed
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
