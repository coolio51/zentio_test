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


@dataclass(frozen=True)
class OperationGraphMetadata:
    """Precomputed structural metadata for a scheduling batch."""

    profiles: tuple[OperationSchedulingProfile, ...]
    dependency_offsets: tuple[int, ...]
    dependency_indices: tuple[int, ...]
    dependency_nodes: tuple[OperationNode, ...]
    successors: tuple[tuple[int, ...], ...]
    indegree: tuple[int, ...]


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
    def _build_operation_graph_metadata(
        operations: list[OperationNode],
    ) -> OperationGraphMetadata:
        """Precompute dependency metadata for each operation."""

        profiles: List[OperationSchedulingProfile] = []
        dependency_offsets: List[int] = [0]
        dependency_indices: List[int] = []
        dependency_nodes: List[OperationNode] = []

        index_by_operation = {operation: idx for idx, operation in enumerate(operations)}
        successors: List[List[int]] = [[] for _ in operations]
        indegree: List[int] = []

        for idx, operation in enumerate(operations):
            dependencies = tuple(operation.dependencies)
            dependency_keys = tuple(id(dependency) for dependency in dependencies)

            profiles.append(
                OperationSchedulingProfile(
                    operation=operation,
                    instance_key=id(operation),
                    dependencies=dependencies,
                    dependency_keys=dependency_keys,
                )
            )

            indegree.append(len(dependencies))

            for dependency in dependencies:
                dep_index = index_by_operation[dependency]
                dependency_indices.append(dep_index)
                dependency_nodes.append(dependency)
                successors[dep_index].append(idx)

            dependency_offsets.append(len(dependency_indices))

        return OperationGraphMetadata(
            profiles=tuple(profiles),
            dependency_offsets=tuple(dependency_offsets),
            dependency_indices=tuple(dependency_indices),
            dependency_nodes=tuple(dependency_nodes),
            successors=tuple(tuple(items) for items in successors),
            indegree=tuple(indegree),
        )

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

        # Counter for generating unique operation instance IDs
        operation_instance_counter: Dict[str, int] = {}

        metadata = SchedulerService._build_operation_graph_metadata(operations)
        profiles = metadata.profiles
        dependency_offsets = metadata.dependency_offsets
        dependency_indices = metadata.dependency_indices
        dependency_nodes = metadata.dependency_nodes

        # Schedule operations in dependency order
        total_operations = len(operations)
        scheduled_flags = [False] * total_operations
        dropped_flags = [False] * total_operations
        completion_times: List[Optional[datetime]] = [None] * total_operations

        def _mark_dropped(index: int, drop: DroppedOperation) -> None:
            if not dropped_flags[index]:
                dropped_operations.append(drop)
                dropped_flags[index] = True

        scheduled_count = 0
        dropped_count = 0

        while scheduled_count + dropped_count < total_operations:
            progress_made = False

            with profile_section("scheduler.dependency_scan"):
                for index, profile in enumerate(profiles):
                    if scheduled_flags[index] or dropped_flags[index]:
                        continue

                    # Check if all dependencies are scheduled (not dropped)
                    dependencies_satisfied = True
                    max_dependency_end_time = None
                    dropped_due_to_dependency = False

                    start = dependency_offsets[index]
                    end = dependency_offsets[index + 1]

                    for offset in range(start, end):
                        dependency_index = dependency_indices[offset]
                        dependency = dependency_nodes[offset]
                        if dropped_flags[dependency_index]:
                            _mark_dropped(
                                index,
                                DroppedOperation(
                                    manufacturing_order_id=profile.operation.manufacturing_order_id,
                                    operation_id=profile.operation.operation_id,
                                    operation_name=profile.operation.operation_name,
                                    reason=DropReason.DEPENDENCY_DROPPED,
                                    dependent_operation_id=dependency.operation_id,
                                ),
                            )
                            dropped_due_to_dependency = True
                            dropped_count += 1
                            progress_made = True
                            break
                        if not scheduled_flags[dependency_index]:
                            dependencies_satisfied = False
                            break

                        dep_end_time = completion_times[dependency_index]
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
                                profile.operation,
                                earliest_start=max_dependency_end_time,
                                operation_instance_counter=operation_instance_counter,
                            )
                        )

                        if operation_dropped:
                            # Operation was dropped, add to dropped list
                            _mark_dropped(index, operation_dropped)
                            dropped_count += 1
                            progress_made = True
                        elif operation_tasks:
                            # Operation was successfully scheduled

                            tasks.extend(operation_tasks)
                            all_idles.extend(operation_idles)
                            # Record the completion time of this operation instance
                            completion_times[index] = max(
                                task.datetime_end for task in operation_tasks
                            )
                            scheduled_flags[index] = True
                            scheduled_count += 1
                            progress_made = True

            if not progress_made:
                # This should not happen if the dependency graph is valid
                remaining_ops = [
                    profiles[idx].operation.operation_id
                    for idx in range(total_operations)
                    if not scheduled_flags[idx] and not dropped_flags[idx]
                ]
                SchedulerService.console.log(
                    f"Warning: Could not schedule operations due to circular dependencies or missing dependencies: {remaining_ops}"
                )
                # Drop the remaining operations with a circular dependency reason
                for idx in range(total_operations):
                    if not scheduled_flags[idx] and not dropped_flags[idx]:
                        op = profiles[idx].operation
                        _mark_dropped(
                            idx,
                            DroppedOperation(
                                manufacturing_order_id=op.manufacturing_order_id,
                                operation_id=op.operation_id,
                                operation_name=op.operation_name,
                                reason=DropReason.CIRCULAR_DEPENDENCY,
                            ),
                        )
                        dropped_count += 1
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
        metadata = SchedulerService._build_operation_graph_metadata(operations)
        profiles = metadata.profiles
        indegree = list(metadata.indegree)
        successors = [list(items) for items in metadata.successors]
        completion_times: List[Optional[datetime]] = [None] * len(operations)
        ready_times: List[Optional[datetime]] = [None] * len(operations)
        dropped_due_to_dependency: Dict[int, OperationNode] = {}

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
