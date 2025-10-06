from datetime import datetime, timedelta
from typing import Optional, List, Tuple, Dict

from scheduler.models import (
    Resource,
    TaskPhase,
    OperationRequirement,
    OperationNode,
    Task,
    DroppedOperation,
    DropReason,
    ResourceType,
    Idle,
    OperatorRequirement,
)
from scheduler.common.profiling import profile_function, profile_section
from scheduler.common.settings import get_slot_search_mode
from scheduler.services.resource_manager import ResourceManager


class OperationScheduler:
    def __init__(self, resource_manager: ResourceManager):
        self.resource_manager = resource_manager

    @profile_function()
    def _find_earliest_operator_availability_in_machine_window(
        self,
        machine: Resource,
        machine_window_start: datetime,
        machine_window_end: datetime,
        operation: OperationNode,
    ) -> Optional[datetime]:
        """
        Find the earliest time when operators are available within the machine window.

        Args:
            machine: The machine resource
            machine_window_start: Start of machine availability window
            machine_window_end: End of machine availability window
            operation: The operation requiring operators

        Returns:
            Earliest datetime when operators are available within machine window, or None if no availability
        """
        if get_slot_search_mode() == "merge":
            with profile_section("op_scheduler.interval_merge"):
                merged_start = self._find_operator_start_merge(
                    machine,
                    machine_window_start,
                    machine_window_end,
                    operation,
                )
            if merged_start is not None:
                return merged_start

        # Get all operator requirements for this operation
        all_operator_requirements = []
        for phase, requirements in operation.operator_requirements.items():
            all_operator_requirements.extend(requirements)

        if not all_operator_requirements:
            return machine_window_start

        # Find operators that can satisfy the requirements
        capable_operators = []
        seen_operator_ids = set()
        for operator_req in all_operator_requirements:
            operators = self.resource_manager.find_capable_resources(
                ResourceType.OPERATOR, operator_req.phase, operation.operation_id
            )
            for operator in operators:
                if operator.resource_id not in seen_operator_ids:
                    capable_operators.append(operator)
                    seen_operator_ids.add(operator.resource_id)

        if not capable_operators:
            return None

        # Check availability of capable operators within machine window
        # We'll check in 30-minute increments for better granularity
        current_time = machine_window_start
        operation_duration = sum(operation.durations.values(), timedelta())
        increment = timedelta(minutes=30)  # More fine-grained search

        with profile_section("op_scheduler.find_slot.step_scan"):
            while current_time + operation_duration <= machine_window_end:
                # Check if any capable operator is available at this time
                for operator in capable_operators:
                    if self.resource_manager.is_resource_available(
                        operator.resource_id,
                        current_time,
                        current_time + operation_duration,
                    ):
                        return current_time

                # Move to next 30-minute slot
                current_time += increment

        return None

    def _find_operator_start_merge(
        self,
        machine: Resource,
        window_start: datetime,
        window_end: datetime,
        operation: OperationNode,
    ) -> Optional[datetime]:
        total_duration = timedelta()
        all_operator_requirements = []
        for phase, requirements in operation.operator_requirements.items():
            if phase in operation.durations:
                duration = (
                    operation.durations[phase] * operation.quantity
                    if phase == TaskPhase.CORE_OPERATION
                    else operation.durations[phase]
                )
                total_duration += duration
            all_operator_requirements.extend(requirements)

        if window_start + total_duration > window_end:
            return None

        candidate_times = {window_start}
        for requirement in all_operator_requirements:
            operators = self.resource_manager.find_capable_resources(
                ResourceType.OPERATOR,
                requirement.phase,
                operation.operation_id,
            )
            for operator in operators:
                for interval_start, _ in self.resource_manager.free_intervals(
                    operator.resource_id,
                    window_start,
                    window_end,
                ):
                    candidate_times.add(interval_start)

        for candidate in sorted(candidate_times):
            candidate_end = candidate + total_duration
            if candidate_end > window_end:
                continue

            if not self.resource_manager.is_resource_available(
                machine.resource_id, candidate, candidate_end
            ):
                continue

            if not all_operator_requirements:
                return candidate

            operators_available = self.resource_manager.find_available_operators_for_machine(
                machine,
                all_operator_requirements,
                candidate,
                candidate_end,
                operation.operation_id,
            )
            if operators_available:
                return candidate

        return None

    def _assign_operators_merge(
        self,
        machine: Resource,
        operator_reqs: List[OperatorRequirement],
        preferred_start: datetime,
        phase_duration: timedelta,
        operation: OperationNode,
        scheduled_idles: List[Idle],
    ) -> List[Resource]:
        if not operator_reqs:
            return []

        total_operators_needed = sum(req.operator_count for req in operator_reqs)
        search_end_time = preferred_start + operation.max_idle_between_phases
        candidate_times = {preferred_start}
        if operation.min_idle_between_phases > timedelta(0):
            candidate_times.add(preferred_start + operation.min_idle_between_phases)

        for requirement in operator_reqs:
            operators = self.resource_manager.find_capable_resources(
                ResourceType.OPERATOR,
                requirement.phase,
                operation.operation_id,
            )
            for operator in operators:
                for interval_start, _ in self.resource_manager.free_intervals(
                    operator.resource_id,
                    preferred_start,
                    search_end_time,
                ):
                    candidate_times.add(interval_start)

        for candidate in sorted(candidate_times):
            if candidate < preferred_start:
                continue
            if candidate > search_end_time:
                continue

            candidate_end = candidate + phase_duration
            if candidate_end > search_end_time + phase_duration:
                continue

            if machine and not self.resource_manager.is_resource_available(
                machine.resource_id, candidate, candidate_end
            ):
                continue

            operators = self.resource_manager.find_available_operators_for_machine(
                machine,
                operator_reqs,
                candidate,
                candidate_end,
                operation.operation_id,
            )

            if operators is not None and len(operators) >= total_operators_needed:
                if candidate > preferred_start:
                    idle_period = Idle(
                        operation=operation,
                        machine=machine,
                        datetime_start=preferred_start,
                        datetime_end=candidate,
                        reason="waiting_for_operator",
                    )
                    scheduled_idles.append(idle_period)
                return operators

        return []

    def _find_operator_windows_merge(
        self,
        machine: Resource,
        operator_reqs: List[OperatorRequirement],
        start_time: datetime,
        operation: OperationNode,
    ) -> List[tuple[datetime, datetime, List[Resource]]]:
        windows: List[tuple[datetime, datetime, List[Resource]]] = []
        total_required = sum(req.operator_count for req in operator_reqs)
        max_search_time = start_time + timedelta(days=60)

        candidate_times = {start_time}
        for requirement in operator_reqs:
            operators = self.resource_manager.find_capable_resources(
                ResourceType.OPERATOR,
                requirement.phase,
                operation.operation_id,
            )
            for operator in operators:
                for interval_start, _ in self.resource_manager.free_intervals(
                    operator.resource_id,
                    start_time,
                    max_search_time,
                ):
                    candidate_times.add(interval_start)

        for candidate in sorted(candidate_times):
            if candidate >= max_search_time:
                break

            candidate_end = candidate + timedelta(hours=1)
            operators = self.resource_manager.find_available_operators_for_machine(
                machine,
                operator_reqs,
                candidate,
                candidate_end,
                operation.operation_id,
            )

            if not operators or len(operators) < total_required:
                continue

            window_end = self._find_operator_availability_end(
                operators, candidate, operation.operation_id
            )

            if window_end - candidate >= timedelta(hours=1):
                windows.append((candidate, window_end, operators))

        return windows

    @profile_function()
    def schedule_operation(
        self,
        operation: OperationNode,
        earliest_start: Optional[datetime] = None,
        operation_instance_counter: Optional[Dict[str, int]] = None,
    ) -> tuple[List[Task], List[Idle], Optional[DroppedOperation]]:
        """
        Schedule all phases of a single operation using machine-first allocation approach.

        1. Calculate total duration needed
        2. If operation requires a machine, find availability window for that machine
        3. Find earliest operator availability within that machine window
        4. Schedule phases sequentially from operator availability time, assigning operators as needed
        5. If no machine required, fall back to operator-only scheduling
        """
        if operation_instance_counter is None:
            operation_instance_counter = {}

        tasks = []

        # Generate unique operation instance ID using embedded manufacturing order info
        # Use existing operation_instance_id if provided, otherwise generate one
        if operation.operation_instance_id:
            operation_instance_id = operation.operation_instance_id
        else:
            mo_id = operation.manufacturing_order_id or "unknown"
            operation_key = f"{mo_id}_{operation.operation_id}"
            if operation_key not in operation_instance_counter:
                operation_instance_counter[operation_key] = 0
            operation_instance_counter[operation_key] += 1
            operation_instance_id = (
                f"{operation_key}_{operation_instance_counter[operation_key]:03d}"
            )

        # Calculate total duration needed for the operation
        total_duration = timedelta()
        phase_order = [TaskPhase.SETUP, TaskPhase.CORE_OPERATION, TaskPhase.CLEANUP]

        for phase in phase_order:
            if phase in operation.durations:
                phase_duration = (
                    operation.durations[phase] * operation.quantity
                    if phase == TaskPhase.CORE_OPERATION
                    else operation.durations[phase]
                )
                total_duration += phase_duration

        # Check if any operators are actually needed for this operation
        total_operators_needed = 0
        for phase in phase_order:
            if phase in operation.operator_requirements:
                operator_reqs = operation.operator_requirements[phase]
                total_operators_needed += (
                    sum(req.operator_count for req in operator_reqs)
                    if operator_reqs
                    else 0
                )

        # Only pass operation_id for operator compatibility if operators are actually needed
        operation_id_for_search = (
            operation.operation_id if total_operators_needed > 0 else None
        )

        # Machine-first allocation approach
        machine = None
        operation_start_time = earliest_start
        operation_end_time = None

        if operation.required_machine_id:
            # Specific machine required - try all available windows
            machine_windows = (
                self.resource_manager.find_all_machine_availability_windows(
                    operation.required_machine_id,
                    total_duration,
                    earliest_start,
                    operation_id_for_search,
                )
            )

            if not machine_windows:
                return (
                    [],
                    [],
                    DroppedOperation(
                        manufacturing_order_id=operation.manufacturing_order_id,
                        operation_id=operation.operation_id,
                        operation_name=operation.operation_name,
                        reason=DropReason.PHASE_SCHEDULING_FAILED,
                        failed_phase=TaskPhase.CORE_OPERATION,
                        error_message=f"Required machine {operation.required_machine_id} not available",
                    ),
                )

            machine = next(
                (
                    r
                    for r in self.resource_manager.resources
                    if r.resource_id == operation.required_machine_id
                ),
                None,
            )

            if not machine:
                return (
                    [],
                    [],
                    DroppedOperation(
                        manufacturing_order_id=operation.manufacturing_order_id,
                        operation_id=operation.operation_id,
                        operation_name=operation.operation_name,
                        reason=DropReason.PHASE_SCHEDULING_FAILED,
                        failed_phase=TaskPhase.CORE_OPERATION,
                        error_message=f"Required machine {operation.required_machine_id} not found",
                    ),
                )

            # Try each machine window until one works
            operator_start_time = None
            selected_machine_window = None

            for window_idx, machine_window in enumerate(machine_windows):
                machine_window_start, machine_window_end = machine_window

                # Find earliest operator availability within this machine window
                operator_start_time = (
                    self._find_earliest_operator_availability_in_machine_window(
                        machine, machine_window_start, machine_window_end, operation
                    )
                )

                if operator_start_time is not None:

                    selected_machine_window = machine_window
                    break
                else:
                    pass

            if operator_start_time is None:
                return (
                    [],
                    [],
                    DroppedOperation(
                        manufacturing_order_id=operation.manufacturing_order_id,
                        operation_id=operation.operation_id,
                        operation_name=operation.operation_name,
                        reason=DropReason.PHASE_SCHEDULING_FAILED,
                        failed_phase=TaskPhase.CORE_OPERATION,
                        error_message=f"No operator availability found in any of the {len(machine_windows)} machine windows",
                    ),
                )

            # At this point, both operator_start_time and selected_machine_window are guaranteed to be set
            assert (
                selected_machine_window is not None
            ), "selected_machine_window should be set when operator_start_time is found"
            machine_window_start, machine_window_end = selected_machine_window
            operation_start_time = operator_start_time
            operation_end_time = machine_window_end
        else:
            # Dynamic machine selection - try to find any available machine
            capable_machines = self.resource_manager.find_capable_resources(
                ResourceType.MACHINE, TaskPhase.CORE_OPERATION, operation.operation_id
            )

            best_machine = None
            best_window = None
            best_operator_start = None

            with profile_section("op_scheduler.interval_merge"):
                for candidate_machine in capable_machines:
                    # Try to find all available windows for this machine, not just the first one
                    all_machine_windows = (
                        self.resource_manager.find_all_machine_availability_windows(
                            candidate_machine.resource_id,
                            total_duration,
                            earliest_start,
                            operation_id_for_search,
                        )
                    )

                    # Try each available window until we find one where operators are available
                    for machine_window_start, machine_window_end in all_machine_windows:
                        # Find operator availability within this machine window
                        operator_start_time = (
                            self._find_earliest_operator_availability_in_machine_window(
                                candidate_machine,
                                machine_window_start,
                                machine_window_end,
                                operation,
                            )
                        )

                        if operator_start_time is not None:
                            best_machine = candidate_machine
                            best_window = (machine_window_start, machine_window_end)
                            best_operator_start = operator_start_time
                            break  # Found a working window for this machine

                    # If we found a working combination, use it
                    if best_machine and best_window and best_operator_start:
                        break

            if best_machine and best_window and best_operator_start:
                machine = best_machine
                operation_start_time = best_operator_start
                operation_end_time = best_window[1]
            else:
                return (
                    [],
                    [],
                    DroppedOperation(
                        manufacturing_order_id=operation.manufacturing_order_id,
                        operation_id=operation.operation_id,
                        operation_name=operation.operation_name,
                        reason=DropReason.PHASE_SCHEDULING_FAILED,
                        failed_phase=TaskPhase.CORE_OPERATION,
                        error_message="No available machine found for operation",
                    ),
                )

        # Schedule phases with idle time support
        current_phase_start: datetime = (
            operation_start_time or earliest_start or datetime.now()
        )
        all_operators_for_operation = []
        scheduled_tasks = []
        scheduled_idles = []

        for phase in phase_order:
            if phase in operation.durations:
                phase_duration = (
                    operation.durations[phase] * operation.quantity
                    if phase == TaskPhase.CORE_OPERATION
                    else operation.durations[phase]
                )

                # Get operator requirements for this phase
                operator_reqs = operation.operator_requirements.get(phase, [])

                # Check if any operators are actually required for this phase
                total_operators_needed = (
                    sum(req.operator_count for req in operator_reqs)
                    if operator_reqs
                    else 0
                )

                # Try to find operators for a contiguous slot first
                if total_operators_needed > 0:
                    operators = self._find_operators_with_idle_support(
                        machine,
                        operator_reqs,
                        current_phase_start,
                        phase_duration,
                        operation,
                        scheduled_idles,
                    )

                    # If contiguous assignment failed and this is core_operation, fall back to splitting
                    if not operators and phase == TaskPhase.CORE_OPERATION:
                        # Calculate work per hour for this operation
                        total_items = operation.quantity
                        total_core_duration = phase_duration
                        items_per_hour = (
                            total_items / (total_core_duration.total_seconds() / 3600)
                            if total_core_duration.total_seconds() > 0
                            else total_items
                        )

                        # Find available operator windows starting from current_phase_start
                        operator_windows = self._find_available_operator_windows(
                            machine, operator_reqs, current_phase_start, operation
                        )

                        if not operator_windows:
                            return (
                                [],
                                [],
                                DroppedOperation(
                                    manufacturing_order_id=operation.manufacturing_order_id,
                                    operation_id=operation.operation_id,
                                    operation_name=operation.operation_name,
                                    reason=DropReason.PHASE_SCHEDULING_FAILED,
                                    failed_phase=phase,
                                    error_message=f"Required operators not available for phase {phase.value}",
                                ),
                            )

                        core_tasks = []
                        remaining_items = total_items
                        last_window_end: datetime = current_phase_start

                        for (
                            window_start,
                            window_end,
                            window_operators,
                        ) in operator_windows:
                            if remaining_items <= 0:
                                break

                            # Ensure machine is available during this operator window
                            if not self.resource_manager.is_resource_available(
                                machine.resource_id, window_start, window_end
                            ):
                                return (
                                    [],
                                    [],
                                    DroppedOperation(
                                        manufacturing_order_id=operation.manufacturing_order_id,
                                        operation_id=operation.operation_id,
                                        operation_name=operation.operation_name,
                                        reason=DropReason.PHASE_SCHEDULING_FAILED,
                                        error_message=f"Machine {machine.resource_id} not available during required operator window",
                                    ),
                                )

                            # Calculate how much work fits in this window
                            window_duration = window_end - window_start
                            window_hours = window_duration.total_seconds() / 3600
                            items_in_window = min(
                                remaining_items,
                                max(1, int(items_per_hour * window_hours)),
                            )

                            if items_in_window > 0:
                                # Create idle period if there's a gap
                                if window_start > last_window_end:
                                    idle_period = Idle(
                                        operation=operation,
                                        machine=machine,
                                        datetime_start=last_window_end,
                                        datetime_end=window_start,
                                        reason="waiting_for_operator_availability",
                                    )
                                    scheduled_idles.append(idle_period)

                                # Calculate actual time needed for these items
                                actual_duration_hours = (
                                    items_in_window / items_per_hour
                                    if items_per_hour > 0
                                    else 0
                                )
                                actual_duration = timedelta(hours=actual_duration_hours)
                                actual_end_time = window_start + actual_duration

                                # Ensure we don't exceed the operator window
                                actual_end_time = min(actual_end_time, window_end)

                                # Create task for this window
                                task = Task(
                                    operation=operation,
                                    machine=machine,
                                    operators=window_operators,
                                    datetime_start=window_start,
                                    datetime_end=actual_end_time,
                                    phase=phase,
                                    quantity=items_in_window,
                                    operation_instance_id=operation_instance_id,
                                )
                                core_tasks.append(task)

                                # Book resources for this chunk (only for actual time needed)
                                self.resource_manager.book_machine_and_operators(
                                    machine,
                                    window_operators,
                                    window_start,
                                    actual_end_time,
                                )

                                remaining_items -= items_in_window
                                last_window_end = actual_end_time

                        if remaining_items > 0:
                            return (
                                [],
                                [],
                                DroppedOperation(
                                    manufacturing_order_id=operation.manufacturing_order_id,
                                    operation_id=operation.operation_id,
                                    operation_name=operation.operation_name,
                                    reason=DropReason.PHASE_SCHEDULING_FAILED,
                                    failed_phase=phase,
                                    error_message=f"Could not schedule all {total_items} items (missing {remaining_items})",
                                ),
                            )

                        scheduled_tasks.extend(core_tasks)
                        current_phase_start = last_window_end
                        continue

                    if not operators:
                        return (
                            [],
                            [],
                            DroppedOperation(
                                manufacturing_order_id=operation.manufacturing_order_id,
                                operation_id=operation.operation_id,
                                operation_name=operation.operation_name,
                                reason=DropReason.PHASE_SCHEDULING_FAILED,
                                failed_phase=phase,
                                error_message=f"Required operators not available for phase {phase.value}",
                            ),
                        )
                elif operation.required_machine_id and not machine:
                    operators = []
                else:
                    operators = []

                # Calculate actual start time based on operator availability
                # If we had to wait for operators, check if any idle periods were created
                actual_phase_start: datetime
                if operators and scheduled_idles:
                    # Check if the last idle period affects this phase's start time
                    last_idle = scheduled_idles[-1]
                    if (
                        last_idle.datetime_end > current_phase_start
                        and last_idle.datetime_start <= current_phase_start
                    ):
                        # This idle period was created for this phase, use its end time
                        actual_phase_start = last_idle.datetime_end
                    else:
                        actual_phase_start = current_phase_start
                else:
                    # No operators needed or no idle periods created
                    actual_phase_start = current_phase_start

                # Standard single task creation for non-core operations or short core operations
                phase_end_time: datetime = actual_phase_start + phase_duration

                # Create the task
                task = Task(
                    operation=operation,
                    phase=phase,
                    quantity=operation.quantity,
                    datetime_start=actual_phase_start,
                    datetime_end=phase_end_time,
                    machine=machine,
                    operators=operators,
                    operation_instance_id=operation_instance_id,
                )
                scheduled_tasks.append(task)

                # Update current start time for next phase
                current_phase_start = phase_end_time

        # Book all resources for the scheduled tasks
        for task in scheduled_tasks:
            if task.machine:
                self.resource_manager.book_machine_and_operators(
                    task.machine,
                    task.operators,
                    task.datetime_start,
                    task.datetime_end,
                )

        # Book machine for idle periods
        for idle in scheduled_idles:
            self.resource_manager.book_machine_idle(
                idle.machine,
                idle.datetime_start,
                idle.datetime_end,
            )

        return scheduled_tasks, scheduled_idles, None

    @profile_function()
    def _find_operators_with_idle_support(
        self,
        machine: Resource,
        operator_reqs: List[OperatorRequirement],
        preferred_start: datetime,
        phase_duration: timedelta,
        operation: OperationNode,
        scheduled_idles: List[Idle],
    ) -> List[Resource]:
        """
        Find operators for a phase, allowing for idle time when operators aren't available.
        """

        if get_slot_search_mode() == "merge":
            with profile_section("op_scheduler.interval_merge"):
                merged_operators = self._assign_operators_merge(
                    machine,
                    operator_reqs,
                    preferred_start,
                    phase_duration,
                    operation,
                    scheduled_idles,
                )
            if merged_operators:
                return merged_operators

        # If no operators required for this phase, return immediately
        if not operator_reqs:
            return []

        total_operators_needed = sum(req.operator_count for req in operator_reqs)

        # Try to find operators at the preferred start time
        operators = self.resource_manager.find_available_operators_for_machine(
            machine,
            operator_reqs,
            preferred_start,
            preferred_start + phase_duration,
            operation.operation_id,
        )

        if operators is not None and len(operators) >= total_operators_needed:
            return operators

        # If operators not available immediately, try to find next available time within idle constraints
        search_end_time = preferred_start + operation.max_idle_between_phases
        current_search_time = preferred_start + operation.min_idle_between_phases

        with profile_section("op_scheduler.find_slot.step_scan"):
            # Search in 1-hour increments for the next available operator time
            while current_search_time <= search_end_time:
                operators = self.resource_manager.find_available_operators_for_machine(
                    machine,
                    operator_reqs,
                    current_search_time,
                    current_search_time + phase_duration,
                    operation.operation_id,
                )

                if operators is not None and len(operators) >= total_operators_needed:
                    # Found operators! Create an idle period if there's a gap
                    if current_search_time > preferred_start:
                        idle_period = Idle(
                            operation=operation,
                            machine=machine,
                            datetime_start=preferred_start,
                            datetime_end=current_search_time,
                            reason="waiting_for_operator",
                        )
                        scheduled_idles.append(idle_period)

                    return operators

                # Move search time forward by 1 hour
                current_search_time += timedelta(hours=1)

        return []

    @profile_function()
    def _find_operator_availability_end(
        self, operators: List[Resource], start_time: datetime, operation_id: str
    ) -> datetime:
        """
        Find when these operators will no longer be available.
        """
        # Simple implementation - find the earliest end time among all operators
        earliest_end = None

        for operator in operators:
            for availability in operator.availabilities:
                if (
                    availability.start_datetime
                    <= start_time
                    < availability.end_datetime
                ):
                    if earliest_end is None or availability.end_datetime < earliest_end:
                        earliest_end = availability.end_datetime
                    break

        return earliest_end or start_time + timedelta(
            hours=8
        )  # Default to 8 hours if not found

    @profile_function()
    def _find_available_operator_windows(
        self,
        machine: Resource,
        operator_reqs: List[OperatorRequirement],
        start_time: datetime,
        operation: OperationNode,
    ) -> List[tuple[datetime, datetime, List[Resource]]]:
        """
        Find all available operator windows starting from start_time.
        Returns list of (window_start, window_end, operators) tuples.
        """
        if get_slot_search_mode() == "merge":
            with profile_section("op_scheduler.interval_merge"):
                merged_windows = self._find_operator_windows_merge(
                    machine,
                    operator_reqs,
                    start_time,
                    operation,
                )
            if merged_windows:
                return merged_windows

        windows = []
        search_time = start_time
        max_search_time = start_time + timedelta(days=60)  # Limit search to 2 months

        with profile_section("op_scheduler.find_slot.step_scan"):
            while search_time < max_search_time:
                # Try to find operators at this time
                operators = self.resource_manager.find_available_operators_for_machine(
                    machine,
                    operator_reqs,
                    search_time,
                    search_time + timedelta(hours=1),  # Check 1-hour window
                    operation.operation_id,
                )

                if operators and len(operators) >= sum(
                    req.operator_count for req in operator_reqs
                ):
                    # Found operators, now determine how long they're available
                    window_end = self._find_operator_availability_end(
                        operators, search_time, operation.operation_id
                    )

                    # Ensure minimum window size (at least 1 hour)
                    if window_end - search_time >= timedelta(hours=1):
                        windows.append((search_time, window_end, operators))

                        # Jump to end of this window for next search
                        search_time = window_end
                    else:
                        # Window too small, advance by 1 hour
                        search_time += timedelta(hours=1)
                else:
                    # No operators found, advance by 1 hour
                    search_time += timedelta(hours=1)

        return windows
