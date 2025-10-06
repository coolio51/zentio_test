from bisect import bisect_left
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Optional, List, Tuple, Dict, Iterable

from scheduler.common.profiling import profile_function, profile_section
from scheduler.models import (
    Resource,
    TaskPhase,
    OperationRequirement,
    ResourceType,
    OperatorRequirement,
    ResourceAvailability,
    Idle,
)

from scheduler.utils.interval_tree import IntervalTree


class ResourceManager:
    def __init__(self, resources: list[Resource]):
        self.original_resources = resources  # Keep reference to original

        # Create our own managed resources with dynamic availabilities
        self.resources = self._create_managed_resources(resources)
        self.resources_by_id = {resource.resource_id: resource for resource in self.resources}

        # Track resource bookings using interval trees for O(log n) operations
        self._resource_booking_trees: Dict[str, IntervalTree] = {
            resource_id: IntervalTree() for resource_id in self.resources_by_id
        }
        # Maintain lightweight ordered views for diagnostics only (built lazily)
        self._resource_booking_cache: Dict[str, List[Tuple[datetime, datetime]]] = {
            resource_id: [] for resource_id in self.resources_by_id
        }
        self._capable_resource_cache: Dict[
            tuple[str, str, Optional[str]], Tuple[Resource, ...]
        ] = {}
        self._availabilities_data: Dict[str, Tuple[Dict[str, object], ...]] = {}
        self._base_availability_templates: Dict[str, List[Tuple[datetime, datetime, float]]] = {
            resource.resource_id: [
                (
                    availability.start_datetime,
                    availability.end_datetime,
                    availability.effort,
                )
                for availability in resource.availabilities
            ]
            for resource in self.resources
        }
        # Build data structures for all operations
        self._build_data_structures()

    def _create_managed_resources(
        self, original_resources: list[Resource]
    ) -> list[Resource]:
        """Create managed copies of resources with dynamic availabilities"""
        managed_resources = []

        for original_resource in original_resources:
            # Create a copy of the resource with the same availabilities
            managed_resource = Resource(
                resource_id=original_resource.resource_id,
                resource_type=original_resource.resource_type,
                resource_name=original_resource.resource_name,
                resource_capabilities=original_resource.resource_capabilities.copy(),
                availabilities=original_resource.availabilities.copy(),  # This will be managed dynamically
            )
            managed_resources.append(managed_resource)

        return managed_resources

    @profile_function()
    def _build_data_structures(self):
        """Build data structures for complex availability operations"""
        self._capable_resource_cache.clear()
        by_type: Dict[str, List[str]] = defaultdict(list)
        by_type_phase: Dict[Tuple[str, str], List[str]] = defaultdict(list)
        by_type_phase_operation: Dict[Tuple[str, str, str], List[str]] = defaultdict(list)

        for resource in self.resources:
            by_type[resource.resource_type.value].append(resource.resource_id)
            for capability in resource.resource_capabilities:
                key_phase = (
                    capability.resource_type.value,
                    capability.phase.value,
                )
                by_type_phase[key_phase].append(resource.resource_id)
                key_full = (
                    capability.resource_type.value,
                    capability.phase.value,
                    capability.operation_id,
                )
                by_type_phase_operation[key_full].append(resource.resource_id)

        self._capabilities_by_type = {
            key: tuple(sorted(values)) for key, values in by_type.items()
        }
        self._capabilities_by_type_phase = {
            key: tuple(sorted(values)) for key, values in by_type_phase.items()
        }
        self._capabilities_by_type_phase_operation = {
            key: tuple(sorted(values)) for key, values in by_type_phase_operation.items()
        }

        for resource in self.resources:
            self._rebuild_resource_availability(resource)

    @profile_function()
    def _rebuild_resource_availability(self, resource: Resource):
        """Rebuild availability data for a single resource."""
        with profile_section("resource_manager.rebuild_availabilities"):
            entries = []
            for availability in resource.availabilities:
                entries.append(
                    {
                        "resource_id": resource.resource_id,
                        "start_datetime": availability.start_datetime,
                        "end_datetime": availability.end_datetime,
                        "effort": availability.effort,
                        "duration_hours": (
                            availability.end_datetime - availability.start_datetime
                        ).total_seconds()
                        / 3600,
                    }
                )
            entries.sort(key=lambda entry: entry["start_datetime"])
            self._availabilities_data[resource.resource_id] = tuple(entries)

    def _book_interval(
        self, resource_id: str, start_time: datetime, end_time: datetime
    ) -> None:
        tree = self._resource_booking_trees[resource_id]
        tree.insert(start_time, end_time)
        cache = self._resource_booking_cache[resource_id]
        idx = bisect_left(cache, (start_time, end_time))
        cache.insert(idx, (start_time, end_time))

    def _remove_interval(
        self, resource_id: str, start_time: datetime, end_time: datetime
    ) -> None:
        tree = self._resource_booking_trees.get(resource_id)
        if tree is None:
            return
        tree.remove(start_time, end_time)
        cache = self._resource_booking_cache.get(resource_id)
        if not cache:
            return
        idx = bisect_left(cache, (start_time, end_time))
        while idx < len(cache) and cache[idx][0] == start_time:
            if cache[idx][1] == end_time:
                cache.pop(idx)
                break
            idx += 1

    @profile_function()
    def _update_resource_availabilities(
        self, resource_id: str, start_time: datetime, end_time: datetime
    ):
        """Update the actual availability windows by removing/splitting booked time"""
        resource = self.resources_by_id.get(resource_id)
        if not resource:
            return

        new_availabilities = []

        for availability in resource.availabilities:
            # Check if booking overlaps with this availability window
            avail_start = availability.start_datetime
            avail_end = availability.end_datetime

            # No overlap - keep the availability as is
            if end_time <= avail_start or start_time >= avail_end:
                new_availabilities.append(availability)
            # Booking completely contains availability - remove it
            elif start_time <= avail_start and end_time >= avail_end:
                continue  # Skip this availability
            # Booking is at the start of availability - trim the start
            elif start_time <= avail_start and end_time < avail_end:
                new_availabilities.append(
                    ResourceAvailability(
                        start_datetime=end_time,
                        end_datetime=avail_end,
                        effort=availability.effort,
                    )
                )
            # Booking is at the end of availability - trim the end
            elif start_time > avail_start and end_time >= avail_end:
                new_availabilities.append(
                    ResourceAvailability(
                        start_datetime=avail_start,
                        end_datetime=start_time,
                        effort=availability.effort,
                    )
                )
            # Booking is in the middle - split the availability
            elif start_time > avail_start and end_time < avail_end:
                # Add the part before the booking
                new_availabilities.append(
                    ResourceAvailability(
                        start_datetime=avail_start,
                        end_datetime=start_time,
                        effort=availability.effort,
                    )
                )
                # Add the part after the booking
                new_availabilities.append(
                    ResourceAvailability(
                        start_datetime=end_time,
                        end_datetime=avail_end,
                        effort=availability.effort,
                    )
                )

        # Update the resource's availabilities
        resource.availabilities = new_availabilities

        # Rebuild the availability data structures for this resource
        self._rebuild_resource_availability(resource)

    def get_current_availabilities(self) -> list[Resource]:
        """Get the current state of all resources with their dynamic availabilities"""
        return self.resources

    @profile_function()
    def find_overlapping_intervals(
        self, resource_id: str, start: datetime, end: datetime
    ) -> bool:
        """
        Check if a time window (start, end) overlaps with any existing intervals using overlapping interval algorithm

        Args:
            intervals: List of (start_time, end_time) tuples representing booked periods
            start: Start time of the new interval to check
            end: End time of the new interval to check

        Returns:
            True if there's an overlap, False if no overlap
        """
        with profile_section("resource_manager.interval_tree_lookup"):
            tree = self._resource_booking_trees.get(resource_id)
            if tree is not None:
                return tree.overlaps(start, end)

        # Backwards compatibility path (should not happen)
        bookings = self._resource_booking_cache.get(resource_id)
        if isinstance(bookings, list):
            idx = bisect_left(bookings, (start, start))
            if idx > 0 and bookings[idx - 1][1] > start:
                return True
            if idx < len(bookings) and bookings[idx][0] < end:
                return True
        return False

    @profile_function()
    def find_capable_resources(
        self, resource_type: ResourceType, phase: TaskPhase, operation_id: str
    ) -> List[Resource]:
        """
        Find resources capable of handling a specific operation requirement.

        Args:
            resource_type: Type of resource needed
            phase: Task phase the resource needs to support
            operation_id: Operation ID for capability matching (may be split operation ID)

        Returns:
            List of capable resources
        """
        # Extract base operation ID from split operation ID (e.g., "abc_split_1" -> "abc")
        base_operation_id = (
            operation_id.split("_split_")[0]
            if "_split_" in operation_id
            else operation_id
        )
        cache_key = (resource_type.value, phase.value, base_operation_id)
        cached_resources = self._capable_resource_cache.get(cache_key)
        if cached_resources is not None:
            return list(cached_resources)

        with profile_section("resource_manager.lookup_capabilities"):
            candidates = self._capabilities_by_type_phase_operation.get(
                (resource_type.value, phase.value, base_operation_id)
            )
            if not candidates:
                candidates = self._capabilities_by_type_phase.get(
                    (resource_type.value, phase.value),
                    (),
                )

        resources = tuple(self.resources_by_id[rid] for rid in candidates)
        self._capable_resource_cache[cache_key] = resources
        return list(resources)

    @profile_function()
    def get_all_available_slots_in_window(
        self,
        resource_id: str,
        duration: timedelta,
        window_start: datetime,
        window_end: datetime,
    ) -> List[Tuple[datetime, datetime]]:
        """
        Find all available time slots for a resource within a specific time window,
        potentially spanning multiple availability periods.
        """
        # Get resource availability windows within the search window
        resource_availabilities = self._availabilities_data.get(resource_id, ())

        if not resource_availabilities:
            return []

        available_slots = []
        # Check each availability window
        for availability in resource_availabilities:
            avail_start = availability["start_datetime"]
            avail_end = availability["end_datetime"]

            # Only consider windows that overlap with our search window
            if avail_end <= window_start or avail_start >= window_end:
                continue

            # Trim to search window
            slot_start = max(avail_start, window_start)
            slot_end = min(avail_end, window_end)

            # Try to fit the duration in this window
            self._find_slots_in_availability_window(
                resource_id,
                slot_start,
                slot_end,
                duration,
                available_slots,
            )

        return available_slots

    @profile_function()
    def _find_slots_in_availability_window(
        self,
        resource_id: str,
        window_start: datetime,
        window_end: datetime,
        duration: timedelta,
        available_slots: List[Tuple[datetime, datetime]],
    ):
        """Find available slots within a single availability window"""
        if window_end - window_start < duration:
            return

        current_time = window_start

        while current_time + duration <= window_end:
            proposed_end = current_time + duration

            # Check if this slot overlaps with any existing booking
            if not self.find_overlapping_intervals(
                resource_id, current_time, proposed_end
            ):
                available_slots.append((current_time, proposed_end))
                current_time = proposed_end
            else:
                # Find the next potential start time after the conflicting booking
                next_start = current_time + timedelta(minutes=1)
                for booking_start, booking_end in self._resource_booking_cache.get(
                    resource_id, []
                ):
                    if booking_start <= current_time < booking_end:
                        next_start = booking_end
                        break
                current_time = next_start

    @profile_function()
    def book_resources(
        self, resources: List[Resource], start_time: datetime, end_time: datetime
    ) -> None:
        """
        Book resources for a specific time period.

        Args:
            resources: List of resources to book
            start_time: Start time of the booking
            end_time: End time of the booking
        """
        for resource in resources:
            self._book_interval(resource.resource_id, start_time, end_time)
            self._update_resource_availabilities(
                resource.resource_id, start_time, end_time
            )

    @profile_function()
    def unbook_resources(
        self, resources: List[Resource], start_time: datetime, end_time: datetime
    ) -> None:
        """
        Remove a booking for resources.

        Args:
            resources: List of resources to unbook
            start_time: Start time of the booking to remove
            end_time: End time of the booking to remove
        """
        for resource in resources:
            self._remove_interval(resource.resource_id, start_time, end_time)
            self._update_resource_availabilities(
                resource.resource_id, start_time, end_time
            )

    @profile_function()
    def is_resource_available(
        self, resource_id: str, start_time: datetime, end_time: datetime
    ) -> bool:
        """
        Check if a resource is available during a specific time period.

        Args:
            resource_id: ID of the resource to check
            start_time: Start time of the period to check
            end_time: End time of the period to check

        Returns:
            True if the resource is available, False otherwise
        """
        # Check if resource has availability windows covering the time period
        resource_availabilities = self._availabilities_data.get(resource_id, ())

        if not resource_availabilities:
            return False

        # Check if the time period is covered by availability windows
        has_coverage = False
        for availability in resource_availabilities:
            avail_start = availability["start_datetime"]
            avail_end = availability["end_datetime"]

            if avail_start <= start_time and avail_end >= end_time:
                has_coverage = True
                break

        if not has_coverage:
            return False

        # Check if there are any conflicting bookings
        has_conflict = self.find_overlapping_intervals(
            resource_id, start_time, end_time
        )

        return not has_conflict

    @profile_function()
    def get_earliest_availability(self, resource_id: str) -> Optional[datetime]:
        """
        Get the earliest availability time for a resource.

        Args:
            resource_id: ID of the resource

        Returns:
            Earliest availability datetime or None if no availability found
        """
        resource_availabilities = self._availabilities_data.get(resource_id, ())

        if not resource_availabilities:
            return None

        return min(avail["start_datetime"] for avail in resource_availabilities)

    @profile_function()
    def find_available_resources(
        self,
        resource_type: ResourceType,
        phase: TaskPhase,
        required_start: datetime,
        required_end: datetime,
        min_effort: float = 1.0,
        operation_id: Optional[str] = None,
    ) -> List[Resource]:
        """
        Find resources available for a specific time window with complex constraints

        Args:
            resource_type: Type of resource needed
            phase: Task phase the resource needs to support
            required_start: When the resource is needed from
            required_end: When the resource is needed until
            min_effort: Minimum effort/capacity required (0.0 to 1.0)
            operation_id: Specific operation if capability requires it

        Returns:
            List of available resources sorted by utilization (least busy first)
        """
        # Step 1: Filter by capability
        if operation_id is None:
            candidates = self._capabilities_by_type_phase.get(
                (resource_type.value, phase.value),
                (),
            )
        else:
            candidates = self._capabilities_by_type_phase_operation.get(
                (resource_type.value, phase.value, operation_id),
                (),
            )

        if not candidates:
            return []

        # Step 2: Check availability
        available_resource_ids = []
        for resource_id in candidates:
            if self.is_resource_available(resource_id, required_start, required_end):
                available_resource_ids.append(resource_id)

        # Step 3: Convert back to Resource objects and sort by utilization
        available_resources = []
        resource_utilizations = []

        for resource_id in available_resource_ids:
            resource = next(r for r in self.resources if r.resource_id == resource_id)
            available_resources.append(resource)

            # Calculate current utilization for sorting
            utilization = self._calculate_resource_utilization(resource_id)
            resource_utilizations.append(utilization)

        # Sort by utilization (least busy first)
        if resource_utilizations:
            sorted_pairs = sorted(
                zip(available_resources, resource_utilizations), key=lambda x: x[1]
            )
            return [resource for resource, _ in sorted_pairs]

        return available_resources

    @profile_function()
    def _calculate_resource_utilization(self, resource_id: str) -> float:
        """Calculate current utilization rate of a resource (0.0 to 1.0)"""
        resource_availabilities = self._availabilities_data.get(resource_id, [])

        if not resource_availabilities:
            return 0.0

        # Calculate utilization based on bookings vs availability
        total_available_seconds = 0
        total_booked_seconds = 0

        bookings = self._resource_booking_cache.get(resource_id, [])
        for availability in resource_availabilities:
            window_duration = (
                availability["end_datetime"] - availability["start_datetime"]
            ).total_seconds()
            total_available_seconds += window_duration

            # Calculate overlapping bookings with this availability window
            for booking_start, booking_end in bookings:
                overlap_start = max(availability["start_datetime"], booking_start)
                overlap_end = min(availability["end_datetime"], booking_end)

                if overlap_start < overlap_end:
                    total_booked_seconds += (
                        overlap_end - overlap_start
                    ).total_seconds()

        if total_available_seconds == 0:
            return 1.0  # Fully utilized if no availability

        return total_booked_seconds / total_available_seconds

    @profile_function()
    def get_best_available_resource(
        self,
        phase: TaskPhase,
        resource_requirement: OperationRequirement,
        required_start: datetime,
        required_end: datetime,
        operation_id: Optional[str] = None,
    ) -> Resource:
        """
        Get the best available resource considering availability, utilization, and requirements
        """
        available_resources = self.find_available_resources(
            resource_requirement.resource_type,
            phase,
            required_start,
            required_end,
            min_effort=resource_requirement.capacity,
            operation_id=operation_id,
        )

        if not available_resources:
            raise ValueError(
                f"No resource found for requirement: {resource_requirement}"
            )

        return available_resources[0]  # Already sorted by utilization

    @profile_function()
    def get_resource_with_requirement(
        self, phase: TaskPhase, resource_requirement: OperationRequirement
    ) -> Resource:
        """Get resource using capability matching"""
        candidates = self._capabilities_by_type_phase.get(
            (resource_requirement.resource_type.value, phase.value),
            (),
        )

        if not candidates:
            raise ValueError(
                f"No resource found for requirement: {resource_requirement}"
            )

        # Return the first capable resource
        first_id = candidates[0]
        return self.resources_by_id[first_id]

    @profile_function()
    def analyze_resource_conflicts(
        self,
        resource_requirements: List[Tuple[TaskPhase, OperationRequirement]],
        required_start: datetime,
        required_end: datetime,
    ) -> List[Dict]:
        """
        Analyze potential conflicts for multiple resource requirements
        Returns a list of dictionaries with conflict analysis
        """
        conflict_data = []

        for phase, requirement in resource_requirements:
            available_resources = self.find_available_resources(
                requirement.resource_type,
                phase,
                required_start,
                required_end,
                min_effort=requirement.capacity,
            )

            conflict_data.append(
                {
                    "phase": phase.value,
                    "resource_type": requirement.resource_type.value,
                    "capacity_required": requirement.capacity,
                    "available_count": len(available_resources),
                    "has_conflict": len(available_resources) == 0,
                    "available_resource_ids": [
                        r.resource_id for r in available_resources
                    ],
                }
            )

        return conflict_data

    @profile_function()
    def get_resource_bookings(
        self, resource_id: str
    ) -> List[Tuple[datetime, datetime]]:
        """
        Get all bookings for a specific resource.

        Args:
            resource_id: ID of the resource

        Returns:
            List of (start_time, end_time) tuples representing bookings
        """
        return list(self._resource_booking_cache.get(resource_id, ()))

    @profile_function()
    def get_all_bookings(self) -> Dict[str, List[Tuple[datetime, datetime]]]:
        """
        Get all resource bookings.

        Returns:
            Dictionary mapping resource_id to list of (start_time, end_time) tuples
        """
        return {
            resource_id: list(bookings)
            for resource_id, bookings in self._resource_booking_cache.items()
        }

    @profile_function()
    def find_all_machine_availability_windows(
        self,
        machine_id: str,
        total_duration: timedelta,
        earliest_start: Optional[datetime] = None,
        operation_id: Optional[str] = None,
        earliest_feasible_start: Optional[datetime] = None,
    ) -> List[tuple[datetime, datetime]]:
        """
        Find all availability windows for a specific machine that can fit the total duration.
        Returns windows in chronological order to enable retry logic.

        Args:
            machine_id: ID of the required machine
            total_duration: Total duration needed (setup + core_operation * quantity + cleanup)
            earliest_start: Earliest possible start time
            operation_id: Operation ID (unused in this simplified version)
            earliest_feasible_start: Earliest time when operators might be available

        Returns:
            List of (start_time, end_time) tuples for all suitable windows
        """
        # Use earliest_feasible_start if provided, otherwise fall back to earliest_start
        effective_earliest_start = earliest_feasible_start or earliest_start

        # Get machine availability windows
        machine_availabilities = self._availabilities_data.get(machine_id, ())

        if not machine_availabilities:
            return []

        suitable_windows = []

        # Check each machine availability window
        for machine_avail in machine_availabilities:
            machine_window_start = max(
                machine_avail["start_datetime"],
                effective_earliest_start or machine_avail["start_datetime"],
            )
            machine_window_end = machine_avail["end_datetime"]

            # Check if this window is large enough for the total duration
            if machine_window_end - machine_window_start < total_duration:
                continue

            # Check if this time slot is free for the machine (no existing bookings)
            if not self.find_overlapping_intervals(
                machine_id,
                machine_window_start,
                machine_window_start + total_duration,  # Check if operation can fit
            ):
                suitable_windows.append((machine_window_start, machine_window_end))

        return suitable_windows

    @profile_function()
    def find_machine_availability_window(
        self,
        machine_id: str,
        total_duration: timedelta,
        earliest_start: Optional[datetime] = None,
        operation_id: Optional[str] = None,
        earliest_feasible_start: Optional[datetime] = None,
    ) -> Optional[tuple[datetime, datetime]]:
        """
        Find the earliest availability window for a specific machine that can fit the total duration.
        Only checks machine availability - operator scheduling is handled separately in phase scheduling.

        Args:
            machine_id: ID of the required machine
            total_duration: Total duration needed (setup + core_operation * quantity + cleanup)
            earliest_start: Earliest possible start time
            operation_id: Operation ID (unused in this simplified version)
            earliest_feasible_start: Earliest time when operators might be available (priority over earliest_start)

        Returns:
            Tuple of (start_time, end_time) if available, None otherwise
        """
        # Use earliest_feasible_start if provided, otherwise fall back to earliest_start
        effective_earliest_start = earliest_feasible_start or earliest_start

        # Get machine availability windows
        machine_availabilities = self._availabilities_data.get(machine_id, ())

        if not machine_availabilities:

            return None

        # Check each machine availability window
        for machine_avail in machine_availabilities:
            machine_window_start = max(
                machine_avail["start_datetime"],
                effective_earliest_start or machine_avail["start_datetime"],
            )
            machine_window_end = machine_avail["end_datetime"]

            # Check if this window is large enough for the total duration
            if machine_window_end - machine_window_start < total_duration:

                continue

            # Check if this time slot is free for the machine (no existing bookings)
            # We only need to check that the operation can start somewhere in this window
            # Return the full machine availability window for operator scheduling flexibility
            if not self.find_overlapping_intervals(
                machine_id,
                machine_window_start,
                machine_window_start + total_duration,  # Check if operation can fit
            ):

                return machine_window_start, machine_window_end  # Return full window
            else:
                pass

        return None

    @profile_function()
    def find_available_operators_for_machine(
        self,
        machine: Resource,
        operator_requirements: List[OperatorRequirement],
        start_time: datetime,
        end_time: datetime,
        operation_id: str,
    ) -> Optional[List[Resource]]:
        """
        Find available operators that can work with the specified machine during the given time window.

        Args:
            machine: The machine resource that operators need to work with
            operator_requirements: List of operator requirements for this operation
            start_time: Start time of the operation
            end_time: End time of the operation
            operation_id: Operation ID for capability matching

        Returns:
            List of operator resources if all requirements can be met, None otherwise
        """
        assigned_operators = []

        for operator_req in operator_requirements:
            # Step 1: Find operators capable of this PHASE and OPERATION (strict)
            strict_ids = self._capabilities_by_type_phase_operation.get(
                (ResourceType.OPERATOR.value, operator_req.phase.value, operation_id),
                (),
            )
            strict_id_set = set(strict_ids)
            strict_capable_operators = [
                self.resources_by_id[rid] for rid in strict_ids
            ]

            # Step 2: Check availability among strict capable operators
            available_operators = []
            for operator in strict_capable_operators:
                if self.is_resource_available(
                    operator.resource_id, start_time, end_time
                ):
                    available_operators.append(operator)

            # Step 3: If not enough, FALLBACK to phase-only capability (operators with this phase for any operation)
            if len(available_operators) < operator_req.operator_count:
                phase_ids = self._capabilities_by_type_phase.get(
                    (ResourceType.OPERATOR.value, operator_req.phase.value),
                    (),
                )

                for rid in phase_ids:
                    if rid in strict_id_set:
                        continue
                    operator = self.resources_by_id.get(rid)
                    if operator and self.is_resource_available(
                        operator.resource_id, start_time, end_time
                    ):
                        available_operators.append(operator)
                        if len(available_operators) >= operator_req.operator_count:
                            break

            # Step 4: Final check: do we have enough operators?
            if len(available_operators) < operator_req.operator_count:
                return None  # Not enough operators available

            # Step 5: Assign the required number of operators (least utilized first)
            operators_with_utilization = []
            for operator in available_operators:
                utilization = self._calculate_resource_utilization(operator.resource_id)
                operators_with_utilization.append((operator, utilization))

            operators_with_utilization.sort(key=lambda x: x[1])
            selected_operators = [
                w[0] for w in operators_with_utilization[: operator_req.operator_count]
            ]
            assigned_operators.extend(selected_operators)

        return assigned_operators

    @profile_function()
    def book_machine_and_operators(
        self,
        machine: Resource,
        operators: List[Resource],
        start_time: datetime,
        end_time: datetime,
    ) -> None:
        """
        Book the machine and operators for the specified time period.

        Args:
            machine: The machine resource to book
            operators: The operator resources to book
            start_time: Start time of the booking
            end_time: End time of the booking
        """
        # Book the machine
        self._book_interval(machine.resource_id, start_time, end_time)
        self._update_resource_availabilities(machine.resource_id, start_time, end_time)

        # Book all operators
        for operator in operators:
            self._book_interval(operator.resource_id, start_time, end_time)
            self._update_resource_availabilities(
                operator.resource_id, start_time, end_time
            )

    @profile_function()
    def book_machine_idle(
        self,
        machine: Resource,
        start_time: datetime,
        end_time: datetime,
    ) -> None:
        """
        Book machine for idle period (no operators, just reservation).

        Args:
            machine: Machine to book
            start_time: Start time of the idle period
            end_time: End time of the idle period
        """
        self._book_interval(machine.resource_id, start_time, end_time)
        self._update_resource_availabilities(machine.resource_id, start_time, end_time)

    def free_intervals(
        self,
        resource_id: str,
        window_start: datetime,
        window_end: datetime,
    ) -> Iterable[Tuple[datetime, datetime]]:
        """Yield free intervals for a resource within the specified window."""
        availabilities = self._availabilities_data.get(resource_id, ())
        for availability in availabilities:
            start = max(availability["start_datetime"], window_start)
            end = min(availability["end_datetime"], window_end)
            if end > start:
                yield (start, end)

    def earliest_common_slot(
        self,
        resource_ids: List[str],
        start_time: datetime,
        duration: timedelta,
        horizon: datetime,
    ) -> Optional[Tuple[datetime, datetime]]:
        """Find the earliest slot shared by all resources within the horizon."""
        interval_iters: Dict[str, Iterable[Tuple[datetime, datetime]]] = {}
        current_intervals: Dict[str, Tuple[datetime, datetime]] = {}

        for resource_id in resource_ids:
            iterator = self.free_intervals(resource_id, start_time, horizon)
            interval_iters[resource_id] = iterator
            try:
                current_intervals[resource_id] = next(iterator)
            except StopIteration:
                return None

        while True:
            latest_start = max(interval[0] for interval in current_intervals.values())
            earliest_end = min(interval[1] for interval in current_intervals.values())

            if earliest_end - latest_start >= duration:
                return latest_start, latest_start + duration

            resource_to_advance = min(
                current_intervals,
                key=lambda rid: current_intervals[rid][1],
            )
            iterator = interval_iters[resource_to_advance]
            try:
                current_intervals[resource_to_advance] = next(iterator)
            except StopIteration:
                return None

    def clone(self) -> "ResourceManager":
        """Create a fresh ResourceManager copy for reuse."""
        return ResourceManager(self.original_resources)

    def reset_bookings(self) -> None:
        """Reset bookings and restore base availability state."""
        for resource in self.resources:
            template = self._base_availability_templates.get(resource.resource_id, [])
            resource.availabilities = [
                ResourceAvailability(
                    start_datetime=start,
                    end_datetime=end,
                    effort=effort,
                )
                for start, end, effort in template
            ]
            self._resource_booking_trees[resource.resource_id] = IntervalTree()
            self._resource_booking_cache[resource.resource_id].clear()
            self._rebuild_resource_availability(resource)
