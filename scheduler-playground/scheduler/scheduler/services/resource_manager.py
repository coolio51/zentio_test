from datetime import datetime, timedelta
from typing import Optional, List, Tuple, Dict

from scheduler.models import (
    Resource,
    TaskPhase,
    OperationRequirement,
    ResourceType,
    OperatorRequirement,
    ResourceAvailability,
    Idle,
)


class ResourceManager:
    def __init__(self, resources: list[Resource]):
        self.original_resources = resources  # Keep reference to original

        # Create our own managed resources with dynamic availabilities
        self.resources = self._create_managed_resources(resources)

        # Track resource bookings (resource_id -> list of (start, end) tuples)
        self._resource_bookings: Dict[str, List[Tuple[datetime, datetime]]] = {
            resource.resource_id: [] for resource in self.resources
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

    def _build_data_structures(self):
        """Build data structures for complex availability operations"""
        # Resources data
        self._resources_data = {
            resource.resource_id: {
                "resource_id": resource.resource_id,
                "resource_type": resource.resource_type.value,
                "resource_name": resource.resource_name,
            }
            for resource in self.resources
        }

        # Capabilities data
        self._capabilities_data = []
        for resource in self.resources:
            for capability in resource.resource_capabilities:
                self._capabilities_data.append(
                    {
                        "resource_id": resource.resource_id,
                        "resource_type": capability.resource_type.value,
                        "phase": capability.phase.value,
                        "operation_id": capability.operation_id,
                    }
                )

        # Availabilities data - this will be rebuilt when availabilities change
        self._rebuild_availabilities_data()

    def _rebuild_availabilities_data(self):
        """Rebuild availabilities data from current resource states"""
        self._availabilities_data = []
        for resource in self.resources:
            for availability in resource.availabilities:
                self._availabilities_data.append(
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

    def _update_resource_availabilities(
        self, resource_id: str, start_time: datetime, end_time: datetime
    ):
        """Update the actual availability windows by removing/splitting booked time"""
        resource = next(
            (r for r in self.resources if r.resource_id == resource_id), None
        )
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

        # Rebuild the availability data structures
        self._rebuild_availabilities_data()

    def get_current_availabilities(self) -> list[Resource]:
        """Get the current state of all resources with their dynamic availabilities"""
        return self.resources

    def find_overlapping_intervals(
        self, intervals: List[Tuple[datetime, datetime]], start: datetime, end: datetime
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
        for interval_start, interval_end in intervals:
            # Overlapping condition: new_start < existing_end AND new_end > existing_start
            if start < interval_end and end > interval_start:
                return True
        return False

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

        capable_resource_ids = set()
        for capability in self._capabilities_data:
            if (
                capability["resource_type"] == resource_type.value
                and capability["phase"] == phase.value
                and capability["operation_id"] == base_operation_id
            ):
                capable_resource_ids.add(capability["resource_id"])

        if len(capable_resource_ids) == 0:
            return []

        # Return actual Resource objects
        return [r for r in self.resources if r.resource_id in capable_resource_ids]

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
        resource_availabilities = [
            avail
            for avail in self._availabilities_data
            if avail["resource_id"] == resource_id
        ]

        if not resource_availabilities:
            return []

        # Sort by start_datetime
        resource_availabilities.sort(key=lambda x: x["start_datetime"])

        available_slots = []
        existing_bookings = self._resource_bookings.get(resource_id, [])

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
                slot_start, slot_end, duration, existing_bookings, available_slots
            )

        return available_slots

    def _find_slots_in_availability_window(
        self,
        window_start: datetime,
        window_end: datetime,
        duration: timedelta,
        existing_bookings: List[Tuple[datetime, datetime]],
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
                existing_bookings, current_time, proposed_end
            ):
                available_slots.append((current_time, proposed_end))
                current_time = proposed_end
            else:
                # Find the next potential start time after the conflicting booking
                next_start = current_time + timedelta(minutes=1)
                for booking_start, booking_end in existing_bookings:
                    if booking_start <= current_time < booking_end:
                        next_start = booking_end
                        break
                current_time = next_start

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
            self._resource_bookings[resource.resource_id].append((start_time, end_time))
            self._update_resource_availabilities(
                resource.resource_id, start_time, end_time
            )

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
            booking_to_remove = (start_time, end_time)
            if booking_to_remove in self._resource_bookings[resource.resource_id]:
                self._resource_bookings[resource.resource_id].remove(booking_to_remove)
                self._update_resource_availabilities(
                    resource.resource_id, start_time, end_time
                )

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
        resource_availabilities = [
            avail
            for avail in self._availabilities_data
            if avail["resource_id"] == resource_id
        ]

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
        existing_bookings = self._resource_bookings.get(resource_id, [])
        has_conflict = self.find_overlapping_intervals(
            existing_bookings, start_time, end_time
        )

        return not has_conflict

    def get_earliest_availability(self, resource_id: str) -> Optional[datetime]:
        """
        Get the earliest availability time for a resource.

        Args:
            resource_id: ID of the resource

        Returns:
            Earliest availability datetime or None if no availability found
        """
        resource_availabilities = [
            avail
            for avail in self._availabilities_data
            if avail["resource_id"] == resource_id
        ]

        if not resource_availabilities:
            return None

        return min(avail["start_datetime"] for avail in resource_availabilities)

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
        capable_resources = set()
        for capability in self._capabilities_data:
            if (
                capability["resource_type"] == resource_type.value
                and capability["phase"] == phase.value
            ):
                if operation_id is None or capability["operation_id"] == operation_id:
                    capable_resources.add(capability["resource_id"])

        if len(capable_resources) == 0:
            return []

        if not self._availabilities_data:
            # No availability data, return all capable resources
            return [r for r in self.resources if r.resource_id in capable_resources]

        # Step 2: Check availability
        available_resource_ids = []
        for resource_id in capable_resources:
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

    def _calculate_resource_utilization(self, resource_id: str) -> float:
        """Calculate current utilization rate of a resource (0.0 to 1.0)"""
        resource_availabilities = [
            avail
            for avail in self._availabilities_data
            if avail["resource_id"] == resource_id
        ]

        if not resource_availabilities:
            return 0.0

        # Calculate utilization based on bookings vs availability
        total_available_seconds = 0
        total_booked_seconds = 0

        for availability in resource_availabilities:
            window_duration = (
                availability["end_datetime"] - availability["start_datetime"]
            ).total_seconds()
            total_available_seconds += window_duration

            # Calculate overlapping bookings with this availability window
            bookings = self._resource_bookings.get(resource_id, [])
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

    def get_resource_with_requirement(
        self, phase: TaskPhase, resource_requirement: OperationRequirement
    ) -> Resource:
        """Get resource using capability matching"""
        # Use capability matching
        capable_resource_ids = set()
        for capability in self._capabilities_data:
            if (
                capability["resource_type"] == resource_requirement.resource_type.value
                and capability["phase"] == phase.value
            ):
                capable_resource_ids.add(capability["resource_id"])

        if len(capable_resource_ids) == 0:
            raise ValueError(
                f"No resource found for requirement: {resource_requirement}"
            )

        # Return the first capable resource
        return next(r for r in self.resources if r.resource_id in capable_resource_ids)

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
        return self._resource_bookings.get(resource_id, []).copy()

    def get_all_bookings(self) -> Dict[str, List[Tuple[datetime, datetime]]]:
        """
        Get all resource bookings.

        Returns:
            Dictionary mapping resource_id to list of (start_time, end_time) tuples
        """
        return {
            resource_id: bookings.copy()
            for resource_id, bookings in self._resource_bookings.items()
        }

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
        machine_availabilities = [
            avail
            for avail in self._availabilities_data
            if avail["resource_id"] == machine_id
        ]

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
                self._resource_bookings.get(machine_id, []),
                machine_window_start,
                machine_window_start + total_duration,  # Check if operation can fit
            ):
                suitable_windows.append((machine_window_start, machine_window_end))

        return suitable_windows

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
        machine_availabilities = [
            avail
            for avail in self._availabilities_data
            if avail["resource_id"] == machine_id
        ]

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
                self._resource_bookings.get(machine_id, []),
                machine_window_start,
                machine_window_start + total_duration,  # Check if operation can fit
            ):

                return machine_window_start, machine_window_end  # Return full window
            else:
                pass

        return None

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
            strict_capable_operators = []
            for capability in self._capabilities_data:
                if (
                    capability["resource_type"] == ResourceType.OPERATOR.value
                    and capability["phase"] == operator_req.phase.value
                    and capability["operation_id"] == operation_id
                ):
                    operator = next(
                        (
                            r
                            for r in self.resources
                            if r.resource_id == capability["resource_id"]
                        ),
                        None,
                    )
                    if operator:
                        strict_capable_operators.append(operator)

            # Step 2: Check availability among strict capable operators
            available_operators = []
            for operator in strict_capable_operators:
                if self.is_resource_available(
                    operator.resource_id, start_time, end_time
                ):
                    available_operators.append(operator)

            # Step 3: If not enough, FALLBACK to phase-only capability (operators with this phase for any operation)
            if len(available_operators) < operator_req.operator_count:
                phase_only_capable_operators = []
                strict_ids = {w.resource_id for w in strict_capable_operators}
                for capability in self._capabilities_data:
                    if (
                        capability["resource_type"] == ResourceType.OPERATOR.value
                        and capability["phase"] == operator_req.phase.value
                    ):
                        operator = next(
                            (
                                r
                                for r in self.resources
                                if r.resource_id == capability["resource_id"]
                            ),
                            None,
                        )
                        if operator and operator.resource_id not in strict_ids:
                            phase_only_capable_operators.append(operator)

                # Add available phase-only operators until we meet the requirement
                for operator in phase_only_capable_operators:
                    if self.is_resource_available(
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
        if machine.resource_id not in self._resource_bookings:
            self._resource_bookings[machine.resource_id] = []
        self._resource_bookings[machine.resource_id].append((start_time, end_time))
        self._update_resource_availabilities(machine.resource_id, start_time, end_time)

        # Book all operators
        for operator in operators:
            if operator.resource_id not in self._resource_bookings:
                self._resource_bookings[operator.resource_id] = []
            self._resource_bookings[operator.resource_id].append((start_time, end_time))
            self._update_resource_availabilities(
                operator.resource_id, start_time, end_time
            )

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
        if machine.resource_id not in self._resource_bookings:
            self._resource_bookings[machine.resource_id] = []
        self._resource_bookings[machine.resource_id].append((start_time, end_time))
        self._update_resource_availabilities(machine.resource_id, start_time, end_time)
