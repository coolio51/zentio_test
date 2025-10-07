from __future__ import annotations

from typing import TYPE_CHECKING, List

try:
    from rich import box, table
except Exception:  # pragma: no cover - optional dependency fallback
    class _NullTable:
        def __init__(self, *args, **kwargs):
            pass

        def add_column(self, *args, **kwargs):
            return None

        def add_row(self, *args, **kwargs):
            return None

        def add_section(self, *args, **kwargs):
            return None

    class _Box:
        ROUNDED = None
        SIMPLE = None

    class _TableModule:
        Table = _NullTable

    box = _Box()  # type: ignore
    table = _TableModule()  # type: ignore

from scheduler.models import Resource, ResourceType
from scheduler.utils.utils import style_datetime, style_duration

if TYPE_CHECKING:  # pragma: no cover - typing only
    from scheduler.services.resource_manager import ResourceManager

RESOURCE_COLORS = {
    ResourceType.MACHINE: "cyan",
    ResourceType.OPERATOR: "blue",
}


class ResourceLogger:
    @staticmethod
    def initial_availability_table(resources: List[Resource]) -> table.Table:
        """Create a table showing initial resource availability periods"""
        availability_table = table.Table(
            title="Initial Resource Availability",
            title_style="bold green",
            style="dim",
            box=box.ROUNDED,
        )

        availability_table.add_column("Resource ID", style="bold")
        availability_table.add_column("Type")
        availability_table.add_column("Name", style="italic")
        availability_table.add_column("Start Time")
        availability_table.add_column("End Time")
        availability_table.add_column("Duration")
        availability_table.add_column("Effort", justify="right")

        for resource in sorted(
            resources, key=lambda r: (r.resource_type.value, r.resource_name)
        ):
            availability_table.add_section()
            if not resource.availabilities:
                # Show resources with no availability
                availability_table.add_row(
                    f"[{RESOURCE_COLORS[resource.resource_type]}]{resource.resource_id}[/{RESOURCE_COLORS[resource.resource_type]}]",
                    f"[{RESOURCE_COLORS[resource.resource_type]}]{resource.resource_type.value.upper()}[/{RESOURCE_COLORS[resource.resource_type]}]",
                    f"[{RESOURCE_COLORS[resource.resource_type]}]{resource.resource_name}[/{RESOURCE_COLORS[resource.resource_type]}]",
                    "[red]No availability[/red]",
                    "",
                    "",
                    "",
                )
                continue

            # Show first availability period
            first_availability = resource.availabilities[0]
            duration = (
                first_availability.end_datetime - first_availability.start_datetime
            )

            availability_table.add_row(
                f"[{RESOURCE_COLORS[resource.resource_type]}]{resource.resource_id}[/{RESOURCE_COLORS[resource.resource_type]}]",
                f"[{RESOURCE_COLORS[resource.resource_type]}]{resource.resource_type.value.upper()}[/{RESOURCE_COLORS[resource.resource_type]}]",
                f"[{RESOURCE_COLORS[resource.resource_type]}]{resource.resource_name}[/{RESOURCE_COLORS[resource.resource_type]}]",
                style_datetime(first_availability.start_datetime),
                style_datetime(first_availability.end_datetime),
                style_duration(duration),
                f"{first_availability.effort:.0%}",
            )

            # Show additional availability periods if any
            for availability in resource.availabilities[1:]:
                duration = availability.end_datetime - availability.start_datetime
                availability_table.add_row(
                    "",  # Empty resource ID for continuation
                    "",  # Empty type
                    "",  # Empty name
                    style_datetime(availability.start_datetime),
                    style_datetime(availability.end_datetime),
                    style_duration(duration),
                    f"{availability.effort:.0%}",
                )

        return availability_table

    @staticmethod
    def resource_bookings_table(resource_manager: ResourceManager) -> table.Table:
        """Create a table showing current resource bookings"""
        bookings_table = table.Table(
            title="Resource Bookings (After Scheduling)",
            title_style="bold red",
            style="dim",
            box=box.ROUNDED,
        )

        bookings_table.add_column("Resource ID", style="italic")
        bookings_table.add_column("Name", style="italic")
        bookings_table.add_column("Type")
        bookings_table.add_column("Booking Start")
        bookings_table.add_column("Booking End")
        bookings_table.add_column("Duration")
        bookings_table.add_column("Utilization", justify="right")

        for resource in sorted(
            resource_manager.resources,
            key=lambda r: (r.resource_type.value, r.resource_name),
        ):
            bookings_table.add_section()
            bookings = resource_manager._resource_bookings.get(resource.resource_id, [])
            utilization = resource_manager._calculate_resource_utilization(
                resource.resource_id
            )

            if not bookings:
                # Show resources with no bookings
                bookings_table.add_row(
                    f"[{RESOURCE_COLORS[resource.resource_type]}]{resource.resource_id}[/{RESOURCE_COLORS[resource.resource_type]}]",
                    f"[{RESOURCE_COLORS[resource.resource_type]}]{resource.resource_name}[/{RESOURCE_COLORS[resource.resource_type]}]",
                    f"[{RESOURCE_COLORS[resource.resource_type]}]{resource.resource_type.value.upper()}[/{RESOURCE_COLORS[resource.resource_type]}]",
                    "[green]No bookings[/green]",
                    "",
                    "",
                    f"[green]{utilization:.0%}[/green]",
                )
                continue

            # Sort bookings by start time
            sorted_bookings = sorted(bookings, key=lambda b: b[0])

            # Show first booking
            first_booking = sorted_bookings[0]
            duration = first_booking[1] - first_booking[0]

            bookings_table.add_row(
                f"[{RESOURCE_COLORS[resource.resource_type]}]{resource.resource_id}[/{RESOURCE_COLORS[resource.resource_type]}]",
                f"[{RESOURCE_COLORS[resource.resource_type]}]{resource.resource_name}[/{RESOURCE_COLORS[resource.resource_type]}]",
                f"[{RESOURCE_COLORS[resource.resource_type]}]{resource.resource_type.value.upper()}[/{RESOURCE_COLORS[resource.resource_type]}]",
                style_datetime(first_booking[0]),
                style_datetime(first_booking[1]),
                style_duration(duration),
                f"[yellow]{utilization:.0%}[/yellow]",
            )

            # Show additional bookings if any
            for booking_start, booking_end in sorted_bookings[1:]:
                duration = booking_end - booking_start
                bookings_table.add_row(
                    "",  # Empty resource ID for continuation
                    "",  # Empty type
                    "",  # Empty name
                    style_datetime(booking_start),
                    style_datetime(booking_end),
                    style_duration(duration),
                    "",  # Empty utilization for continuation rows
                )

        return bookings_table

    @staticmethod
    def resource_summary_table(resource_manager: ResourceManager) -> table.Table:
        """Create a summary table of resource statistics"""
        summary_table = table.Table(
            title="Resource Summary",
            title_style="bold magenta",
            style="dim",
            box=box.SIMPLE,
        )

        summary_table.add_column("Metric", style="bold")
        summary_table.add_column("Value", justify="right")

        total_resources = len(resource_manager.resources)
        machine_count = sum(
            1
            for r in resource_manager.resources
            if r.resource_type == ResourceType.MACHINE
        )
        operator_count = sum(
            1
            for r in resource_manager.resources
            if r.resource_type == ResourceType.OPERATOR
        )

        # Count resources with bookings
        booked_resources = sum(
            1 for bookings in resource_manager._resource_bookings.values() if bookings
        )

        # Calculate average utilization
        utilizations = [
            resource_manager._calculate_resource_utilization(r.resource_id)
            for r in resource_manager.resources
        ]
        avg_utilization = sum(utilizations) / len(utilizations) if utilizations else 0

        summary_table.add_row("Total Resources", str(total_resources))
        summary_table.add_row("Machines", str(machine_count))
        summary_table.add_row("Operators", str(operator_count))
        summary_table.add_row("Resources with Bookings", str(booked_resources))
        summary_table.add_row("Average Utilization", f"{avg_utilization:.1%}")

        return summary_table

    @staticmethod
    def resource_timeline_table(
        resource_manager: ResourceManager, resource_id: str
    ) -> table.Table:
        """Create a detailed timeline for a specific resource showing availability and bookings"""
        resource = next(
            (r for r in resource_manager.resources if r.resource_id == resource_id),
            None,
        )
        if not resource:
            raise ValueError(f"Resource {resource_id} not found")

        timeline_table = table.Table(
            title=f"Resource Timeline: {resource.resource_name} ({resource_id})",
            title_style="bold cyan",
            style="dim",
            box=box.ROUNDED,
        )

        timeline_table.add_column("Type", style="bold")
        timeline_table.add_column("Start Time")
        timeline_table.add_column("End Time")
        timeline_table.add_column("Duration")
        timeline_table.add_column("Details")

        # Get bookings for this resource
        bookings = resource_manager._resource_bookings.get(resource_id, [])

        # Create a proper timeline by processing each availability window
        timeline_events = []

        for availability in resource.availabilities:
            avail_start = availability.start_datetime
            avail_end = availability.end_datetime

            # Find bookings that overlap with this availability window
            overlapping_bookings = []
            for booking_start, booking_end in bookings:
                # Check if booking overlaps with availability window
                if booking_start < avail_end and booking_end > avail_start:
                    # Trim booking to availability window bounds
                    trimmed_start = max(booking_start, avail_start)
                    trimmed_end = min(booking_end, avail_end)
                    overlapping_bookings.append((trimmed_start, trimmed_end))

            # Sort bookings by start time
            overlapping_bookings.sort()

            # Create timeline segments within this availability window
            current_time = avail_start

            for booking_start, booking_end in overlapping_bookings:
                # Add available period before booking (if any)
                if current_time < booking_start:
                    timeline_events.append(
                        {
                            "type": "availability",
                            "start": current_time,
                            "end": booking_start,
                            "details": f"Available (effort: {availability.effort:.0%})",
                        }
                    )

                # Add booking period
                timeline_events.append(
                    {
                        "type": "booking",
                        "start": booking_start,
                        "end": booking_end,
                        "details": "Booked",
                    }
                )

                current_time = booking_end

            # Add remaining available period after last booking (if any)
            if current_time < avail_end:
                timeline_events.append(
                    {
                        "type": "availability",
                        "start": current_time,
                        "end": avail_end,
                        "details": f"Available (effort: {availability.effort:.0%})",
                    }
                )

        # Sort all events by start time
        timeline_events.sort(key=lambda x: x["start"])

        # Add events to table
        for event in timeline_events:
            duration = event["end"] - event["start"]
            event_type = (
                "[green]AVAILABLE[/green]"
                if event["type"] == "availability"
                else "[red]BOOKED[/red]"
            )

            timeline_table.add_row(
                event_type,
                style_datetime(event["start"]),
                style_datetime(event["end"]),
                style_duration(duration),
                event["details"],
            )

        return timeline_table
