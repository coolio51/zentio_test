from typing import Optional
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

from scheduler.models import (
    Resource,
    Schedule,
    TaskPhase,
    ResourceType,
    DropReason,
)
from scheduler.utils.utils import style_datetime
from scheduler.common.console import get_console

PHASE_COLORS = {
    TaskPhase.SETUP: "yellow",
    TaskPhase.CORE_OPERATION: "green",
    TaskPhase.CLEANUP: "blue",
}

RESOURCE_COLORS = {
    ResourceType.MACHINE: "white",
    ResourceType.OPERATOR: "blue",
}

DROP_REASON_COLORS = {
    DropReason.PHASE_SCHEDULING_FAILED: "red",
    DropReason.DEPENDENCY_DROPPED: "yellow",
    DropReason.CIRCULAR_DEPENDENCY: "magenta",
}


class ScheduleLogger:

    @staticmethod
    def style_phase(phase: TaskPhase) -> str:
        """Apply conditional styling to a task phase."""
        color = PHASE_COLORS.get(phase, "white")
        return f"[{color}]{phase.value}[/{color}]"

    @staticmethod
    def style_resource(resource: Resource) -> str:
        """Apply conditional styling to a resource."""
        color = RESOURCE_COLORS.get(resource.resource_type, "white")
        return f"[{color}]{resource.resource_name}[/{color}]"

    @staticmethod
    def style_drop_reason(reason: DropReason) -> str:
        """Apply conditional styling to a drop reason."""
        color = DROP_REASON_COLORS.get(reason, "white")
        reason_display = {
            DropReason.PHASE_SCHEDULING_FAILED: "Phase Scheduling Failed",
            DropReason.DEPENDENCY_DROPPED: "Dependency Dropped",
            DropReason.CIRCULAR_DEPENDENCY: "Circular Dependency",
        }.get(reason, reason.value)
        return f"[{color}]{reason_display}[/{color}]"

    @staticmethod
    def schedule_summary_table(schedule: Schedule) -> table.Table:
        # Pretty print summary, fitness score, generation number, individual number
        summary_table = table.Table(style="dim", box=box.SIMPLE)
        summary_table.add_column("Tasks (phases)")
        summary_table.add_column("Exec. Units (unique op instances)")
        summary_table.add_column("Unique Ops (MO,Op)")
        summary_table.add_column("Resources")
        summary_table.add_column("Fitness")
        summary_table.add_column("Makespan")

        # Compute richer counts
        tasks_count = len(schedule.tasks)
        unique_instance_ids = {
            task.operation_instance_id
            for task in schedule.tasks
            if task.operation_instance_id
        }
        exec_units = len(unique_instance_ids)

        unique_mo_op = set()
        for task in schedule.tasks:
            mo_id = task.operation.manufacturing_order_id
            op_id = task.operation.operation_id
            if mo_id and op_id:
                unique_mo_op.add((mo_id, op_id))
        unique_ops = len(unique_mo_op)

        summary_table.add_row(
            str(tasks_count),
            str(exec_units),
            str(unique_ops),
            str(schedule.number_of_resources_scheduled),
            str(schedule.fitness_score),
            str(schedule.makespan),
        )
        return summary_table

    @staticmethod
    def schedule_table(schedule: Schedule, title: Optional[str] = None) -> table.Table:
        tasks_table = table.Table(
            title=title,
            title_style="bold red",
            style="dim",
            box=box.ROUNDED,
        )
        tasks_table.add_column("Manufacturing order", style="italic")
        tasks_table.add_column("Operation Instance", style="cyan")
        tasks_table.add_column("Operation Name", style="italic")
        tasks_table.add_column("Phase")
        tasks_table.add_column("Quantity", style="italic")
        tasks_table.add_column("Datetime start")
        tasks_table.add_column("Datetime end")
        tasks_table.add_column("Resources", style="italic")

        # Group tasks by manufacturing order, then by operation instance
        tasks_by_mo = {}
        for task in schedule.tasks:
            mo_name = task.operation.manufacturing_order_name or "Unknown MO"
            if mo_name not in tasks_by_mo:
                tasks_by_mo[mo_name] = {}

            op_instance_id = task.operation_instance_id
            if op_instance_id not in tasks_by_mo[mo_name]:
                tasks_by_mo[mo_name][op_instance_id] = []

            tasks_by_mo[mo_name][op_instance_id].append(task)

        # Add tasks to table with hierarchical grouping
        for mo_index, (mo_name, operations) in enumerate(tasks_by_mo.items()):
            mo_displayed = False
            tasks_table.add_section()

            for op_index, (op_instance_id, tasks) in enumerate(operations.items()):
                # Sort tasks within each operation instance by start time
                tasks.sort(key=lambda x: x.datetime_start)
                op_instance_displayed = False

                for task_index, task in enumerate(tasks):
                    # Display MO name only for the first task of the manufacturing order
                    mo_display = mo_name if not mo_displayed else ""
                    if mo_display:
                        mo_displayed = True

                    # Display operation instance ID only for the first task of each operation instance
                    op_instance_display = (
                        op_instance_id if not op_instance_displayed else ""
                    )
                    op_name_display = (
                        task.operation.operation_name
                        if not op_instance_displayed
                        else ""
                    )
                    if op_instance_display:
                        op_instance_displayed = True

                    tasks_table.add_row(
                        mo_display,
                        op_instance_display,
                        op_name_display,
                        ScheduleLogger.style_phase(task.phase),
                        str(task.quantity),
                        style_datetime(task.datetime_start),
                        style_datetime(task.datetime_end),
                        ", ".join(
                            [
                                ScheduleLogger.style_resource(resource)
                                for resource in ([task.machine] if task.machine else [])
                                + task.operators
                            ]
                        ),
                    )

                if op_index < len(operations) - 1:
                    tasks_table.add_row("", "", "", "", "", "", "", "", style="dim")

        return tasks_table

    @staticmethod
    def dropped_operations_table(
        schedule: Schedule, title: Optional[str] = None
    ) -> Optional[table.Table]:
        """Create a table for dropped operations."""
        if not schedule.dropped_operations:
            return None

        dropped_table = table.Table(
            title=title or "Dropped Operations",
            title_style="bold red",
            style="dim",
            box=box.ROUNDED,
        )
        dropped_table.add_column("Manufacturing Order", style="italic")
        dropped_table.add_column("Operation ID", style="cyan")
        dropped_table.add_column("Operation Name", style="italic")
        dropped_table.add_column("Reason")
        dropped_table.add_column("Failed Phase", style="yellow")
        dropped_table.add_column("Dependent Operation", style="yellow")
        dropped_table.add_column("Error Details", style="dim")

        # Group by manufacturing order for better organization
        dropped_by_mo = {}
        for dropped_op in schedule.dropped_operations:
            mo_id = dropped_op.manufacturing_order_id
            if mo_id not in dropped_by_mo:
                dropped_by_mo[mo_id] = []
            dropped_by_mo[mo_id].append(dropped_op)

        for mo_index, (mo_id, operations) in enumerate(dropped_by_mo.items()):
            if mo_index > 0:
                dropped_table.add_section()

            mo_displayed = False
            for op in operations:
                mo_display = mo_id if not mo_displayed else ""
                if mo_display:
                    mo_displayed = True

                failed_phase_display = (
                    ScheduleLogger.style_phase(op.failed_phase)
                    if op.failed_phase
                    else ""
                )

                dependent_op_display = op.dependent_operation_id or ""

                error_details = op.error_message or ""
                # Truncate long error messages
                if len(error_details) > 50:
                    error_details = error_details[:47] + "..."

                dropped_table.add_row(
                    mo_display,
                    op.operation_id,
                    op.operation_name,
                    ScheduleLogger.style_drop_reason(op.reason),
                    failed_phase_display,
                    dependent_op_display,
                    error_details,
                )

        return dropped_table

    @staticmethod
    def print(schedule: Schedule, title: Optional[str] = None) -> None:
        console = get_console()
        # console.print(ScheduleLogger.schedule_table(schedule, title=title))
        # console.print(ScheduleLogger.schedule_summary_table(schedule))

        # Print dropped operations table if there are any
        dropped_table = ScheduleLogger.dropped_operations_table(schedule)
        if dropped_table:
            # console.print()  # Add spacing
            # console.print(dropped_table)
            pass
