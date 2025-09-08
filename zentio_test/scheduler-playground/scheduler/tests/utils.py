from datetime import datetime, timedelta, date
from enum import Enum
from typing import Dict, List, Optional

from scheduler.models import (
    TaskPhase,
    OperationRequirement,
    OperatorRequirement,
    ResourceAvailability,
    ResourceCapability,
    ResourceType,
)


class Shift(Enum):
    MORNING = "morning"
    AFTERNOON = "afternoon"
    NIGHT = "night"
    ALL_DAY = "all_day"


def generate_availabilities(
    shift: Shift, start_date: date, span_days: int
) -> List[ResourceAvailability]:
    match shift:
        case Shift.MORNING:
            # 6:00 - 14:00
            return [
                ResourceAvailability(
                    start_datetime=datetime.combine(
                        start_date + timedelta(days=i), datetime.min.time()
                    )
                    + timedelta(hours=6),
                    end_datetime=datetime.combine(
                        start_date + timedelta(days=i), datetime.min.time()
                    )
                    + timedelta(hours=14),
                    effort=1.0,
                )
                for i in range(span_days)
            ]
        case Shift.AFTERNOON:
            # 14:00 - 22:00
            return [
                ResourceAvailability(
                    start_datetime=datetime.combine(
                        start_date + timedelta(days=i), datetime.min.time()
                    )
                    + timedelta(hours=14),
                    end_datetime=datetime.combine(
                        start_date + timedelta(days=i), datetime.min.time()
                    )
                    + timedelta(hours=22),
                    effort=1.0,
                )
                for i in range(span_days)
            ]
        case Shift.NIGHT:
            # 22:00 - 6:00
            return [
                ResourceAvailability(
                    start_datetime=datetime.combine(
                        start_date + timedelta(days=i), datetime.min.time()
                    )
                    + timedelta(hours=22),
                    end_datetime=datetime.combine(
                        start_date + timedelta(days=i), datetime.min.time()
                    )
                    + timedelta(hours=6),
                    effort=1.0,
                )
                for i in range(span_days)
            ]
        case Shift.ALL_DAY:
            # 00:00 - 23:59
            return [
                ResourceAvailability(
                    start_datetime=datetime.combine(start_date, datetime.min.time()),
                    end_datetime=datetime.combine(
                        start_date + timedelta(days=span_days), datetime.min.time()
                    ),
                    effort=1.0,
                )
            ]


def generate_capabilities(
    resource_type: ResourceType,
    operation_id: str,
    phases: Optional[List[TaskPhase]] = None,
) -> List[ResourceCapability]:
    phases = [phase for phase in TaskPhase] if phases is None else phases
    return [
        ResourceCapability(
            resource_type=resource_type, operation_id=operation_id, phase=phase
        )
        for phase in phases
    ]


def generate_requirements(
    setup: Optional[List[ResourceType]] = None,
    core_operation: Optional[List[ResourceType]] = None,
    cleanup: Optional[List[ResourceType]] = None,
) -> Dict[TaskPhase, List[OperationRequirement]]:
    return {
        TaskPhase.SETUP: (
            []
            if setup is None
            else [
                OperationRequirement(
                    resource_type=resource_type, phase=TaskPhase.SETUP, capacity=1.0
                )
                for resource_type in setup
            ]
        ),
        TaskPhase.CORE_OPERATION: (
            []
            if core_operation is None
            else [
                OperationRequirement(
                    resource_type=resource_type,
                    phase=TaskPhase.CORE_OPERATION,
                    capacity=1.0,
                )
                for resource_type in core_operation
            ]
        ),
        TaskPhase.CLEANUP: (
            []
            if cleanup is None
            else [
                OperationRequirement(
                    resource_type=resource_type, phase=TaskPhase.CLEANUP, capacity=1.0
                )
                for resource_type in cleanup
            ]
        ),
    }


def generate_operator_requirements(
    setup_operators: int = 0,
    core_operation_operators: int = 0,
    cleanup_operators: int = 0,
    effort: float = 1.0,  # Default 100% effort
) -> Dict[TaskPhase, List[OperatorRequirement]]:
    """Generate operator requirements for operation phases"""
    requirements = {}

    if setup_operators > 0:
        requirements[TaskPhase.SETUP] = [
            OperatorRequirement(
                phase=TaskPhase.SETUP, operator_count=setup_operators, effort=effort
            )
        ]

    if core_operation_operators > 0:
        requirements[TaskPhase.CORE_OPERATION] = [
            OperatorRequirement(
                phase=TaskPhase.CORE_OPERATION,
                operator_count=core_operation_operators,
                effort=effort,
            )
        ]

    if cleanup_operators > 0:
        requirements[TaskPhase.CLEANUP] = [
            OperatorRequirement(
                phase=TaskPhase.CLEANUP, operator_count=cleanup_operators, effort=effort
            )
        ]

    return requirements
