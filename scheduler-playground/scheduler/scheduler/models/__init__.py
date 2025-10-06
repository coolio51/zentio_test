from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Optional

# MARK: - Enums


class ResourceType(Enum):
    MACHINE = "machine"
    OPERATOR = "operator"


class TaskPhase(Enum):
    SETUP = "setup"
    CORE_OPERATION = "core_operation"
    CLEANUP = "cleanup"


class DropReason(Enum):
    PHASE_SCHEDULING_FAILED = "phase_scheduling_failed"
    DEPENDENCY_DROPPED = "dependency_dropped"
    CIRCULAR_DEPENDENCY = "circular_dependency"


class Priority(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    URGENT = "urgent"


# MARK: - Models


@dataclass
class ResourceAvailability:
    start_datetime: datetime
    end_datetime: datetime
    effort: float


@dataclass
class ResourceCapability:
    resource_type: ResourceType
    phase: TaskPhase
    operation_id: str


@dataclass
class Resource:
    resource_id: str
    resource_type: ResourceType
    resource_name: str
    resource_capabilities: list[ResourceCapability]
    availabilities: list[ResourceAvailability]


@dataclass
class OperationRequirement:
    phase: TaskPhase
    resource_type: ResourceType
    capacity: float  # 0.0 -> 1.0


@dataclass
class OperatorRequirement:
    phase: TaskPhase
    operator_count: int
    effort: float = 1.0  # 0.0 -> 1.0, defaults to 100% effort
    skills_required: Optional[list[str]] = None  # Optional skills/capabilities required


@dataclass(eq=False)
class OperationNode:
    operation_id: str
    operation_name: str
    durations: dict[TaskPhase, timedelta]  # Duration for each phase
    quantity: int
    dependencies: list["OperationNode"]
    operator_requirements: dict[TaskPhase, list[OperatorRequirement]]
    required_machine_id: Optional[str] = (
        None  # Specific machine required for this operation
    )
    min_idle_between_phases: timedelta = timedelta(
        0
    )  # Minimum idle time allowed between phases
    max_idle_between_phases: timedelta = timedelta(
        hours=48
    )  # Maximum idle time allowed between phases
    # Manufacturing order context (embedded in operation)
    operation_instance_id: Optional[str] = None
    manufacturing_order_id: Optional[str] = None
    manufacturing_order_name: Optional[str] = None
    article_id: Optional[str] = None
    article_name: Optional[str] = None


@dataclass
class OperationGraph:
    nodes: list[OperationNode]


@dataclass
class ManufacturingOrder:
    manufacturing_order_id: str
    manufacturing_order_name: str
    article_id: str
    article_name: str
    quantity: int
    operations_graph: Optional[OperationGraph] = None
    required_by_date: Optional[datetime] = (
        None  # Required completion date for scheduling
    )
    priority: Optional[Priority] = None  # Scheduling priority


@dataclass
class Task:
    operation: OperationNode
    phase: TaskPhase
    quantity: int
    datetime_start: datetime
    datetime_end: datetime
    machine: Optional[Resource]  # The machine used for this task
    operators: list[Resource]  # The operators assigned to this task
    operation_instance_id: (
        str  # Unique identifier for grouping phases of the same operation instance
    )


@dataclass
class DroppedOperation:
    operation_id: str
    operation_name: str
    reason: DropReason
    manufacturing_order_id: Optional[str] = None  # Optional for backward compatibility
    failed_phase: Optional[TaskPhase] = None  # The phase that failed to schedule
    error_message: Optional[str] = None  # Detailed error message from resource manager
    dependent_operation_id: Optional[str] = (
        None  # ID of the operation this depends on (for dependency drops)
    )


@dataclass
class Idle:
    operation: OperationNode
    machine: Resource
    datetime_start: datetime
    datetime_end: datetime
    reason: str = "waiting_for_operator"  # Reason for the idle time
    operation_instance_id: Optional[str] = None


@dataclass
class Schedule:
    fitness_score: float
    tasks: list[Task]
    number_of_operations_scheduled: int
    number_of_resources_scheduled: int
    makespan: Optional[timedelta]
    dropped_operations: Optional[list[DroppedOperation]] = None
    idles: Optional[list[Idle]] = None
