"""Synthetic workload generators for stress testing scheduler algorithms."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Iterable, List, Sequence, Tuple

from scheduler.models import (
    ManufacturingOrder,
    OperationGraph,
    OperationNode,
    OperatorRequirement,
    Resource,
    ResourceAvailability,
    ResourceCapability,
    ResourceType,
    TaskPhase,
)


_DEFAULT_START = datetime(2024, 1, 1, 8, 0, 0)


def _build_availabilities(
    count: int,
    *,
    start: datetime,
    window_hours: int,
    gap_hours: int,
) -> List[ResourceAvailability]:
    availabilities: List[ResourceAvailability] = []
    for index in range(count):
        window_start = start + timedelta(days=index, hours=(index % 3) * gap_hours)
        window_end = window_start + timedelta(hours=window_hours)
        availabilities.append(
            ResourceAvailability(
                start_datetime=window_start,
                end_datetime=window_end,
                effort=1.0,
            )
        )
    return availabilities


def generate_synthetic_manufacturing_orders(
    mo_count: int = 200,
    operations_per_mo: int = 6,
    *,
    start: datetime | None = None,
) -> List[ManufacturingOrder]:
    """Create deterministic manufacturing orders with consistent dependency chains."""

    base_start = (start or _DEFAULT_START).replace(minute=0, second=0, microsecond=0)
    manufacturing_orders: List[ManufacturingOrder] = []

    for mo_index in range(mo_count):
        operations: List[OperationNode] = []
        previous_operation: OperationNode | None = None

        for step in range(operations_per_mo):
            op_id = f"op-{mo_index:04d}-{step:02d}"
            durations = {
                TaskPhase.SETUP: timedelta(minutes=15 + (step % 3) * 5),
                TaskPhase.CORE_OPERATION: timedelta(minutes=60 + (step % 4) * 10),
                TaskPhase.CLEANUP: timedelta(minutes=10),
            }
            operator_requirements = {
                TaskPhase.CORE_OPERATION: [
                    OperatorRequirement(
                        phase=TaskPhase.CORE_OPERATION,
                        operator_count=1 + (step % 2),
                        effort=1.0,
                    )
                ]
            }

            operation = OperationNode(
                operation_id=op_id,
                operation_name=f"Synthetic Operation {mo_index:04d}-{step:02d}",
                durations=durations,
                quantity=1 + (step % 3),
                dependencies=[previous_operation] if previous_operation else [],
                operator_requirements=operator_requirements,
                manufacturing_order_id=f"MO-{mo_index:04d}",
                manufacturing_order_name=f"Synthetic MO {mo_index:04d}",
                article_id=f"ART-{mo_index:04d}",
                article_name=f"Article {mo_index:04d}",
            )
            operations.append(operation)
            previous_operation = operation

        operations_graph = OperationGraph(nodes=operations)
        required_date = base_start + timedelta(days=mo_index % 30)
        manufacturing_orders.append(
            ManufacturingOrder(
                manufacturing_order_id=f"MO-{mo_index:04d}",
                manufacturing_order_name=f"Synthetic MO {mo_index:04d}",
                article_id=f"ART-{mo_index:04d}",
                article_name=f"Article {mo_index:04d}",
                quantity=1 + (mo_index % 5),
                operations_graph=operations_graph,
                required_by_date=required_date,
            )
        )

    return manufacturing_orders


def generate_synthetic_operations(
    manufacturing_orders: Sequence[ManufacturingOrder],
) -> List[OperationNode]:
    """Flatten manufacturing orders into a single operations list."""

    operations: List[OperationNode] = []
    for mo in manufacturing_orders:
        if mo.operations_graph and mo.operations_graph.nodes:
            operations.extend(mo.operations_graph.nodes)
    return operations


def _build_capability_assignments(
    item_ids: Sequence[str],
    resource_slots: int,
    fan_out: int,
) -> List[List[str]]:
    assignments: List[List[str]] = [[] for _ in range(resource_slots)]
    if not item_ids:
        return assignments

    for index, item_id in enumerate(item_ids):
        for offset in range(max(1, fan_out)):
            slot = (index + offset) % resource_slots
            assignments[slot].append(item_id)
    return assignments


def generate_synthetic_resources(
    operation_ids: Iterable[str],
    *,
    machine_count: int = 80,
    operator_count: int = 120,
    availability_days: int = 14,
    start: datetime | None = None,
) -> List[Resource]:
    """Generate large synthetic resource pools covering all provided operations."""

    op_ids = sorted(set(operation_ids))
    base_start = (start or _DEFAULT_START).replace(minute=0, second=0, microsecond=0)

    machine_assignments = _build_capability_assignments(op_ids, machine_count, fan_out=max(1, machine_count // 8))
    operator_assignments = _build_capability_assignments(op_ids, operator_count, fan_out=max(1, operator_count // 10))

    resources: List[Resource] = []

    for machine_index in range(machine_count):
        capabilities = [
            ResourceCapability(
                resource_type=ResourceType.MACHINE,
                phase=TaskPhase.CORE_OPERATION,
                operation_id=operation_id,
            )
            for operation_id in machine_assignments[machine_index]
        ]
        if not capabilities:
            # Ensure every machine can handle at least one operation
            capabilities.append(
                ResourceCapability(
                    resource_type=ResourceType.MACHINE,
                    phase=TaskPhase.CORE_OPERATION,
                    operation_id=op_ids[machine_index % len(op_ids)] if op_ids else "fallback",
                )
            )

        availabilities = _build_availabilities(
            availability_days,
            start=base_start,
            window_hours=10,
            gap_hours=2,
        )
        resources.append(
            Resource(
                resource_id=f"machine-{machine_index:04d}",
                resource_type=ResourceType.MACHINE,
                resource_name=f"Machine {machine_index:04d}",
                resource_capabilities=capabilities,
                availabilities=availabilities,
            )
        )

    for operator_index in range(operator_count):
        capabilities = [
            ResourceCapability(
                resource_type=ResourceType.OPERATOR,
                phase=TaskPhase.CORE_OPERATION,
                operation_id=operation_id,
            )
            for operation_id in operator_assignments[operator_index]
        ]
        if not capabilities:
            capabilities.append(
                ResourceCapability(
                    resource_type=ResourceType.OPERATOR,
                    phase=TaskPhase.CORE_OPERATION,
                    operation_id=op_ids[operator_index % len(op_ids)] if op_ids else "fallback",
                )
            )

        availabilities = _build_availabilities(
            availability_days,
            start=base_start,
            window_hours=8,
            gap_hours=1,
        )
        resources.append(
            Resource(
                resource_id=f"operator-{operator_index:04d}",
                resource_type=ResourceType.OPERATOR,
                resource_name=f"Operator {operator_index:04d}",
                resource_capabilities=capabilities,
                availabilities=availabilities,
            )
        )

    return resources


def generate_workload(
    *,
    mo_count: int = 200,
    operations_per_mo: int = 6,
    machine_count: int = 80,
    operator_count: int = 120,
    availability_days: int = 14,
    start: datetime | None = None,
) -> Tuple[List[ManufacturingOrder], List[Resource], List[OperationNode]]:
    """Convenience helper returning manufacturing orders, resources and operations."""

    manufacturing_orders = generate_synthetic_manufacturing_orders(
        mo_count,
        operations_per_mo,
        start=start,
    )
    operations = generate_synthetic_operations(manufacturing_orders)
    resources = generate_synthetic_resources(
        [operation.operation_id for operation in operations],
        machine_count=machine_count,
        operator_count=operator_count,
        availability_days=availability_days,
        start=start,
    )
    return manufacturing_orders, resources, operations


__all__ = [
    "generate_synthetic_manufacturing_orders",
    "generate_synthetic_operations",
    "generate_synthetic_resources",
    "generate_workload",
]
