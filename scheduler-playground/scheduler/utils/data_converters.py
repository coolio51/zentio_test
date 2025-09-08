"""
Data conversion utilities for converting between zentio-v1 and scheduler-v3 formats.

This module provides functions to convert manufacturing orders and resources
from the zentio-v1 API format to the scheduler-v3 internal format.
"""

from datetime import datetime, timedelta
from typing import List

from scheduler.models import (
    TaskPhase,
    ManufacturingOrder,
    OperationGraph,
    OperationNode,
    Priority,
    Resource,
    ResourceAvailability,
    ResourceCapability,
    ResourceType,
    OperatorRequirement,
)


def convert_manufacturing_orders(data) -> List[ManufacturingOrder]:
    """
    Convert manufacturing orders data from zentio-v1 format to scheduler-v3 format.
    """
    manufacturing_orders = []

    # Handle both wrapped and direct array formats
    if isinstance(data, list):
        mo_list = data
    elif isinstance(data, dict):
        mo_list = data.get("manufacturingOrdersRequirements", [])
    else:
        return manufacturing_orders

    if not mo_list:
        return manufacturing_orders

    print(f"🔄 Converting {len(mo_list)} manufacturing orders...")

    for mo_data in mo_list:
        # Convert operations
        operations = []
        for op_req in mo_data.get("operations_requirements", []):
            # Convert durations from operation resource capacities
            durations = {}
            operator_requirements = {}

            for capacity in op_req.get("operation_resource_capacities", []):
                phase_str = capacity["phase"]
                phase = TaskPhase(phase_str)

                # Convert duration (scheduler expects timedelta, data has seconds)
                # IMPORTANT: Do NOT multiply core_operation by quantity here.
                # The scheduler multiplies core_operation by quantity during scheduling.
                duration_seconds = capacity["duration_seconds"]
                durations[phase] = timedelta(seconds=duration_seconds)

                # Set operator requirements
                if capacity["resource_type"] == "operator":
                    if phase not in operator_requirements:
                        operator_requirements[phase] = []

                    operator_requirements[phase].append(
                        OperatorRequirement(
                            phase=phase,
                            operator_count=1,  # Default to 1 operator
                            effort=capacity["capacity"]
                            / 100.0,  # Convert percentage to 0-1
                        )
                    )

            operation = OperationNode(
                operation_id=op_req["operation_id"],
                operation_name=op_req["operation_name"],
                durations=durations,
                quantity=op_req["quantity"],
                dependencies=[],  # Will be filled based on actual dependency graph
                operator_requirements=operator_requirements,
                manufacturing_order_id=mo_data["manufacturing_order_id"],
                manufacturing_order_name=mo_data.get("manufacturing_order_name", ""),
                article_id=mo_data["article_id"],
                article_name=mo_data["article_name"],
            )

            # Fallback-safe: support both assignment_id and operation_id as graph keys
            graph_key = op_req.get("assignment_id") or op_req.get("operation_id")
            dependency_ids = op_req.get("dependencies") or []
            operations.append((graph_key, operation, dependency_ids))

        # Create mapping from assignment_id to operation for dependency resolution
        assignment_to_operation = {
            assignment_id: operation for assignment_id, operation, _ in operations
        }

        # Set actual dependencies based on dependency graph
        for assignment_id, operation, dependency_ids in operations:
            operation.dependencies = [
                assignment_to_operation[dep_id]
                for dep_id in dependency_ids
                if dep_id in assignment_to_operation and dep_id is not None
            ]

        # Extract operations without dependency metadata
        sorted_operations = [operation for _, operation, _ in operations]

        # Convert priority
        priority_str = mo_data.get("priority", "medium")
        priority = (
            Priority(priority_str)
            if priority_str in ["low", "medium", "high", "urgent"]
            else Priority.MEDIUM
        )

        # Convert required date
        required_by_date = None
        if "required_date" in mo_data:
            try:
                required_by_date = datetime.fromisoformat(
                    mo_data["required_date"].replace("Z", "+00:00")
                )
            except (ValueError, AttributeError):
                pass

        manufacturing_order = ManufacturingOrder(
            manufacturing_order_id=mo_data["manufacturing_order_id"],
            manufacturing_order_name=mo_data.get("manufacturing_order_name", ""),
            article_id=mo_data["article_id"],
            article_name=mo_data["article_name"],
            quantity=mo_data["quantity"],
            operations_graph=OperationGraph(nodes=sorted_operations),
            required_by_date=required_by_date,
            priority=priority,
        )

        manufacturing_orders.append(manufacturing_order)

    return manufacturing_orders


def convert_resources(data) -> List[Resource]:
    """
    Convert resources data from zentio-v1 format to scheduler-v3 format.
    """
    resources = []

    # Handle both wrapped and direct array formats
    if isinstance(data, list):
        resource_list = data
    elif isinstance(data, dict):
        resource_list = data.get("availableResources", [])
    else:
        return resources

    if not resource_list:
        return resources

    print(f"🔄 Converting {len(resource_list)} resources...")

    for resource_data in resource_list:
        # Convert resource type
        resource_type_str = resource_data["resource_type"]
        resource_type = ResourceType(resource_type_str)

        # Convert availabilities (zentio-v1 uses "periods_available")
        availabilities = []
        for avail in resource_data.get("periods_available", []):
            availability = ResourceAvailability(
                start_datetime=datetime.fromisoformat(
                    avail["start_datetime"].replace("Z", "+00:00")
                ),
                end_datetime=datetime.fromisoformat(
                    avail["end_datetime"].replace("Z", "+00:00")
                ),
                effort=avail.get("capacity", 100) / 100.0,  # Convert percentage to 0-1
            )
            availabilities.append(availability)

        # Convert capabilities (zentio-v1 uses "operations_skills")
        capabilities = []
        for skill in resource_data.get("operations_skills", []):
            capability = ResourceCapability(
                resource_type=resource_type,
                phase=TaskPhase(skill["operation_phase"]),
                operation_id=skill["operation_id"],
            )
            capabilities.append(capability)

        resource = Resource(
            resource_id=resource_data["resource_id"],
            resource_type=resource_type,
            resource_name=resource_data.get(
                "resource_label", resource_data.get("resource_name", "Unknown")
            ),
            resource_capabilities=capabilities,
            availabilities=availabilities,
        )

        resources.append(resource)

    return resources
