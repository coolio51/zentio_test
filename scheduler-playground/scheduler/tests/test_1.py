from datetime import datetime, timedelta
from copy import deepcopy

from rich.console import Console
from scheduler.common.console import get_console

from scheduler.models import (
    TaskPhase,
    ResourceType,
    OperationNode,
    Resource,
    OperatorRequirement,
)
from scheduler.services.resource_manager import ResourceManager
from scheduler.services.scheduler import SchedulerService

from scheduler.utils.resource_logger import ResourceLogger as rr
from scheduler.utils.schedule_logger import ScheduleLogger as sr

from .utils import (
    generate_availabilities,
    Shift,
    generate_capabilities,
    generate_operator_requirements,
)

# MARK: Default Operations

operation_cnc_machining = OperationNode(
    operation_id="cnc_machining",
    operation_name="CNC Machining",
    operator_requirements=generate_operator_requirements(
        setup_operators=1,
        cleanup_operators=1,
    ),
    durations={
        TaskPhase.SETUP: timedelta(minutes=10),
        TaskPhase.CORE_OPERATION: timedelta(minutes=10),
        TaskPhase.CLEANUP: timedelta(minutes=10),
    },
    quantity=0,
    dependencies=[],
    required_machine_id="machine-1",
    min_idle_between_phases=timedelta(0),  # No minimum idle time
    max_idle_between_phases=timedelta(hours=16),  # Allow up to 16 hours between phases
)

operation_deburring = OperationNode(
    operation_id="deburring",
    operation_name="Deburring",
    operator_requirements=generate_operator_requirements(
        setup_operators=1,
        core_operation_operators=1,
        cleanup_operators=0,  # No cleanup phase
    ),
    durations={
        TaskPhase.SETUP: timedelta(minutes=10),
        TaskPhase.CORE_OPERATION: timedelta(minutes=5),
    },
    quantity=0,
    dependencies=[],
    required_machine_id=None,  # Let scheduler choose between available machines
    min_idle_between_phases=timedelta(0),  # No minimum idle time
    max_idle_between_phases=timedelta(hours=16),  # Allow up to 16 hours between phases
)


# MARK: Manufacturing Orders

# MO-00001
mo1_manufacturing_order_id = "mo-00001"
mo1_manufacturing_order_name = "MO-00001"
mo1_article_id = "article_1"
mo1_article_name = "Article 1"
mo1_quantity = 10

mo1_cnc_machining = deepcopy(operation_cnc_machining)
mo1_cnc_machining.operation_instance_id = "mo-00001-cnc-machining-1"
mo1_cnc_machining.article_id = mo1_article_id
mo1_cnc_machining.article_name = mo1_article_name
mo1_cnc_machining.manufacturing_order_id = mo1_manufacturing_order_id
mo1_cnc_machining.manufacturing_order_name = mo1_manufacturing_order_name
mo1_cnc_machining.quantity = mo1_quantity

mo1_deburring = deepcopy(operation_deburring)
mo1_deburring.operation_instance_id = "mo-00001-deburring-1"
mo1_deburring.article_id = mo1_article_id
mo1_deburring.article_name = mo1_article_name
mo1_deburring.manufacturing_order_id = mo1_manufacturing_order_id
mo1_deburring.manufacturing_order_name = mo1_manufacturing_order_name
mo1_deburring.dependencies = [mo1_cnc_machining]
mo1_deburring.quantity = mo1_quantity

mo1_operations = [
    mo1_cnc_machining,
    mo1_deburring,
]


# MO-00002
mo2_manufacturing_order_id = "mo-00002"
mo2_manufacturing_order_name = "MO-00002"
mo2_article_id = "article_2"
mo2_article_name = "Article 2"
mo2_quantity = 10

mo2_cnc_machining = deepcopy(operation_cnc_machining)
mo2_cnc_machining.operation_instance_id = "mo-00002-cnc-machining-1"
mo2_cnc_machining.article_id = mo2_article_id
mo2_cnc_machining.article_name = mo2_article_name
mo2_cnc_machining.manufacturing_order_id = mo2_manufacturing_order_id
mo2_cnc_machining.manufacturing_order_name = mo2_manufacturing_order_name
mo2_cnc_machining.quantity = mo2_quantity
mo2_cnc_machining.required_machine_id = "machine-2"

mo2_deburring_1 = deepcopy(operation_deburring)
mo2_deburring_1.operation_instance_id = "mo-00002-deburring-1"
mo2_deburring_1.article_id = mo2_article_id
mo2_deburring_1.article_name = mo2_article_name
mo2_deburring_1.manufacturing_order_id = mo2_manufacturing_order_id
mo2_deburring_1.manufacturing_order_name = mo2_manufacturing_order_name
mo2_deburring_1.quantity = mo2_quantity // 2
mo2_deburring_1.required_machine_id = "machine-2"
mo2_deburring_1.dependencies = [mo2_cnc_machining]

mo2_deburring_2 = deepcopy(operation_deburring)
mo2_deburring_2.operation_instance_id = "mo-00002-deburring-2"
mo2_deburring_2.article_id = mo2_article_id
mo2_deburring_2.article_name = mo2_article_name
mo2_deburring_2.manufacturing_order_id = mo2_manufacturing_order_id
mo2_deburring_2.manufacturing_order_name = mo2_manufacturing_order_name
mo2_deburring_2.quantity = mo2_quantity // 2
mo2_deburring_2.required_machine_id = "machine-2"
mo2_deburring_2.dependencies = [mo2_cnc_machining]

mo2_operations = [
    mo2_cnc_machining,
    mo2_deburring_1,
    mo2_deburring_2,
]


# MO-00003
mo3_manufacturing_order_id = "mo-00003"
mo3_manufacturing_order_name = "MO-00003"
mo3_article_id = "article_3"
mo3_article_name = "Article 3"
mo3_quantity = 100

mo3_cnc_machining = deepcopy(operation_cnc_machining)
mo3_cnc_machining.operation_instance_id = "mo-00003-cnc-machining-1"
mo3_cnc_machining.article_id = mo3_article_id
mo3_cnc_machining.article_name = mo3_article_name
mo3_cnc_machining.manufacturing_order_id = mo3_manufacturing_order_id
mo3_cnc_machining.manufacturing_order_name = mo3_manufacturing_order_name
mo3_cnc_machining.quantity = mo3_quantity
mo3_cnc_machining.operator_requirements[TaskPhase.CORE_OPERATION] = [
    OperatorRequirement(
        phase=TaskPhase.CORE_OPERATION,
        operator_count=1,
    )
]
mo3_cnc_machining.required_machine_id = "machine-2"

mo3_deburring = deepcopy(operation_deburring)
mo3_deburring.operation_instance_id = "mo-00003-deburring-1"
mo3_deburring.article_id = mo3_article_id
mo3_deburring.article_name = mo3_article_name
mo3_deburring.manufacturing_order_id = mo3_manufacturing_order_id
mo3_deburring.manufacturing_order_name = mo3_manufacturing_order_name
mo3_deburring.quantity = mo3_quantity
mo3_deburring.dependencies = [mo3_cnc_machining]
mo3_deburring.required_machine_id = "machine-2"

mo3_operations = [
    mo3_cnc_machining,
    mo3_deburring,
]


# MO-00004
mo4_manufacturing_order_id = "mo-00004"
mo4_manufacturing_order_name = "MO-00004"
mo4_article_id = "article_4"
mo4_article_name = "Article 4"
mo4_quantity = 100

mo4_cnc_machining = deepcopy(operation_cnc_machining)
mo4_cnc_machining.operation_instance_id = "mo-00004-cnc-machining-1"
mo4_cnc_machining.article_id = mo4_article_id
mo4_cnc_machining.article_name = mo4_article_name
mo4_cnc_machining.manufacturing_order_id = mo4_manufacturing_order_id
mo4_cnc_machining.manufacturing_order_name = mo4_manufacturing_order_name
mo4_cnc_machining.quantity = mo4_quantity

mo4_deburring = deepcopy(operation_deburring)
mo4_deburring.operation_instance_id = "mo-00004-deburring-1"
mo4_deburring.article_id = mo4_article_id
mo4_deburring.article_name = mo4_article_name
mo4_deburring.manufacturing_order_id = mo4_manufacturing_order_id
mo4_deburring.manufacturing_order_name = mo4_manufacturing_order_name
mo4_deburring.dependencies = [mo4_cnc_machining]
mo4_deburring.quantity = mo4_quantity

mo4_operations = [
    mo4_cnc_machining,
    mo4_deburring,
]


# MO-00005
mo5_manufacturing_order_id = "mo-00005"
mo5_manufacturing_order_name = "MO-00005"
mo5_article_id = "article_5"
mo5_article_name = "Article 5"
mo5_quantity = 100

mo5_cnc_machining = deepcopy(operation_cnc_machining)
mo5_cnc_machining.operation_instance_id = "mo-00005-cnc-machining-1"
mo5_cnc_machining.article_id = mo5_article_id
mo5_cnc_machining.article_name = mo5_article_name
mo5_cnc_machining.manufacturing_order_id = mo5_manufacturing_order_id
mo5_cnc_machining.manufacturing_order_name = mo5_manufacturing_order_name
mo5_cnc_machining.quantity = mo5_quantity

mo5_deburring = deepcopy(operation_deburring)
mo5_deburring.operation_instance_id = "mo-00005-deburring-1"
mo5_deburring.article_id = mo5_article_id
mo5_deburring.article_name = mo5_article_name
mo5_deburring.manufacturing_order_id = mo5_manufacturing_order_id
mo5_deburring.manufacturing_order_name = mo5_manufacturing_order_name
mo5_deburring.dependencies = [mo5_cnc_machining]
mo5_deburring.quantity = mo5_quantity

mo5_operations = [
    mo5_cnc_machining,
    mo5_deburring,
]


# MARK: Resources

machine_1 = Resource(
    resource_id="machine-1",
    resource_type=ResourceType.MACHINE,
    resource_name="CNC Machine 1",
    resource_capabilities=[
        *generate_capabilities(
            resource_type=ResourceType.MACHINE, operation_id="cnc_machining"
        ),
        *generate_capabilities(
            resource_type=ResourceType.MACHINE, operation_id="deburring"
        ),
    ],
    availabilities=[
        *generate_availabilities(Shift.ALL_DAY, datetime.now().date(), 10),
    ],
)

machine_2 = Resource(
    resource_id="machine-2",
    resource_type=ResourceType.MACHINE,
    resource_name="CNC Machine 2",
    resource_capabilities=[
        *generate_capabilities(
            resource_type=ResourceType.MACHINE, operation_id="cnc_machining"
        ),
        *generate_capabilities(
            resource_type=ResourceType.MACHINE, operation_id="deburring"
        ),
    ],
    availabilities=[
        *generate_availabilities(Shift.ALL_DAY, datetime.now().date(), 10),
    ],
)

operator_1 = Resource(
    resource_id="operator-1",
    resource_type=ResourceType.OPERATOR,
    resource_name="John Doe",
    resource_capabilities=[
        *generate_capabilities(
            resource_type=ResourceType.OPERATOR, operation_id="cnc_machining"
        ),
        *generate_capabilities(
            resource_type=ResourceType.OPERATOR, operation_id="deburring"
        ),
    ],
    availabilities=[
        *generate_availabilities(Shift.MORNING, datetime.now().date(), 10),
    ],
)

operator_2 = Resource(
    resource_id="operator-2",
    resource_type=ResourceType.OPERATOR,
    resource_name="Frank Tarkenton",
    resource_capabilities=[
        *generate_capabilities(
            resource_type=ResourceType.OPERATOR, operation_id="cnc_machining"
        ),
        *generate_capabilities(
            resource_type=ResourceType.OPERATOR, operation_id="deburring"
        ),
    ],
    availabilities=[
        *generate_availabilities(Shift.AFTERNOON, datetime.now().date(), 10),
    ],
)

# MARK: Test

resources = [machine_1, machine_2, operator_1, operator_2]

# Test with just one simple manufacturing order
mo_debug_manufacturing_order_id = "mo-debug"
mo_debug_manufacturing_order_name = "Debug MO"
mo_debug_article_id = "debug"
mo_debug_article_name = "Debug Article"
mo_debug_quantity = 1

mo_debug_cnc_machining = deepcopy(operation_cnc_machining)
mo_debug_cnc_machining.operation_instance_id = "mo-debug-cnc-machining-1"
mo_debug_cnc_machining.article_id = mo_debug_article_id
mo_debug_cnc_machining.article_name = mo_debug_article_name
mo_debug_cnc_machining.manufacturing_order_id = mo_debug_manufacturing_order_id
mo_debug_cnc_machining.manufacturing_order_name = mo_debug_manufacturing_order_name
mo_debug_cnc_machining.quantity = mo_debug_quantity
mo_debug_cnc_machining.dependencies = []

mo_debug_operations = [
    mo_debug_cnc_machining,
]


operations = [
    *mo_debug_operations,
    *mo1_operations,
    *mo2_operations,
    *mo3_operations,
    *mo4_operations,
    *mo5_operations,
]


def test_1(console: Console):
    resource_manager = ResourceManager(resources)

    schedule = SchedulerService.schedule(
        operations,
        resource_manager,
    )

    console.print(rr.initial_availability_table(resource_manager.resources))
    console.print(rr.resource_bookings_table(resource_manager))
    console.print(sr.schedule_table(schedule))
    console.print(sr.schedule_summary_table(schedule))
    console.print(sr.dropped_operations_table(schedule))
    # console.print(rr.resource_timeline_table(resource_manager, machine_1.resource_id))
    # console.print(rr.resource_timeline_table(resource_manager, operator_1.resource_id))
