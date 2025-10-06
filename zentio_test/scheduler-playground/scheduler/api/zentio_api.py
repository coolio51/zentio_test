"""
FastAPI application that provides scheduler endpoints compatible with zentio-v1 server.

This module replaces the old scheduler and provides:
- /run-genetic: Genetic algorithm optimization
- /run: Simple FIFO optimization

Both endpoints integrate with the zentio-v1 DAL system for data access.
"""

import asyncio
import json
import logging
import os
import sys
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import httpx
from fastapi import FastAPI, HTTPException

# Set up logger
logger = logging.getLogger(__name__)
from pydantic import BaseModel

# Add the project root to Python path to import scheduler modules
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from scheduler.models import ManufacturingOrder, Resource
from scheduler.services.genetic_optimizer import (
    GeneticAlgorithmConfig,
    GeneticSchedulerOptimizer,
    SchedulingChromosome,
    optimize_schedule,
)
from scheduler.services.resource_manager import ResourceManager
from scheduler.common.profiling import profile_enabled
from scheduler.common.profiling_middleware import ProfilingMiddleware
from scheduler.services.scheduler import SchedulerService
from utils.data_converters import convert_manufacturing_orders, convert_resources


# MARK: - Pydantic Models for API


class OptimizationRequest(BaseModel):
    organization_id: str
    start_date: str  # ISO format
    end_date: str  # ISO format


class ScheduledTask(BaseModel):
    order: str
    operation: str
    phase: str
    amount: int
    start_time: str  # ISO datetime
    end_time: str  # ISO datetime
    duration: str
    assigned_resources: List[str]
    priority: int


class OptimizationResponse(BaseModel):
    status: str
    message: str
    tasks_created: int
    runtime: float
    mo_ontime: int
    mo_optimized: int
    date: str
    makespan: str
    fitness_score: float
    tasks: List[ScheduledTask]


# MARK: - FastAPI App

app = FastAPI(title="Zentio Scheduler v3", version="3.0.0")

if profile_enabled():
    app.add_middleware(ProfilingMiddleware)

from scheduler.services.run_processor import start_run_processor, stop_run_processor


@app.on_event("startup")
async def startup_event():
    """Start background run processor on startup."""
    import asyncio

    asyncio.create_task(start_run_processor())


@app.on_event("shutdown")
async def shutdown_event():
    """Stop background run processor on shutdown."""
    stop_run_processor()


# MARK: - Helper Functions


def compute_mo_on_time_from_schedule(
    schedule, manufacturing_orders: List[ManufacturingOrder]
) -> int:
    """Compute MOs on time based on latest task end per MO and required_by_date (EOD)."""
    try:
        if not schedule or not getattr(schedule, "tasks", None):
            return 0
        # Map MO id -> latest end datetime
        mo_latest_end: Dict[str, datetime] = {}
        for task in schedule.tasks:
            mo_id = getattr(
                getattr(task, "operation", None), "manufacturing_order_id", None
            )
            end_dt = getattr(task, "datetime_end", None)
            if not mo_id or not end_dt:
                continue
            prev = mo_latest_end.get(mo_id)
            if prev is None or end_dt > prev:
                mo_latest_end[mo_id] = end_dt
        if not mo_latest_end:
            return 0
        # Map MO id -> due date (end-of-day). If due is None, treat as on-time when tasks exist.
        mo_due: Dict[str, Optional[datetime]] = {}
        for mo in manufacturing_orders:
            due = getattr(mo, "required_by_date", None)
            if due is None:
                mo_due[mo.manufacturing_order_id] = None
            else:
                try:
                    mo_due[mo.manufacturing_order_id] = due.replace(
                        hour=23, minute=59, second=59, microsecond=999000
                    )
                except Exception:
                    mo_due[mo.manufacturing_order_id] = due
        on_time = 0
        for mo_id, latest_end in mo_latest_end.items():
            due = mo_due.get(mo_id)
            if due is None:
                on_time += 1
            elif latest_end <= due:
                on_time += 1
        return on_time
    except Exception:
        return 0


def compute_mo_scheduled_count(schedule) -> int:
    """Count distinct manufacturing orders that have at least one scheduled task."""
    try:
        if not schedule or not getattr(schedule, "tasks", None):
            return 0
        mo_ids = set()
        for task in schedule.tasks:
            mo_id = getattr(
                getattr(task, "operation", None), "manufacturing_order_id", None
            )
            if mo_id:
                mo_ids.add(mo_id)
        return len(mo_ids)
    except Exception:
        return 0


def compute_mo_on_time_breakdown(
    schedule, manufacturing_orders: List[ManufacturingOrder]
) -> Dict:
    """Return detailed MO on-time breakdown for debugging UI/DB: on-time ids and late details."""
    result = {"on_time_ids": [], "late": []}
    try:
        if not schedule or not getattr(schedule, "tasks", None):
            return result
        # reuse logic from compute_mo_on_time_from_schedule
        mo_latest_end: Dict[str, datetime] = {}
        for task in schedule.tasks:
            mo_id = getattr(
                getattr(task, "operation", None), "manufacturing_order_id", None
            )
            end_dt = getattr(task, "datetime_end", None)
            if not mo_id or not end_dt:
                continue
            prev = mo_latest_end.get(mo_id)
            if prev is None or end_dt > prev:
                mo_latest_end[mo_id] = end_dt
        mo_due: Dict[str, Optional[datetime]] = {}
        for mo in manufacturing_orders:
            due = getattr(mo, "required_by_date", None)
            if due is None:
                mo_due[mo.manufacturing_order_id] = None
            else:
                try:
                    mo_due[mo.manufacturing_order_id] = due.replace(
                        hour=23, minute=59, second=59, microsecond=999000
                    )
                except Exception:
                    mo_due[mo.manufacturing_order_id] = due
        for mo_id, latest_end in mo_latest_end.items():
            due = mo_due.get(mo_id)
            if due is None or latest_end <= due:
                result["on_time_ids"].append(mo_id)
            else:
                result["late"].append(
                    {
                        "mo_id": mo_id,
                        "latest_end": latest_end.isoformat() if latest_end else None,
                        "due": due.isoformat() if due else None,
                    }
                )
        return result
    except Exception:
        return result


async def fetch_input_data(
    organization_id: str, start_date: str, end_date: str
) -> Dict:
    """
    Fetch manufacturing orders and available resources from zentio-v1 DAL system.
    """

    # Get the zentio-v1 server URL from environment or use default
    server_url = os.getenv("ZENTIO_V1_SERVER_URL", "http://localhost:8080")

    async with httpx.AsyncClient() as client:
        try:
            # Fetch manufacturing orders
            mo_response = await client.get(
                f"{server_url}/api/scheduler/manufacturing-orders",
                headers={
                    "X-Organization-ID": organization_id,
                    "Content-Type": "application/json",
                },
                params={
                    "start_date": start_date,
                    "end_date": end_date,
                },
            )
            mo_response.raise_for_status()
            manufacturing_orders_data = mo_response.json()

            # Fetch available resources
            resources_response = await client.get(
                f"{server_url}/api/scheduler/available-resources",
                headers={
                    "X-Organization-ID": organization_id,
                    "Content-Type": "application/json",
                },
                params={
                    "start_date": start_date,
                    "end_date": end_date,
                },
            )
            resources_response.raise_for_status()
            resources_data = resources_response.json()

            return {
                "manufacturing_orders": manufacturing_orders_data,
                "resources": resources_data,
            }

        except httpx.HTTPStatusError as e:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to fetch data from zentio-v1 server: {e.response.status_code} - {e.response.text}",
            )
        except httpx.RequestError as e:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to connect to zentio-v1 server: {str(e)}",
            )


def convert_schedule_to_tasks(schedule, organization_id: str) -> List[ScheduledTask]:
    """
    Convert scheduler-v3 schedule to the task format expected by zentio-v1.
    """
    tasks = []

    for task in schedule.tasks:
        # Convert resource assignments to list of resource IDs
        assigned_resources = []
        if task.machine:
            assigned_resources.append(task.machine.resource_id)
        for operator in task.operators:
            assigned_resources.append(operator.resource_id)

        scheduled_task = ScheduledTask(
            order=task.operation.manufacturing_order_id or "",
            operation=task.operation.operation_id,
            phase=task.phase.value,
            amount=task.quantity,
            start_time=task.datetime_start.isoformat(),
            end_time=task.datetime_end.isoformat(),
            duration=str(task.datetime_end - task.datetime_start),
            assigned_resources=assigned_resources,
            priority=0,  # Default priority
        )

        tasks.append(scheduled_task)

    return tasks


# MARK: - API Endpoints


class ChromosomeInput(BaseModel):
    mo_order: List[str]
    operation_splits: Dict[int, List[int]] = {}


class ScheduleWithDnaRequest(BaseModel):
    organization_id: str
    start_date: str
    end_date: str
    dna: ChromosomeInput


async def evaluate_dna_schedule(
    organization_id: str, start_date: str, end_date: str, dna: ChromosomeInput
) -> OptimizationResponse:
    """
    Core DNA evaluation logic - reusable for both direct DNA and resolve endpoints.
    This function bypasses the queue and evaluates chromosomes immediately.
    """
    start_time = time.time()

    try:
        # Fetch input data
        input_data = await fetch_input_data(organization_id, start_date, end_date)

        # Convert to internal types
        manufacturing_orders = convert_manufacturing_orders(
            input_data["manufacturing_orders"]
        )
        resources = convert_resources(input_data["resources"])

        # Extract operations
        operations = []
        for mo in manufacturing_orders:
            if mo.operations_graph and mo.operations_graph.nodes:
                operations.extend(mo.operations_graph.nodes)

        # Resource manager
        resource_manager = ResourceManager(resources)

        # Rehydrate SchedulingChromosome from input DNA
        chromosome = SchedulingChromosome(
            mo_order=list(dna.mo_order or []),
            operation_splits=dict(dna.operation_splits or {}),
        )

        # Use optimizer internals to apply chromosome and schedule
        optimizer = GeneticSchedulerOptimizer()
        optimizer.operations = operations
        optimizer.original_resource_manager = resource_manager
        # No machine assignment cache needed with new DNA

        schedule = optimizer._evaluate_chromosome(chromosome)
        fitness = optimizer._calculate_fitness(schedule, chromosome)

        # Convert to output
        tasks = convert_schedule_to_tasks(schedule, organization_id)
        mo_ontime_count = compute_mo_on_time_from_schedule(
            schedule, manufacturing_orders
        )
        runtime = time.time() - start_time
        makespan_str = str(schedule.makespan) if schedule.makespan else "00:00:00"

        return OptimizationResponse(
            status="success",
            message="Scheduled with provided chromosome",
            tasks_created=len(tasks),
            runtime=runtime,
            mo_ontime=mo_ontime_count,
            mo_optimized=compute_mo_scheduled_count(schedule),
            date=datetime.now().strftime("%Y-%m-%d"),
            makespan=makespan_str,
            fitness_score=fitness,
            tasks=tasks,
        )

    except Exception as e:
        return OptimizationResponse(
            status="error",
            message=f"DNA evaluation failed: {str(e)}",
            tasks_created=0,
            runtime=time.time() - start_time,
            mo_ontime=0,
            mo_optimized=0,
            date=datetime.now().strftime("%Y-%m-%d"),
            makespan="00:00:00",
            fitness_score=0.0,
            tasks=[],
        )


@app.post("/schedule-with-dna", response_model=OptimizationResponse)
async def schedule_with_dna(request: ScheduleWithDnaRequest):
    """
    Schedule using a provided chromosome (DNA) for a single individual.
    Bypasses the optimization runs queue since DNA evaluation is fast and doesn't require queueing.
    """
    return await evaluate_dna_schedule(
        request.organization_id, request.start_date, request.end_date, request.dna
    )


@app.post("/run-genetic", response_model=OptimizationResponse)
async def run_genetic_optimization(request: OptimizationRequest):
    """
    Run genetic algorithm optimization.
    """
    start_time = time.time()

    try:
        # Fetch input data from zentio-v1 DAL
        input_data = await fetch_input_data(
            request.organization_id, request.start_date, request.end_date
        )

        # Convert data to scheduler-v3 format
        manufacturing_orders = convert_manufacturing_orders(
            input_data["manufacturing_orders"]
        )
        resources = convert_resources(input_data["resources"])

        if not manufacturing_orders:
            return OptimizationResponse(
                status="success",
                message="No manufacturing orders to schedule",
                tasks_created=0,
                runtime=time.time() - start_time,
                mo_ontime=0,
                mo_optimized=0,
                date=datetime.now().strftime("%Y-%m-%d"),
                makespan="00:00:00",
                fitness_score=0.0,
                tasks=[],
            )

        # Configure genetic algorithm for faster testing
        config = GeneticAlgorithmConfig(
            population_size=30,
            max_generations=20,
            crossover_rate=0.8,
            elite_size=5,
            mutation_rate=0.2,
            max_operation_splits=3,
            min_split_quantity=5,
        )

        # Create resource manager
        resource_manager = ResourceManager(resources)

        # Extract operations from manufacturing orders for genetic algorithm
        operations = []
        for mo in manufacturing_orders:
            if mo.operations_graph and mo.operations_graph.nodes:
                operations.extend(mo.operations_graph.nodes)

        # Run optimization
        print(
            f"🚀 Running genetic optimization with {len(operations)} operations from {len(manufacturing_orders)} orders and {len(resources)} resources..."
        )

        optimized_schedule, chromosome, optimizer = optimize_schedule(
            operations, resource_manager, manufacturing_orders, config
        )

        # Convert results to expected format
        tasks = convert_schedule_to_tasks(optimized_schedule, request.organization_id)
        mo_ontime_count = compute_mo_on_time_from_schedule(
            optimized_schedule, manufacturing_orders
        )

        runtime = time.time() - start_time
        makespan_str = (
            str(optimized_schedule.makespan)
            if optimized_schedule.makespan
            else "00:00:00"
        )

        return OptimizationResponse(
            status="success",
            message=f"Genetic optimization completed successfully",
            tasks_created=len(tasks),
            runtime=runtime,
            mo_ontime=mo_ontime_count,
            mo_optimized=compute_mo_scheduled_count(optimized_schedule),
            date=datetime.now().strftime("%Y-%m-%d"),
            makespan=makespan_str,
            fitness_score=optimized_schedule.fitness_score,
            tasks=tasks,
        )

    except Exception as e:
        print(f"❌ Error in genetic optimization: {e}")
        return OptimizationResponse(
            status="error",
            message=f"Optimization failed: {str(e)}",
            tasks_created=0,
            runtime=time.time() - start_time,
            mo_ontime=0,
            mo_optimized=0,
            date=datetime.now().strftime("%Y-%m-%d"),
            makespan="00:00:00",
            fitness_score=0.0,
            tasks=[],
        )


@app.post("/run", response_model=OptimizationResponse)
async def run_simple_optimization(request: OptimizationRequest):
    """
    Run simple FIFO optimization.
    """
    start_time = time.time()

    try:
        # Fetch input data from zentio-v1 DAL
        input_data = await fetch_input_data(
            request.organization_id, request.start_date, request.end_date
        )

        # Convert data to scheduler-v3 format
        manufacturing_orders = convert_manufacturing_orders(
            input_data["manufacturing_orders"]
        )
        resources = convert_resources(input_data["resources"])

        if not manufacturing_orders:
            return OptimizationResponse(
                status="success",
                message="No manufacturing orders to schedule",
                tasks_created=0,
                runtime=time.time() - start_time,
                mo_ontime=0,
                mo_optimized=0,
                date=datetime.now().strftime("%Y-%m-%d"),
                makespan="00:00:00",
                fitness_score=0.0,
                tasks=[],
            )

        # Create resource manager
        resource_manager = ResourceManager(resources)

        # Extract operations from manufacturing orders
        operations = []
        for mo in manufacturing_orders:
            if mo.operations_graph and mo.operations_graph.nodes:
                operations.extend(mo.operations_graph.nodes)

        # Run simple scheduling (FIFO)
        print(
            f"🚀 Running simple scheduler with {len(operations)} operations from {len(manufacturing_orders)} orders and {len(resources)} resources..."
        )

        schedule = SchedulerService.schedule(operations, resource_manager)

        # Convert results to expected format
        tasks = convert_schedule_to_tasks(schedule, request.organization_id)
        mo_ontime_count = compute_mo_on_time_from_schedule(
            schedule, manufacturing_orders
        )

        runtime = time.time() - start_time
        makespan_str = str(schedule.makespan) if schedule.makespan else "00:00:00"

        return OptimizationResponse(
            status="success",
            message=f"Simple optimization completed successfully",
            tasks_created=len(tasks),
            runtime=runtime,
            mo_ontime=mo_ontime_count,
            mo_optimized=compute_mo_scheduled_count(schedule),
            date=datetime.now().strftime("%Y-%m-%d"),
            makespan=makespan_str,
            fitness_score=schedule.fitness_score,
            tasks=tasks,
        )

    except Exception as e:
        print(f"❌ Error in simple optimization: {e}")
        return OptimizationResponse(
            status="error",
            message=f"Optimization failed: {str(e)}",
            tasks_created=0,
            runtime=time.time() - start_time,
            mo_ontime=0,
            mo_optimized=0,
            date=datetime.now().strftime("%Y-%m-%d"),
            makespan="00:00:00",
            fitness_score=0.0,
            tasks=[],
        )


@app.get("/health")
async def health_check():
    """
    Health check endpoint.
    """
    return {
        "status": "healthy",
        "message": "Zentio Scheduler v3 is running",
        "version": "3.0.0",
    }


# MARK: - Run Polling Endpoints


class RunUpdateRequest(BaseModel):
    run_id: str
    organization_id: str  # Required for organization scoping
    status: str  # 'running', 'completed', 'failed'
    current_generation: Optional[int] = None
    total_generations: Optional[int] = None
    results: Optional[Dict] = None
    error_message: Optional[str] = None


@app.get("/pending-runs")
async def get_pending_runs(organization_id: Optional[str] = None):
    """
    Get pending scheduler runs for background processing.
    If organization_id is provided, filter by that org.
    If not provided, get pending runs from all organizations.
    """
    server_url = os.getenv("ZENTIO_V1_SERVER_URL", "http://localhost:8080")

    async with httpx.AsyncClient() as client:
        try:
            if organization_id:
                # If specific org is provided, only check that one
                orgs_to_check = [organization_id]
            else:
                # Get all organizations that have pending runs dynamically
                try:
                    orgs_response = await client.get(
                        f"{server_url}/api/scheduler/organizations-with-pending-runs"
                    )
                    orgs_to_check = orgs_response.json()
                    logger.info(
                        f"Found {len(orgs_to_check)} organizations with pending runs"
                    )
                except Exception as e:
                    logger.error(
                        f"Failed to fetch organizations with pending runs: {e}"
                    )
                    # Fallback to empty list - no runs will be processed
                    orgs_to_check = []

            all_pending_runs = []

            for org_id in orgs_to_check:
                try:
                    response = await client.get(
                        f"{server_url}/api/scheduler/runs",
                        params={"status": "pending"},
                        headers={
                            "X-Organization-ID": org_id,
                        },
                        timeout=30.0,
                    )
                    if response.status_code == 200:
                        runs = response.json()
                        # Add organization_id to each run for context
                        for run in runs:
                            run["organizationId"] = org_id
                        all_pending_runs.extend(runs)
                except httpx.HTTPError:
                    # Skip organizations that have errors, continue with others
                    continue

            return all_pending_runs

        except Exception as e:
            raise HTTPException(
                status_code=500, detail=f"Failed to fetch pending runs: {e}"
            )


@app.post("/update-run")
async def update_run(request: RunUpdateRequest):
    """
    Update run status and progress.
    """
    server_url = os.getenv("ZENTIO_V1_SERVER_URL", "http://localhost:8080")

    async with httpx.AsyncClient() as client:
        try:
            # Update the scheduler run record
            response = await client.put(
                f"{server_url}/api/scheduler/runs/{request.run_id}",
                json={
                    "status": request.status,
                    "current_generation": request.current_generation,
                    "total_generations": request.total_generations,
                    "results": request.results,
                    "error_message": request.error_message,
                },
                headers={
                    "X-Organization-ID": request.organization_id,
                    "Content-Type": "application/json",
                },
                timeout=30.0,
            )
            response.raise_for_status()
            return {"success": True}
        except httpx.HTTPError as e:
            raise HTTPException(status_code=500, detail=f"Failed to update run: {e}")


@app.get("/")
async def root():
    """
    Root endpoint.
    """
    return {
        "message": "Zentio Scheduler v3 API",
        "version": "3.0.0",
        "endpoints": [
            "/run-genetic",
            "/run",
            "/schedule-with-dna",
            "/health",
            "/pending-runs",
            "/update-run",
        ],
    }
