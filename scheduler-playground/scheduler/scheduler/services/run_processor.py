"""
Background run processor for handling async genetic algorithm optimization.

This module polls for pending runs from the zentio-v1 server and processes them
asynchronously without blocking HTTP requests.
"""

import asyncio
import logging
import os
import time
from typing import Dict, List, Optional
from datetime import datetime

import httpx

# Configure logging
logging.basicConfig(level=logging.INFO)

from scheduler.services.genetic_optimizer import GeneticSchedulerOptimizer
from scheduler.services.resource_manager import ResourceManager
from scheduler.models import ManufacturingOrder, Resource
from utils.data_converters import convert_manufacturing_orders, convert_resources

logger = logging.getLogger(__name__)


class RunProcessor:
    """Background run processor for genetic algorithm optimization."""

    def __init__(self, scheduler_api_url: str = "http://localhost:8002"):
        self.scheduler_api_url = scheduler_api_url
        self.server_url = os.getenv("ZENTIO_V1_SERVER_URL", "http://localhost:8080")
        self.poll_interval = 30  # Poll every 30 seconds
        self.running = False

    async def start(self):
        """Start the background run processor."""
        self.running = True
        logger.info("🚀 Starting background run processor...")

        while self.running:
            try:
                await self.process_pending_runs()
            except Exception as e:
                logger.error(f"Error in run processor loop: {e}")

            await asyncio.sleep(self.poll_interval)

    def stop(self):
        """Stop the background run processor."""
        self.running = False
        logger.info("🛑 Stopping background run processor...")

    async def process_pending_runs(self):
        """Poll for and process pending runs."""
        try:
            async with httpx.AsyncClient() as client:
                # Get pending runs from all organizations
                response = await client.get(
                    f"{self.scheduler_api_url}/pending-runs",
                    timeout=30.0,
                )

                if response.status_code == 200:
                    runs = response.json()

                    if runs:
                        logger.info(
                            f"📋 Found {len(runs)} pending runs across all organizations"
                        )

                        for run in runs:
                            await self.process_run(run)

        except httpx.HTTPError as e:
            logger.warning(f"Failed to fetch pending runs: {e}")
        except Exception as e:
            logger.error(f"Error processing pending runs: {e}")

    async def process_run(self, run: Dict):
        """Process a single genetic optimization run."""
        run_id = run.get("id")
        organization_id = run.get("organizationId")
        settings = run.get("settings", {})

        if not run_id or not organization_id:
            logger.error(f"Invalid run data: {run}")
            return

        start_date = settings.get("start_date")
        end_date = settings.get("end_date")
        generations = settings.get("generations", 100)
        population_size = settings.get("population_size", 30)

        logger.info(
            f"🧬 Processing genetic optimization run {run_id} for org {organization_id}"
        )

        try:
            # Update run status to running
            await self.update_run_status(
                run_id, organization_id, "running", total_generations=generations
            )

            # Fetch run details to get the queued input snapshot (self-contained)
            input_data = None
            try:
                async with httpx.AsyncClient() as client:
                    run_resp = await client.get(
                        f"{self.server_url}/api/scheduler/runs/{run_id}",
                        headers={"X-Organization-ID": organization_id},
                        timeout=30.0,
                    )
                    run_resp.raise_for_status()
                    run_payload = run_resp.json()
                    # DAL returns camelCase JSON property for jsonb('input_data') → inputData
                    input_data = run_payload.get("inputData")
            except Exception as e:
                logger.error(f"Failed to fetch run details: {e}")

            if not input_data:
                await self.update_run_status(
                    run_id,
                    organization_id,
                    "failed",
                    error_message="Failed to fetch input data",
                )
                return

            # Convert to scheduler format
            # Converters accept either wrapped dict with manufacturingOrdersRequirements/availableResources
            manufacturing_orders = convert_manufacturing_orders(input_data)
            resources = convert_resources(input_data)

            # 🔍 DEBUG: Print resource availability tables
            self.print_resource_availability_tables(resources, start_date, end_date)

            if not manufacturing_orders:
                await self.update_run_status(
                    run_id,
                    organization_id,
                    "failed",
                    error_message="No manufacturing orders found",
                )
                return

            # Run genetic algorithm with progress updates
            results = await self.run_genetic_optimization_with_progress(
                run_id,
                organization_id,
                manufacturing_orders,
                resources,
                generations,
                population_size,
            )

            # Update run with results
            await self.update_run_status(
                run_id, organization_id, "completed", results=results
            )

            logger.info(f"✅ Completed genetic optimization run {run_id}")

        except Exception as e:
            logger.error(f"❌ Failed to process run {run_id}: {e}")
            await self.update_run_status(
                run_id, organization_id, "failed", error_message=str(e)
            )

    # Removed live input fetching – runs are self-contained via inputData snapshot

    def print_resource_availability_tables(
        self, resources: List, start_date: str, end_date: str
    ):
        """Print detailed resource availability tables for debugging."""
        try:
            from datetime import datetime, timedelta
            from rich.console import Console
            from scheduler.utils.resource_logger import ResourceLogger

            console = Console()

            logger.info("=" * 80)
            logger.info("🔍 RESOURCE AVAILABILITY ANALYSIS")
            logger.info(f"📅 Planning Period: {start_date} → {end_date}")
            logger.info("=" * 80)

            # Parse date strings for analysis
            start_dt = datetime.fromisoformat(start_date.replace("Z", "+00:00"))
            end_dt = datetime.fromisoformat(end_date.replace("Z", "+00:00"))
            total_days = (end_dt - start_dt).days

            logger.info(f"📊 Total planning horizon: {total_days} days")
            logger.info(f"🏭 Total resources found: {len(resources)}")

            # Use ResourceLogger to create and print the availability table
            availability_table = ResourceLogger.initial_availability_table(resources)

            # Print the table using rich console
            console.print(availability_table)

            # Also create a summary table using ResourceManager
            from scheduler.services.resource_manager import ResourceManager

            resource_manager = ResourceManager(resources)
            summary_table = ResourceLogger.resource_summary_table(resource_manager)
            console.print(summary_table)

            # Add time-based coverage analysis
            logger.info("")
            logger.info("📈 AVAILABILITY COVERAGE ANALYSIS:")
            logger.info("-" * 60)

            # Sample dates throughout the planning period
            sample_dates = []
            for i in range(0, min(total_days, 60), 7):  # Sample every 7 days
                sample_date = start_dt + timedelta(days=i)
                sample_dates.append(sample_date)

            for sample_date in sample_dates:
                available_count = 0
                for resource in resources:
                    if hasattr(resource, "availabilities") and resource.availabilities:
                        for availability in resource.availabilities:
                            if (
                                availability.start_datetime
                                <= sample_date
                                <= availability.end_datetime
                            ):
                                available_count += 1
                                break

                logger.info(
                    f"  {sample_date.strftime('%Y-%m-%d')}: {available_count}/{len(resources)} resources available"
                )

            logger.info("=" * 80)

        except Exception as e:
            logger.error(f"Error printing resource availability tables: {e}")
            import traceback

            logger.error(f"Traceback: {traceback.format_exc()}")

    async def run_genetic_optimization_with_progress(
        self,
        run_id: str,
        organization_id: str,
        manufacturing_orders: List[ManufacturingOrder],
        resources: List[Resource],
        generations: int,
        population_size: int,
    ) -> Dict:
        """Run genetic optimization with progress updates."""

        # Initialize components
        resource_manager = ResourceManager(resources)
        # OperationScheduler not required here; GA optimizer directly schedules via ResourceManager

        # Extract all operations
        all_operations = []
        for mo in manufacturing_orders:
            if mo.operations_graph and mo.operations_graph.nodes:
                for operation in mo.operations_graph.nodes:
                    all_operations.append(operation)

        logger.info(
            f"🔬 Starting genetic optimization with {len(all_operations)} operations, {generations} generations"
        )

        # Create genetic optimizer with progress callback and logger
        progress_data = {"current_generation": 0, "total_generations": generations}

        def progress_callback(generation: int, current_generations: int):
            """Update run progress during optimization."""
            logger.info(
                f"🧬 Generation {generation}/{current_generations} in progress..."
            )
            # Update progress data that can be accessed by async polling
            progress_data["current_generation"] = generation
            progress_data["total_generations"] = current_generations
            print(f"📊 GA Progress: Generation {generation}/{current_generations}")

        async def update_progress_periodically():
            while (
                progress_data["current_generation"] < progress_data["total_generations"]
            ):
                if progress_data["current_generation"] > 0:
                    await self.update_run_status(
                        run_id,
                        organization_id,
                        "running",
                        current_generation=progress_data["current_generation"],
                        total_generations=progress_data["total_generations"],
                    )
                await asyncio.sleep(5)  # Update every 5 seconds

        # Start the progress updater
        progress_task = asyncio.create_task(update_progress_periodically())

        from scheduler.services.genetic_optimizer import GeneticAlgorithmConfig
        from scheduler.utils.genetic_algorithm_logger import GeneticAlgorithmLogger

        # Set up genetic algorithm logger for real-time progress
        ga_logger = GeneticAlgorithmLogger()
        ga_logger.configure_evolution_logging("INFO")

        # Create config for genetic optimizer
        config = GeneticAlgorithmConfig(
            population_size=population_size,
            max_generations=generations,
            elite_size=5,
            max_operation_splits=5,
        )

        genetic_optimizer = GeneticSchedulerOptimizer(config=config)

        start_time = time.time()

        # Run optimization with progress callback and logger
        try:
            schedule, best_chromosome = genetic_optimizer.optimize(
                operations=all_operations,
                resource_manager=resource_manager,
                manufacturing_orders=manufacturing_orders,
                logger=ga_logger,
                progress_callback=progress_callback,
            )
        finally:
            # Cancel the progress updater task
            progress_task.cancel()
            try:
                await progress_task
            except asyncio.CancelledError:
                pass

        runtime = time.time() - start_time

        logger.info(f"✅ Genetic optimization completed in {runtime:.2f} seconds")

        # Build results
        scheduled_tasks = []
        if schedule and schedule.tasks:
            for task in schedule.tasks:
                # Calculate duration in minutes
                duration_minutes = 0
                if task.datetime_start and task.datetime_end:
                    duration_minutes = (
                        task.datetime_end - task.datetime_start
                    ).total_seconds() / 60

                # Build assigned resources list
                assigned_resources = []
                if task.machine:
                    assigned_resources.append(task.machine.resource_id)
                if task.operators:
                    for operator in task.operators:
                        assigned_resources.append(operator.resource_id)

                scheduled_tasks.append(
                    {
                        "order": task.operation.manufacturing_order_id or "",
                        "operation": task.operation.operation_id or "",
                        "phase": task.phase.value if task.phase else "production",
                        "amount": task.quantity or 1,
                        "start_time": (
                            task.datetime_start.isoformat()
                            if task.datetime_start
                            else ""
                        ),
                        "end_time": (
                            task.datetime_end.isoformat() if task.datetime_end else ""
                        ),
                        "duration": str(int(duration_minutes)),
                        "assigned_resources": assigned_resources,
                        "priority": 1,
                    }
                )

        # Calculate makespan string from timedelta
        makespan_str = "0 days, 0:00:00"
        if schedule and schedule.makespan:
            total_seconds = int(schedule.makespan.total_seconds())
            days = total_seconds // 86400
            hours = (total_seconds % 86400) // 3600
            minutes = (total_seconds % 3600) // 60
            seconds = total_seconds % 60
            makespan_str = f"{days} days, {hours:02d}:{minutes:02d}:{seconds:02d}"

        # Calculate start date from first operation
        start_date_str = datetime.now().isoformat()
        if manufacturing_orders:
            for mo in manufacturing_orders:
                if mo.operations_graph and mo.operations_graph.nodes:
                    first_op = mo.operations_graph.nodes[0]
                    # Use current datetime as default since operations don't have earliest_start_time
                    start_date_str = datetime.now().isoformat()
                    break

        # Compute number of manufacturing orders on time (latest task end <= required_by_date end-of-day)
        def compute_mo_on_time() -> int:
            try:
                if not schedule or not getattr(schedule, "tasks", None):
                    return 0

                # Map MO id -> latest end datetime from scheduled tasks
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

        mo_ontime_count = compute_mo_on_time()

        # Build on-time breakdown for debugging (ids and late details)
        def build_mo_on_time_breakdown() -> Dict:
            result = {"on_time_ids": [], "late": []}
            try:
                if not schedule or not getattr(schedule, "tasks", None):
                    return result
                # Latest end per MO
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
                # Due per MO (EOD)
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
                    if due is None or (latest_end and latest_end <= due):
                        result["on_time_ids"].append(mo_id)
                    else:
                        result["late"].append(
                            {
                                "mo_id": mo_id,
                                "latest_end": (
                                    latest_end.isoformat() if latest_end else None
                                ),
                                "due": due.isoformat() if due else None,
                            }
                        )
                return result
            except Exception:
                return result

        mo_on_time_breakdown = build_mo_on_time_breakdown()

        # Compute number of MOs actually scheduled (distinct MOs present in tasks)
        def compute_mo_scheduled_count() -> int:
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

        mo_scheduled_count = compute_mo_scheduled_count()

        results = {
            "status": "completed",
            "message": f"Genetic algorithm optimization completed successfully",
            "tasks_created": len(scheduled_tasks),
            "runtime": runtime,
            "mo_ontime": mo_ontime_count,
            # Count only MOs that actually received tasks in the schedule
            "mo_optimized": mo_scheduled_count,
            "date": datetime.now().isoformat(),
            "makespan": makespan_str,
            "fitness_score": schedule.fitness_score if schedule else 0,
            "tasks": scheduled_tasks,
            "mo_ontime_breakdown": mo_on_time_breakdown,
            "start_date": start_date_str,
            "end_date": datetime.now().isoformat(),
            # Include serialized chromosome of best schedule for persistence
            "best_chromosome": {
                "mo_order": (best_chromosome.mo_order if best_chromosome else []),
                "operation_splits": (
                    best_chromosome.operation_splits if best_chromosome else {}
                ),
            },
        }

        return results

    async def update_run_status(
        self,
        run_id: str,
        organization_id: str,
        status: str,
        current_generation: Optional[int] = None,
        total_generations: Optional[int] = None,
        results: Optional[Dict] = None,
        error_message: Optional[str] = None,
    ):
        """Update run status via the scheduler API."""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.scheduler_api_url}/update-run",
                    json={
                        "run_id": run_id,
                        "organization_id": organization_id,
                        "status": status,
                        "current_generation": current_generation,
                        "total_generations": total_generations,
                        "results": results,
                        "error_message": error_message,
                    },
                    timeout=30.0,
                )
                response.raise_for_status()
        except Exception as e:
            logger.error(f"Failed to update run status: {e}")


# Global run processor instance
run_processor: Optional[RunProcessor] = None


async def start_run_processor():
    """Start the global run processor."""
    global run_processor
    if run_processor is None:
        run_processor = RunProcessor()
        await run_processor.start()


def stop_run_processor():
    """Stop the global run processor."""
    global run_processor
    if run_processor:
        run_processor.stop()
        run_processor = None
