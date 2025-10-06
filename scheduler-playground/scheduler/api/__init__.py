from fastapi import FastAPI
from scheduler.models import ManufacturingOrder, Resource
from scheduler.services.resource_manager import ResourceManager
from scheduler.services.scheduler import SchedulerService
from scheduler.utils.schedule_logger import ScheduleLogger

app = FastAPI()


@app.get("/schedule")
def schedule(
    manufacturing_orders: list[ManufacturingOrder],
    resources: list[Resource],
):
    resource_manager = ResourceManager(resources)
    schedule = SchedulerService.schedule(manufacturing_orders, resource_manager)
    ScheduleLogger.print(schedule)
    return {"message": "Schedule generated"}


@app.get("/")
def read_root():
    return {"message": "Hello, World!"}
