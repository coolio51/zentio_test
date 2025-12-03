# Playground Performance Tasks

Reviewing the current codebase showed several of the earlier suggestions are already in place (interval trees for bookings, cached machine windows, etc.). The items below remain open and should still move the needle:

## 1. Profiling, Benchmarks, and Guardrails
- [ ] Build a deterministic benchmark harness that loads `scheduler-playground/data` and runs the converters → scheduler pipeline under `scheduler.common.profiling` so we can compare wall-clock numbers between commits.
- [ ] Extend `scheduler.common.profiling` to aggregate wall-clock summaries per run and persist them (CSV/JSON) so CI can flag regressions automatically.
- [ ] Provide fixture generators that emit large synthetic workloads (hundreds of operations/resources) for unit/integration tests to expose quadratic paths early.
- [ ] Plumb `get_logging_verbosity()` through the Rich/console helpers so benchmarks can run with minimal logging overhead.

## 2. Scheduler Service (`scheduler/services/scheduler.py`)
- [ ] Reuse or pool an `OperationScheduler` instance per scheduling run to keep its caches warm instead of instantiating a fresh scheduler on every call to `SchedulerService.schedule`.

## 3. Operation Scheduler (`scheduler/services/operation_scheduler.py`)
- [ ] Replace the repeated `is_resource_available` probes in `_find_earliest_operator_availability_in_machine_window` with merged operator/machine interval scans so each boundary is checked once.
- [ ] Collapse the double machine-window traversal between `find_all_machine_availability_windows` and `_find_available_operator_windows` by iterating the machine free list once and evaluating operator availability inline.
- [ ] Actually reuse `_core_task_buffer` / `_idle_buffer` (or another preallocated structure) when splitting long core phases so we avoid allocating new lists for every call.
- [ ] Detect contiguous operator windows before the phase loop so idle periods are appended in one shot rather than constructed incrementally inside the loop.

## 4. Resource Manager (`scheduler/services/resource_manager.py`)
- [ ] Make `_update_resource_availabilities` update only overlapping segments (difference lists / interval updates) instead of rebuilding the entire availability list for each booking.
- [ ] Teach `_find_slots_in_availability_window` to advance directly to the end of conflicting bookings using the interval tree or bisect lookups instead of the current minute-by-minute fallback.
- [ ] Avoid constructing temporary `set(...)` objects in `find_available_operators_for_machine`; operate on the cached tuples of capable IDs with index-based tracking or bitsets.
- [ ] Cache or incrementally update `_calculate_resource_utilization` so operator selection does not rescan every availability window on each call.
- [ ] Speed up `reset_bookings`/`clone` by snapshotting booking trees and availability arrays rather than recreating every `ResourceAvailability` object.
- [ ] Add batch booking helpers so contiguous phase reservations can be written once per machine/operator rather than looping over individual resources.

## 5. Genetic Optimizer (`scheduler/services/genetic_optimizer.py`)
- [ ] Replace the heavy `deepcopy` usage in `_apply_chromosome_to_operations` with reusable templates or lightweight copy helpers.
- [ ] Cache per-manufacturing-order topological orders and dependency maps so `_topological_order_for_mo` does not rebuild graphs during every evaluation.
- [ ] Memoise fitness components (dropped counts, makespan, etc.) for chromosomes that only tweak splits/sequencing to avoid recomputing the full metrics stack.

## 6. API & Data Conversion (`scheduler/api/zentio_api.py`, `scheduler/utils/data_converters.py`)
- [ ] Remove unconditional `print` statements in the converters; route messages through the logger with lazy formatting.
- [ ] Use precompiled datetime parsers or vectorised helpers when converting large ISO timestamp batches in `convert_manufacturing_orders` / `convert_resources`.
- [ ] Cache conversion results when the same snapshot is processed multiple times (API + background processor) to skip redundant parsing.
- [ ] Reuse a single `httpx.AsyncClient` per request lifecycle instead of recreating clients each time GA progress is streamed.

## 7. Background Run Processor (`scheduler/services/run_processor.py`)
- [ ] Process pending runs concurrently via an async task pool so `process_pending_runs` is no longer fully sequential.
- [ ] Memoise resource availability tables (and respect verbosity settings) so `print_resource_availability_tables` only renders Rich tables when explicitly requested.
- [ ] Batch `update_run_status` calls when several runs change state together to cut HTTP round-trips.

## 8. Logging & Diagnostics (`scheduler/utils/resource_logger.py`, `scheduler/utils/schedule_logger.py`)
- [ ] Add lazy formatting / lightweight text-mode paths so expensive Rich table generation only happens when a caller opts in.

## 9. Testing & Tooling
- [ ] Add performance regression tests (e.g., `pytest-benchmark`) that assert scheduling N operations stays under a target budget.
- [ ] Add large-scale interval tests for `ResourceManager` to guard upcoming tree/bitset optimisations.
- [ ] Provide profiling visualisation scripts (flamegraphs) to help interpret profiler output.

# Check for wrong splitting and randomness in GA
