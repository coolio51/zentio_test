# Playground Performance To-Do

Comprehensive list of opportunities to speed up the scheduling playground while preserving behaviour.

## 1. Profiling, Benchmarks, and Guardrails
- Create a deterministic benchmark harness that loads the sample data in `scheduler-playground/data` and runs the end-to-end pipeline (converters → scheduler) under `scheduler.common.profiling` so regressions are measurable.
- Extend `scheduler.common.profiling` to emit aggregated wall-clock summaries and store them per commit; wire it into CI to catch slowdowns automatically.
- Provide fixture generators that build synthetic large workloads (hundreds of operations/resources) to surface quadratic paths in unit/integration tests.
- Make logging verbosity configurable (especially Rich console output) so tight loops do not spend time formatting strings during benchmarks.

## 2. Scheduler Service (`scheduler/services/scheduler.py`)
- Retire the `_schedule_operations_naive` `O(n²)` dependency scan: default to the topological scheduler and guard the old path behind a debug flag if it must remain.
- Precompute dependency metadata once: store `indegree`, successor lists, and earliest-ready timestamps in arrays keyed by index instead of Python `dict`/`set` lookups on `OperationNode` objects.
- Track completion times per operation in primitive `list[datetime|None]` structures (index-based) to avoid repeated `dict` hashing and `datetime.min` sentinel use in the heap.
- Batch-drop propagation: when an operation is dropped, queue successors immediately rather than revisiting them in later heap pops to shrink the number of scheduler iterations.
- Cache `OperationScheduler` instances per run (currently recreated for each scheduling call) if reentrant state allows, or pool them to avoid repeated initialisation overhead.

## 3. Operation Scheduler (`scheduler/services/operation_scheduler.py`)
- Replace the 30-minute step scan in `_find_earliest_operator_availability_in_machine_window` with interval-set intersections: use `ResourceManager.free_intervals` for both machines and operators and advance directly to the next conflicting boundary.
- Precompute per-operation `phase_candidates` / `capable_operators` only once per scheduling call (cache by `(operation_id, phase)`), instead of rebuilding lists for every window search and every phase.
- Avoid repeated `ResourceManager.is_resource_available` calls while iterating candidate windows: when scanning machine windows, carry forward the booking cursor and stop checking slots that are already known busy.
- Collapse repeated machine window loops: `find_all_machine_availability_windows` and `_find_available_operator_windows` both iterate over identical availability data; unify the scan so operators are checked during machine window enumeration to avoid nested loops.
- Cache `_get_phase_cache`, `_get_total_duration`, and `_get_all_operator_requirements` results per `OperationNode` across scheduler invocations (current implementation only memoises within a process via attributes; ensure they survive when nodes are copied/split).
- When splitting core tasks for partial operator availability, reuse a single preallocated list/array for `core_tasks` and `scheduled_idles` to minimise per-item allocations in long operations.
- Short-circuit idle insertion: detect contiguous operator windows ahead of time instead of appending and trimming `Idle` entries inside the phase loop.

## 4. Resource Manager (`scheduler/services/resource_manager.py`)
- Replace `_resource_bookings` lists with balanced search trees or a binary indexed tree to make `find_overlapping_intervals` and `_book_interval` `O(log n)` instead of repeated linear scans plus manual minute increments.
- `_update_resource_availabilities` currently rebuilds every availability segment after each booking/unbooking; maintain a difference-list or interval tree so updates only touch overlapping nodes.
- Precompute and cache `Resource` lookups (`self.resources_by_id`) everywhere: eliminate `next(...)` scans in `find_available_resources` and machine selection by reusing the dictionary.
- Optimise `get_all_available_slots_in_window` and `_find_slots_in_availability_window`: instead of advancing by minutes when a conflict is found, jump directly to the end of the blocking booking using the sorted bookings list.
- Avoid repeatedly constructing `set(...)` in `find_available_operators_for_machine`; keep capability lists as frozen tuples and intersect with availability via indices/bitsets.
- Memoise `_calculate_resource_utilization` per booking mutation, or maintain utilisation counters incrementally, so selection of least-used resources does not rescan entire availability arrays each time.
- Speed up cloning/resetting: implement lightweight snapshotting of `_resource_bookings` and availability arrays to support `clone()` / `reset_bookings()` without reconstructing every `ResourceAvailability` object from scratch.
- Introduce batched booking/unbooking operations for multi-phase scheduling so machine/operator reservations are written once per contiguous block.

## 5. Genetic Optimizer (`scheduler/services/genetic_optimizer.py`)
- Reduce `deepcopy` usage in `_apply_chromosome_to_operations`: cache pre-split operation templates and only adjust quantity/IDs, or implement a custom copy that reuses immutable fields.
- Hoist manufacturing-order topological orders and dependency maps outside the per-chromosome loop; they are recomputed every time an individual is evaluated.
- Pool `ResourceManager` clones: maintain a small pool reset via `reset_bookings()` rather than cloning a fresh manager for every chromosome and every process worker.
- Make `_evaluate_population_parallel` reuse initialised worker state by shipping only chromosome deltas, not the entire operations list, when possible.
- Cache fitness components (e.g., dropped operation counts, makespan) between similar chromosomes to avoid recomputing metrics after small mutations.
- When stagnation is detected, shrink population or mutate elites in place instead of running full scheduling on unchanged genomes (skip redundant evaluations).
- Optimise mutation/crossover bookkeeping: avoid allocating intermediate lists (e.g., `offspring.extend`) by preallocating arrays sized to the population.

## 6. API & Data Conversion (`scheduler/api/zentio_api.py`, `scheduler/utils/data_converters.py`)
- Remove unconditional `print` statements in converters; switch to lazy logging so large payload conversions do not incur console overhead.
- Convert manufacturing orders/resources using vectorised/parsing helpers (e.g., map ISO timestamps via precompiled `datetime.fromisoformat` wrappers) to reduce repeated string handling.
- Cache results of `convert_manufacturing_orders` / `convert_resources` when the same snapshot is processed multiple times (run processor + API endpoint) to avoid re-parsing identical payloads.
- In FastAPI handlers, avoid recreating HTTP clients (`httpx.AsyncClient`) per request when streaming GA progress; reuse a single session or connection pool.

## 7. Background Run Processor (`scheduler/services/run_processor.py`)
- Parallelise `process_run` across pending runs using an async task pool instead of handling them sequentially within `process_pending_runs`.
- Replace repeated conversions/logging in `print_resource_availability_tables` with memoised tables and gate Rich formatting behind debug flags to prevent expensive rendering during normal runs.
- Batch remote status updates (`update_run_status`) when multiple runs progress simultaneously to cut HTTP round-trips.

## 8. Logging & Diagnostics (`scheduler/utils/resource_logger.py`, `scheduler/utils/schedule_logger.py`)
- Add lazy string formatting (e.g., only build tables when the caller requests them) so schedule/resource logs do not spend milliseconds on Rich table generation during automated runs.
- Provide lightweight text-mode summaries for automated performance tests to avoid importing Rich entirely when not needed.

## 9. Testing & Tooling
- Expand `scheduler/tests` to include performance assertions (e.g., `pytest-benchmark`) verifying that scheduling N operations stays within target time bounds.
- Add unit tests for `ResourceManager` interval operations with large booking counts to ensure future optimisations (trees/bitsets) remain correct.
- Provide tooling scripts that visualise profiler output (flamegraphs) to guide further optimisation work.
