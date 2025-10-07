# Decoder Performance Follow-ups

Most of the earlier hotspots (machine horizon scans, per-call demand rebuilds, worker filtering) are already handled in `zentio_test/scheduler.py`. Remaining opportunities that still look valuable:

- [ ] Capture fresh profiling traces for `decode_schedule` on the stress-test dataset so we can confirm the new bottlenecks before any further tuning.
- [ ] Prototype Numba `@njit` versions of `find_machine_start` and `worker_assign` (plus their small helpers) to see whether JIT can accelerate the tight loops without regressing behaviour.
- [ ] If Numba proves worthwhile, add a guarded code path (env flag or setting) that switches between the accelerated routines and the pure NumPy implementation while keeping the caching structures compatible.
