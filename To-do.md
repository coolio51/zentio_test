To-Do list:
1. focus first on finding inefficiencies in the codes - scheduler.py
2. look for alternatives to reaching the same goal for each of the functions with special focus on computation time
3. complete the profiling code and make measurements after each improvement

# GPT suggestions:
## Key opportunities to speed up scheduler.py
Avoid full-horizon scans when picking machine start times – For every candidate machine we rebuild free_m and call sliding_all_true over the entire bucket horizon, even though only a small prefix after the precedence-ready time matters. Maintaining an interval/range structure per machine (initialised from A_M) and updating it incrementally after each assignment would let us query the next feasible slot in roughly O(log n) instead of O(nT) per check.

Suggested task
* Replace machine availability scans with interval scheduling:
1. In `build_matrices` (or a helper), derive per-machine free-interval lists from `A_M` and include them in the returned structure (e.g., `machine_free_windows`).
2. In `decode_schedule`, replace the `free_m`/`sliding_all_true` logic with an interval query that finds the earliest window ≥ `earliest_pred_finish(oi)` whose length covers `total_len`.
3. After committing to `(oi, best_mi, best_t)`, update that machine’s interval list by splitting/removing the used segment.
4. Remove the now-obsolete `sliding_all_true` calls for machine feasibility and drop the temporary boolean allocations.

Vectorise worker selection – worker_assign repeatedly builds availability masks with np.where for every skill and bucket, leading to heavy Python/NumPy overhead inside the innermost loop. Precomputing the worker indices per skill and computing the availability mask once per time bucket would drastically cut the number of array scans.

Suggested task
* Vectorise worker availability checks
1. Extend `build_matrices` to store `skill_workers = [np.where(Q[:, si])[0] …]` (or similar) alongside the existing matrices.
2. In `worker_assign`, replace each `np.where(np.logical_and(...))` with:

   * a single `available = (C_W[:, abs_t] > 0) & (~W_busy[:, abs_t])` mask,
   * followed by masking `skill_workers[si]` against `available` to count and pick workers.
3. Adjust the assignment step to use the precomputed index arrays instead of slicing the whole workforce each time.
4. Profile to confirm the reduced number of temporary allocations.

Precompute demand traces instead of rebuilding req – Every call to worker_assign walks each (si, off, Ls, demand) tuple, allocates a zero array, and increments it per bucket just to rebuild the same demand pattern. Preparing dense per-skill demand arrays (or cumulative deltas) for each (oi, mi) during matrix construction would eliminate the nested Python loops.

Suggested task
* Precompute per-operation skill demand tensors
1. When building `NeedKernels`, also produce—for each `(oi, mi)`—a compact structure such as:

   * `skill_ids`, `offsets`, `lengths`, `demands` NumPy arrays, or
   * an already expanded `demand[s, relative_t]` array stored in a cache.
2. Pass these precomputed arrays into `worker_assign` so the function can slice or reuse them directly instead of allocating `np.zeros` per call.
3. Rewrite the demand accumulation in `worker_assign` to use vectorised `np.add.at` (or direct slice addition) against a reusable scratch buffer sized to the operation length.
4. Ensure the scratch buffer is reset between calls to avoid repeated allocations.

Cut per-operation Python overhead for static data – We rebuild pred_list and pull operation lengths from dicts on every decode, and we recompute Rem = C_S_base - S_used inside the inner machine loop even though it is unchanged while evaluating candidates. Moving these structures to precomputed NumPy arrays and hoisting invariant calculations out of inner loops will shave significant interpreter time.

Suggested task
* Precompute static scheduling metadata
1. In `build_matrices`, convert `preds` into an array-of-arrays representation (e.g., CSR-style) and store it, along with a dense `total_len_arr[oi, mi]` (fill with 0 where ineligible).
2. Update `decode_schedule` to reference the precomputed predecessors directly, eliminating the per-call construction of `pred_list`.
3. Before looping over candidate machines, compute `Rem = C_S_base - S_used` once and reuse it for each machine, avoiding repeated allocations.
4. Audit other `.get()` dictionary accesses in hot loops and switch them to array indexing using the new dense structures.



## More suggestions - there way be some overlap
Machine availability search repeatedly scans the full horizon.
Every machine candidate rebuilds a boolean timeline and runs sliding_all_true, even though the decoder only needs the earliest feasible start. This O(nT) scan happens inside the per-operation loop, so the cost grows with both machines and time buckets.

Suggested task
* Replace sliding window scans with interval tracking
1. In `decode_schedule`, derive a list of free intervals per machine from `mats["A_M"]` (e.g., pairs of start/end indices where the machine is available).
2. Rewrite the inner feasibility loop to search those interval lists for the first slot ≥ `est` whose length covers `tot`, rather than calling `sliding_all_true`.
3. When an operation is scheduled, update the machine’s interval list by splitting/removing the used segment, and keep returning `M_busy` by marking the scheduled range after the fact.
4. Remove the now-unused `sliding_all_true` path for machines and adjust any helper functions/tests accordingly.

Skill-capacity feasibility clones whole matrices for each candidate.
decode_schedule recomputes Rem = C_S_base - S_used and allocates aligned buffers on every machine attempt, even though only a handful of skills and time windows are relevant per operation.

Suggested task
* Make skill checks slice-based
1. Track remaining capacity per skill/time (e.g., `skill_free = mats["C_S"].copy()`) and update it only after a successful assignment.
2. During feasibility checks, iterate candidate start times and evaluate required skill segments using direct slices like `skill_free[si, start+off:start+off+Ls] >= demand`, eliminating `Rem` and `aligned` allocations.
3. Adjust `worker_assign` to consume/restore `skill_free` slices instead of writing into `S_used` during feasibility probing.
4. Verify downstream code still reports `S_used` by reconstructing it from consumed capacity if necessary.

Worker demand profiles are rebuilt on every assignment.
worker_assign expands NeedKernels into a fresh req dictionary of numpy arrays for each operation, duplicating the same per-op/per-skill demand vectors across all calls.

Suggested task
* Precompute per-operation demand tensors
1. Extend `build_matrices` to store, for each `(op, machine)`, a dense `{skill: np.ndarray}` map of demand per local time step (length = `total_len[(oi, mi)]`).
2. Persist this structure in `mats` (e.g., `NeedProfiles`) so the decoder can reuse it.
3. Update `worker_assign` to read the precomputed arrays directly, removing the nested `req.setdefault` and inner loops.
4. Ensure serialization/export paths (if any) remain compatible with the new structure.

Candidate worker selection recomputes identical filters.
For every skill/time pair, the decoder runs np.where over the full worker matrix even though the skill mask Q[:, si] never changes, causing redundant boolean operations across thousands of time buckets.

Suggested task
* Cache worker lists per skill
1. Precompute `workers_by_skill[si] = np.flatnonzero(Q[:, si])` (or bitmasks) inside `build_matrices`, storing the result in `mats`.
2. In `worker_assign`, build the time-specific availability mask `(C_W[:, t] > 0) & (~W_busy[:, t])` once per `abs_t`, then intersect it with `workers_by_skill[si]` via boolean indexing or in-place filtering.
3. Reuse the filtered worker list between the counting and selection steps to avoid calling `np.where` twice.
4. Re-run any benchmarks to confirm the updated selection logic preserves behavior.

Decoder reinitializes large structures on every call.
decode_schedule copies mats["A_M"]/mats["C_S"] and rebuilds the predecessor adjacency list for each evaluation, even though the GA invokes the decoder hundreds of times per run.

Suggested task
* Reuse static decoder structures
1. Move the `pred_list` computation into `build_matrices` (store as `mats["pred_list"]`) so the decoder can reuse it directly.
2. Replace the `A_M.copy()`/`M_busy` tandem with a single mutable availability matrix cloned once per decode (or, when combined with the interval-tracking change, eliminate one of the matrices entirely).
3. Do the same for skill capacity (`C_S`), only copying the portions that actually mutate.
4. Adjust decoder outputs (`M_busy`, `S_used`) to reflect the new storage without duplicating data.

### Explore JIT acceleration for inner loops
* Profile the decoder first to confirm hotspots, focusing on `worker_assign`’s nested loops and the candidate-machine feasibility scans.
* Prototype Numba `@njit` versions of the tight loops using typed NumPy arrays to validate achievable speedups before committing to larger refactors or Cython equivalents.
* Prioritise JITing the worker selection and machine feasibility paths, which are prime candidates for acceleration due to their repetitive numerical operations.

Parallelise GA decoding
decode_schedule calls inside the GA fitness loop are independent per individual. We could dispatch those evaluations across worker processes or threads, but we must check and, if needed, limit NumPy’s internal threading to avoid oversubscribing cores.

Next steps
* Prototype parallel decoding via `concurrent.futures` (ThreadPoolExecutor vs ProcessPoolExecutor), `joblib`, or `multiprocessing` to compare overheads.
* Ensure deterministic behaviour by seeding the GA and NumPy RNG per worker/task and avoiding shared global RNG state.

## Skip redundant GA decodes
Repeatedly decoding elites every generation wastes time once the genomes stabilise.
* Issue: During GA evolution, the top-ranked individuals are copied forward unchanged, but the decoder still recomputes their schedules/evaluations on each generation, duplicating work.
* Remedy: Cache decoder outputs per genome or detect unchanged individuals so we can reuse prior results.
* Implementation ideas:
  * Memoise `decode_schedule` using the genome (or a stable hash) as the key, storing the resulting `(sched, eval)` pair for reuse.
  * When elites survive unchanged, bypass decoding and attach the cached schedule/evaluation directly.
  * Invalidate cache entries only when mutation/crossover alters the genome or when decoder parameters change.
