# Decoder Performance Follow-ups

Most of the earlier hotspots (machine horizon scans, per-call demand rebuilds, worker filtering) are already handled in `zentio_test/scheduler.py`. Remaining opportunities that still look valuable:

- [ ] Capture fresh profiling traces for `decode_schedule` on the stress-test dataset so we can confirm the new bottlenecks before any further tuning.
- [ ] Prototype Numba `@njit` versions of `find_machine_start` and `worker_assign` (plus their small helpers) to see whether JIT can accelerate the tight loops without regressing behaviour.
- [ ] If Numba proves worthwhile, add a guarded code path (env flag or setting) that switches between the accelerated routines and the pure NumPy implementation while keeping the caching structures compatible.


Tough list to attack:

1. Drop-on-first-failure everywhere
OperationScheduler::schedule_operation returns a DroppedOperation at the first serious failure:
* No machine windows → drop.
* Machine exists but no operators in any machine window → drop. 
* Operator-only search fails → drop.
* Core split disabled and no contiguous operators → drop.
And in the split path, if some items can’t be scheduled, the whole operation is dropped.
On top of that, your higher-level scheduler (in TS/C++) drops the entire MO when one op is dropped, which multiplies the damage. Combined, this makes the system very eager to give up instead of “schedule late / partially / on alternative resources.”

2. Priority and due dates are basically ignored
Inside schedule_operation:
* There is no use of MO priority or due date at all.
* The only temporal scaffolding is planning_start + planning_days → a global planning_end.
So:
* A low-priority MO that happens to be early in whatever list GA/decoder feeds will happily consume capacity.
* A high-priority MO with a tight due date is treated exactly the same as everything else.
* The decision “drop vs. schedule late” never considers due dates, lateness penalties, or priority.
That’s why you see “high” orders not being prioritised: the scheduler simply doesn’t know they’re high.

3. Machine-first allocation can starve operator-limited ops
The core pattern is:
1. Pick a machine window (specific or any capable) for total_duration.
2. Then look for operators inside that machine window. If none, drop (or flip to operator-only mode).
When operators are the real bottleneck, this is upside-down:
* You might have a perfectly fine operator window a bit later or on another machine, but because step 1 picked one machine window and stuck to it, the operation is dropped with NO_OPERATOR_CAPACITY.
* Operator-only fallback tries to find operator windows without a machine, but that’s only used in specific situations and can’t fix the core “wrong machine window chosen” problem.
Net effect: lots of artificial “no operator capacity” drops even when valid (machine, operator) pairs exist somewhere else in the horizon.

4. Search horizon + increments are expensive and blunt
Two hot spots:
1. find_available_operator_windows
    * Steps through time in 15-minute increments from start_time to window_end_time.
    * At each step, calls find_available_operators_for_machine.
    * For each hit, calls find_operator_availability_end.
2. find_operator_availability_end
    * For each operator, checks availability using is_resource_available over increasing intervals, up to 60 days ahead. 
On realistic data (many ops × many operators × long planning horizon), that’s huge:
* Complexity is roughly “days × operators × ops,” and every GA evaluation re-runs this.
* There’s no pruning by due date, MO priority, or “max lateness” – only a blunt “up to planning_end or 60 days.”
So you pay a lot of CPU to search in regions of time that may be completely irrelevant for the business objective.

5. Splitting is extremely aggressive and unbounded
When contiguous core fails:
* It gathers all operator windows from current_phase_start up to window_end_bound.
* Then greedily fills each window with items_per_hour * window_hours, with:
    * No real minimum chunk size beyond “at least 1 item.”
    * No maximum number of segments.
Consequences:
* A long core phase can become dozens of micro-tasks spread over days.
* Task list becomes noisy and hard to reason about (what you’re already seeing in the UI).
* Each extra segment adds booking work + idle calculations → more slowness.
Splitting is also disconnected from business rules like “this machine really shouldn’t start/stop 20 times for the same MO” or “prefer fewer, larger chunks.”

6. Inconsistent transactional behaviour
For split core:
* There’s a proper checkpoint (create_checkpoint) and rollback (restore_checkpoint) on failure.
For the normal (non-split) path:
* Phases are scheduled sequentially; tasks and idles are pushed into result and booked as we go.
* If a later phase or fallback fails, we just set result.dropped_operation and return result – no undo of earlier bookings.
That means:
* Capacity can be partially eaten by tasks from operations that are conceptually “dropped,” which in turn hurts later operations and GA fitness.
* Debugging is harder because logs say “operation dropped” but some tasks may still exist.

7. Fallback searches are huge but still “all or nothing”
For non-core phases, if find_operators_with_idle_support fails:
* It does a fallback up to 60 days forward, scanning machine slots and then checking operators for each slot. 
* If nothing is found, it drops the operation.
There is no concept of:
* “OK, we couldn’t find a slot before the due date, but we could do it 2 days late.”
* “Partially schedule some of the quantity now, rest later.”
So you pay for an enormous search and still end up with a binary outcome: either perfect slot, or complete drop.

8. Idle creation is blind to priority
When operators aren’t available at the preferred time, the scheduler happily creates waiting_for_operator idles before the task.
* This happens even for low-value MOs.
* Idles can block machines for long stretches, preventing other MOs (possibly higher priority) from using that capacity.
Because “priority” never enters the picture, you effectively get “whoever got there first keeps the machine warm,” regardless of business importance.

9. Operator-only path can mis-predict feasibility
determine_operator_only_start:
* Picks the earliest operator availability (per phase) and returns the first window it finds. 
When called via operator_only_mode, this can:
* Decide on a start time that looks good from the operators’ perspective.
* But not necessarily align with machine availability (if the operation actually needs a machine; this depends on your data).
The code tries to protect against that by using operator-only mode mostly when no machines are found, but the design leaves room for weird “operators are fine but machine later says no” situations that end in yet another drop.

10. Same op_id across many MOs competes with no batching
You pointed at lines like:
stage=SCHEDULE_OPERATION_START ... op_id=5383ab5f-... repeated for multiple MOs.
In code:
* Uniqueness is enforced as operation_instance_id = mo_id + "_" + op_id + "_NNN".
* The scheduler does not aggregate same op_id across MOs into a batched job.
So:
* Several MOs with the same operation are treated as separate competing jobs.
* There is no batching (“run all T31432BK cores together”), which often is what you’d want in a moulding/printing context.
* Whether one wins or not depends entirely on decode order and whatever capacity happens to be available, not on global “we have 5 MOs of the same thing, group them.”
That’s not a bug per se, but it’s a structural limitation that explains some of the fragmentation and non-intuitive allocations you’re seeing.

If I compress all of this into one sentence:
The scheduler is over-eager to drop, under-aware of priority/due dates, and overly brute-force in its search, with an aggressive splitting strategy and inconsistent rollback, so it both works very hard and gives up very quickly.


Yes, there are a few more structural weaknesses I’d flag beyond the first 0–10 list. I’d treat these as secondary, but still meaningful, especially once you’ve fixed the big ones.
I’ll number them from 11 so you can reference them cleanly later.

11. “All-or-nothing core duration” inflates required windows
What it is
* For each phase, you compute a single contiguous duration:
    * SETUP / CLEANUP: duration_it->second
    * CORE: duration_it->second * quantity then you round that to buckets and use it as the required contiguous block for find_operators_with_idle_support. 
Why it’s a weakness
* For large quantities this creates absurdly long required windows (e.g. 19 hours in your trace) that essentially force splitting or dropping, instead of allowing “intelligent chunking” by design.
* GA never gets the option “same MO, same machine, but just schedule it as 4 reasonable chunks from the start”; instead, the scheduler insists “give me 1 giant window or I’ll go through a complex split fallback.”
Symptoms
* Very big required_block in IDLE_SUPPORT_ENTER logs (18–24h+).
* Lots of IDLE_SUPPORT_FAIL followed by SPLIT or drop for large MOs even when capacity exists in reasonable blocks.

12. Splitting logic and quantity math are opaque and hard to control
What it is
* When contiguous core fails and splitting is allowed, scheduler:
    * Computes items_per_hour from the full duration and quantity.
    * Then creates chunks based on operator windows and that rate, pushing tasks and idles into result.
Why it’s a weakness
* The chunking is emergent, not explicitly controlled by business rules:
    * No explicit “max segments per MO” or “minimum chunk size”.
    * No clear guarantee that chunks line up nicely with shifts / breaks / changeovers.
* Because of bucket rounding + items_per_hour based on the full core duration, you can get:
    * Slight quantity over/under allocation per window.
    * Strange patterns (e.g. 3 huge tasks covering multiple days) that are technically legal but operationally ugly.
Symptoms
* Logs where one MO shows 2–3 very long tasks for the same core op, plus weird long idles.
* Schedulers that “technically” use capacity, but the plan looks like spaghetti in the Gantt.

13. Operator-only scheduling can ignore due dates and GA intent
What it is
* determine_operator_only_start walks phases, finds earliest operator availability, and chooses the first window that can host the phase, up to planning_start + planning_days. 
* It doesn’t look at:
    * Order priority,
    * Required/planned completion dates,
    * Or the GA chromosome’s intent.
Why it’s a weakness
* For operations without a machine, the scheduler may pull work very early simply because operators are free, regardless of whether GA wanted it early or whether there are higher-priority orders waiting.
* This can consume operator capacity that GA “wanted” to keep for other MOs, but GA never sees this distinction because fitness is evaluated on the final schedule only.
Symptoms
* Operator-only tasks starting way before they “need” to, pushing other MOs later and indirectly increasing drops elsewhere.
* Hard to explain “why did this low-priority operator-only job start on Monday, when a high-priority machine job was dropped on Wednesday?”

14. Horizon and bucket config are global, not per-machine / per-family
What it is
* planning_start, planning_days, bucket_size are global in bucket_config. All search horizons (for machine slots, operator windows, idle support, fallback) reference that same config.
Why it’s a weakness
* Different resource families typically need different policies:
    * Bottleneck machines (presses, injection molding) may want a shorter planning horizon with tighter windows.
    * Auxiliary machines or generic operators can be scheduled further out.
* Because everything shares a single horizon:
    * You can end up scanning huge ranges for “cheap” resources while still giving expensive bottlenecks too much freedom.
    * Tuning the horizon for one use-case can regress another.
Symptoms
* When you shorten planning_days to tame runtime, some operations become impossible to schedule; when you extend it, searches become very long and logs show massive ranges (attempted_horizon_hours in OPERATION_DROPPED_NO_SLOT).

15. Fallback search for non-core phases can create long “waiting” idles
What it is
* For non-core phases, if no contiguous slot is found, there’s a fallback that:
    * Asks ResourceManager for all machine slots up to 60 days ahead,
    * For each slot, checks operator availability,
    * If found, inserts an idle from current_phase_start to slot start and uses it. 
Why it’s a weakness
* That idle is always “waiting_for_resources”, but you never ask:
    * Does it make sense to wait this long?
    * Would dropping this MO be cheaper than idling the machine/operator for days?
* GA doesn’t explicitly penalize “giant idle before cleanup” any differently from smaller gaps unless you’ve wired that into fitness.
Symptoms
* Schedules where a machine finishes core one day, then sits idle for 1–2 days, then does a short CLEANUP / SETUP for the same MO.
* Hard for planners to accept, because it looks like an obvious opportunity for resequencing.

16. Limited use of order priority and due dates in the inner scheduler
What it is
* The core C++ scheduler mostly takes a fixed order of operations (provided by upstream) and works greedily:
    * It doesn’t reorder operations by priority within the horizon.
    * Priority/due dates only affect GA if the outer fitness function uses them aggressively.
Why it’s a weakness
* You effectively have two places where priorities could live:
    * In the GA chromosome.
    * Inside the scheduler.
* Right now, the C++ scheduler behaves as if “priority is already handled somewhere else”, but your traces show this isn’t reliably happening (high-priority orders not being clearly favored).
Symptoms
* HIGH priority and NORMAL priority MOs interleaved in logs without a clear preference.
* High-priority MOs still being dropped while lower-priority ones get capacity.

17. Diagnostics are still mostly low-level; not enough high-level “why” signals
You’ve made a lot of progress with TRACE-MO and the new IDLE_SUPPORT_* logs, but there’s still a structural diagnostic gap:
* You see what windows were scanned and where it failed, but not:
    * “MO X dropped because: operator group G overloaded between D1–D2 (utilization > 110%).”
    * “GA tried 30 individuals; best one still left 20% of orders unscheduled; main bottleneck: machine M on days D1–D3.”
This isn’t a bug in scheduling logic, but it is a weakness for making the system tunable.








Comments on how to solve:

Guiding principles for the roadmap
* Don’t touch everything at once. Each step should be togglable/configurable so you can A/B it.
* Measure after every change: at minimum, per run:
    * #MOs total, #MOs fully scheduled, #MOs partially scheduled (if we add that), #MOs dropped
    * avg tasks per MO, #IDLE_SUPPORT_FAIL, runtime
* Start with visibility & behavior that doesn’t change results, then move to logic that changes scheduling decisions, then to GA-level stuff.

Phase 0 – Make observability solid (no behavior change)
Goal: You can see clearly why an operation failed/succeeded, with consistent logs.
What to implement
1. Normalize all TRACE points:
    * Ensure every call to schedule_operation emits:
        * SCHEDULE_OPERATION_START
        * On success → at least one TASK_CREATED
        * On failure → OPERATION_DROPPED_* with a reason (no machine, no operator, idle fail, split fail, etc.).
    * Ensure every idle-search path emits:
        * IDLE_SUPPORT_ENTER
        * repeated CANDIDATE_WINDOW
        * either IDLE_SUPPORT_OK or IDLE_SUPPORT_FAIL.
2. Run-level summary log (one line per run):
    * At the end of a scheduler run, log something like: [SCHEDULER_SUMMARY] run_id=... mos_total=... mos_scheduled=... mos_dropped=...
    *   tasks=... avg_tasks_per_mo=... idle_failures=... runtime_ms=...
    * 
    * This can live in the TS side after the C++ call or inside the C++ worker.
3. No logic change – just logging.
How to evaluate
* Use your current dataset, run once before and once after Phase 0.
* Confirm:
    * You can pick any MO and see its entire life: INPUT → OPS_CREATED → each SCHEDULE_OPERATION_START → IDLE_* → TASK_CREATED or DROP reason.
    * You see exactly which “reason buckets” dominate drops.
👉 Once Phase 0 is done, every later change is much easier to understand.


Phase 1 – Respect priority / due dates in scheduling order
Goal: Stop the most absurd cases where LOW priority eats “premium” capacity before HIGH.
Key idea: Change the order in which operations/MOs are fed to the scheduler, without yet changing the inner scheduling algorithm.
What to implement
1. Define a deterministic sort key for MOs:
    * e.g., (priority desc, due_date asc, external_id asc).  <— we need to do a double ranking here: first by required_date and then by priority. So, an MO with HIGH priority and a closer required_date is higher on the list than a HIGH priority and a further date. Same for the low priority. The goal is to cram in as many MOs within their required dates but by priority. The required date of high priority should be really forced  one way or another.
2. Apply it in the place that prepares the input list for the C++ scheduler / GA:
    * If you have a TS layer that builds manufacturing_orders_requirements, sort there.
    * If GA uses a “base order” to build chromosomes, use this as the base.
3. Optionally (config flag) allow:  <— not an option! A MUST!
    * sort_by_priority_enabled = true/false
    * So you can roll back if GA diversity suffers too much.
Why this is early in the roadmap
* It’s cheap and high impact: no deep C++ surgery.
* You immediately see if high-priority MOs at least get attempted earlier (SCHEDULE_OPERATION_START order).
How to evaluate
* Compare two runs (flag off vs on) on the same input:
    * For each priority bucket (HIGH/MED/LOW): scheduled fraction vs dropped fraction.
    * Check logs: do HIGH MOs appear earlier in SCHEDULE_OPERATION_START sequences?
    * Make sure runtime doesn’t get worse (it shouldn’t).


Phase 2 – Tame the time search horizon (runtime & weird far-future windows)
Goal: Reduce “225 candidate windows over 3 days” type searches and cut useless scanning.
What to implement (high-level)
1. Introduce per-operation search caps:
    * max_idle_search_candidates (e.g. 100–150) – after that, give up for that contiguous search, move to split / return fail.  <— I do not understand this variable. Is this the maximum number of candidates you can look at for an operation? And does it take into account the skills (or ability) of the operator? 
    * max_idle_search_span (e.g. X hours or due_date + slack).  <— we have the due date AND the horizon. We need to consider both areas for search while pushing to always complete before due date
2. Stop scanning beyond “horizon + block” like you described:  <— My mistake. It should be “horizon - block”. If it take 1 hour to produce something, looking at less than 1 hour before horizon is useless because there won’t be enough time to complete the work.
    * In find_operators_with_idle_support and find_available_operator_windows:
        * Ensure search_end_time ≤ min(planning_end, window_end) minus required_block, not plus.  <— Correct!
        * You don’t need to look at starts where the full block already can’t fit.  <— Exactly!
3. Make step increments configurable:
    * Currently 15 or 30 min.  <— This should already be the case. The UI has an option for Bucket (in minutes) which is set by default to 30. If this is not wired in the backend, then it needs to be.
    * Add a config like operator_search_step_minutes, with a sensible default (maybe 30).
    * This gives you a knob if you need to coarsen search.
How to evaluate
* Measure:
    * Average and max candidates= per IDLE_SUPPORT_FAIL or IDLE_SUPPORT_OK.
    * Overall scheduler runtime for same data.
* Confirm that:
    * You still schedule roughly as many MOs as before (Phase 1).
    * You no longer see insane spans like 3–5 days scanned in 15-min steps for one op.


Phase 3 – Make splitting sane and bounded
Goal: Keep splitting but stop it from producing insane fragmentation and very long segments.  <— Is this sensible? There are 2 things: if we split a lot and get the final product within the required due date, this should be acceptable especially if doing so allows us to complete more MOs.
What to implement
1. Config for splitting policy:
    * max_core_splits_per_operation  <— this should already exist in the config or somewhere in the code. For this and all others, make sure to check what already exists so that you don’t create a new variable for the same thing
    * min_core_chunk_duration (in minutes or hours)
    * Possibly min_core_chunk_qty (if your domain thinks in pieces).
2. Apply these in the split core path:
    * When iterating over operator windows:
        * Only create a chunk if it can meet min_core_chunk_duration and min_qty.
    * Stop splitting if:
        * number of chunks hits max_core_splits_per_operation, or
        * remaining quantity < min_qty → mark remainder unscheduled instead of forcing micro-chunks.
3. Always emit a clear split summary log:
    * e.g. SPLIT_SUMMARY op=... chunks=3 qty_scheduled=... qty_unscheduled=....
How to evaluate
* Before vs after:
    * Distribution of “tasks per MO” (max, avg).
    * Number of ops that got > X segments (say 5, 10).
    * Visual sanity check in your Gantt UI.
* You should see:
    * Fewer silly micro-tasks.
    * Still roughly similar #MOs scheduled (maybe a slight drop in edge cases—but more realistic behavior).

Phase 4 – Fix the “whole-MO drop” semantics  <— we need even more precision here: not only do we not want to drop all but if we drop something, we should still be able to use the space where we dropped something from for the other not-yet scheduled MOs. Otherwise, after the drop, we still have empty slots that could be used. Overall, this section is not well described enough for Codex to move forward
Goal: Stop nuking entire MOs when a single operation cannot be scheduled, and avoid ghost bookings.
This is the first big behavior change; I’d keep it after Phases 1–3.
What to implement
1. Change semantics in TS service / C++ wrapper:
    * Instead of:
        * “If any op dropped, wipe all tasks for that MO and mark MO as dropped”
    * Move to:
        * “Track for each op: scheduled_qty vs unscheduled_qty”
        * MO can be:
            * FULLY_SCHEDULED
            * PARTIALLY_SCHEDULED
            * UNSCHEDULED
2. Ensure transactional behavior:
    * Either:
        * Give each operation its own ResourceManager checkpoint (full rollback if op fails), but keep already-committed ops of the same MO.
    * Or:
        * Allow partial ops (some chunks) and mark unscheduled remainder separately.
3. Expose this to GA fitness:
    * Fitness already knows “unscheduled” vs “scheduled”; now give it more granularity:
        * Heavier penalty for unscheduled HIGH priority quantity.
        * Lower penalty for unscheduled LOW priority.
How to evaluate
* Compare:
    * #MOs with at least one scheduled operation before and after.
    * Total scheduled quantity vs previous version.
* Check that capacity is not polluted by ghost tasks from dropped MOs (your new logs from Phase 0 will help).

Phase 5 – Refine GA ↔ scheduler contract (fairness & coverage)
Goal: Use GA to improve which MOs get capacity, now that the scheduler logic is less pathological.
This is where “chromosome → decode → schedule” needs to be tuned.
What to implement (conceptually)
1. Make sure GA’s chromosome actually controls order:
    * Validate where the operation/MO sequence is created.
    * Confirm it’s not being resorted after decode (you can instrument decode order in logs for a single individual).
2. Adjust fitness to reward “breadth” of coverage:
    * Penalize schedules that only fully cover a handful of MOs but completely drop many others, especially HIGH priority ones.
    * Reward:
        * Higher count of MOs with at least X% of quantity scheduled.
        * Better priority-weighted coverage.
3. Optional fairness heuristics:
    * Soft penalty for schedules where:
        * Any LOW priority MO is fully scheduled while a HIGH priority from same family is < Y% scheduled.
    * This pushes GA away from “5 fat MOs hog everything” and toward more balanced schedules.
How to evaluate
* For a fixed dataset and config:
    * Compare before vs after:
        * #MOs significantly covered (e.g. ≥80% qty) per priority
        * #MOs with 0% scheduled per priority
        * GA runtime (generations × evaluation cost).
* Confirm you’re not seeing the “only 5 MOs get all the love” pattern anymore.

Phase 6 – Second-order niceties (batching identical ops, smarter idles, etc.)
Once 0–5 are in place and stable, then you can think about:
* Batching identical op_id across MOs for better setup time utilization.
* Smarter idle creation:
    * Prevent cheap idles from blocking expensive capacity for days.
    * Allow pre-emption of low-priority idles by high-priority MOs.
* Diagnostics / “why” views:
    * Per-resource bottleneck reports,
    * Simple “capacity vs demand” heatmaps per day.
These are nice-to-haves after the core behavior is under control.



We need to add early stopping. Very often the optimization continues without any gain. We have a very long patience (I think it was 50). It needs to be tweaked to stop much earlier.


Current Scheduler Log example:
[2025-12-05T14:41:29.963Z] [TRACE-MO] run_id=fda873fc-d35f-40da-b09a-f0622ea403dc mo_id=27245f85-dea0-47a8-8548-9c26b26c5b17 stage=SCHEDULE_OPERATION_START mo_name=MO-000115 op_id=e07d6701-9773-4117-a54d-507237afbcea
[2025-12-05T14:41:29.963Z] [TRACE-MO] [TRACE-MO-DETAIL] run_id=fda873fc-d35f-40da-b09a-f0622ea403dc mo_id=27245f85-dea0-47a8-8548-9c26b26c5b17 stage=IDLE_SUPPORT_ENTER required_block=57
[2025-12-05T14:41:29.963Z] [TRACE-MO] [TRACE-MO-DETAIL] run_id=fda873fc-d35f-40da-b09a-f0622ea403dc mo_id=27245f85-dea0-47a8-8548-9c26b26c5b17 stage=CANDIDATE_WINDOW machine=b9c068bd-2764-4f55-9c66-901b1d1c837f idle_start=1764946800000 idle_end=1764968400000 window=21600000
[2025-12-05T14:41:29.963Z] [TRACE-MO] [TRACE-MO-DETAIL] run_id=fda873fc-d35f-40da-b09a-f0622ea403dc mo_id=27245f85-dea0-47a8-8548-9c26b26c5b17 stage=IDLE_SUPPORT_OK machine=b9c068bd-2764-4f55-9c66-901b1d1c837f start=1764946800000 end=1764968400000
[2025-12-05T14:41:29.963Z] [TRACE-MO] [TRACE-MO-DETAIL] run_id=fda873fc-d35f-40da-b09a-f0622ea403dc mo_id=27245f85-dea0-47a8-8548-9c26b26c5b17 stage=CANDIDATE_WINDOW machine=b9c068bd-2764-4f55-9c66-901b1d1c837f idle_start=1765198800000 idle_end=1765227600000 window=28800000
[2025-12-05T14:41:29.963Z] [TRACE-MO] [TRACE-MO-DETAIL] run_id=fda873fc-d35f-40da-b09a-f0622ea403dc mo_id=27245f85-dea0-47a8-8548-9c26b26c5b17 stage=CANDIDATE_WINDOW machine=b9c068bd-2764-4f55-9c66-901b1d1c837f idle_start=1765200600000 idle_end=1765227600000 window=27000000
[2025-12-05T14:41:29.963Z] [TRACE-MO] [TRACE-MO-DETAIL] run_id=fda873fc-d35f-40da-b09a-f0622ea403dc mo_id=27245f85-dea0-47a8-8548-9c26b26c5b17 stage=CANDIDATE_WINDOW machine=b9c068bd-2764-4f55-9c66-901b1d1c837f idle_start=1765202400000 idle_end=1765227600000 window=25200000
[2025-12-05T14:41:29.963Z] [TRACE-MO] [TRACE-MO-DETAIL] run_id=fda873fc-d35f-40da-b09a-f0622ea403dc mo_id=27245f85-dea0-47a8-8548-9c26b26c5b17 stage=CANDIDATE_WINDOW machine=b9c068bd-2764-4f55-9c66-901b1d1c837f idle_start=1765204200000 idle_end=1765227600000 window=23400000
[2025-12-05T14:41:29.963Z] [TRACE-MO] [TRACE-MO-DETAIL] run_id=fda873fc-d35f-40da-b09a-f0622ea403dc mo_id=27245f85-dea0-47a8-8548-9c26b26c5b17 stage=CANDIDATE_WINDOW machine=b9c068bd-2764-4f55-9c66-901b1d1c837f idle_start=1765206000000 idle_end=1765227600000 window=21600000
[2025-12-05T14:41:29.963Z] [TRACE-MO] [TRACE-MO-DETAIL] run_id=fda873fc-d35f-40da-b09a-f0622ea403dc mo_id=27245f85-dea0-47a8-8548-9c26b26c5b17 stage=CANDIDATE_WINDOW machine=b9c068bd-2764-4f55-9c66-901b1d1c837f idle_start=1765207800000 idle_end=1765227600000 window=19800000
