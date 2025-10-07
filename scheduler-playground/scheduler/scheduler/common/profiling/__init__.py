"""Runtime profiling helpers for scheduler services.

The profiling utilities in this module are designed to be opt-in via the
``ZENTIO_PROFILE`` environment variable. When profiling is disabled the
wrappers fall back to near-zero overhead passthrough implementations.

When enabled the profiler captures wall-clock time, CPU time and memory deltas
for decorated functions or profiled sections. Results are written to per-process
JSONL and CSV artifacts inside ``./.profile`` and can be merged later using the
``scheduler.common.profiling.merge`` CLI entry point.
"""

from __future__ import annotations

import atexit
import csv
import json
import logging
import math
import os
import signal
import threading
import tracemalloc
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import contextvars
import functools
import inspect
import multiprocessing as mp
import threading as threading_module
import time

try:  # Optional dependency for RSS tracking
    import psutil  # type: ignore
except Exception:  # pragma: no cover - psutil optional
    psutil = None  # type: ignore

profile_logger = logging.getLogger("profiling")
profile_logger.setLevel(logging.WARNING)

_PROFILE_ENABLED_CACHE: Optional[bool] = None
_PROFILE_DIR = Path(".profile")

_correlation_id_var: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "profile_correlation_id", default=None
)


def profile_enabled() -> bool:
    """Return True when profiling is enabled via ``ZENTIO_PROFILE``."""

    global _PROFILE_ENABLED_CACHE
    if _PROFILE_ENABLED_CACHE is None:
        value = os.getenv("ZENTIO_PROFILE", "").lower()
        _PROFILE_ENABLED_CACHE = value in {"1", "true", "yes", "on"}
    return _PROFILE_ENABLED_CACHE


def set_correlation_id(value: Optional[str]):
    """Assign the active correlation id for the current context."""

    return _correlation_id_var.set(value)


def reset_correlation_id(token: Optional[contextvars.Token]):
    """Reset the correlation id using the provided context token."""

    if token is not None:
        try:
            _correlation_id_var.reset(token)
        except LookupError:  # pragma: no cover - defensive
            pass


def get_correlation_id() -> Optional[str]:
    """Return the current correlation id if set."""

    try:
        return _correlation_id_var.get()
    except LookupError:  # pragma: no cover - defensive
        return None


@dataclass
class _Measurement:
    name: str
    module: str
    type: str
    start_wall: float
    start_cpu: float
    start_alloc: Optional[int]
    start_rss: Optional[int]
    thread_id: int


class _NullProfiler:
    """No-op profiler used when profiling is disabled."""

    enabled: bool = False

    def before(self, *_: Any, **__: Any) -> Optional[_Measurement]:  # pragma: no cover - trivial
        return None

    def after(self, *_: Any, **__: Any) -> None:  # pragma: no cover - trivial
        return None

    def flush(self) -> None:  # pragma: no cover - trivial
        return None


class Profiler:
    """Singleton profiler used to accumulate profiling samples."""

    _instance: Optional["Profiler"] = None
    _instance_lock = threading.Lock()

    def __init__(self) -> None:
        self.enabled = True
        self.pid = os.getpid()
        self.records: List[Dict[str, Any]] = []
        self._summary: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
        self._lock = threading.Lock()
        self._profile_dir = _PROFILE_DIR
        self._profile_dir.mkdir(parents=True, exist_ok=True)
        self._jsonl_path = self._profile_dir / f"profile_{self.pid}.jsonl"
        self._summary_path = self._profile_dir / f"summary_{self.pid}.csv"
        self._meta_path = self._profile_dir / f"meta_{self.pid}.json"
        self._run_summary_json_path = self._profile_dir / f"run_{self.pid}.json"
        self._run_summary_csv_path = self._profile_dir / f"run_{self.pid}.csv"
        self._started_at = datetime.now(timezone.utc)
        self._banner_printed = False
        self._previous_sigint_handler = signal.getsignal(signal.SIGINT)

        self._mp_start_method = mp.get_start_method(allow_none=True)
        if self._mp_start_method is None:
            try:
                mp.set_start_method("spawn", force=False)
                self._mp_start_method = mp.get_start_method(allow_none=True)
            except RuntimeError:  # pragma: no cover - already started
                self._mp_start_method = "unknown"

        self._tracemalloc_started = False
        if profile_enabled() and not tracemalloc.is_tracing():
            tracemalloc.start()
            self._tracemalloc_started = True

        self._process = psutil.Process(os.getpid()) if psutil else None

        # Register shutdown handlers
        try:
            signal.signal(signal.SIGINT, self._handle_sigint)
        except ValueError:  # pragma: no cover - signal only in main thread
            pass
        atexit.register(self.flush)

        self._log_banner()
        self._write_meta()

    @classmethod
    def instance(cls) -> "Profiler | _NullProfiler":
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    if profile_enabled():
                        cls._instance = cls()
                    else:
                        cls._instance = _NullProfiler()  # type: ignore
        return cls._instance  # type: ignore[return-value]

    def before(self, name: str, module: str, typ: str) -> _Measurement:
        start_wall = time.perf_counter()
        start_cpu = time.process_time()
        start_alloc: Optional[int] = None
        start_rss: Optional[int] = None

        if tracemalloc.is_tracing():
            start_alloc = tracemalloc.get_traced_memory()[0]

        if self._process is not None:
            try:
                start_rss = self._process.memory_info().rss
            except Exception:  # pragma: no cover - defensive
                start_rss = None

        return _Measurement(
            name=name,
            module=module,
            type=typ,
            start_wall=start_wall,
            start_cpu=start_cpu,
            start_alloc=start_alloc,
            start_rss=start_rss,
            thread_id=threading_module.get_ident(),
        )

    def after(self, measurement: Optional[_Measurement]) -> None:
        if measurement is None:
            return

        end_wall = time.perf_counter()
        end_cpu = time.process_time()
        end_alloc: Optional[int] = None
        end_rss: Optional[int] = None

        if tracemalloc.is_tracing():
            end_alloc = tracemalloc.get_traced_memory()[0]

        if self._process is not None:
            try:
                end_rss = self._process.memory_info().rss
            except Exception:  # pragma: no cover - defensive
                end_rss = None

        wall_ms = (end_wall - measurement.start_wall) * 1000.0
        cpu_ms = (end_cpu - measurement.start_cpu) * 1000.0
        alloc_kb_delta = None
        rss_kb_delta = None

        if measurement.start_alloc is not None and end_alloc is not None:
            alloc_kb_delta = (end_alloc - measurement.start_alloc) / 1024.0

        if measurement.start_rss is not None and end_rss is not None:
            rss_kb_delta = (end_rss - measurement.start_rss) / 1024.0

        record: Dict[str, Any] = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "pid": self.pid,
            "thread": measurement.thread_id,
            "module": measurement.module,
            "name": measurement.name,
            "type": measurement.type,
            "calls": 1,
            "wall_ms": wall_ms,
            "cpu_ms": cpu_ms,
            "alloc_kb_delta": alloc_kb_delta,
            "rss_kb_delta": rss_kb_delta,
        }

        corr_id = get_correlation_id()
        if corr_id:
            record["corr_id"] = corr_id

        with self._lock:
            self.records.append(record)
            self._update_summary(record)

    def _update_summary(self, record: Dict[str, Any]) -> None:
        key = (record["module"], record["name"], record["type"])
        stats = self._summary.setdefault(
            key,
            {
                "module": record["module"],
                "name": record["name"],
                "type": record["type"],
                "calls": 0,
                "wall_ms_total": 0.0,
                "wall_ms_values": [],
                "cpu_ms_total": 0.0,
                "alloc_kb_total": 0.0,
                "rss_kb_total": 0.0,
            },
        )

        stats["calls"] += 1
        stats["wall_ms_total"] += record["wall_ms"]
        stats["wall_ms_values"].append(record["wall_ms"])
        stats["cpu_ms_total"] += record["cpu_ms"]

        alloc_delta = record.get("alloc_kb_delta")
        if alloc_delta is not None:
            stats["alloc_kb_total"] += alloc_delta

        rss_delta = record.get("rss_kb_delta")
        if rss_delta is not None:
            stats["rss_kb_total"] += rss_delta

    def _handle_sigint(self, signum: int, frame: Optional[Any]) -> None:  # pragma: no cover - signal path
        try:
            self.flush()
        finally:
            if callable(self._previous_sigint_handler):
                self._previous_sigint_handler(signum, frame)
            elif self._previous_sigint_handler == signal.SIG_DFL:
                signal.signal(signal.SIGINT, signal.SIG_DFL)
                os.kill(os.getpid(), signal.SIGINT)

    def _log_banner(self) -> None:
        if self._banner_printed:
            return
        self._banner_printed = True
        profile_logger.info(
            "Profiling enabled (pid=%s, dir=%s, profiles=%s, summary=%s, mp_start=%s)",
            self.pid,
            self._profile_dir,
            self._jsonl_path,
            self._summary_path,
            self._mp_start_method,
        )

    def _write_meta(self) -> None:
        data = {
            "pid": self.pid,
            "created": self._started_at.isoformat(),
            "profile_dir": str(self._profile_dir),
            "mp_start_method": self._mp_start_method,
        }
        try:
            with self._meta_path.open("w", encoding="utf-8") as fh:
                json.dump(data, fh, indent=2)
        except OSError:  # pragma: no cover - filesystem issues
            profile_logger.warning("Failed to write profiling meta file: %s", self._meta_path)

    def flush(self) -> None:
        if not self.enabled:
            return
        with self._lock:
            if not self.records:
                # Still create summary/header files if profiling is enabled
                self._write_records([])
                self._write_summary({})
                return
            records_copy = list(self.records)
            summary_copy = {k: v.copy() for k, v in self._summary.items()}
        self._write_records(records_copy)
        self._write_summary(summary_copy)
        self._write_run_summary(records_copy, summary_copy)

    def _write_records(self, records: Iterable[Dict[str, Any]]) -> None:
        try:
            with self._jsonl_path.open("w", encoding="utf-8") as fh:
                for record in records:
                    fh.write(json.dumps(record, default=_json_default) + "\n")
        except OSError:  # pragma: no cover - filesystem issues
            profile_logger.warning("Failed to write profiling JSONL: %s", self._jsonl_path)

    def _write_summary(self, summary: Dict[Tuple[str, str, str], Dict[str, Any]]) -> None:
        try:
            with self._summary_path.open("w", encoding="utf-8", newline="") as fh:
                writer = csv.writer(fh)
                writer.writerow(
                    [
                        "module",
                        "name",
                        "type",
                        "calls",
                        "wall_ms_total",
                        "wall_ms_mean",
                        "wall_ms_p95",
                        "cpu_ms_total",
                        "alloc_kb_total",
                        "rss_kb_total",
                    ]
                )
                for stats in summary.values():
                    calls = max(stats.get("calls", 0), 1)
                    wall_total = stats.get("wall_ms_total", 0.0)
                    wall_mean = wall_total / calls
                    wall_p95 = _percentile(stats.get("wall_ms_values", []), 95)
                    cpu_total = stats.get("cpu_ms_total", 0.0)
                    alloc_total = stats.get("alloc_kb_total", 0.0)
                    rss_total = stats.get("rss_kb_total", 0.0)
                    writer.writerow(
                        [
                            stats.get("module"),
                            stats.get("name"),
                            stats.get("type"),
                            calls,
                            wall_total,
                            wall_mean,
                            wall_p95,
                            cpu_total,
                            alloc_total,
                            rss_total,
                        ]
                    )
        except OSError:  # pragma: no cover - filesystem issues
            profile_logger.warning("Failed to write profiling summary CSV: %s", self._summary_path)


    def _write_run_summary(
        self,
        records: Iterable[Dict[str, Any]],
        summary: Dict[Tuple[str, str, str], Dict[str, Any]],
    ) -> None:
        records_list = list(records)
        finished_at = datetime.now(timezone.utc)
        wall_values = [record.get("wall_ms", 0.0) for record in records_list]
        cpu_values = [record.get("cpu_ms", 0.0) for record in records_list]

        total_wall = sum(wall_values)
        total_cpu = sum(cpu_values)
        sample_count = len(records_list)
        unique_sections = len(summary)

        run_summary: Dict[str, Any] = {
            "pid": self.pid,
            "started_at": self._started_at.isoformat(),
            "finished_at": finished_at.isoformat(),
            "duration_s": (finished_at - self._started_at).total_seconds(),
            "records": sample_count,
            "unique_sections": unique_sections,
            "total_wall_ms": total_wall,
            "total_cpu_ms": total_cpu,
            "mean_wall_ms": (total_wall / sample_count) if sample_count else 0.0,
            "p95_wall_ms": _percentile(wall_values, 95),
            "max_wall_ms": max(wall_values) if wall_values else 0.0,
        }

        top_sections = sorted(
            summary.values(), key=lambda item: item.get("wall_ms_total", 0.0), reverse=True
        )[:10]
        run_summary["top_sections"] = [
            {
                "module": section.get("module"),
                "name": section.get("name"),
                "type": section.get("type"),
                "calls": section.get("calls", 0),
                "wall_ms_total": section.get("wall_ms_total", 0.0),
                "cpu_ms_total": section.get("cpu_ms_total", 0.0),
            }
            for section in top_sections
        ]

        try:
            with self._run_summary_json_path.open("w", encoding="utf-8") as fh:
                json.dump(run_summary, fh, indent=2)
        except OSError:  # pragma: no cover - filesystem issues
            profile_logger.warning(
                "Failed to write profiling run summary JSON: %s", self._run_summary_json_path
            )

        try:
            with self._run_summary_csv_path.open("w", encoding="utf-8", newline="") as fh:
                writer = csv.writer(fh)
                writer.writerow(
                    [
                        "pid",
                        "started_at",
                        "finished_at",
                        "duration_s",
                        "records",
                        "unique_sections",
                        "total_wall_ms",
                        "total_cpu_ms",
                        "mean_wall_ms",
                        "p95_wall_ms",
                        "max_wall_ms",
                    ]
                )
                writer.writerow(
                    [
                        run_summary["pid"],
                        run_summary["started_at"],
                        run_summary["finished_at"],
                        run_summary["duration_s"],
                        run_summary["records"],
                        run_summary["unique_sections"],
                        run_summary["total_wall_ms"],
                        run_summary["total_cpu_ms"],
                        run_summary["mean_wall_ms"],
                        run_summary["p95_wall_ms"],
                        run_summary["max_wall_ms"],
                    ]
                )
        except OSError:  # pragma: no cover - filesystem issues
            profile_logger.warning(
                "Failed to write profiling run summary CSV: %s", self._run_summary_csv_path
            )


def _json_default(value: Any) -> Any:  # pragma: no cover - JSON fallback
    if isinstance(value, (set, tuple)):
        return list(value)
    raise TypeError(f"Object of type {type(value)!r} is not JSON serializable")


def _percentile(values: List[float], percentile: float) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(values)
    k = (len(sorted_values) - 1) * (percentile / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return sorted_values[int(k)]
    d0 = sorted_values[int(f)] * (c - k)
    d1 = sorted_values[int(c)] * (k - f)
    return d0 + d1


def profile_function(name: Optional[str] = None):
    """Decorator that profiles the wrapped function when profiling is enabled."""

    def decorator(func):
        if not callable(func):  # pragma: no cover - defensive
            raise TypeError("@profile_function can only wrap callables")

        if inspect.iscoroutinefunction(func):

            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                profiler = Profiler.instance()
                if not getattr(profiler, "enabled", False):
                    return await func(*args, **kwargs)

                event_name = name or func.__qualname__
                measurement = profiler.before(event_name, func.__module__, "function")
                try:
                    return await func(*args, **kwargs)
                finally:
                    profiler.after(measurement)

            return async_wrapper

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            profiler = Profiler.instance()
            if not getattr(profiler, "enabled", False):
                return func(*args, **kwargs)

            event_name = name or func.__qualname__
            measurement = profiler.before(event_name, func.__module__, "function")
            try:
                return func(*args, **kwargs)
            finally:
                profiler.after(measurement)

        return wrapper

    return decorator


class _ProfileSectionContext:
    def __init__(self, profiler: Profiler, name: str, module: str):
        self._profiler = profiler
        self._name = name
        self._module = module
        self._measurement: Optional[_Measurement] = None

    def __enter__(self):
        self._measurement = self._profiler.before(self._name, self._module, "section")
        return self

    def __exit__(self, exc_type, exc, tb):
        self._profiler.after(self._measurement)
        return False


def profile_section(name: str):
    """Context manager to profile an arbitrary code block."""

    profiler = Profiler.instance()
    if not getattr(profiler, "enabled", False):
        return nullcontext()

    caller_frame = inspect.currentframe().f_back if inspect.currentframe() else None
    module_name = (
        caller_frame.f_globals.get("__name__", "__main__") if caller_frame else "__main__"
    )
    return _ProfileSectionContext(profiler, name, module_name)


# ---------------------------------------------------------------------------
# Merge utilities
# ---------------------------------------------------------------------------


def merge_profile_records(by_corr_id: bool = False) -> Path:
    """Merge per-process profile files into a single summary CSV.

    Args:
        by_corr_id: When True, the merged output groups records by correlation id
            in addition to the module/name/type triple.

    Returns:
        Path to the merged CSV file.
    """

    dir_path = _PROFILE_DIR
    dir_path.mkdir(parents=True, exist_ok=True)
    grouped: Dict[Tuple[str, str, str, Optional[str]], Dict[str, Any]] = {}

    for jsonl_path in sorted(dir_path.glob("profile_*.jsonl")):
        try:
            with jsonl_path.open("r", encoding="utf-8") as fh:
                for line in fh:
                    if not line.strip():
                        continue
                    record = json.loads(line)
                    corr_id = record.get("corr_id") if by_corr_id else None
                    key = (record.get("module"), record.get("name"), record.get("type"), corr_id)
                    stats = grouped.setdefault(
                        key,
                        {
                            "module": record.get("module"),
                            "name": record.get("name"),
                            "type": record.get("type"),
                            "corr_id": corr_id,
                            "calls": 0,
                            "wall_ms_total": 0.0,
                            "wall_ms_values": [],
                            "cpu_ms_total": 0.0,
                            "alloc_kb_total": 0.0,
                            "rss_kb_total": 0.0,
                        },
                    )
                    stats["calls"] += record.get("calls", 1)
                    wall_ms = record.get("wall_ms", 0.0)
                    stats["wall_ms_total"] += wall_ms
                    stats["wall_ms_values"].append(wall_ms)
                    stats["cpu_ms_total"] += record.get("cpu_ms", 0.0)
                    alloc_delta = record.get("alloc_kb_delta")
                    if alloc_delta is not None:
                        stats["alloc_kb_total"] += alloc_delta
                    rss_delta = record.get("rss_kb_delta")
                    if rss_delta is not None:
                        stats["rss_kb_total"] += rss_delta
        except OSError:  # pragma: no cover - filesystem issues
            profile_logger.warning("Failed to read profiling data: %s", jsonl_path)

    merged_path = dir_path / "summary_merged.csv"
    with merged_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        headers = [
            "module",
            "name",
            "type",
            "corr_id" if by_corr_id else None,
            "calls",
            "wall_ms_total",
            "wall_ms_mean",
            "wall_ms_p95",
            "cpu_ms_total",
            "alloc_kb_total",
            "rss_kb_total",
        ]
        writer.writerow([h for h in headers if h is not None])

        for stats in grouped.values():
            calls = max(stats.get("calls", 0), 1)
            wall_total = stats.get("wall_ms_total", 0.0)
            wall_mean = wall_total / calls
            wall_p95 = _percentile(stats.get("wall_ms_values", []), 95)
            row = [
                stats.get("module"),
                stats.get("name"),
                stats.get("type"),
            ]
            if by_corr_id:
                row.append(stats.get("corr_id"))
            row.extend(
                [
                    calls,
                    wall_total,
                    wall_mean,
                    wall_p95,
                    stats.get("cpu_ms_total", 0.0),
                    stats.get("alloc_kb_total", 0.0),
                    stats.get("rss_kb_total", 0.0),
                ]
            )
            writer.writerow(row)

    return merged_path


__all__ = [
    "profile_enabled",
    "profile_function",
    "profile_section",
    "Profiler",
    "merge_profile_records",
    "set_correlation_id",
    "reset_correlation_id",
    "get_correlation_id",
]
