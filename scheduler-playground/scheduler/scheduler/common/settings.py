"""Environment-driven feature flag helpers for scheduler services."""

from __future__ import annotations

import os
from functools import lru_cache


@lru_cache(maxsize=None)
def allow_naive_scheduler() -> bool:
    """Return True when the legacy ``naive`` scheduler path is explicitly allowed."""

    return os.getenv("ZENTIO_ALLOW_NAIVE", "0").lower() in {"1", "true", "yes", "on"}


@lru_cache(maxsize=None)
def get_scheduler_mode() -> str:
    """Return the requested scheduler implementation.

    The topological scheduler is the default.  The legacy ``naive`` mode can only be
    re-enabled by setting ``ZENTIO_ALLOW_NAIVE`` in addition to
    ``ZENTIO_SCHEDULER_MODE=naive`` which prevents accidental regressions when the
    slow path is left configured in production or CI environments.
    """

    requested = os.getenv("ZENTIO_SCHEDULER_MODE", "topo").lower()
    if requested == "naive" and not allow_naive_scheduler():
        return "topo"
    return requested


@lru_cache(maxsize=None)
def get_slot_search_mode() -> str:
    return os.getenv("ZENTIO_SLOT_SEARCH", "step").lower()


@lru_cache(maxsize=None)
def use_resource_manager_clone() -> bool:
    return os.getenv("ZENTIO_RM_CLONE", "0") in {"1", "true", "yes", "on"}


@lru_cache(maxsize=None)
def debug_print_enabled() -> bool:
    return os.getenv("ZENTIO_DEBUG_PRINT", "0") in {"1", "true", "yes", "on"}


@lru_cache(maxsize=None)
def get_log_verbosity() -> str:
    """Return the configured log verbosity for playground runs."""

    return os.getenv("ZENTIO_LOG_VERBOSITY", "info").lower()


__all__ = [
    "allow_naive_scheduler",
    "get_scheduler_mode",
    "get_slot_search_mode",
    "use_resource_manager_clone",
    "debug_print_enabled",
    "get_log_verbosity",
]
