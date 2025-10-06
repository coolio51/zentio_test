"""Environment-driven feature flag helpers for scheduler services."""

from __future__ import annotations

import os
from functools import lru_cache


@lru_cache(maxsize=None)
def get_scheduler_mode() -> str:
    return os.getenv("ZENTIO_SCHEDULER_MODE", "topo").lower()


@lru_cache(maxsize=None)
def get_slot_search_mode() -> str:
    return os.getenv("ZENTIO_SLOT_SEARCH", "step").lower()


@lru_cache(maxsize=None)
def use_resource_manager_clone() -> bool:
    return os.getenv("ZENTIO_RM_CLONE", "0") in {"1", "true", "yes", "on"}


@lru_cache(maxsize=None)
def get_logging_verbosity() -> str:
    return os.getenv("ZENTIO_LOG_VERBOSITY", "info").lower()


@lru_cache(maxsize=None)
def debug_print_enabled() -> bool:
    return os.getenv("ZENTIO_DEBUG_PRINT", "0") in {"1", "true", "yes", "on"}


__all__ = [
    "get_scheduler_mode",
    "get_slot_search_mode",
    "use_resource_manager_clone",
    "get_logging_verbosity",
    "debug_print_enabled",
]
