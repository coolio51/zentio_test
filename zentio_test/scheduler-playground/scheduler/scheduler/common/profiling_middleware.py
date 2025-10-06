"""FastAPI middleware for propagating profiling correlation ids."""

from __future__ import annotations

import uuid
from typing import Callable, Optional

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware

from .profiling import (
    get_correlation_id,
    profile_enabled,
    profile_section,
    reset_correlation_id,
    set_correlation_id,
)


class ProfilingMiddleware(BaseHTTPMiddleware):
    """Attach a correlation id to each request for profiling traces."""

    async def dispatch(self, request: Request, call_next: Callable):  # type: ignore[override]
        if not profile_enabled():
            return await call_next(request)

        incoming = request.headers.get("X-Profile-Correlation-ID") or request.headers.get(
            "X-Request-ID"
        )
        corr_id = incoming or str(uuid.uuid4())
        token = set_correlation_id(corr_id)
        try:
            with profile_section("api.request"):
                response = await call_next(request)
            response.headers.setdefault("X-Profile-Correlation-ID", corr_id)
            return response
        finally:
            reset_correlation_id(token)


def get_request_correlation_id() -> Optional[str]:
    """Expose the current request correlation id."""

    return get_correlation_id()


__all__ = ["ProfilingMiddleware", "get_request_correlation_id"]
