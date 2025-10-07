"""Helpers for constructing Rich consoles with environment-driven verbosity."""

from __future__ import annotations

from typing import Any, Optional

try:  # Rich is an optional dependency in some environments
    from rich.console import Console
except Exception:  # pragma: no cover - optional dependency
    Console = None  # type: ignore

from .settings import get_logging_verbosity

_SILENT_KWARGS = {
    "quiet": True,
    "highlight": False,
    "markup": False,
    "emoji": False,
    "color_system": None,
    "soft_wrap": True,
}

_VERBOSE_KWARGS = {"soft_wrap": True}


def get_console(level: Optional[str] = None, **kwargs: Any) -> "Console":
    """Return a Rich ``Console`` configured for the requested verbosity.

    The console respects ``ZENTIO_LOG_VERBOSITY`` so benchmarks and other
    non-interactive tooling can disable expensive rich rendering when the
    verbosity is set to ``warning``/``error``/``quiet``.

    Args:
        level: Optional verbosity override (``debug``/``info``/... ). When not
            provided the value returned by :func:`get_logging_verbosity` is used.
        **kwargs: Additional keyword arguments forwarded to ``Console``.

    Returns:
        A configured ``Console`` instance. When Rich is not installed a minimal
        stub that mimics the ``Console`` API is returned.
    """

    verbosity = (level or get_logging_verbosity()).lower()

    # Rich is optional for some deployments; fall back to a lightweight stub
    if Console is None:  # pragma: no cover - executed when Rich is unavailable
        return _FallbackConsole()

    base_kwargs = _VERBOSE_KWARGS if verbosity in {"debug", "info"} else _SILENT_KWARGS
    config = {**base_kwargs, **kwargs}
    return Console(**config)


class _FallbackConsole:
    """Fallback console used when Rich is not available."""

    def print(self, *args: Any, **kwargs: Any) -> None:  # pragma: no cover - trivial
        print(*args, **kwargs)

    def log(self, *args: Any, **kwargs: Any) -> None:  # pragma: no cover - trivial
        print(*args, **kwargs)


__all__ = ["get_console"]
