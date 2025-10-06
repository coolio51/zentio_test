from datetime import datetime, timedelta

from scheduler.models import TaskPhase


def style_datetime(dt: datetime) -> str:
    """Format datetime with styling"""
    return dt.strftime(f"%d.%m.%Y [bold italic]%H:%M[/bold italic]")


def style_duration(duration: timedelta) -> str:
    """Format duration in a readable way"""
    total_hours = duration.total_seconds() / 3600
    if total_hours < 24:
        return f"{total_hours:.1f}h"
    else:
        days = int(total_hours // 24)
        hours = total_hours % 24
        return f"{days}d {hours:.1f}h"


def style_phase(phase: TaskPhase) -> str:
    """Style phase with color"""
    return f"[{phase.value}]{phase.value.upper()}[/{phase.value}]"
