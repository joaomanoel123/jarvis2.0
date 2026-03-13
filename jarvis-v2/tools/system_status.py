"""
tools/system_status.py
======================
system_status — returns safe system diagnostics for the JARVIS runtime.

Only exposes metrics that are safe to surface to an AI agent:
CPU %, RAM usage, disk usage, Python version, uptime.
Never exposes credentials, environment variables, or process list.
"""

from __future__ import annotations

import asyncio
import platform
import sys
import time
from typing import Any

_START_TIME = time.time()


async def system_status() -> dict:
    """
    Return a snapshot of current system health metrics.

    Returns:
        {
            "cpu_percent":    float,
            "ram_used_mb":    float,
            "ram_total_mb":   float,
            "ram_percent":    float,
            "disk_used_gb":   float,
            "disk_total_gb":  float,
            "disk_percent":   float,
            "uptime_seconds": float,
            "python_version": str,
            "platform":       str,
        }
    """
    loop  = asyncio.get_running_loop()
    stats = await loop.run_in_executor(None, _collect_sync)
    return stats


def _collect_sync() -> dict[str, Any]:
    try:
        import psutil  # optional dependency
        cpu  = psutil.cpu_percent(interval=0.5)
        ram  = psutil.virtual_memory()
        disk = psutil.disk_usage("/")
        return {
            "cpu_percent":    round(cpu, 1),
            "ram_used_mb":    round(ram.used / 1024 ** 2, 1),
            "ram_total_mb":   round(ram.total / 1024 ** 2, 1),
            "ram_percent":    round(ram.percent, 1),
            "disk_used_gb":   round(disk.used / 1024 ** 3, 2),
            "disk_total_gb":  round(disk.total / 1024 ** 3, 2),
            "disk_percent":   round(disk.percent, 1),
            "uptime_seconds": round(time.time() - _START_TIME, 1),
            "python_version": sys.version.split()[0],
            "platform":       platform.system(),
        }
    except ImportError:
        # psutil not installed — return basic info only
        return {
            "cpu_percent":    None,
            "ram_used_mb":    None,
            "ram_total_mb":   None,
            "ram_percent":    None,
            "disk_used_gb":   None,
            "disk_total_gb":  None,
            "disk_percent":   None,
            "uptime_seconds": round(time.time() - _START_TIME, 1),
            "python_version": sys.version.split()[0],
            "platform":       platform.system(),
            "note":           "Install psutil for full metrics: pip install psutil",
        }
