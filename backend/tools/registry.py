"""
tools/registry.py
=================
Central tool registry.  Every tool must be:

  1. Registered here (name → callable + schema).
  2. Listed in the ALLOWED_TOOLS setting to be executable.
  3. Called through execute_tool() — never directly by agents.

Security model
──────────────
• execute_tool() validates the name against the runtime allowlist.
• Each tool runs with a per-call asyncio.wait_for() timeout.
• Tool output is truncated to TOOL_MAX_OUTPUT_BYTES.
• run_python_code is restricted via a separate sandbox (no __builtins__ escape).
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from config.settings import get_settings

log = logging.getLogger(__name__)
cfg = get_settings()

AsyncToolFn = Callable[..., Awaitable[dict]]


@dataclass
class ToolSpec:
    name:        str
    fn:          AsyncToolFn
    description: str
    parameters:  dict               # JSON-Schema style
    tags:        list[str] = field(default_factory=list)


# ── Registry ───────────────────────────────────────────────────────────────────

class ToolRegistry:
    """
    Manages tool registration, discovery, and sandboxed execution.
    """

    def __init__(self) -> None:
        self._tools: dict[str, ToolSpec] = {}

    def register(self, spec: ToolSpec) -> None:
        self._tools[spec.name] = spec
        log.debug("ToolRegistry: registered '%s'", spec.name)

    def get(self, name: str) -> ToolSpec | None:
        return self._tools.get(name)

    def list_names(self) -> list[str]:
        return list(self._tools.keys())

    def list_allowed(self) -> list[str]:
        """Return only tools that are in the runtime allowlist."""
        return [n for n in self._tools if n in cfg.allowed_tools_set]

    def schema_for_llm(self) -> list[dict]:
        """Return tool descriptors formatted for LLM prompt injection."""
        return [
            {
                "name":        spec.name,
                "description": spec.description,
                "parameters":  spec.parameters,
            }
            for spec in self._tools.values()
            if spec.name in cfg.allowed_tools_set
        ]

    async def execute(self, name: str, kwargs: dict) -> dict:
        """
        Execute a tool with full sandboxing:

        1. Allowlist check   — refuses unknown / disallowed names.
        2. Schema check      — validates required parameters are present.
        3. Timeout wrap      — cancels after TOOL_TIMEOUT_SECONDS.
        4. Output cap        — truncates string values over the byte limit.
        5. Exception barrier — converts all exceptions to error dicts.

        Returns:
            {"tool": str, "result": dict, "success": bool}
        """
        # 1 — allowlist
        if name not in cfg.allowed_tools_set:
            log.warning("ToolRegistry: blocked disallowed tool '%s'", name)
            return {
                "tool":    name,
                "result":  {"error": f"Tool '{name}' is not in the allowed list."},
                "success": False,
            }

        spec = self.get(name)
        if spec is None:
            return {
                "tool":    name,
                "result":  {"error": f"Tool '{name}' is not registered."},
                "success": False,
            }

        # 2 — required parameters
        required = spec.parameters.get("required", [])
        missing  = [r for r in required if r not in kwargs]
        if missing:
            return {
                "tool":    name,
                "result":  {"error": f"Missing required parameters: {missing}"},
                "success": False,
            }

        # 3 — execute with timeout
        try:
            log.info("Tool '%s' executing | args: %s", name, list(kwargs.keys()))
            result: dict = await asyncio.wait_for(
                spec.fn(**kwargs),
                timeout=cfg.TOOL_TIMEOUT_SECONDS,
            )
        except asyncio.TimeoutError:
            log.warning("Tool '%s' timed out after %ds", name, cfg.TOOL_TIMEOUT_SECONDS)
            return {
                "tool":    name,
                "result":  {"error": f"Tool timed out after {cfg.TOOL_TIMEOUT_SECONDS}s"},
                "success": False,
            }
        except Exception as exc:
            log.exception("Tool '%s' raised: %s", name, exc)
            return {
                "tool":    name,
                "result":  {"error": str(exc)},
                "success": False,
            }

        # 4 — output cap
        result = _truncate_output(result, cfg.TOOL_MAX_OUTPUT_BYTES)
        log.info("Tool '%s' succeeded", name)
        return {"tool": name, "result": result, "success": True}


def _truncate_output(obj: Any, max_bytes: int) -> Any:
    """Recursively truncate long strings in a dict to stay under max_bytes."""
    if isinstance(obj, dict):
        return {k: _truncate_output(v, max_bytes) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_truncate_output(v, max_bytes) for v in obj]
    if isinstance(obj, str) and len(obj.encode()) > max_bytes:
        return obj.encode()[:max_bytes].decode("utf-8", errors="ignore") + " … [truncated]"
    return obj


# Module-level singleton
tool_registry = ToolRegistry()
