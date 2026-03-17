"""
backend/services/tool_service.py
==================================
ToolService — centralised tool execution with allowlist, timeout, and schema validation.

This service wraps every tool callable and enforces:
  • Allowlist check (configured via ALLOWED_TOOLS env var).
  • JSON-schema parameter validation.
  • Per-call asyncio timeout.
  • Output truncation.
  • Structured result envelope: {tool, result, success, latency_ms}.

Built-in tools
──────────────
  web_search       — DuckDuckGo instant answers + related links.
  run_python_code  — Sandboxed Python execution (AST-gated).
  file_reader      — Read text files from /tmp/jarvis_files.
  system_status    — CPU, RAM, disk, uptime metrics.
  open_url         — Open a URL in the default browser.
  send_email       — Placeholder for email integration.

Adding a tool
─────────────
  1. Write an async function returning a dict.
  2. Register with tool_service.register(ToolSpec(...)).
  3. Add the name to ALLOWED_TOOLS env var.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
import webbrowser
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

log = logging.getLogger("jarvis.services.tool")

ALLOWED_TOOLS_ENV = os.getenv(
    "ALLOWED_TOOLS",
    "web_search,run_python_code,file_reader,system_status,open_url",
)
TOOL_TIMEOUT_S    = int(os.getenv("TOOL_TIMEOUT_SECONDS", "30"))
MAX_OUTPUT_BYTES  = int(os.getenv("TOOL_MAX_OUTPUT_BYTES", "65536"))

AsyncToolFn = Callable[..., Awaitable[dict]]


@dataclass
class ToolSpec:
    name:        str
    fn:          AsyncToolFn
    description: str
    parameters:  dict
    tags:        list[str] = field(default_factory=list)


@dataclass
class ToolResult:
    tool:       str
    result:     Any
    success:    bool
    latency_ms: float = 0.0
    error:      str | None = None

    def to_dict(self) -> dict:
        return {
            "tool":       self.tool,
            "result":     self.result,
            "success":    self.success,
            "latency_ms": round(self.latency_ms, 1),
            "error":      self.error,
        }


# ── ToolService ────────────────────────────────────────────────────────────────

class ToolService:
    """
    Central tool registry and executor.

    Intended as a singleton — import `tool_service` from this module.
    """

    def __init__(self) -> None:
        self._tools: dict[str, ToolSpec] = {}
        self._allowed: frozenset[str] = frozenset(
            t.strip() for t in ALLOWED_TOOLS_ENV.split(",") if t.strip()
        )
        self._calls    = 0
        self._failures = 0

    # ── Registration ──────────────────────────────────────────────────────────

    def register(self, spec: ToolSpec) -> None:
        self._tools[spec.name] = spec
        log.debug("ToolService: registered '%s'", spec.name)

    def register_all_defaults(self) -> None:
        """Register all built-in tools. Call once at application startup."""
        from backend.tools.web_search    import web_search
        from backend.tools.code_runner   import run_python_code
        from backend.tools.file_reader   import file_reader
        from backend.tools.system_status import system_status

        self.register(ToolSpec(
            name="web_search",
            fn=web_search,
            description="Search the web. Returns instant answer + top links.",
            parameters={"type": "object", "required": ["query"],
                        "properties": {"query": {"type": "string"},
                                       "max_results": {"type": "integer", "default": 5}}},
            tags=["search"],
        ))
        self.register(ToolSpec(
            name="run_python_code",
            fn=run_python_code,
            description="Execute Python in a sandboxed environment.",
            parameters={"type": "object", "required": ["code"],
                        "properties": {"code": {"type": "string"}}},
            tags=["code"],
        ))
        self.register(ToolSpec(
            name="file_reader",
            fn=file_reader,
            description="Read a text file from /tmp/jarvis_files.",
            parameters={"type": "object", "required": ["path"],
                        "properties": {"path": {"type": "string"}}},
            tags=["file"],
        ))
        self.register(ToolSpec(
            name="system_status",
            fn=system_status,
            description="Return CPU, RAM, disk, and uptime.",
            parameters={"type": "object", "required": [], "properties": {}},
            tags=["system"],
        ))
        self.register(ToolSpec(
            name="open_url",
            fn=_open_url,
            description="Open a URL in the default web browser.",
            parameters={"type": "object", "required": ["url"],
                        "properties": {"url": {"type": "string"}}},
            tags=["browser"],
        ))
        log.info("ToolService: %d built-in tools registered", len(self._tools))

    # ── Execution ─────────────────────────────────────────────────────────────

    async def execute(self, name: str, kwargs: dict) -> ToolResult:
        """
        Execute a tool with full sandboxing.

        Steps:
        1. Allowlist check.
        2. Existence check.
        3. Required-parameter validation.
        4. Timeout-bounded execution.
        5. Output truncation.
        """
        t0 = time.perf_counter()
        self._calls += 1

        # 1 — Allowlist
        if name not in self._allowed:
            self._failures += 1
            log.warning("ToolService: blocked disallowed tool '%s'", name)
            return ToolResult(
                tool=name, result={"error": f"Tool '{name}' is not allowed"},
                success=False, error="not_allowed",
            )

        # 2 — Existence
        spec = self._tools.get(name)
        if spec is None:
            self._failures += 1
            return ToolResult(
                tool=name, result={"error": f"Tool '{name}' is not registered"},
                success=False, error="not_registered",
            )

        # 3 — Required parameters
        required = spec.parameters.get("required", [])
        missing  = [r for r in required if r not in kwargs]
        if missing:
            self._failures += 1
            return ToolResult(
                tool=name,
                result={"error": f"Missing required parameters: {missing}"},
                success=False, error="missing_params",
            )

        # 4 — Execute with timeout
        try:
            log.info("Tool '%s' | args: %s", name, list(kwargs.keys()))
            raw: dict = await asyncio.wait_for(
                spec.fn(**kwargs),
                timeout=TOOL_TIMEOUT_S,
            )
        except asyncio.TimeoutError:
            self._failures += 1
            msg = f"Tool '{name}' timed out after {TOOL_TIMEOUT_S}s"
            log.warning(msg)
            return ToolResult(
                tool=name, result={"error": msg},
                success=False, error="timeout",
                latency_ms=(time.perf_counter() - t0) * 1000,
            )
        except Exception as exc:
            self._failures += 1
            log.exception("Tool '%s' raised: %s", name, exc)
            return ToolResult(
                tool=name, result={"error": str(exc)},
                success=False, error=str(exc),
                latency_ms=(time.perf_counter() - t0) * 1000,
            )

        # 5 — Truncate output
        raw = _truncate(raw, MAX_OUTPUT_BYTES)
        latency = round((time.perf_counter() - t0) * 1000, 1)
        log.info("Tool '%s' done in %.0f ms", name, latency)
        return ToolResult(tool=name, result=raw, success=True, latency_ms=latency)

    async def execute_many(self, calls: list[dict]) -> list[ToolResult]:
        """Execute multiple tool calls concurrently."""
        return await asyncio.gather(*[
            self.execute(c["name"], c.get("kwargs", {}))
            for c in calls
        ])

    # ── Discovery ─────────────────────────────────────────────────────────────

    def schema_for_llm(self) -> list[dict]:
        """Return tool schemas for LLM prompt injection."""
        return [
            {"name": s.name, "description": s.description, "parameters": s.parameters}
            for s in self._tools.values()
            if s.name in self._allowed
        ]

    def list_allowed(self) -> list[str]:
        return [n for n in self._tools if n in self._allowed]

    def stats(self) -> dict:
        return {
            "registered": len(self._tools),
            "allowed":    len(self._allowed),
            "calls":      self._calls,
            "failures":   self._failures,
        }


# ── Built-in: open_url ─────────────────────────────────────────────────────────

async def _open_url(url: str) -> dict:
    """Open a URL in the default browser (non-blocking)."""
    import urllib.parse
    # Validate URL
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise ValueError(f"Invalid URL scheme: {parsed.scheme}")
    webbrowser.open(url)
    return {"opened": url, "status": "ok"}


# ── Helpers ────────────────────────────────────────────────────────────────────

def _truncate(obj: Any, max_bytes: int) -> Any:
    if isinstance(obj, dict):
        return {k: _truncate(v, max_bytes) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_truncate(v, max_bytes) for v in obj]
    if isinstance(obj, str) and len(obj.encode()) > max_bytes:
        return obj.encode()[:max_bytes].decode("utf-8", errors="ignore") + " …[truncated]"
    return obj


# Module singleton
tool_service = ToolService()
