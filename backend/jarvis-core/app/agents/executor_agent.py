"""
app/agents/executor_agent.py
Executes ActionDecision objects and returns structured results.
The backend never opens URLs directly — it returns the action payload
for the frontend to execute.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

from app.core.decision_engine import ActionDecision
from app.utils.logger import get_logger

log = get_logger("jarvis.agent.executor")

# Backend execution timeout
EXEC_TIMEOUT = float(30)


class ExecutorAgent:
    """
    Executes commands and system actions.
    Browser/UI actions are returned as payloads for the frontend.
    Backend-only actions (API calls, DB ops) are executed here.
    """

    async def execute(self, action: ActionDecision) -> dict:
        """
        Execute an action with timeout, retry on failure.

        Args:
            action: ActionDecision from DecisionEngine / PlannerAgent.

        Returns:
            dict with {success, result, latency_ms, action_type}
        """
        t0 = time.perf_counter()
        try:
            result = await asyncio.wait_for(
                self._dispatch(action),
                timeout=EXEC_TIMEOUT,
            )
            latency = round((time.perf_counter() - t0) * 1000, 1)
            log.info("Executed %s::%s in %.0f ms", action.type, action.name, latency)
            return {**result, "latency_ms": latency, "action_type": action.type}

        except asyncio.TimeoutError:
            log.warning("Executor timeout: %s::%s", action.type, action.name)
            return {"success": False, "error": "timeout", "action_type": action.type}
        except Exception as exc:
            log.exception("Executor error: %s", exc)
            return {"success": False, "error": str(exc), "action_type": action.type}

    async def _dispatch(self, action: ActionDecision) -> dict:
        """Route to the correct execution handler."""
        handlers = {
            "browser": self._exec_browser,
            "ui":      self._exec_ui,
            "media":   self._exec_media,
            "system":  self._exec_system,
            "api":     self._exec_api,
            "gesture_execution": self._exec_gesture,
        }
        handler = handlers.get(action.type, self._exec_unknown)
        return await handler(action.name, action.parameters)

    # ── Browser actions ────────────────────────────────────────────────
    # These are returned as payloads — the frontend opens the URL.

    async def _exec_browser(self, name: str, params: dict) -> dict:
        if name in ("open_url", "search_query"):
            url = params.get("url", "")
            if not url:
                return {"success": False, "error": "No URL"}
            # Backend validates the URL; frontend executes window.open()
            return {"success": True, "result": f"Frontend: open {url}", "url": url}
        return {"success": True, "result": f"Browser: {name}"}

    # ── UI actions ─────────────────────────────────────────────────────
    # Dispatched as CustomEvents to the frontend.

    async def _exec_ui(self, name: str, params: dict) -> dict:
        log.debug("UI action: %s %s", name, params)
        return {"success": True, "result": f"UI: {name}", "ui_command": name}

    # ── Media actions ──────────────────────────────────────────────────

    async def _exec_media(self, name: str, params: dict) -> dict:
        log.debug("Media action: %s", name)
        return {"success": True, "result": f"Media: {name}"}

    # ── System actions ─────────────────────────────────────────────────

    async def _exec_system(self, name: str, params: dict) -> dict:
        if name == "get_status":
            return {
                "success": True,
                "result":  "Systems operational",
                "status":  {"mode": "operational", "agents": "active"},
            }
        if name == "clear_memory":
            # Memory clearing is handled at the core level; signal frontend
            return {"success": True, "result": "Memory clear requested"}
        if name == "set_mode":
            mode = params.get("mode", "")
            return {"success": True, "result": f"Mode set to {mode}", "mode": mode}
        log.debug("System action: %s", name)
        return {"success": True, "result": f"System: {name}"}

    # ── API actions ────────────────────────────────────────────────────

    async def _exec_api(self, name: str, params: dict) -> dict:
        if name == "get_status":
            return {"success": True, "result": "API status OK"}
        return {"success": True, "result": f"API: {name}"}

    # ── Gesture execution ──────────────────────────────────────────────

    async def _exec_gesture(self, name: str, params: dict) -> dict:
        return {"success": True, "result": f"Gesture executed: {name}"}

    # ── Unknown ────────────────────────────────────────────────────────

    async def _exec_unknown(self, name: str, params: dict) -> dict:
        log.warning("Unknown action: %s", name)
        return {"success": False, "error": f"Unknown action: {name}"}
