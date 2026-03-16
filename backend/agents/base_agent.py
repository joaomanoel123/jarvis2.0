"""
agents/base_agent.py
====================
Abstract base class that every JARVIS agent inherits from.

Contract
────────
Every agent must implement:

    async def run(self, task: AgentTask) -> AgentResult

Agents receive a fully-hydrated AgentTask (intent, history, context)
and return an AgentResult (text, tool_calls, metadata, next_agent hint).

Agents are stateless – all session state lives in MemoryManager.
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


# ── Agent identity ─────────────────────────────────────────────────────────────

class AgentType(str, Enum):
    PLANNER   = "planner"
    EXECUTOR  = "executor"
    KNOWLEDGE = "knowledge"
    GESTURE   = "gesture"
    BRAIN     = "brain"          # reserved for JarvisBrain itself


# ── Task / Result data containers ──────────────────────────────────────────────

@dataclass
class AgentTask:
    """
    Fully-hydrated task handed to an agent by JarvisBrain.

    Attributes:
        session_id:  Active memory session.
        user_input:  Raw user message (text or gesture descriptor).
        intent:      Parsed intent string (e.g. "code", "search", "gesture").
        history:     Recent conversation messages [{role, content}].
        context:     Session context bag (arbitrary metadata).
        metadata:    Extra per-task data (e.g. gesture coordinates).
        created_at:  Unix timestamp when the task was created.
    """
    session_id:  str
    user_input:  str
    intent:      str                    = "chat"
    history:     list[dict]             = field(default_factory=list)
    context:     dict[str, Any]         = field(default_factory=dict)
    metadata:    dict[str, Any]         = field(default_factory=dict)
    created_at:  float                  = field(default_factory=time.time)


@dataclass
class AgentResult:
    """
    Structured output from an agent.

    Attributes:
        text:         Final assistant response text.
        agent:        Which agent produced this result.
        success:      Whether the agent completed without error.
        tool_calls:   List of tool invocations (name + result dicts).
        steps:        Intermediate reasoning steps (for Planner).
        next_agent:   Optional hint for the Brain to chain to another agent.
        latency_ms:   Wall-clock time in milliseconds.
        metadata:     Arbitrary extra fields.
        error:        Error message if success=False.
    """
    text:        str
    agent:       AgentType
    success:     bool                   = True
    tool_calls:  list[dict]             = field(default_factory=list)
    steps:       list[str]              = field(default_factory=list)
    next_agent:  AgentType | None       = None
    latency_ms:  float                  = 0.0
    metadata:    dict[str, Any]         = field(default_factory=dict)
    error:       str | None             = None


# ── Base agent ─────────────────────────────────────────────────────────────────

class BaseAgent(ABC):
    """
    Abstract base class for all JARVIS agents.

    Subclasses implement run() and declare their agent_type.
    Timing, logging, and error containment are handled by execute()
    so subclasses only need to focus on business logic.
    """

    agent_type: AgentType   # must be set at class level by subclasses

    def __init__(self) -> None:
        self.log = logging.getLogger(
            f"jarvis.agent.{self.__class__.__name__}"
        )

    @abstractmethod
    async def run(self, task: AgentTask) -> AgentResult:
        """
        Core agent logic.  Implemented by every concrete agent.

        Args:
            task: Fully-hydrated task with user input, history, and context.

        Returns:
            AgentResult with the final response and any tool call records.
        """
        ...

    async def execute(self, task: AgentTask) -> AgentResult:
        """
        Public entry-point.  Wraps run() with timing and error containment.

        Callers (JarvisBrain) always use execute(), never run() directly.
        """
        t0 = time.perf_counter()
        self.log.info("→ %s.execute | session=%s | intent=%s",
                      self.__class__.__name__, task.session_id[:8], task.intent)
        try:
            result = await self.run(task)
        except Exception as exc:
            self.log.exception("%s failed: %s", self.__class__.__name__, exc)
            result = AgentResult(
                text=f"Agent {self.__class__.__name__} encountered an error: {exc}",
                agent=self.agent_type,
                success=False,
                error=str(exc),
            )
        result.latency_ms = round((time.perf_counter() - t0) * 1000, 1)
        self.log.info("← %s finished | %.0f ms | success=%s",
                      self.__class__.__name__, result.latency_ms, result.success)
        return result

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} type={self.agent_type.value}>"
