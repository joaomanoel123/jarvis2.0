"""
brain/jarvis_brain.py
=====================
JarvisBrain — the central autonomous orchestrator.

                         ┌───────────────────────────────┐
                         │          JarvisBrain           │
                         │                               │
                         │  1. Understand intent         │
                         │  2. Select agent(s)           │
                         │  3. Manage memory             │
                         │  4. Orchestrate execution     │
                         │  5. Assemble final response   │
                         └───────────────────────────────┘
                                        │
               ┌───────────────────────┼───────────────────────┐
               ▼                       ▼                       ▼
         PlannerAgent           ExecutorAgent          KnowledgeAgent
         (decompose)            (tools + actions)      (retrieval)
                                                              │
                                                       GestureAgent
                                                       (MediaPipe)

Intent routing table
────────────────────
  chat          → ExecutorAgent  (direct LLM answer)
  code          → PlannerAgent → ExecutorAgent
  search / info → KnowledgeAgent
  analyse       → PlannerAgent → ExecutorAgent
  gesture       → GestureAgent
  system        → ExecutorAgent (system_status tool)
  *             → ExecutorAgent (fallback)
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from agents.base_agent import AgentResult, AgentTask, AgentType
from agents.executor_agent import ExecutorAgent
from agents.gesture_agent import GestureAgent
from agents.knowledge_agent import KnowledgeAgent
from agents.planner_agent import PlannerAgent
from memory.memory_manager import memory
from services.llm_service import llm_service

log = logging.getLogger("jarvis.brain")


# ── Intent classification ──────────────────────────────────────────────────────

class Intent(str, Enum):
    CHAT      = "chat"
    CODE      = "code"
    SEARCH    = "search"
    ANALYSE   = "analyse"
    GESTURE   = "gesture"
    SYSTEM    = "system"
    PLAN      = "plan"
    MEMORY    = "memory"


# Regex fast-path rules — ordered by specificity
_INTENT_RULES: list[tuple[re.Pattern, Intent]] = [
    (re.compile(r"\b(write|generate|create|build|code|script|function|class|refactor|debug|fix bug)\b", re.I), Intent.CODE),
    (re.compile(r"\b(search|find|look up|what is|who is|when|where|news|latest|research)\b", re.I), Intent.SEARCH),
    (re.compile(r"\b(analyse|analyze|examine|explore|dataset|csv|statistics|insights)\b", re.I), Intent.ANALYSE),
    (re.compile(r"\b(plan|break down|steps to|how do i|how should i|strategy)\b", re.I), Intent.PLAN),
    (re.compile(r"\b(remember|save|store|what did i|recall|note that)\b", re.I), Intent.MEMORY),
    (re.compile(r"\b(system|status|health|cpu|memory|ram|disk|uptime)\b", re.I), Intent.SYSTEM),
]


def _classify_intent(text: str) -> Intent:
    """Fast regex classification; falls back to CHAT for unknown patterns."""
    for pattern, intent in _INTENT_RULES:
        if pattern.search(text):
            return intent
    return Intent.CHAT


# ── Orchestration result ───────────────────────────────────────────────────────

@dataclass
class BrainResponse:
    """
    Top-level response returned to the API layer.

    Attributes:
        session_id:  Active session UUID.
        text:        Final assistant text to show the user.
        intent:      Classified intent.
        agent_path:  Ordered list of agents that were invoked.
        tool_calls:  All tool invocations across all agents.
        steps:       Plan steps (populated for CODE/ANALYSE/PLAN intents).
        latency_ms:  Total wall-clock time in milliseconds.
        metadata:    Arbitrary extra fields.
    """
    session_id: str
    text:       str
    intent:     str
    agent_path: list[str]                   = field(default_factory=list)
    tool_calls: list[dict]                  = field(default_factory=list)
    steps:      list[str]                   = field(default_factory=list)
    latency_ms: float                       = 0.0
    metadata:   dict[str, Any]              = field(default_factory=dict)
    success:    bool                        = True
    error:      str | None                  = None


# ── JarvisBrain ────────────────────────────────────────────────────────────────

class JarvisBrain:
    """
    Central orchestrator for the JARVIS multi-agent system.

    Responsibilities
    ────────────────
    1. Resolve/create a memory session.
    2. Classify user intent (regex fast-path + optional LLM fallback).
    3. Select and invoke the appropriate agent pipeline.
    4. Persist the turn to conversation memory.
    5. Return a structured BrainResponse to the API layer.

    The Brain is stateless — all session state lives in MemoryManager.
    It can be instantiated as a singleton at startup.
    """

    def __init__(self) -> None:
        self._planner   = PlannerAgent()
        self._executor  = ExecutorAgent()
        self._knowledge = KnowledgeAgent()
        self._gesture   = GestureAgent()

        # Maps intent enum → async method
        self._routes: dict[Intent, Any] = {
            Intent.CHAT:    self._route_chat,
            Intent.CODE:    self._route_plan_execute,
            Intent.ANALYSE: self._route_plan_execute,
            Intent.PLAN:    self._route_plan_execute,
            Intent.SEARCH:  self._route_knowledge,
            Intent.SYSTEM:  self._route_system,
            Intent.MEMORY:  self._route_memory,
            Intent.GESTURE: self._route_gesture,
        }
        log.info("JarvisBrain initialised")

    # ── Public API ─────────────────────────────────────────────────────────────

    async def process(
        self,
        user_input:  str,
        session_id:  str | None = None,
        metadata:    dict | None = None,
    ) -> BrainResponse:
        """
        Process a user message end-to-end.

        Args:
            user_input:  Raw user text (or gesture descriptor).
            session_id:  Existing session UUID, or None to create a new one.
            metadata:    Optional extra context (gesture data, client info …).

        Returns:
            BrainResponse with text, intent, agent_path, tool_calls, etc.
        """
        t0 = time.perf_counter()
        meta = metadata or {}

        # 1 — Resolve session
        sid = await memory.get_or_create(session_id)

        # 2 — Classify intent
        intent = _classify_intent(user_input)
        # Gesture is always explicit from the /gesture endpoint
        if meta.get("gesture_id"):
            intent = Intent.GESTURE

        log.info("Brain.process | session=%s | intent=%s | input=%.60r",
                 sid[:8], intent.value, user_input)

        # 3 — Fetch history + context for this session
        history = await memory.get_messages(
            sid, roles=["user", "assistant"], last_n=20
        )
        context = await memory.get_full_context(sid)

        # 4 — Build task
        task = AgentTask(
            session_id=sid,
            user_input=user_input,
            intent=intent.value,
            history=history,
            context=context,
            metadata=meta,
        )

        # 5 — Route to agent pipeline
        route_fn = self._routes.get(intent, self._route_chat)
        try:
            result: AgentResult = await route_fn(task)
        except Exception as exc:
            log.exception("Brain routing failed: %s", exc)
            result = AgentResult(
                text=f"I encountered an unexpected error: {exc}",
                agent=AgentType.BRAIN,
                success=False,
                error=str(exc),
            )

        # 6 — Persist to memory
        await memory.add_message(sid, "user", user_input)
        await memory.add_message(
            sid, "assistant", result.text,
            metadata={"agent": result.agent.value, "intent": intent.value},
        )

        # 7 — Evict expired short-term entries
        await memory.evict_short_term(sid)

        latency_ms = round((time.perf_counter() - t0) * 1000, 1)
        log.info("Brain.process done | %.0f ms | agent_path=%s",
                 latency_ms, result.metadata.get("agent_path", [result.agent.value]))

        return BrainResponse(
            session_id=sid,
            text=result.text,
            intent=intent.value,
            agent_path=result.metadata.get("agent_path", [result.agent.value]),
            tool_calls=result.tool_calls,
            steps=result.steps,
            latency_ms=latency_ms,
            metadata={
                "model_id": llm_service.info().get("model_id"),
                **result.metadata,
            },
            success=result.success,
            error=result.error,
        )

    async def process_gesture(
        self,
        gesture_id:  str,
        session_id:  str | None = None,
        confidence:  float = 1.0,
        landmarks:   list[dict] | None = None,
        context:     dict | None = None,
    ) -> BrainResponse:
        """
        Convenience wrapper for gesture input.

        Args:
            gesture_id:  Gesture name from MediaPipe (e.g. "swipe_right").
            session_id:  Existing session UUID.
            confidence:  Classifier confidence [0–1].
            landmarks:   List of MediaPipe hand landmark dicts [{x, y, z}].
            context:     Extra UI context (active_widget, coordinates …).
        """
        meta = {
            "gesture_id": gesture_id,
            "confidence": confidence,
            "landmarks":  landmarks or [],
            **(context or {}),
        }
        return await self.process(
            user_input=f"[gesture:{gesture_id}]",
            session_id=session_id,
            metadata=meta,
        )

    # ── Routing methods ────────────────────────────────────────────────────────

    async def _route_chat(self, task: AgentTask) -> AgentResult:
        """Direct chat — single ExecutorAgent call (no planner overhead)."""
        result = await self._executor.execute(task)
        result.metadata["agent_path"] = ["executor"]
        return result

    async def _route_plan_execute(self, task: AgentTask) -> AgentResult:
        """
        Two-stage pipeline: Planner → Executor.

        Used for CODE, ANALYSE, PLAN intents where decomposition adds value.
        """
        # Stage 1: Plan
        plan_result = await self._planner.execute(task)

        # Stage 2: Execute (inject plan into task metadata)
        exec_task = AgentTask(
            session_id=task.session_id,
            user_input=task.user_input,
            intent=task.intent,
            history=task.history,
            context=task.context,
            metadata={
                **task.metadata,
                "plan":  plan_result.metadata.get("plan", {}),
                "steps": plan_result.steps,
                "tools": plan_result.metadata.get("tools", []),
            },
        )
        exec_result = await self._executor.execute(exec_task)
        exec_result.steps       = plan_result.steps
        exec_result.tool_calls  += plan_result.tool_calls
        exec_result.metadata["agent_path"] = ["planner", "executor"]
        exec_result.metadata["plan"]       = plan_result.metadata.get("plan")
        return exec_result

    async def _route_knowledge(self, task: AgentTask) -> AgentResult:
        """Knowledge retrieval — KnowledgeAgent (search + synthesis)."""
        result = await self._knowledge.execute(task)
        result.metadata["agent_path"] = ["knowledge"]
        return result

    async def _route_gesture(self, task: AgentTask) -> AgentResult:
        """Gesture pipeline — GestureAgent with optional follow-up."""
        result = await self._gesture.execute(task)
        result.metadata["agent_path"] = ["gesture"]

        # If the gesture implies a follow-up action, chain to executor
        if result.success and result.metadata.get("follow_up_intent"):
            follow_task = AgentTask(
                session_id=task.session_id,
                user_input=result.metadata["follow_up_intent"],
                intent=result.metadata["follow_up_intent"],
                history=task.history,
                context=task.context,
            )
            follow_result = await self._executor.execute(follow_task)
            result.text = follow_result.text
            result.metadata["agent_path"].append("executor")
        return result

    async def _route_system(self, task: AgentTask) -> AgentResult:
        """System status — executor with system_status tool pre-selected."""
        sys_task = AgentTask(
            session_id=task.session_id,
            user_input=task.user_input,
            intent="system",
            history=task.history,
            context=task.context,
            metadata={
                **task.metadata,
                "steps": ["Call the system_status tool and report results"],
                "tools": ["system_status"],
            },
        )
        result = await self._executor.execute(sys_task)
        result.metadata["agent_path"] = ["executor(system)"]
        return result

    async def _route_memory(self, task: AgentTask) -> AgentResult:
        """Memory operations — parse key/value from user input and persist."""
        # Try to extract what the user wants to remember
        prompt = (
            f"The user said: '{task.user_input}'\n\n"
            "Extract what they want to remember. "
            "Respond ONLY with JSON: {\"key\": \"<label>\", \"value\": \"<thing to remember>\"}"
        )
        result = await llm_service.generate(
            prompt=prompt,
            max_new_tokens=128,
            temperature=0.1,
        )

        import json, re as _re
        m = _re.search(r"\{.*\}", result.text, _re.DOTALL)
        if m:
            try:
                parsed = json.loads(m.group(0))
                key    = parsed.get("key", "note")
                value  = parsed.get("value", task.user_input)
                await memory.set_context(task.session_id, key, value)
                text = f"Got it — I've saved that as '{key}': {value}"
            except Exception:
                text = "I've noted that down."
        else:
            text = "I've noted that down."

        return AgentResult(
            text=text,
            agent=AgentType.BRAIN,
            metadata={"agent_path": ["brain(memory)"]},
        )

    # ── Diagnostics ────────────────────────────────────────────────────────────

    async def status(self) -> dict:
        """Return diagnostic information about all subsystems."""
        mem_stats  = await memory.stats()
        model_info = llm_service.info()
        return {
            "brain":   "operational",
            "memory":  mem_stats,
            "model":   model_info,
            "agents":  [
                self._planner.agent_type.value,
                self._executor.agent_type.value,
                self._knowledge.agent_type.value,
                self._gesture.agent_type.value,
            ],
            "intents_supported": [i.value for i in Intent],
        }


# Module-level singleton
brain = JarvisBrain()
