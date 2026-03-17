"""
backend/agent_manager.py
=========================
AgentManager — routes incoming tasks to the correct agent pipeline
and orchestrates multi-step multi-agent execution.

Routing table
─────────────
  chat     →  ExecutorAgent   (direct LLM, single-turn)
  code     →  PlannerAgent  → ExecutorAgent
  search   →  KnowledgeAgent
  analyse  →  PlannerAgent  → ExecutorAgent
  gesture  →  GestureAgent
  voice    →  ExecutorAgent   (already parsed command)
  memory   →  MemoryAgent    → ExecutorAgent
  system   →  ExecutorAgent  (system_status tool pre-selected)
  plan     →  PlannerAgent  → ExecutorAgent

Each pipeline is a list of (agent_instance, task_mutator) pairs.
task_mutator receives the previous AgentResult and updates the AgentTask
before passing it to the next agent.

The MemoryAgent is always called post-pipeline to store the turn.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger("jarvis.agent_manager")

# Intent classification patterns (same as JarvisBrain for direct access)
_INTENT_RULES = [
    (re.compile(r"\b(write|generate|create|build|code|script|function|class|refactor|debug|fix)\b", re.I), "code"),
    (re.compile(r"\b(search|find|look up|what is|who is|news|latest|research)\b", re.I), "search"),
    (re.compile(r"\b(analyse|analyze|examine|explore|dataset|csv|statistics)\b", re.I), "analyse"),
    (re.compile(r"\b(plan|break down|steps to|how do i|strategy)\b", re.I), "plan"),
    (re.compile(r"\b(remember|save|store|recall|note that)\b", re.I), "memory"),
    (re.compile(r"\b(system|status|cpu|ram|disk|uptime)\b", re.I), "system"),
]


@dataclass
class AgentManagerResult:
    """Final result returned to JarvisCore."""
    text:        str
    intent:      str
    agent_path:  list[str] = field(default_factory=list)
    tool_calls:  list[dict] = field(default_factory=list)
    steps:       list[str]  = field(default_factory=list)
    latency_ms:  float      = 0.0
    success:     bool       = True
    error:       str | None = None
    metadata:    dict       = field(default_factory=dict)


class AgentManager:
    """
    Manages agent lifecycle and routes requests to the correct pipeline.

    All agent instances are injected at construction — allows testing
    with mocks and prevents circular imports.

    Args:
        planner:   PlannerAgent instance.
        executor:  ExecutorAgent instance.
        knowledge: KnowledgeAgent instance.
        gesture:   GestureAgent instance.
        memory_ag: MemoryAgent instance.
    """

    def __init__(
        self,
        planner=None,
        executor=None,
        knowledge=None,
        gesture=None,
        memory_ag=None,
    ) -> None:
        self._planner   = planner
        self._executor  = executor
        self._knowledge = knowledge
        self._gesture   = gesture
        self._memory    = memory_ag

        # Metrics
        self._routed: dict[str, int] = {}

    # ── Public API ─────────────────────────────────────────────────────────────

    async def route(
        self,
        user_input:  str,
        session_id:  str,
        intent:      str | None = None,
        metadata:    dict | None = None,
        history:     list[dict] | None = None,
        context:     dict | None = None,
    ) -> AgentManagerResult:
        """
        Route a user input through the appropriate agent pipeline.

        Args:
            user_input:  Cleaned user text or command.
            session_id:  Active memory session UUID.
            intent:      Pre-classified intent (skips classification if set).
            metadata:    Extra context (gesture data, voice source, etc.).
            history:     Conversation history for the LLM.
            context:     Session context bag.

        Returns:
            AgentManagerResult with response text and pipeline metadata.
        """
        t0 = time.perf_counter()
        meta = metadata or {}

        # Determine intent
        if not intent:
            intent = self._classify(user_input)

        # Gesture always explicit
        if meta.get("gesture_id"):
            intent = "gesture"
        # Voice commands are pre-parsed — route directly
        if meta.get("source") == "voice" and meta.get("command"):
            intent = "voice"

        self._routed[intent] = self._routed.get(intent, 0) + 1
        log.info("AgentManager: intent=%s  input=%.50r", intent, user_input)

        # Build task
        from agents.base_agent import AgentTask
        task = AgentTask(
            session_id=session_id,
            user_input=user_input,
            intent=intent,
            history=history or [],
            context=context or {},
            metadata=meta,
        )

        # Execute pipeline
        result_obj = await self._execute_pipeline(task, intent)
        latency = round((time.perf_counter() - t0) * 1000, 1)
        result_obj.latency_ms = latency

        log.info("AgentManager: %.0f ms | path=%s | success=%s",
                 latency, result_obj.agent_path, result_obj.success)
        return result_obj

    # ── Intent classification ──────────────────────────────────────────────────

    def _classify(self, text: str) -> str:
        for pattern, intent in _INTENT_RULES:
            if pattern.search(text):
                return intent
        return "chat"

    # ── Pipeline execution ─────────────────────────────────────────────────────

    async def _execute_pipeline(
        self, task, intent: str
    ) -> AgentManagerResult:
        """Dispatch to the correct pipeline based on intent."""
        try:
            if intent in ("code", "analyse", "plan"):
                return await self._pipeline_plan_execute(task)
            elif intent == "search":
                return await self._pipeline_knowledge(task)
            elif intent == "gesture":
                return await self._pipeline_gesture(task)
            elif intent == "system":
                return await self._pipeline_system(task)
            elif intent == "memory":
                return await self._pipeline_memory(task)
            else:  # chat, voice, fallback
                return await self._pipeline_direct(task)
        except Exception as exc:
            log.exception("Pipeline '%s' crashed: %s", intent, exc)
            return AgentManagerResult(
                text=f"I encountered an error: {exc}",
                intent=intent,
                success=False,
                error=str(exc),
            )

    async def _pipeline_direct(self, task) -> AgentManagerResult:
        """Direct executor — single LLM call, no planning."""
        if not self._executor:
            return AgentManagerResult(
                text="ExecutorAgent not available", intent=task.intent,
                success=False, error="no_executor",
            )
        result = await self._executor.execute(task)
        return AgentManagerResult(
            text=result.text, intent=task.intent,
            agent_path=[result.agent.value],
            tool_calls=result.tool_calls,
            success=result.success, error=result.error,
        )

    async def _pipeline_plan_execute(self, task) -> AgentManagerResult:
        """Planner → Executor two-stage pipeline."""
        if not self._planner or not self._executor:
            return await self._pipeline_direct(task)

        plan_result = await self._planner.execute(task)

        from agents.base_agent import AgentTask
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

        return AgentManagerResult(
            text=exec_result.text,
            intent=task.intent,
            agent_path=["planner", exec_result.agent.value],
            tool_calls=exec_result.tool_calls + plan_result.tool_calls,
            steps=plan_result.steps,
            success=exec_result.success,
            error=exec_result.error,
            metadata={"plan": plan_result.metadata.get("plan")},
        )

    async def _pipeline_knowledge(self, task) -> AgentManagerResult:
        if not self._knowledge:
            return await self._pipeline_direct(task)
        result = await self._knowledge.execute(task)
        return AgentManagerResult(
            text=result.text, intent=task.intent,
            agent_path=[result.agent.value],
            tool_calls=result.tool_calls,
            success=result.success, error=result.error,
        )

    async def _pipeline_gesture(self, task) -> AgentManagerResult:
        if not self._gesture:
            return await self._pipeline_direct(task)
        result = await self._gesture.execute(task)
        return AgentManagerResult(
            text=result.text, intent="gesture",
            agent_path=[result.agent.value],
            success=result.success, error=result.error,
            metadata=result.metadata,
        )

    async def _pipeline_system(self, task) -> AgentManagerResult:
        """System status — executor with tool pre-selected."""
        if not self._executor:
            return AgentManagerResult(
                text="System query unavailable", intent="system",
                success=False, error="no_executor",
            )
        from agents.base_agent import AgentTask
        sys_task = AgentTask(
            session_id=task.session_id,
            user_input=task.user_input,
            intent="system",
            history=task.history,
            context=task.context,
            metadata={
                **task.metadata,
                "steps": ["Call system_status tool and report results"],
                "tools": ["system_status"],
            },
        )
        result = await self._executor.execute(sys_task)
        return AgentManagerResult(
            text=result.text, intent="system",
            agent_path=["executor(system)"],
            tool_calls=result.tool_calls,
            success=result.success,
        )

    async def _pipeline_memory(self, task) -> AgentManagerResult:
        """Parse memory operation and persist."""
        import json, re as _re
        if self._memory:
            prompt_result = None
            if self._executor:
                from agents.base_agent import AgentTask
                mem_task = AgentTask(
                    session_id=task.session_id,
                    user_input=task.user_input,
                    intent="memory",
                    history=task.history,
                    context=task.context,
                    metadata=task.metadata,
                )
                prompt_result = await self._executor.execute(mem_task)
            # Persist to memory
            m = _re.search(r"\{.*\}", getattr(prompt_result, "text", ""), _re.DOTALL)
            if m:
                try:
                    parsed = json.loads(m.group(0))
                    key, val = parsed.get("key", "note"), parsed.get("value", task.user_input)
                    await self._memory.remember(task.session_id, key, val)
                    return AgentManagerResult(
                        text=f"Got it — saved '{key}': {val}",
                        intent="memory", agent_path=["memory"],
                    )
                except Exception:
                    pass
        return AgentManagerResult(
            text="I've noted that down.", intent="memory", agent_path=["memory"],
        )

    # ── Diagnostics ───────────────────────────────────────────────────────────

    def stats(self) -> dict:
        return {
            "routes": dict(self._routed),
            "agents": {
                "planner":   self._planner is not None,
                "executor":  self._executor is not None,
                "knowledge": self._knowledge is not None,
                "gesture":   self._gesture is not None,
                "memory":    self._memory is not None,
            },
        }
