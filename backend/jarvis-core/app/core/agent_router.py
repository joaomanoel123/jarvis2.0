"""
app/core/agent_router.py
═══════════════════════════════════════════════════════════════
AgentRouter — selects and sequences the correct agent pipeline
for each classified intent + action decision.

Pipeline dispatch table:
  conversational  → [MemoryAgent(retrieve), ReasoningAgent, MemoryAgent(store)]
  query           → [MemoryAgent(retrieve), ReasoningAgent, MemoryAgent(store)]
  command         → [PlannerAgent, ExecutorAgent, MemoryAgent(store)]
  system_control  → [PlannerAgent, ExecutorAgent]
  gesture         → [ExecutorAgent]
═══════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from app.core.decision_engine import DecisionResult, CONVERSATIONAL, COMMAND, QUERY, SYSTEM_CONTROL, GESTURE
from app.utils.logger import get_logger

log = get_logger("jarvis.router")


@dataclass
class AgentResult:
    """Aggregated result from the agent pipeline."""
    text:         str
    intent:       str
    decision:     str
    action:       dict              = field(default_factory=dict)
    agent_path:   list[str]        = field(default_factory=list)
    tool_calls:   list[dict]       = field(default_factory=list)
    risk:         str              = "low"
    confidence:   float            = 0.85
    proactive:    str | None       = None
    requires_confirmation: bool    = False
    success:      bool             = True
    latency_ms:   float            = 0.0


class AgentRouter:
    """
    Routes a DecisionResult through the appropriate agent pipeline.
    All agents are injected at construction to allow testing with mocks.
    """

    def __init__(
        self,
        planner:   Any = None,
        reasoner:  Any = None,
        executor:  Any = None,
        memory_ag: Any = None,
        observer:  Any = None,
    ) -> None:
        self._planner  = planner
        self._reasoner = reasoner
        self._executor = executor
        self._memory   = memory_ag
        self._observer = observer

    async def route(
        self,
        decision:   DecisionResult,
        text:       str,
        session_id: str,
        history:    list[dict],
    ) -> AgentResult:
        """
        Execute the correct agent pipeline for the given decision.

        Args:
            decision:   Output of DecisionEngine.decide()
            text:       Raw user input
            session_id: Active session UUID
            history:    Recent conversation turns

        Returns:
            AgentResult with response and full pipeline metadata
        """
        t0 = time.perf_counter()

        intent = decision.intent

        if intent in (CONVERSATIONAL, QUERY):
            result = await self._pipeline_converse(decision, text, session_id, history)
        elif intent == COMMAND:
            result = await self._pipeline_command(decision, text, session_id)
        elif intent == SYSTEM_CONTROL:
            result = await self._pipeline_system(decision, text, session_id)
        elif intent == GESTURE:
            result = await self._pipeline_gesture(decision, session_id)
        else:
            result = await self._pipeline_converse(decision, text, session_id, history)

        result.latency_ms = round((time.perf_counter() - t0) * 1000, 1)

        # Observer: record metrics
        if self._observer:
            try:
                await self._observer.record(intent, result.latency_ms, result.success)
            except Exception:
                pass

        log.info("Routed %s → %s (%.0f ms)", intent, result.agent_path, result.latency_ms)
        return result

    # ── Pipeline: conversational / query ───────────────────────────────

    async def _pipeline_converse(
        self, decision: DecisionResult, text: str, session_id: str, history: list[dict]
    ) -> AgentResult:
        path = []

        # Memory: retrieve relevant context
        extra_context = ""
        if self._memory:
            try:
                ctx = await self._memory.retrieve(session_id, text)
                extra_context = ctx
                path.append("memory[retrieve]")
            except Exception as e:
                log.debug("Memory retrieve failed: %s", e)

        # Reasoning: generate LLM response
        response_text = decision.decision  # fallback
        if self._reasoner:
            try:
                response_text = await self._reasoner.respond(
                    text=text,
                    history=history,
                    context=extra_context,
                    intent=decision.intent,
                )
                path.append("reasoner")
            except Exception as e:
                log.warning("Reasoner failed: %s", e)
                response_text = "I understood your message. Please stand by."

        # Memory: store turn
        if self._memory:
            try:
                await self._memory.store(session_id, text, response_text, decision.intent)
                path.append("memory[store]")
            except Exception as e:
                log.debug("Memory store failed: %s", e)

        return AgentResult(
            text=response_text,
            intent=decision.intent,
            decision=decision.decision,
            action=decision.action_dict(),
            agent_path=path,
            risk=decision.risk,
            confidence=decision.confidence,
            proactive=decision.proactive,
        )

    # ── Pipeline: command ──────────────────────────────────────────────

    async def _pipeline_command(
        self, decision: DecisionResult, text: str, session_id: str
    ) -> AgentResult:
        path = []
        tool_calls = []

        # Planner: decompose if needed
        if self._planner and decision.action:
            try:
                await self._planner.plan(decision.action, text)
                path.append("planner")
            except Exception as e:
                log.debug("Planner failed: %s", e)

        # Executor: run the command
        exec_result = {"success": True, "result": decision.decision}
        if self._executor and decision.action:
            try:
                exec_result = await self._executor.execute(decision.action)
                path.append("executor")
                tool_calls.append({
                    "name":    decision.action.name,
                    "params":  decision.action.parameters,
                    "success": exec_result.get("success", True),
                })
            except Exception as e:
                log.warning("Executor failed: %s", e)
                exec_result = {"success": False, "error": str(e)}

        # Memory: log command
        if self._memory:
            try:
                await self._memory.log_command(
                    session_id=session_id,
                    name=decision.action.name if decision.action else "unknown",
                    parameters=decision.action.parameters if decision.action else {},
                    result=exec_result,
                    risk=decision.risk,
                )
                path.append("memory[log]")
            except Exception as e:
                log.debug("Memory log failed: %s", e)

        response_text = decision.decision
        if not exec_result.get("success"):
            response_text = f"Command failed: {exec_result.get('error', 'unknown error')}"
            if decision.action:
                # Retry once
                log.info("Retrying command: %s", decision.action.name)
                if self._executor:
                    try:
                        exec_result = await self._executor.execute(decision.action)
                        if exec_result.get("success"):
                            response_text = decision.decision + " (retried)"
                    except Exception:
                        pass

        return AgentResult(
            text=response_text,
            intent=decision.intent,
            decision=decision.decision,
            action=decision.action_dict(),
            agent_path=path,
            tool_calls=tool_calls,
            risk=decision.risk,
            confidence=decision.confidence,
            success=exec_result.get("success", True),
            requires_confirmation=decision.requires_confirmation,
        )

    # ── Pipeline: system control ───────────────────────────────────────

    async def _pipeline_system(
        self, decision: DecisionResult, text: str, session_id: str
    ) -> AgentResult:
        path = ["planner"]
        if decision.requires_confirmation:
            return AgentResult(
                text=f"This action requires confirmation: {decision.decision}. Please confirm.",
                intent=decision.intent,
                decision=decision.decision,
                action=decision.action_dict(),
                agent_path=path,
                risk=decision.risk,
                confidence=decision.confidence,
                requires_confirmation=True,
            )

        if self._executor and decision.action:
            try:
                await self._executor.execute(decision.action)
                path.append("executor")
            except Exception as e:
                log.warning("System executor failed: %s", e)

        return AgentResult(
            text=decision.decision,
            intent=decision.intent,
            decision=decision.decision,
            action=decision.action_dict(),
            agent_path=path,
            risk=decision.risk,
            confidence=decision.confidence,
        )

    # ── Pipeline: gesture ──────────────────────────────────────────────

    async def _pipeline_gesture(
        self, decision: DecisionResult, session_id: str
    ) -> AgentResult:
        path = []
        if self._executor and decision.action:
            try:
                await self._executor.execute(decision.action)
                path.append("executor")
            except Exception as e:
                log.debug("Gesture executor: %s", e)

        return AgentResult(
            text=decision.decision,
            intent=decision.intent,
            decision=decision.decision,
            action=decision.action_dict(),
            agent_path=path,
            risk=decision.risk,
            confidence=decision.confidence,
        )

    # ── Diagnostics ────────────────────────────────────────────────────

    def status(self) -> dict:
        return {
            "agents": {
                "planner":  self._planner  is not None,
                "reasoner": self._reasoner is not None,
                "executor": self._executor is not None,
                "memory":   self._memory   is not None,
                "observer": self._observer is not None,
            }
        }
