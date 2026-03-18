"""
app/core/jarvis_core.py
═══════════════════════════════════════════════════════════════
JarvisCore — the autonomous intelligence engine.

The primary loop (per request):

    OBSERVE  → receive input, restore session + memory context
    THINK    → DecisionEngine classifies intent + selects action
    DECIDE   → risk evaluation, proactive suggestions
    ACT      → AgentRouter dispatches to correct agent pipeline
    LEARN    → persist result to memory, update state

This class is the single entry point from all API routes.
It wires DecisionEngine → AgentRouter → all agents together
and manages the full lifecycle of each interaction.
═══════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from app.core.decision_engine import decision_engine, DecisionResult
from app.core.state_manager    import state_manager, MODE_THINKING, MODE_EXECUTING, MODE_IDLE, MODE_ERROR
from app.core.agent_router     import AgentRouter, AgentResult
from app.memory.database       import db
from app.utils.logger          import get_logger

log = get_logger("jarvis.core")

# ── Response type ──────────────────────────────────────────────────────

@dataclass
class CoreResponse:
    """
    Unified response returned to every API endpoint.

    This is the "Frontend Integration Contract" the system spec requires.
    """
    session_id:  str
    intent:      str
    decision:    str
    response:    str
    action:      dict              = field(default_factory=dict)
    state:       dict              = field(default_factory=dict)
    confidence:  float             = 0.85
    risk:        str               = "low"
    proactive:   str | None        = None
    agent_path:  list[str]         = field(default_factory=list)
    tool_calls:  list[dict]        = field(default_factory=list)
    latency_ms:  float             = 0.0
    success:     bool              = True
    requires_confirmation: bool    = False
    error:       str | None        = None

    def to_dict(self) -> dict:
        return {
            "session_id":  self.session_id,
            "intent":      self.intent,
            "decision":    self.decision,
            "response":    self.response,
            "action":      self.action,
            "state": {
                "mode":       self.state.get("mode", MODE_IDLE),
                "intent":     self.intent,
                "confidence": round(self.confidence, 3),
            },
            "confidence":  round(self.confidence, 3),
            "risk":        self.risk,
            "proactive":   self.proactive,
            "agent_path":  self.agent_path,
            "tool_calls":  self.tool_calls,
            "latency_ms":  self.latency_ms,
            "success":     self.success,
            "requires_confirmation": self.requires_confirmation,
            "error":       self.error,
        }


# ── JarvisCore ─────────────────────────────────────────────────────────

class JarvisCore:
    """
    Autonomous intelligence engine — single entry point for all input.

    Constructed via JarvisCore.build() which wires all agents together.
    Can also be constructed with None agents for testing.

    Args:
        router:      Wired AgentRouter instance.
        llm_service: LLMService for the ReasoningAgent.
    """

    def __init__(self, router: AgentRouter) -> None:
        self._router    = router
        self._requests  = 0
        self._errors    = 0
        self._start_ts  = time.time()

    # ── Factory ────────────────────────────────────────────────────────

    @classmethod
    def build(cls) -> "JarvisCore":
        """
        Wire all agents and services together.
        Call once in the FastAPI lifespan.
        """
        from app.agents.planner_agent   import PlannerAgent
        from app.agents.reasoning_agent import ReasoningAgent
        from app.agents.executor_agent  import ExecutorAgent
        from app.agents.memory_agent    import MemoryAgent
        from app.agents.observer_agent  import ObserverAgent

        planner   = PlannerAgent()
        reasoner  = ReasoningAgent()
        executor  = ExecutorAgent()
        memory_ag = MemoryAgent(db=db)
        observer  = ObserverAgent(db=db)

        router = AgentRouter(
            planner=planner,
            reasoner=reasoner,
            executor=executor,
            memory_ag=memory_ag,
            observer=observer,
        )

        log.info("JarvisCore built — all agents wired")
        return cls(router=router)

    # ══════════════════════════════════════════════════════════════════
    #  OBSERVE → THINK → DECIDE → ACT → LEARN
    # ══════════════════════════════════════════════════════════════════

    async def process(
        self,
        text:        str,
        session_id:  str | None = None,
        source:      str        = "text",   # text | voice | gesture
        metadata:    dict | None = None,
    ) -> CoreResponse:
        """
        Full autonomous pipeline for one input.

        Args:
            text:       Raw user input (text, transcript, or gesture ID).
            session_id: Existing session UUID. None creates a new session.
            source:     Input modality.
            metadata:   Extra data (gesture landmarks, voice confidence, etc.)

        Returns:
            CoreResponse — the structured JSON response.
        """
        t0 = time.perf_counter()
        self._requests += 1
        sid = session_id or str(uuid.uuid4())
        meta = metadata or {}

        log.info("OBSERVE [%s] source=%s: %.60r", sid[:8], source, text)

        # ── OBSERVE: restore context ───────────────────────────────────
        session  = await state_manager.get_or_create(sid)
        history  = await db.get_history(sid, limit=20)
        prefs    = await db.all_preferences()

        # ── THINK: classify intent + evaluate risk ─────────────────────
        await state_manager.set_mode(sid, MODE_THINKING)

        try:
            decision: DecisionResult = decision_engine.decide(
                text=text,
                history=history,
                preferences=prefs,
                session_state=session.to_dict(),
            )
        except Exception as exc:
            log.exception("DecisionEngine failed: %s", exc)
            await state_manager.set_mode(sid, MODE_ERROR)
            return self._error_response(sid, str(exc))

        await state_manager.set_intent(sid, decision.intent)

        log.info("THINK  [%s] intent=%s risk=%s conf=%.0f%%",
                 sid[:8], decision.intent, decision.risk, decision.confidence * 100)

        # ── DECIDE: broadcast state to WS clients ──────────────────────
        await state_manager.push_event(sid, {
            "type":    "thinking",
            "intent":  decision.intent,
            "risk":    decision.risk,
            "ts":      time.time(),
        })

        # ── ACT: dispatch to agent pipeline ───────────────────────────
        await state_manager.set_mode(sid, MODE_EXECUTING)

        try:
            result: AgentResult = await self._router.route(
                decision=decision,
                text=text,
                session_id=sid,
                history=history,
            )
        except Exception as exc:
            log.exception("AgentRouter failed: %s", exc)
            self._errors += 1
            await state_manager.set_mode(sid, MODE_ERROR)
            return self._error_response(sid, str(exc))

        # ── LEARN: update state + push final event ─────────────────────
        await state_manager.set_mode(sid, MODE_IDLE)

        total_ms = round((time.perf_counter() - t0) * 1000, 1)

        # Broadcast final response to WS layer
        await state_manager.push_event(sid, {
            "type":      "response_ready",
            "intent":    result.intent,
            "response":  result.text,
            "action":    result.action,
            "latency_ms": total_ms,
            "ts":        time.time(),
        })

        log.info("LEARN  [%s] %.0f ms | path=%s | success=%s",
                 sid[:8], total_ms, result.agent_path, result.success)

        return CoreResponse(
            session_id=sid,
            intent=result.intent,
            decision=result.decision,
            response=result.text,
            action=result.action,
            state={"mode": MODE_IDLE},
            confidence=result.confidence,
            risk=result.risk,
            proactive=result.proactive,
            agent_path=result.agent_path,
            tool_calls=result.tool_calls,
            latency_ms=total_ms,
            success=result.success,
            requires_confirmation=result.requires_confirmation,
        )

    # ── Status / health ────────────────────────────────────────────────

    async def status(self) -> dict:
        state_summary = await state_manager.summary()
        db_stats      = await db.stats()
        uptime        = round(time.time() - self._start_ts, 1)
        return {
            "status":    "operational",
            "uptime_s":  uptime,
            "requests":  self._requests,
            "errors":    self._errors,
            "state":     state_summary,
            "memory":    db_stats,
            "agents":    self._router.status(),
        }

    async def health(self) -> dict:
        return {
            "status":   "ok",
            "uptime_s": round(time.time() - self._start_ts, 1),
        }

    # ── Helper ─────────────────────────────────────────────────────────

    def _error_response(self, session_id: str, error: str) -> CoreResponse:
        self._errors += 1
        return CoreResponse(
            session_id=session_id,
            intent="error",
            decision="System error",
            response=f"A system error occurred: {error}",
            state={"mode": MODE_ERROR},
            success=False,
            error=error,
        )


# Module singleton (populated by build() in lifespan)
jarvis: JarvisCore | None = None
