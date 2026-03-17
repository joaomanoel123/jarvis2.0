"""
backend/jarvis_core.py
=======================
JarvisCore — the central brain of the JARVIS 2.0 system.

                    ┌──────────────────────────────────────────┐
                    │              JarvisCore                   │
                    │                                          │
                    │  receive(text | voice | gesture)         │
                    │       ↓                                  │
                    │  ConversationMemory.retrieve_context()   │
                    │       ↓                                  │
                    │  AgentManager.route()                    │
                    │       ↓ (multi-agent pipeline)           │
                    │  MemoryAgent.store_turn()                │
                    │       ↓                                  │
                    │  CoreResponse → API → Frontend           │
                    └──────────────────────────────────────────┘

JarvisCore is the single entry point from the FastAPI layer.
It handles:
  • Session management (creates / restores sessions from memory)
  • Context assembly (conversation history + vector hits)
  • Agent pipeline delegation (via AgentManager)
  • Post-turn memory persistence (via MemoryAgent)
  • Voice command processing (calls CommandParser + CommandExecutor)
  • Gesture command processing (calls GestureAgent via AgentManager)
  • System-wide metrics and health reporting

Singleton pattern: import `core` from this module.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger("jarvis.core")

# ── Result type ────────────────────────────────────────────────────────────────

@dataclass
class CoreResponse:
    """
    Unified response from JarvisCore for all input types.

    Fields
    ──────
    session_id:  Active session UUID (create and pass back to frontend).
    text:        Final assistant response text.
    intent:      Classified intent (chat, code, search, gesture, voice, …).
    agent_path:  Ordered agent names that handled this request.
    tool_calls:  Tool invocations that occurred.
    steps:       Plan steps (populated for code/analyse intents).
    latency_ms:  Total wall-clock processing time.
    success:     False if an unrecoverable error occurred.
    error:       Error message when success=False.
    metadata:    Extra data (model_id, voice command, gesture intent, …).
    """
    session_id: str
    text:       str
    intent:     str              = "chat"
    agent_path: list[str]        = field(default_factory=list)
    tool_calls: list[dict]       = field(default_factory=list)
    steps:      list[str]        = field(default_factory=list)
    latency_ms: float            = 0.0
    success:    bool             = True
    error:      str | None       = None
    metadata:   dict[str, Any]   = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "text":       self.text,
            "intent":     self.intent,
            "agent_path": self.agent_path,
            "tool_calls": self.tool_calls,
            "steps":      self.steps,
            "latency_ms": self.latency_ms,
            "success":    self.success,
            "error":      self.error,
            "metadata":   self.metadata,
        }


# ── JarvisCore ─────────────────────────────────────────────────────────────────

class JarvisCore:
    """
    Central brain of JARVIS 2.0.

    All FastAPI routes funnel through this class.
    Components are injected to allow testing without real models.

    Args:
        agent_manager: AgentManager instance.
        memory_agent:  MemoryAgent instance.
        conv_memory:   ConversationMemory instance.
        llm_service:   LLMService instance (for metadata only here).
    """

    def __init__(
        self,
        agent_manager  = None,
        memory_agent   = None,
        conv_memory    = None,
        llm_service    = None,
    ) -> None:
        self._agents   = agent_manager
        self._memory   = memory_agent
        self._conv     = conv_memory
        self._llm      = llm_service

        # Metrics
        self._requests = 0
        self._errors   = 0
        self._start_ts = time.time()

    # ── Initialisation ─────────────────────────────────────────────────────────

    @classmethod
    def build(cls) -> "JarvisCore":
        """
        Factory method — wires all components together.

        Import and call this in the FastAPI lifespan to get the production instance.
        """
        from memory.conversation_memory import ConversationMemory
        from memory.vector_memory       import VectorMemory
        from agents.memory_agent        import MemoryAgent
        from agent_manager              import AgentManager

        # Memory tier
        conv_mem = ConversationMemory()
        vec_mem  = VectorMemory()

        # LLM service (shared across agents)
        llm_svc = None
        try:
            from services.llm_service import llm_service as _llm
            llm_svc = _llm
        except Exception as exc:
            log.warning("LLMService not available: %s", exc)

        # Agents
        mem_agent = MemoryAgent(
            conv_memory=conv_mem,
            vec_memory=vec_mem,
            llm_service=llm_svc,
        )

        try:
            from agents.planner_agent   import PlannerAgent
            from agents.executor_agent  import ExecutorAgent
            from agents.knowledge_agent import KnowledgeAgent
            from agents.gesture_agent   import GestureAgent

            agent_mgr = AgentManager(
                planner=PlannerAgent(),
                executor=ExecutorAgent(),
                knowledge=KnowledgeAgent(),
                gesture=GestureAgent(),
                memory_ag=mem_agent,
            )
        except Exception as exc:
            log.error("Agent init failed: %s", exc)
            agent_mgr = AgentManager(memory_ag=mem_agent)

        instance = cls(
            agent_manager=agent_mgr,
            memory_agent=mem_agent,
            conv_memory=conv_mem,
            llm_service=llm_svc,
        )
        log.info("JarvisCore built  agents=%s", agent_mgr.stats()["agents"])
        return instance

    # ── Chat ───────────────────────────────────────────────────────────────────

    async def chat(
        self,
        text:       str,
        session_id: str | None = None,
        metadata:   dict | None = None,
    ) -> CoreResponse:
        """
        Process a text message end-to-end.

        Args:
            text:       User message.
            session_id: Existing session UUID, or None to create a new one.
            metadata:   Optional extra context.

        Returns:
            CoreResponse with assistant text and full pipeline metadata.
        """
        t0 = time.perf_counter()
        self._requests += 1
        meta = metadata or {}

        # 1 — Resolve session
        sid = await self._ensure_session(session_id)

        # 2 — Retrieve context (history + vector hits)
        ctx = await self._get_context(sid, text)

        # 3 — Route through agent pipeline
        result = await self._route(
            text, sid,
            intent=None,
            metadata=meta,
            history=ctx["messages"],
            context=ctx["context_bag"],
        )

        # 4 — Store turn in memory
        await self._store_turn(sid, text, result.text, result.intent, result)

        # 5 — Attach model metadata
        result.metadata["model_id"]    = self._model_id()
        result.metadata["session_id"]  = sid
        result.latency_ms = round((time.perf_counter() - t0) * 1000, 1)

        return result

    # ── Voice command ──────────────────────────────────────────────────────────

    async def voice_command(
        self,
        text:       str,
        session_id: str | None = None,
        confidence: float      = 1.0,
        source:     str        = "voice",
    ) -> CoreResponse:
        """
        Process a voice-transcribed command.

        The voice-system pre-parses the text into a command structure.
        JarvisCore routes it through the agent pipeline as a 'voice' intent.

        Args:
            text:       Transcribed and wake-word-stripped command text.
            session_id: Existing session UUID.
            confidence: STT confidence [0–1].
            source:     "voice" (default).
        """
        meta = {
            "source":     source,
            "confidence": confidence,
        }

        # Try to parse as a structured voice command
        parsed_cmd = await self._try_parse_voice(text)
        if parsed_cmd:
            meta["command"]        = parsed_cmd.to_dict()
            meta["parsed_intent"]  = parsed_cmd.intent
            meta["parsed_action"]  = parsed_cmd.action

            # Execute immediately for non-AI commands (open URL, launch app, etc.)
            if parsed_cmd.intent in ("open_url", "web_search", "launch_app",
                                      "media_control", "volume_control", "system_control"):
                exec_result = await self._execute_parsed_command(parsed_cmd)
                sid = await self._ensure_session(session_id)
                await self._conv_add(sid, "user", text, source="voice")
                await self._conv_add(sid, "assistant", exec_result.message)
                return CoreResponse(
                    session_id=sid,
                    text=exec_result.message,
                    intent=parsed_cmd.intent,
                    agent_path=["command_executor"],
                    success=exec_result.success,
                    metadata=meta,
                )

        # Route through JARVIS AI agent pipeline
        return await self.chat(text=text, session_id=session_id, metadata=meta)

    # ── Gesture command ────────────────────────────────────────────────────────

    async def gesture_command(
        self,
        gesture_id: str,
        session_id: str | None   = None,
        confidence: float        = 1.0,
        landmarks:  list[dict]   | None = None,
        context:    dict | None  = None,
    ) -> CoreResponse:
        """
        Process a MediaPipe gesture event.

        Args:
            gesture_id: Gesture name (e.g. "swipe_right", "open_palm").
            session_id: Existing session UUID.
            confidence: Classifier confidence [0–1].
            landmarks:  List of MediaPipe hand landmark dicts [{x, y, z}].
            context:    Optional UI context (active_widget, coordinates, …).
        """
        meta = {
            "gesture_id": gesture_id,
            "confidence": confidence,
            "landmarks":  landmarks or [],
            "source":     "gesture",
            **(context or {}),
        }
        return await self.chat(
            text=f"[gesture:{gesture_id}]",
            session_id=session_id,
            metadata=meta,
        )

    # ── Internal helpers ───────────────────────────────────────────────────────

    async def _ensure_session(self, session_id: str | None) -> str:
        if self._conv:
            return await self._conv.get_or_create(session_id)
        import uuid
        return session_id or str(uuid.uuid4())

    async def _get_context(self, session_id: str, query: str) -> dict:
        if self._memory:
            return await self._memory.retrieve_context(
                session_id, query, history_n=20, vec_top_k=3
            )
        if self._conv:
            msgs = await self._conv.get_messages(session_id, last_n=20)
            ctx  = await self._conv.get_full_context(session_id)
            return {"messages": msgs, "relevant": [], "context_bag": ctx}
        return {"messages": [], "relevant": [], "context_bag": {}}

    async def _route(self, text, sid, intent, metadata, history, context) -> CoreResponse:
        if not self._agents:
            return CoreResponse(
                session_id=sid,
                text="JARVIS agents not initialised — check startup logs.",
                intent="error", success=False, error="agents_not_init",
            )
        try:
            agent_result = await self._agents.route(
                user_input=text,
                session_id=sid,
                intent=intent,
                metadata=metadata,
                history=history,
                context=context,
            )
            return CoreResponse(
                session_id=sid,
                text=agent_result.text,
                intent=agent_result.intent,
                agent_path=agent_result.agent_path,
                tool_calls=[
                    tc if isinstance(tc, dict) else tc
                    for tc in agent_result.tool_calls
                ],
                steps=agent_result.steps,
                success=agent_result.success,
                error=agent_result.error,
                metadata=agent_result.metadata,
            )
        except Exception as exc:
            self._errors += 1
            log.exception("Route failed: %s", exc)
            return CoreResponse(
                session_id=sid,
                text=f"I encountered an unexpected error: {exc}",
                intent=intent or "error",
                success=False, error=str(exc),
            )

    async def _store_turn(self, sid, user_text, bot_text, intent, result) -> None:
        if self._memory:
            await self._memory.store_turn(
                session_id=sid,
                user_text=user_text,
                bot_text=bot_text,
                intent=intent,
                agent=result.agent_path[-1] if result.agent_path else "core",
                tool_calls=[tc if isinstance(tc, dict) else {} for tc in result.tool_calls],
            )
        elif self._conv:
            await self._conv.add_message(sid, "user", user_text)
            await self._conv.add_message(sid, "assistant", bot_text,
                                          agent=result.agent_path[-1] if result.agent_path else None)

    async def _conv_add(self, sid, role, content, **meta) -> None:
        if self._conv:
            await self._conv.add_message(sid, role, content, **meta)

    async def _try_parse_voice(self, text: str):
        """Attempt to parse voice command text into a structured VoiceCommand."""
        try:
            import sys, os
            # Add voice-system to path if running locally
            vs_path = os.path.join(os.path.dirname(__file__), "..", "voice-system")
            if vs_path not in sys.path:
                sys.path.insert(0, vs_path)
            from command_parser import CommandParser
            parser = CommandParser(use_llm_fallback=False)
            cmd = parser.parse(text)
            # Only return for non-AI intents — AI intents go through agents
            if cmd and cmd.intent not in ("jarvis_command",):
                return cmd
        except Exception as exc:
            log.debug("Voice parse skipped: %s", exc)
        return None

    async def _execute_parsed_command(self, cmd):
        """Execute a pre-parsed voice command directly."""
        try:
            import sys, os
            vs_path = os.path.join(os.path.dirname(__file__), "..", "voice-system")
            if vs_path not in sys.path:
                sys.path.insert(0, vs_path)
            from command_executor import CommandExecutor
            executor = CommandExecutor(dry_run=False)
            return executor.execute(cmd)
        except Exception as exc:
            log.warning("Direct command execution failed: %s", exc)
            from types import SimpleNamespace
            return SimpleNamespace(success=False, message=str(exc))

    def _model_id(self) -> str | None:
        try:
            return self._llm.info().get("model_id")
        except Exception:
            return None

    # ── Diagnostics ───────────────────────────────────────────────────────────

    async def status(self) -> dict:
        """Return full system status for GET /status."""
        mem_stats   = await self._memory.stats()  if self._memory else {}
        agent_stats = self._agents.stats()         if self._agents else {}
        model_info  = self._llm.info()             if self._llm    else {}
        uptime      = round(time.time() - self._start_ts, 1)

        return {
            "status":    "operational" if model_info.get("loaded") else "degraded",
            "uptime_s":  uptime,
            "requests":  self._requests,
            "errors":    self._errors,
            "memory":    mem_stats,
            "agents":    agent_stats,
            "model":     model_info,
        }

    async def health(self) -> dict:
        """Lightweight liveness check for GET /health."""
        return {
            "status":       "ok",
            "model_loaded": bool(self._llm and self._llm.info().get("loaded")),
            "uptime_s":     round(time.time() - self._start_ts, 1),
        }
