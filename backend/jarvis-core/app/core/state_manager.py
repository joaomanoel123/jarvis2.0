"""
app/core/state_manager.py
═══════════════════════════════════════════════════════════════
StateManager — JARVIS internal state machine.

System mode:  idle → listening → thinking → executing → idle
Session:      per-connection session data
Broadcast:    asyncio.Queue per session for WS push
═══════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from app.utils.logger import get_logger

log = get_logger("jarvis.state")

# ── Mode constants ─────────────────────────────────────────────────────
MODE_IDLE      = "idle"
MODE_LISTENING = "listening"
MODE_THINKING  = "thinking"
MODE_EXECUTING = "executing"
MODE_ERROR     = "error"

VALID_MODES = {MODE_IDLE, MODE_LISTENING, MODE_THINKING, MODE_EXECUTING, MODE_ERROR}


@dataclass
class SessionState:
    """Per-connection session data."""
    session_id:   str
    mode:         str             = MODE_IDLE
    intent:       str | None      = None
    last_command: str | None      = None
    context:      dict            = field(default_factory=dict)
    created_at:   float           = field(default_factory=time.time)
    last_active:  float           = field(default_factory=time.time)
    msg_count:    int             = 0
    error_count:  int             = 0
    queue:        asyncio.Queue   = field(default_factory=lambda: asyncio.Queue(maxsize=32))

    def to_dict(self) -> dict:
        return {
            "session_id":   self.session_id,
            "mode":         self.mode,
            "intent":       self.intent,
            "last_command": self.last_command,
            "msg_count":    self.msg_count,
            "error_count":  self.error_count,
            "uptime_s":     round(time.time() - self.created_at, 1),
        }


class StateManager:
    """
    Thread-safe state store for all active sessions.

    Key operations:
        get_or_create(session_id)  → SessionState
        set_mode(session_id, mode) → broadcast state change
        push_event(session_id, ev) → add to session WS queue
    """

    def __init__(self) -> None:
        self._sessions: dict[str, SessionState] = {}
        self._global_mode: str = MODE_IDLE
        self._lock = asyncio.Lock()

    # ── Session lifecycle ──────────────────────────────────────────────

    async def get_or_create(self, session_id: str | None = None) -> SessionState:
        sid = session_id or str(uuid.uuid4())
        async with self._lock:
            if sid not in self._sessions:
                self._sessions[sid] = SessionState(session_id=sid)
                log.debug("New session: %s", sid[:8])
            else:
                self._sessions[sid].last_active = time.time()
        return self._sessions[sid]

    async def destroy(self, session_id: str) -> None:
        async with self._lock:
            self._sessions.pop(session_id, None)

    async def prune_idle(self, ttl_s: float = 3600.0) -> int:
        """Remove sessions idle beyond ttl_s. Returns count evicted."""
        cutoff = time.time() - ttl_s
        async with self._lock:
            dead = [sid for sid, s in self._sessions.items() if s.last_active < cutoff]
            for sid in dead:
                del self._sessions[sid]
        if dead:
            log.info("Pruned %d idle sessions", len(dead))
        return len(dead)

    # ── Mode transitions ───────────────────────────────────────────────

    async def set_mode(self, session_id: str, mode: str) -> None:
        assert mode in VALID_MODES, f"Unknown mode: {mode}"
        session = await self.get_or_create(session_id)
        prev = session.mode
        session.mode = mode
        if mode == MODE_ERROR:
            session.error_count += 1
        log.debug("Session %s: %s → %s", session_id[:8], prev, mode)
        await self.push_event(session_id, {"type": "state_change", "mode": mode, "ts": time.time()})

    async def set_intent(self, session_id: str, intent: str) -> None:
        session = await self.get_or_create(session_id)
        session.intent = intent
        session.msg_count += 1

    async def set_context(self, session_id: str, key: str, value: Any) -> None:
        session = await self.get_or_create(session_id)
        session.context[key] = value

    async def get_context(self, session_id: str, key: str, default: Any = None) -> Any:
        session = await self.get_or_create(session_id)
        return session.context.get(key, default)

    # ── WebSocket event queue ──────────────────────────────────────────

    async def push_event(self, session_id: str, event: dict) -> None:
        """Push an event into the session's WebSocket queue (non-blocking)."""
        session = await self.get_or_create(session_id)
        try:
            session.queue.put_nowait(event)
        except asyncio.QueueFull:
            log.debug("Event queue full for session %s — dropping", session_id[:8])

    async def get_event(self, session_id: str, timeout: float = 30.0) -> dict | None:
        """
        Wait for the next event from the session queue.
        Returns None on timeout (send keepalive ping instead).
        """
        session = await self.get_or_create(session_id)
        try:
            return await asyncio.wait_for(session.queue.get(), timeout=timeout)
        except asyncio.TimeoutError:
            return None

    # ── Diagnostics ────────────────────────────────────────────────────

    async def summary(self) -> dict:
        async with self._lock:
            sessions = list(self._sessions.values())
        return {
            "active_sessions": len(sessions),
            "modes":           {m: sum(1 for s in sessions if s.mode == m) for m in VALID_MODES},
            "total_messages":  sum(s.msg_count for s in sessions),
            "total_errors":    sum(s.error_count for s in sessions),
        }

    def get_session_data(self, session_id: str) -> dict:
        s = self._sessions.get(session_id)
        return s.to_dict() if s else {}


# Module singleton
state_manager = StateManager()
