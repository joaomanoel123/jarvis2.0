"""
memory/memory_manager.py
========================
Three-tier async memory for JARVIS.

Tier 1 – Conversation history
    Ordered {role, content} messages, trimmed to MAX_HISTORY_MESSAGES.

Tier 2 – Short-term memory
    Arbitrary key/value store with per-entry TTL.
    Evicted automatically during session access.

Tier 3 – Context bag
    Persistent per-session metadata (user prefs, last tool output, …).
    Cleared only when the session is destroyed.

Production upgrade
    Replace the in-process dict with redis.asyncio.
    The public async API is identical, so the swap is a one-file change.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from typing import Any

from config.settings import get_settings

log = logging.getLogger(__name__)
cfg = get_settings()


# ── Internal session model ─────────────────────────────────────────────────────

class _Session:
    __slots__ = (
        "session_id", "messages", "context",
        "short_term", "created_at", "last_active",
    )

    def __init__(self, session_id: str) -> None:
        now = time.monotonic()
        self.session_id:  str                       = session_id
        self.messages:    list[dict]                = []
        self.context:     dict[str, Any]            = {}
        self.short_term:  dict[str, tuple[Any, float]] = {}  # key → (value, expires_at)
        self.created_at   = now
        self.last_active  = now

    def touch(self) -> None:
        self.last_active = time.monotonic()

    def is_expired(self) -> bool:
        return (time.monotonic() - self.last_active) > cfg.SESSION_TTL_SECONDS


# ── Memory manager ─────────────────────────────────────────────────────────────

class MemoryManager:
    """
    Central async memory store.

    All public methods are coroutines and acquire a per-session asyncio.Lock
    to prevent concurrent modification of the same session's state.
    """

    def __init__(self) -> None:
        self._sessions: dict[str, _Session]      = {}
        self._locks:    dict[str, asyncio.Lock]  = {}
        self._meta_lock = asyncio.Lock()          # guards _sessions/_locks dicts

    # ── Lock helpers ───────────────────────────────────────────────────────────

    async def _session_lock(self, sid: str) -> asyncio.Lock:
        async with self._meta_lock:
            if sid not in self._locks:
                self._locks[sid] = asyncio.Lock()
            return self._locks[sid]

    # ── Session lifecycle ──────────────────────────────────────────────────────

    async def get_or_create(self, session_id: str | None = None) -> str:
        """
        Return an active session ID, creating a new session when needed.
        A new UUID is generated when session_id is None.
        """
        sid = session_id or str(uuid.uuid4())
        lock = await self._session_lock(sid)
        async with lock:
            session = self._sessions.get(sid)
            if session is None or session.is_expired():
                self._sessions[sid] = _Session(sid)
                log.info("Memory: new session %s", sid[:8])
            else:
                session.touch()
        return sid

    async def destroy(self, session_id: str) -> None:
        """Permanently remove a session and release its lock."""
        async with self._meta_lock:
            self._sessions.pop(session_id, None)
            self._locks.pop(session_id, None)
        log.info("Memory: destroyed session %s", session_id[:8])

    # ── Tier 1: Conversation history ───────────────────────────────────────────

    async def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        metadata: dict | None = None,
    ) -> None:
        """
        Append a message and enforce the rolling-window size cap.

        Args:
            role:     "user" | "assistant" | "system" | "tool" | "agent"
            metadata: Optional extra fields (agent_name, tool, confidence …)
        """
        await self.get_or_create(session_id)
        lock = await self._session_lock(session_id)
        async with lock:
            session = self._sessions[session_id]
            msg: dict = {"role": role, "content": content, "ts": time.time()}
            if metadata:
                msg.update(metadata)
            session.messages.append(msg)
            if len(session.messages) > cfg.MAX_HISTORY_MESSAGES:
                session.messages = session.messages[-cfg.MAX_HISTORY_MESSAGES:]

    async def get_messages(
        self,
        session_id: str,
        roles: list[str] | None = None,
        last_n: int | None = None,
    ) -> list[dict]:
        """
        Return conversation history, optionally filtered.

        Args:
            roles:  Only return messages whose role is in this list.
            last_n: Only return the most recent N messages.
        """
        await self.get_or_create(session_id)
        lock = await self._session_lock(session_id)
        async with lock:
            msgs = list(self._sessions[session_id].messages)
        if roles:
            msgs = [m for m in msgs if m["role"] in roles]
        if last_n is not None:
            msgs = msgs[-last_n:]
        return msgs

    async def clear_history(self, session_id: str) -> None:
        """Wipe conversation history while keeping context and short-term."""
        lock = await self._session_lock(session_id)
        async with lock:
            if session_id in self._sessions:
                self._sessions[session_id].messages = []

    # ── Tier 2: Short-term memory ──────────────────────────────────────────────

    async def set_short_term(
        self,
        session_id: str,
        key: str,
        value: Any,
        ttl: int | None = None,
    ) -> None:
        """Store a transient value that auto-expires after ttl seconds."""
        await self.get_or_create(session_id)
        expires_at = time.monotonic() + (ttl or cfg.SHORT_TERM_TTL)
        lock = await self._session_lock(session_id)
        async with lock:
            self._sessions[session_id].short_term[key] = (value, expires_at)

    async def get_short_term(self, session_id: str, key: str, default: Any = None) -> Any:
        """Retrieve a short-term value, returning default if absent or expired."""
        await self.get_or_create(session_id)
        lock = await self._session_lock(session_id)
        async with lock:
            entry = self._sessions[session_id].short_term.get(key)
        if entry is None:
            return default
        value, expires_at = entry
        return value if time.monotonic() <= expires_at else default

    async def evict_short_term(self, session_id: str) -> int:
        """Purge expired short-term entries. Returns count removed."""
        lock = await self._session_lock(session_id)
        async with lock:
            if session_id not in self._sessions:
                return 0
            now  = time.monotonic()
            st   = self._sessions[session_id].short_term
            dead = [k for k, (_, exp) in st.items() if now > exp]
            for k in dead:
                del st[k]
        return len(dead)

    # ── Tier 3: Context bag ────────────────────────────────────────────────────

    async def set_context(self, session_id: str, key: str, value: Any) -> None:
        """Persist a value for the lifetime of the session."""
        await self.get_or_create(session_id)
        lock = await self._session_lock(session_id)
        async with lock:
            self._sessions[session_id].context[key] = value

    async def get_context(self, session_id: str, key: str, default: Any = None) -> Any:
        await self.get_or_create(session_id)
        lock = await self._session_lock(session_id)
        async with lock:
            return self._sessions[session_id].context.get(key, default)

    async def get_full_context(self, session_id: str) -> dict:
        """Return the entire context snapshot."""
        await self.get_or_create(session_id)
        lock = await self._session_lock(session_id)
        async with lock:
            return dict(self._sessions[session_id].context)

    # ── Housekeeping ───────────────────────────────────────────────────────────

    async def evict_expired_sessions(self) -> int:
        """Remove sessions whose idle TTL has elapsed. Returns count evicted."""
        async with self._meta_lock:
            dead = [sid for sid, s in self._sessions.items() if s.is_expired()]
            for sid in dead:
                del self._sessions[sid]
                self._locks.pop(sid, None)
        if dead:
            log.info("Memory: evicted %d expired session(s)", len(dead))
        return len(dead)

    async def stats(self) -> dict:
        async with self._meta_lock:
            sessions = list(self._sessions.values())
        return {
            "active_sessions":  len(sessions),
            "total_messages":   sum(len(s.messages) for s in sessions),
            "total_short_term": sum(len(s.short_term) for s in sessions),
        }


# Module-level singleton
memory = MemoryManager()
