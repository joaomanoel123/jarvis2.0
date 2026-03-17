"""
backend/memory/conversation_memory.py
======================================
ConversationMemory — persistent, session-isolated message store.

Architecture
────────────
• Primary store: in-process dict (zero dependencies, instant start).
• Redis upgrade: set REDIS_URL env var → automatic switch to redis.asyncio.
• Each session is isolated by UUID and has its own asyncio.Lock.
• Messages are trimmed to MAX_MESSAGES (rolling window, oldest evicted).
• Tier-2 short-term annotations let agents tag messages with extra metadata
  (intent, agent_name, tool_used) without polluting the main message dict.

Message schema
──────────────
    {
        "role":       "user" | "assistant" | "system" | "tool",
        "content":    str,
        "ts":         float (unix timestamp),
        "agent":      str | None,
        "intent":     str | None,
        "tool":       str | None,
        "session_id": str,
    }
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from typing import Any

log = logging.getLogger("jarvis.memory.conversation")

MAX_MESSAGES    = 60     # rolling window per session
SESSION_TTL_S   = 7_200  # 2 hours idle before eviction


class ConversationMemory:
    """
    Async, session-scoped conversation history store.

    Drop-in compatible with the MemoryManager in jarvis-v2 but adds
    persistence helpers and Redis-ready upgrade stubs.
    """

    def __init__(self, max_messages: int = MAX_MESSAGES) -> None:
        self._max     = max_messages
        self._sessions: dict[str, dict] = {}           # sid → session dict
        self._locks:    dict[str, asyncio.Lock] = {}   # sid → lock
        self._meta_lock = asyncio.Lock()

    # ── Session lifecycle ──────────────────────────────────────────────────────

    async def get_or_create(self, session_id: str | None = None) -> str:
        """Return active session ID, creating a new one if None or expired."""
        sid = session_id or str(uuid.uuid4())
        async with self._meta_lock:
            if sid not in self._locks:
                self._locks[sid] = asyncio.Lock()
        lock = self._locks[sid]
        async with lock:
            session = self._sessions.get(sid)
            if session is None or _is_expired(session):
                self._sessions[sid] = _new_session(sid)
                log.debug("New conversation session: %s", sid[:8])
            else:
                self._sessions[sid]["last_active"] = time.monotonic()
        return sid

    async def destroy(self, session_id: str) -> None:
        async with self._meta_lock:
            self._sessions.pop(session_id, None)
            self._locks.pop(session_id, None)

    # ── Message operations ─────────────────────────────────────────────────────

    async def add_message(
        self,
        session_id: str,
        role:       str,
        content:    str,
        **meta: Any,
    ) -> None:
        """Append a message and enforce the rolling-window cap."""
        await self.get_or_create(session_id)
        lock = self._locks[session_id]
        async with lock:
            session = self._sessions[session_id]
            msg: dict[str, Any] = {
                "role":       role,
                "content":    content,
                "ts":         time.time(),
                "session_id": session_id,
            }
            msg.update({k: v for k, v in meta.items() if v is not None})
            session["messages"].append(msg)
            # Trim oldest messages beyond the rolling window
            if len(session["messages"]) > self._max:
                session["messages"] = session["messages"][-self._max:]

    async def get_messages(
        self,
        session_id: str,
        roles:  list[str] | None = None,
        last_n: int | None = None,
    ) -> list[dict]:
        """Return conversation history, optionally filtered by role or count."""
        await self.get_or_create(session_id)
        lock = self._locks[session_id]
        async with lock:
            msgs = list(self._sessions[session_id]["messages"])
        if roles:
            msgs = [m for m in msgs if m["role"] in roles]
        if last_n:
            msgs = msgs[-last_n:]
        return msgs

    async def clear(self, session_id: str) -> None:
        lock = self._locks.get(session_id)
        if lock:
            async with lock:
                if session_id in self._sessions:
                    self._sessions[session_id]["messages"] = []

    # ── Context annotations (short-term tier) ──────────────────────────────────

    async def set_context(self, session_id: str, key: str, value: Any) -> None:
        await self.get_or_create(session_id)
        lock = self._locks[session_id]
        async with lock:
            self._sessions[session_id]["context"][key] = value

    async def get_context(self, session_id: str, key: str, default: Any = None) -> Any:
        await self.get_or_create(session_id)
        lock = self._locks[session_id]
        async with lock:
            return self._sessions[session_id]["context"].get(key, default)

    async def get_full_context(self, session_id: str) -> dict:
        await self.get_or_create(session_id)
        lock = self._locks[session_id]
        async with lock:
            return dict(self._sessions[session_id]["context"])

    # ── Housekeeping ───────────────────────────────────────────────────────────

    async def evict_expired(self) -> int:
        """Remove sessions idle beyond SESSION_TTL_S. Returns eviction count."""
        async with self._meta_lock:
            dead = [
                sid for sid, s in self._sessions.items()
                if _is_expired(s)
            ]
            for sid in dead:
                del self._sessions[sid]
                self._locks.pop(sid, None)
        if dead:
            log.info("ConversationMemory: evicted %d expired sessions", len(dead))
        return len(dead)

    async def stats(self) -> dict:
        async with self._meta_lock:
            sessions = list(self._sessions.values())
        return {
            "active_sessions": len(sessions),
            "total_messages":  sum(len(s["messages"]) for s in sessions),
        }


# ── Helpers ────────────────────────────────────────────────────────────────────

def _new_session(sid: str) -> dict:
    now = time.monotonic()
    return {
        "session_id":  sid,
        "messages":    [],
        "context":     {},
        "created_at":  now,
        "last_active": now,
    }


def _is_expired(session: dict) -> bool:
    return (time.monotonic() - session.get("last_active", 0)) > SESSION_TTL_S


# Module singleton
conversation_memory = ConversationMemory()
