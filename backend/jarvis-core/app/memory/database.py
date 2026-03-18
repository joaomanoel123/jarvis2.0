"""
app/memory/database.py
═══════════════════════════════════════════════════════════════
Persistent memory layer for JARVIS 2.0.

Storage: aiosqlite (SQLite, zero-dependency, Render-compatible).
Upgrade path: swap _get_conn() for an asyncpg pool → PostgreSQL.

Tables
──────
  conversations   — full turn-by-turn history per session
  commands        — every executed command + result
  preferences     — user key/value preference store
  observations    — observer agent system metrics log

All public methods are async and safe for concurrent access.
═══════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import asyncio
import json
import os
import time
import uuid
from pathlib import Path
from typing import Any

import aiosqlite

from app.utils.logger import get_logger

log = get_logger("jarvis.memory")

DB_PATH = Path(os.getenv("DB_PATH", "./data/jarvis.db"))

# ── Schema ──────────────────────────────────────────────────────────────
_DDL = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS conversations (
    id          TEXT PRIMARY KEY,
    session_id  TEXT NOT NULL,
    role        TEXT NOT NULL,          -- user | assistant | system
    content     TEXT NOT NULL,
    intent      TEXT,
    confidence  REAL,
    ts          REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_conv_session ON conversations(session_id, ts DESC);

CREATE TABLE IF NOT EXISTS commands (
    id          TEXT PRIMARY KEY,
    session_id  TEXT NOT NULL,
    name        TEXT NOT NULL,
    parameters  TEXT NOT NULL,          -- JSON
    result      TEXT,                   -- JSON
    success     INTEGER NOT NULL,       -- 1 | 0
    latency_ms  REAL,
    risk        TEXT,
    ts          REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_cmd_session ON commands(session_id, ts DESC);

CREATE TABLE IF NOT EXISTS preferences (
    key         TEXT PRIMARY KEY,
    value       TEXT NOT NULL,          -- JSON
    updated_at  REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS observations (
    id          TEXT PRIMARY KEY,
    metric      TEXT NOT NULL,
    value       REAL NOT NULL,
    meta        TEXT,                   -- JSON
    ts          REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_obs_metric ON observations(metric, ts DESC);
"""


# ── Database manager ────────────────────────────────────────────────────

class Database:
    """
    Async SQLite wrapper — module singleton pattern.

    Usage:
        await db.init()
        await db.save_turn(session_id, "user", "hello", intent="conversational")
        history = await db.get_history(session_id, limit=20)
    """

    def __init__(self) -> None:
        self._conn: aiosqlite.Connection | None = None
        self._lock = asyncio.Lock()

    # ── Lifecycle ──────────────────────────────────────────────────────

    async def init(self) -> None:
        """Open the database and apply schema migrations."""
        DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        async with self._lock:
            if self._conn is not None:
                return
            self._conn = await aiosqlite.connect(str(DB_PATH))
            self._conn.row_factory = aiosqlite.Row
            await self._conn.executescript(_DDL)
            await self._conn.commit()
        log.info("Database ready: %s", DB_PATH)

    async def close(self) -> None:
        async with self._lock:
            if self._conn:
                await self._conn.close()
                self._conn = None

    def _c(self) -> aiosqlite.Connection:
        assert self._conn is not None, "Database not initialised — call db.init()"
        return self._conn

    # ── Conversations ──────────────────────────────────────────────────

    async def save_turn(
        self,
        session_id: str,
        role:       str,
        content:    str,
        intent:     str | None    = None,
        confidence: float | None  = None,
    ) -> str:
        """Persist one conversation turn. Returns the new row id."""
        row_id = str(uuid.uuid4())
        await self._c().execute(
            "INSERT INTO conversations VALUES (?,?,?,?,?,?,?)",
            (row_id, session_id, role, content, intent, confidence, time.time()),
        )
        await self._c().commit()
        return row_id

    async def get_history(
        self,
        session_id: str,
        limit:      int  = 20,
        roles:      list[str] | None = None,
    ) -> list[dict]:
        """Return recent conversation turns for a session."""
        if roles:
            placeholders = ",".join("?" * len(roles))
            rows = await self._c().execute_fetchall(
                f"SELECT role, content, intent, ts FROM conversations "
                f"WHERE session_id=? AND role IN ({placeholders}) "
                f"ORDER BY ts DESC LIMIT ?",
                (session_id, *roles, limit),
            )
        else:
            rows = await self._c().execute_fetchall(
                "SELECT role, content, intent, ts FROM conversations "
                "WHERE session_id=? ORDER BY ts DESC LIMIT ?",
                (session_id, limit),
            )
        return [dict(r) for r in reversed(rows)]

    async def search_history(self, session_id: str, query: str) -> list[dict]:
        """Full-text search within a session's conversation."""
        rows = await self._c().execute_fetchall(
            "SELECT role, content, intent, ts FROM conversations "
            "WHERE session_id=? AND content LIKE ? ORDER BY ts DESC LIMIT 10",
            (session_id, f"%{query}%"),
        )
        return [dict(r) for r in rows]

    # ── Commands ───────────────────────────────────────────────────────

    async def log_command(
        self,
        session_id: str,
        name:       str,
        parameters: dict,
        result:     dict | None  = None,
        success:    bool         = True,
        latency_ms: float        = 0.0,
        risk:       str          = "low",
    ) -> str:
        """Record a command execution."""
        row_id = str(uuid.uuid4())
        await self._c().execute(
            "INSERT INTO commands VALUES (?,?,?,?,?,?,?,?,?)",
            (
                row_id, session_id, name,
                json.dumps(parameters),
                json.dumps(result) if result else None,
                int(success), latency_ms, risk, time.time(),
            ),
        )
        await self._c().commit()
        return row_id

    async def get_recent_commands(self, session_id: str, limit: int = 10) -> list[dict]:
        rows = await self._c().execute_fetchall(
            "SELECT name, parameters, success, latency_ms, ts FROM commands "
            "WHERE session_id=? ORDER BY ts DESC LIMIT ?",
            (session_id, limit),
        )
        result = []
        for r in rows:
            d = dict(r)
            d["parameters"] = json.loads(d["parameters"] or "{}")
            result.append(d)
        return result

    async def command_stats(self) -> dict:
        rows = await self._c().execute_fetchall(
            "SELECT COUNT(*) total, SUM(success) succeeded, AVG(latency_ms) avg_ms FROM commands"
        )
        return dict(rows[0]) if rows else {}

    # ── Preferences ────────────────────────────────────────────────────

    async def set_preference(self, key: str, value: Any) -> None:
        await self._c().execute(
            "INSERT OR REPLACE INTO preferences VALUES (?,?,?)",
            (key, json.dumps(value), time.time()),
        )
        await self._c().commit()

    async def get_preference(self, key: str, default: Any = None) -> Any:
        rows = await self._c().execute_fetchall(
            "SELECT value FROM preferences WHERE key=?", (key,)
        )
        if rows:
            return json.loads(rows[0]["value"])
        return default

    async def all_preferences(self) -> dict:
        rows = await self._c().execute_fetchall("SELECT key, value FROM preferences")
        return {r["key"]: json.loads(r["value"]) for r in rows}

    # ── Observations ───────────────────────────────────────────────────

    async def record_observation(self, metric: str, value: float, meta: dict | None = None) -> None:
        await self._c().execute(
            "INSERT INTO observations VALUES (?,?,?,?,?)",
            (str(uuid.uuid4()), metric, value, json.dumps(meta) if meta else None, time.time()),
        )
        await self._c().commit()

    async def get_observations(self, metric: str, limit: int = 20) -> list[dict]:
        rows = await self._c().execute_fetchall(
            "SELECT metric, value, meta, ts FROM observations "
            "WHERE metric=? ORDER BY ts DESC LIMIT ?",
            (metric, limit),
        )
        result = []
        for r in rows:
            d = dict(r)
            d["meta"] = json.loads(d["meta"]) if d.get("meta") else {}
            result.append(d)
        return result

    # ── Stats ──────────────────────────────────────────────────────────

    async def stats(self) -> dict:
        conv = await self._c().execute_fetchall("SELECT COUNT(*) n FROM conversations")
        cmds = await self._c().execute_fetchall("SELECT COUNT(*) n FROM commands")
        pref = await self._c().execute_fetchall("SELECT COUNT(*) n FROM preferences")
        return {
            "conversations": conv[0]["n"],
            "commands":      cmds[0]["n"],
            "preferences":   pref[0]["n"],
            "db_path":       str(DB_PATH),
        }


# Module singleton
db = Database()
