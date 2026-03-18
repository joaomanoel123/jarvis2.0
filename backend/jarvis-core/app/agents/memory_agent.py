"""
app/agents/memory_agent.py
Retrieves context from the database and stores every interaction.
"""

from __future__ import annotations

from app.memory.database import Database
from app.utils.logger    import get_logger

log = get_logger("jarvis.agent.memory")


class MemoryAgent:
    """Wraps the Database for agent-friendly retrieve / store operations."""

    def __init__(self, db: Database) -> None:
        self._db = db

    async def retrieve(self, session_id: str, query: str) -> str:
        """
        Build a context string from recent history + semantic search.

        Returns:
            Multi-line string of relevant past exchanges.
        """
        history = await self._db.get_history(session_id, limit=6, roles=["user", "assistant"])
        if not history:
            return ""

        lines = []
        for turn in history:
            role    = turn["role"].upper()
            content = turn["content"][:200]
            lines.append(f"{role}: {content}")

        # Recent commands context
        commands = await self._db.get_recent_commands(session_id, limit=3)
        if commands:
            lines.append("\nRecent commands:")
            for cmd in commands:
                lines.append(f"  {cmd['name']} — {'OK' if cmd['success'] else 'FAILED'}")

        return "\n".join(lines)

    async def store(
        self,
        session_id: str,
        user_text:  str,
        bot_text:   str,
        intent:     str,
        confidence: float = 0.85,
    ) -> None:
        """Persist a complete conversation turn."""
        await self._db.save_turn(session_id, "user",      user_text, intent, confidence)
        await self._db.save_turn(session_id, "assistant", bot_text,  intent, confidence)

    async def log_command(
        self,
        session_id: str,
        name:       str,
        parameters: dict,
        result:     dict,
        risk:       str = "low",
    ) -> None:
        """Record a command execution in the database."""
        success    = result.get("success", True)
        latency_ms = result.get("latency_ms", 0.0)
        await self._db.log_command(
            session_id=session_id,
            name=name,
            parameters=parameters,
            result=result,
            success=success,
            latency_ms=latency_ms,
            risk=risk,
        )
