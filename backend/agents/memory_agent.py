"""
backend/agents/memory_agent.py
================================
MemoryAgent — manages three-tier memory: conversation, short-term, and vector.

Responsibilities
────────────────
1. Store user / assistant messages in ConversationMemory.
2. Persist important facts / tool results to VectorMemory.
3. Retrieve semantically-relevant context before LLM calls.
4. Summarise long conversations when history exceeds the token budget.
5. Expose memory state to other agents through the AgentTask context bag.

Integration
───────────
JarvisCore calls memory_agent.store_turn() after every interaction and
memory_agent.retrieve_context() before building the LLM prompt.
Other agents can inject into the context bag via AgentTask.context.
"""

from __future__ import annotations

import logging
import re
import time
from typing import Any

log = logging.getLogger("jarvis.agent.memory")

# Importance keywords — messages containing these are stored in vector memory
_IMPORTANT_PATTERNS = re.compile(
    r"\b(remember|note|important|always|never|prefer|my name|i am|i'm|i work|"
    r"i like|i hate|i want|i need|save this|keep in mind)\b",
    re.IGNORECASE,
)

# Maximum tokens in history before summarisation is triggered
MAX_HISTORY_TOKENS = 3000
AVG_CHARS_PER_TOKEN = 4


class MemoryAgent:
    """
    Manages all memory operations for JARVIS.

    Args:
        conv_memory:   ConversationMemory instance.
        vec_memory:    VectorMemory instance.
        llm_service:   LLMService for summarisation (injected at runtime).
    """

    def __init__(self, conv_memory=None, vec_memory=None, llm_service=None) -> None:
        self._conv   = conv_memory
        self._vec    = vec_memory
        self._llm    = llm_service
        self._calls  = 0
        self._stores = 0

    # ── Store ──────────────────────────────────────────────────────────────────

    async def store_turn(
        self,
        session_id: str,
        user_text:  str,
        bot_text:   str,
        intent:     str = "chat",
        agent:      str = "executor",
        tool_calls: list[dict] | None = None,
    ) -> None:
        """
        Persist one complete conversation turn.

        Stores both messages in conversation memory.
        If the user message contains important content, also stores it
        in vector memory for future semantic retrieval.

        Args:
            session_id: Active session UUID.
            user_text:  What the user said.
            bot_text:   What JARVIS replied.
            intent:     Classified intent for this turn.
            agent:      Agent that produced the response.
            tool_calls: List of tool invocations this turn.
        """
        self._stores += 1

        if self._conv:
            await self._conv.add_message(
                session_id, "user", user_text,
                intent=intent,
            )
            await self._conv.add_message(
                session_id, "assistant", bot_text,
                agent=agent,
                tool_calls=len(tool_calls or []),
            )

        # Conditionally persist to vector memory
        if self._vec and _IMPORTANT_PATTERNS.search(user_text):
            await self._vec.add(
                text=f"User said: {user_text}\nJARVIS replied: {bot_text}",
                session_id=session_id,
                source="conversation",
                metadata={"intent": intent, "agent": agent},
            )
            log.debug("MemoryAgent: stored turn to vector memory (important content)")

        # Store tool results if significant
        if self._vec and tool_calls:
            for tc in tool_calls:
                if tc.get("success") and tc.get("tool") in ("web_search", "file_reader"):
                    snippet = str(tc.get("result", ""))[:500]
                    if snippet:
                        await self._vec.add(
                            text=snippet,
                            session_id=session_id,
                            source="tool",
                            metadata={"tool": tc["tool"]},
                        )

    # ── Retrieve ───────────────────────────────────────────────────────────────

    async def retrieve_context(
        self,
        session_id: str,
        query:      str,
        history_n:  int = 10,
        vec_top_k:  int = 3,
    ) -> dict:
        """
        Assemble context for the LLM prompt.

        Returns:
            {
                "messages":   list of recent conversation messages,
                "relevant":   semantically relevant past entries,
                "context_bag": session context annotations,
            }
        """
        self._calls += 1

        messages   = []
        relevant   = []
        ctx_bag    = {}

        if self._conv:
            messages = await self._conv.get_messages(
                session_id,
                roles=["user", "assistant"],
                last_n=history_n,
            )
            ctx_bag = await self._conv.get_full_context(session_id)

        if self._vec and query:
            hits = await self._vec.search(
                query=query,
                top_k=vec_top_k,
                session_id=session_id,
            )
            relevant = [
                {"text": h.text, "score": h.score, "source": h.source}
                for h in hits
            ]

        return {
            "messages":    messages,
            "relevant":    relevant,
            "context_bag": ctx_bag,
        }

    # ── Summarise ──────────────────────────────────────────────────────────────

    async def maybe_summarise(self, session_id: str) -> str | None:
        """
        Summarise conversation history if it's too long for the LLM context window.

        Returns a summary string (stored back into context), or None.
        """
        if not self._conv or not self._llm:
            return None

        messages = await self._conv.get_messages(session_id)
        total_chars = sum(len(m["content"]) for m in messages)
        if total_chars / AVG_CHARS_PER_TOKEN < MAX_HISTORY_TOKENS:
            return None

        log.info("MemoryAgent: history too long (%d chars) — summarising", total_chars)

        history_text = "\n".join(
            f"{m['role'].upper()}: {m['content']}"
            for m in messages[-20:]
        )
        prompt = (
            f"Summarise this conversation in 3–5 bullet points:\n\n{history_text}\n\n"
            "Summary:"
        )
        try:
            result = await self._llm.generate(prompt, max_new_tokens=200, temperature=0.3)
            summary = result.text.strip()
            await self._conv.set_context(session_id, "summary", summary)
            # Clear old messages, keep summary as a system message
            await self._conv.clear(session_id)
            await self._conv.add_message(
                session_id, "system",
                f"[Previous conversation summary]\n{summary}",
            )
            log.info("MemoryAgent: summarised %d messages into %d chars",
                     len(messages), len(summary))
            return summary
        except Exception as exc:
            log.warning("Summarisation failed: %s", exc)
            return None

    # ── Remember explicit facts ────────────────────────────────────────────────

    async def remember(self, session_id: str, key: str, value: Any) -> None:
        """Store an explicit fact in the session context bag."""
        if self._conv:
            await self._conv.set_context(session_id, key, value)
        if self._vec:
            await self._vec.add(
                text=f"Fact: {key} = {value}",
                session_id=session_id,
                source="user",
                metadata={"type": "explicit_fact"},
            )

    async def recall(self, session_id: str, key: str, default: Any = None) -> Any:
        """Retrieve an explicit fact from the session context bag."""
        if self._conv:
            return await self._conv.get_context(session_id, key, default)
        return default

    # ── Diagnostics ───────────────────────────────────────────────────────────

    async def stats(self) -> dict:
        conv_stats = await self._conv.stats() if self._conv else {}
        vec_stats  = await self._vec.stats()  if self._vec  else {}
        return {
            "calls":  self._calls,
            "stores": self._stores,
            "conversation": conv_stats,
            "vector": vec_stats,
        }
