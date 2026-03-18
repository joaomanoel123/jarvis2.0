"""
app/agents/reasoning_agent.py
Generates natural-language responses using the LLMService.
Falls back to a rule-based stub when LLM is unavailable.
"""

from __future__ import annotations

import os
from app.utils.logger import get_logger

log = get_logger("jarvis.agent.reasoner")

_SYSTEM_PROMPT = (
    "You are JARVIS 2.0, an advanced AI assistant inspired by Iron Man's J.A.R.V.I.S. "
    "You are precise, intelligent, and slightly futuristic in tone. "
    "You are concise — never verbose. You speak as a high-level AI system, not a chatbot. "
    "You are both a conversational partner AND an execution engine."
)


class ReasoningAgent:
    """
    Wraps LLMService for reasoning and response generation.
    Uses the in-session history to maintain conversation continuity.
    """

    async def respond(
        self,
        text:    str,
        history: list[dict],
        context: str   = "",
        intent:  str   = "conversational",
    ) -> str:
        """
        Generate a response for the given input.

        Args:
            text:    User input.
            history: Recent conversation turns [{role, content}].
            context: Optional retrieved memory context.
            intent:  Classified intent (affects system prompt tone).

        Returns:
            Response text string.
        """
        # Augment system prompt with retrieved context
        system = _SYSTEM_PROMPT
        if context:
            system += f"\n\nRelevant memory context:\n{context}"

        # Try LLM service
        try:
            from app.services.llm_service import llm_service
            if llm_service.is_loaded():
                result = await llm_service.generate(
                    user_message=text,
                    system_prompt=system,
                    history=history,
                )
                return result.text
        except Exception as exc:
            log.warning("LLM unavailable: %s — using stub", exc)

        # Stub responses when LLM is not loaded
        return self._stub(text, intent)

    def _stub(self, text: str, intent: str) -> str:
        """Rule-based fallback responses."""
        lower = text.lower()
        if any(w in lower for w in ["hello", "hi", "hey"]):
            return "JARVIS online. Systems nominal. How can I assist you?"
        if "status" in lower or "system" in lower:
            return "All systems operational. Memory, decision engine, and agent pipeline active."
        if any(w in lower for w in ["thank", "thanks"]):
            return "At your service."
        if "what" in lower and "ai" in lower:
            return ("Artificial Intelligence is the simulation of human cognitive processes "
                    "by computer systems, including learning, reasoning, and self-correction.")
        if any(w in lower for w in ["who are you", "what are you"]):
            return ("I am JARVIS 2.0 — Just A Rather Very Intelligent System. "
                    "A fully autonomous AI platform with multi-agent reasoning and command execution.")
        return (f"Understood: '{text}'. Language model is initialising. "
                "Processing via rule-based fallback.")
