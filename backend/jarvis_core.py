"""
backend/jarvis_core.py
Central pipeline: receives input, detects intent, routes to LLM or executor,
returns a structured JSON-compatible response.
"""

import time
import logging
from dataclasses import dataclass, field

from intent_detector  import detect, CONVERSATIONAL, COMMAND, GESTURE_ACTION, SYSTEM_CONTROL
from command_executor import command_executor, CommandResult
from llm_service      import llm_service, LLMResult

log = logging.getLogger("jarvis.core")


@dataclass
class JarvisResponse:
    """Unified response object for all pipeline outputs."""
    intent:     str
    response:   str
    action:     dict       = field(default_factory=dict)
    confidence: float      = 1.0
    session_id: str | None = None
    latency_ms: float      = 0.0
    success:    bool       = True

    def to_dict(self) -> dict:
        return {
            "intent":     self.intent,
            "response":   self.response,
            "action":     self.action,
            "confidence": round(self.confidence, 3),
            "session_id": self.session_id,
            "latency_ms": self.latency_ms,
            "success":    self.success,
        }


class JarvisCore:
    """
    Main processing pipeline for JARVIS 2.0.

    Pipeline:
        1. Detect intent
        2. If COMMAND or GESTURE → CommandExecutor
        3. If CONVERSATIONAL or SYSTEM_CONTROL → LLMService
        4. Return JarvisResponse
    """

    def __init__(self) -> None:
        # Simple in-memory session history
        self._sessions: dict[str, list[dict]] = {}

    async def process(
        self,
        text:       str,
        session_id: str | None = None,
        source:     str        = "text",  # "text" | "voice" | "gesture"
    ) -> JarvisResponse:
        """
        Process a user input end-to-end.

        Args:
            text:       Raw user input.
            session_id: Existing session UUID for conversation continuity.
            source:     Input modality.

        Returns:
            JarvisResponse
        """
        t0 = time.perf_counter()
        sid = session_id or _generate_sid()

        log.info("Processing [%s] source=%s: %.60r", sid[:8], source, text)

        # ── 1. Intent detection ───────────────────────────────────────
        detection = detect(text)
        intent    = detection.intent
        log.info("Intent: %s (%.0f%%)", intent, detection.confidence * 100)

        # ── 2. Route ──────────────────────────────────────────────────
        response_text = ""
        action_dict   = {}

        if intent in (COMMAND, GESTURE_ACTION):
            # --- Command / gesture path ---
            cmd_result: CommandResult = command_executor.execute(
                detection.command or "unknown",
                detection.parameters,
            )
            response_text = cmd_result.response
            action_dict   = cmd_result.action_dict()

            # For commands that have no LLM component, we're done.
            # For unrecognised commands, fall through to LLM.
            if not cmd_result.success:
                llm_result: LLMResult = await llm_service.generate(
                    text, history=self._get_history(sid)
                )
                response_text = llm_result.text

        else:
            # --- Conversational / system control path ---
            llm_result: LLMResult = await llm_service.generate(
                text, history=self._get_history(sid)
            )
            response_text = llm_result.text

            # Store turn in history
            self._add_turn(sid, text, response_text)

        latency = round((time.perf_counter() - t0) * 1000, 1)

        return JarvisResponse(
            intent=intent,
            response=response_text,
            action=action_dict,
            confidence=detection.confidence,
            session_id=sid,
            latency_ms=latency,
            success=True,
        )

    # ── Session helpers ────────────────────────────────────────────────

    def _get_history(self, sid: str) -> list[dict]:
        return self._sessions.get(sid, [])[-20:]   # last 20 turns

    def _add_turn(self, sid: str, user: str, assistant: str) -> None:
        if sid not in self._sessions:
            self._sessions[sid] = []
        self._sessions[sid].append({"role": "user",      "content": user})
        self._sessions[sid].append({"role": "assistant", "content": assistant})
        # Keep rolling window
        if len(self._sessions[sid]) > 60:
            self._sessions[sid] = self._sessions[sid][-60:]

    def clear_session(self, sid: str) -> None:
        self._sessions.pop(sid, None)

    def session_count(self) -> int:
        return len(self._sessions)


def _generate_sid() -> str:
    import uuid
    return str(uuid.uuid4())


# Module singleton
jarvis_core = JarvisCore()
