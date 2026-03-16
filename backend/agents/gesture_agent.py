"""
agents/gesture_agent.py
=======================
GestureAgent — interprets hardware/UI gesture signals and converts
them into natural-language intents that flow through the standard
agent pipeline.

MediaPipe integration
──────────────────────
The API endpoint receives a GesturePayload containing:
  • gesture_id   — string key (e.g. "swipe_right", "pinch")
  • landmarks    — optional list of 3-D hand landmarks from MediaPipe Hands
  • confidence   — optional float [0–1] from the classifier
  • metadata     — screen context (active_widget, coordinates, …)

Landmark-based refinement (when landmarks are supplied)
────────────────────────────────────────────────────────
The agent applies simple geometric rules on top of the MediaPipe
hand-landmark model (21 keypoints, normalised to wrist).
  • Finger extension detection  — tips above PIP joints → extended
  • Thumb direction             → left/right disambiguation

This layer runs synchronously on CPU; no GPU inference required here
because MediaPipe already did the heavy lifting on the client.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any

from agents.base_agent import AgentTask, AgentResult, AgentType, BaseAgent
from services.llm_service import llm_service

log = logging.getLogger(__name__)

# ── Gesture → intent map ───────────────────────────────────────────────────────
GESTURE_MAP: dict[str, str] = {
    "open_palm":    "Show available options and what JARVIS can do",
    "fist":         "Stop and cancel the current operation",
    "pinch":        "Select and confirm the highlighted item",
    "spread":       "Expand and show more detail",
    "swipe_right":  "Navigate to the next item",
    "swipe_left":   "Navigate to the previous item",
    "swipe_up":     "Show more information about this topic",
    "swipe_down":   "Show a quick summary",
    "tap":          "Select and activate this",
    "double_tap":   "Open or execute this item",
    "long_press":   "Open the context menu",
    "shake":        "Reset and start a fresh conversation",
    "wave":         "Wake up — greet me and show what you can do",
    "thumbs_up":    "That is correct, please continue",
    "thumbs_down":  "That is wrong, please try a different approach",
    "point_up":     "Scroll to the top of the list",
    "point_down":   "Scroll to the bottom of the list",
    "zoom_in":      "Increase the detail level",
    "zoom_out":     "Give me a high-level overview",
    "rotate_cw":    "Move forward through history",
    "rotate_ccw":   "Move back through history",
}

# Confidence threshold below which a gesture is rejected
CONFIDENCE_THRESHOLD = 0.65


# ── Landmark helpers ────────────────────────────────────────────────────────────

@dataclass
class Landmark:
    x: float
    y: float
    z: float = 0.0


def _parse_landmarks(raw: list[dict]) -> list[Landmark]:
    return [Landmark(**p) for p in raw if {"x", "y"}.issubset(p)]


def _finger_extended(lms: list[Landmark], tip: int, pip: int) -> bool:
    """True when the finger tip is above (smaller y) the PIP joint."""
    if tip >= len(lms) or pip >= len(lms):
        return False
    return lms[tip].y < lms[pip].y


def _count_extended_fingers(lms: list[Landmark]) -> int:
    """
    Count how many of the 4 non-thumb fingers are extended.
    MediaPipe hand landmark indices: tips [8,12,16,20], PIPs [6,10,14,18].
    """
    pairs = [(8, 6), (12, 10), (16, 14), (20, 18)]
    return sum(_finger_extended(lms, tip, pip) for tip, pip in pairs)


def _refine_gesture(gesture_id: str, landmarks: list[Landmark]) -> str:
    """
    Apply geometric checks to improve gesture classification.
    Returns possibly-updated gesture_id.
    """
    if not landmarks:
        return gesture_id

    n_ext = _count_extended_fingers(landmarks)

    # Open palm: all 4 fingers extended
    if gesture_id in ("open_palm", "spread") and n_ext >= 3:
        return "open_palm"

    # Fist: no fingers extended
    if gesture_id == "fist" and n_ext == 0:
        return "fist"

    # Pinch: index + thumb close (landmarks 4 & 8)
    if gesture_id == "pinch" and len(landmarks) >= 9:
        dist = math.hypot(
            landmarks[4].x - landmarks[8].x,
            landmarks[4].y - landmarks[8].y,
        )
        if dist > 0.12:    # too far apart — not a real pinch
            return gesture_id   # keep original

    return gesture_id


# ── GestureAgent ───────────────────────────────────────────────────────────────

_GESTURE_SYSTEM = """\
You are JARVIS responding to a physical gesture command.
The user performed a gesture that maps to the following intent:

  Gesture:  {gesture}
  Intent:   {intent}
  Context:  {context}

Respond naturally and helpfully, treating this as if the user typed
the intent text.  Be concise.
"""


class GestureAgent(BaseAgent):
    """
    Interprets gesture payloads from MediaPipe and returns a conversational
    response aligned with the gesture's mapped intent.
    """

    agent_type = AgentType.GESTURE

    async def run(self, task: AgentTask) -> AgentResult:
        gesture_id = task.metadata.get("gesture_id", "").lower().strip()
        raw_lms    = task.metadata.get("landmarks", [])
        confidence = float(task.metadata.get("confidence", 1.0))

        # Confidence gate
        if confidence < CONFIDENCE_THRESHOLD:
            return AgentResult(
                text=(
                    f"I wasn't confident enough in the '{gesture_id}' gesture "
                    f"(confidence: {confidence:.0%}). Could you try again more clearly?"
                ),
                agent=self.agent_type,
                success=False,
                metadata={"gesture": gesture_id, "confidence": confidence},
            )

        # Landmark refinement
        lms = _parse_landmarks(raw_lms)
        if lms:
            gesture_id = _refine_gesture(gesture_id, lms)
            log.debug("GestureAgent: refined → %s", gesture_id)

        # Look up intent
        intent = GESTURE_MAP.get(gesture_id)
        if not intent:
            known = ", ".join(sorted(GESTURE_MAP.keys()))
            return AgentResult(
                text=(
                    f"Gesture '{gesture_id}' is not recognised. "
                    f"Known gestures: {known}."
                ),
                agent=self.agent_type,
                success=False,
                metadata={"gesture": gesture_id},
            )

        # Enrich intent with screen context
        ctx_parts = []
        if screen := task.metadata.get("active_widget"):
            ctx_parts.append(f"active widget: {screen}")
        if coords := task.metadata.get("coordinates"):
            ctx_parts.append(f"at {coords}")
        context_str = ", ".join(ctx_parts) if ctx_parts else "no additional context"

        log.info("GestureAgent: %r → intent=%r", gesture_id, intent)

        system = _GESTURE_SYSTEM.format(
            gesture=gesture_id,
            intent=intent,
            context=context_str,
        )
        result = await llm_service.chat(
            messages=task.history + [{"role": "user", "content": intent}],
            system=system,
            temperature=0.6,
        )

        return AgentResult(
            text=result.text,
            agent=self.agent_type,
            metadata={
                "gesture":    gesture_id,
                "intent":     intent,
                "confidence": confidence,
                "landmarks":  len(lms),
            },
        )
