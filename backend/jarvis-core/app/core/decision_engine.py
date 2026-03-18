"""
app/core/decision_engine.py
═══════════════════════════════════════════════════════════════
Decision Engine — the analytical brain of JARVIS 2.0.

Pipeline for every input:
    classify_intent()   → conversational | command | query | system_control
    evaluate_risk()     → low | medium | high
    select_action()     → ActionDecision (command + parameters)

Risk framework
──────────────
  low    → execute directly, no confirmation needed
  medium → ask user for confirmation before executing
  high   → block and warn (shutdown, delete, destructive ops)

Intent classification uses a two-pass approach:
  Pass 1: fast regex/keyword matching (< 1 ms)
  Pass 2: semantic heuristic scoring
═══════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from app.utils.logger import get_logger

log = get_logger("jarvis.decision")

# ── Intent constants ───────────────────────────────────────────────────
CONVERSATIONAL = "conversational"
COMMAND        = "command"
QUERY          = "query"
SYSTEM_CONTROL = "system_control"
GESTURE        = "gesture"

# ── Risk constants ─────────────────────────────────────────────────────
RISK_LOW    = "low"
RISK_MEDIUM = "medium"
RISK_HIGH   = "high"

# ── Known gesture IDs ──────────────────────────────────────────────────
_GESTURE_IDS = {
    "swipe_right", "swipe_left", "swipe_up", "swipe_down",
    "open_hand", "fist", "pinch", "grab", "release",
    "zoom_in", "zoom_out", "thumbs_up", "thumbs_down",
    "wave", "circle_clockwise", "circle_counterclockwise",
}

# ── Classification patterns ────────────────────────────────────────────
_CMD_PATTERNS = re.compile(
    r"\b(open|launch|start|go to|navigate|search|find|look up|play|run|"
    r"close|stop|pause|resume|skip|next|previous|volume|mute|download|"
    r"create|make|set|send|call|message|remind|schedule|show|display|"
    r"screenshot|take a screenshot|capture screen)\b",
    re.IGNORECASE,
)
_QUERY_PATTERNS = re.compile(
    r"\b(what|who|why|how|when|where|which|tell me|give me|list|"
    r"summarize|calculate|convert|translate|compare)\b",
    re.IGNORECASE,
)
_CONV_PATTERNS = re.compile(
    r"\b(explain|describe|define|meaning of|talk about|discuss|help me understand)\b",
    re.IGNORECASE,
)
_SYS_PATTERNS = re.compile(
    r"\b(activate|deactivate|enable|disable|switch(?: to)?|voice mode|"
    r"gesture mode|dark mode|light mode|restart|reboot|shutdown|"
    r"reset|clear(?: memory| history| session)?|update settings|"
    r"configure|status|health check)\b",
    re.IGNORECASE,
)
_HIGH_RISK = re.compile(
    r"\b(delete|remove|format|wipe|destroy|shutdown|reboot|restart|"
    r"uninstall|kill process|terminate|factory reset)\b",
    re.IGNORECASE,
)
_MEDIUM_RISK = re.compile(
    r"\b(send|message|email|post|share|upload|purchase|buy|pay|confirm|"
    r"submit|sign|agree|accept|install|update|modify|change password)\b",
    re.IGNORECASE,
)

# ── URL / site catalog ─────────────────────────────────────────────────
_URL_MAP: dict[str, str] = {
    "youtube":       "https://www.youtube.com",
    "google":        "https://www.google.com",
    "github":        "https://github.com",
    "gmail":         "https://mail.google.com",
    "netflix":       "https://www.netflix.com",
    "spotify":       "https://open.spotify.com",
    "twitter":       "https://twitter.com",
    "instagram":     "https://www.instagram.com",
    "reddit":        "https://www.reddit.com",
    "linkedin":      "https://www.linkedin.com",
    "amazon":        "https://www.amazon.com",
    "twitch":        "https://www.twitch.tv",
    "discord":       "https://discord.com",
    "notion":        "https://notion.so",
    "chatgpt":       "https://chat.openai.com",
    "claude":        "https://claude.ai",
    "wikipedia":     "https://www.wikipedia.org",
    "stackoverflow": "https://stackoverflow.com",
    "arxiv":         "https://arxiv.org",
    "news":          "https://news.google.com",
}

_SEARCH_ENGINES: dict[str, str] = {
    "google":    "https://www.google.com/search?q=",
    "youtube":   "https://www.youtube.com/results?search_query=",
    "github":    "https://github.com/search?q=",
    "reddit":    "https://www.reddit.com/search/?q=",
    "amazon":    "https://www.amazon.com/s?k=",
    "bing":      "https://www.bing.com/search?q=",
    "duckduckgo":"https://duckduckgo.com/?q=",
}


# ── Decision result types ──────────────────────────────────────────────

@dataclass
class ActionDecision:
    """Structured action the executor will carry out."""
    type:       str                    # "browser" | "ui" | "media" | "system" | "api"
    name:       str                    # command name
    parameters: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"type": self.type, "name": self.name, "parameters": self.parameters}


@dataclass
class DecisionResult:
    """
    Full classification result for one input.

    Fields:
        intent:     Primary intent category.
        decision:   One-line human-readable decision description.
        action:     Structured action (None for conversational/query).
        risk:       Risk level of the proposed action.
        confidence: Classification confidence [0–1].
        proactive:  Optional proactive suggestion text.
        requires_confirmation: True when risk == medium.
    """
    intent:                 str
    decision:               str
    action:                 ActionDecision | None = None
    risk:                   str   = RISK_LOW
    confidence:             float = 0.85
    proactive:              str | None = None
    requires_confirmation:  bool  = False

    def action_dict(self) -> dict:
        return self.action.to_dict() if self.action else {}


# ── DecisionEngine ─────────────────────────────────────────────────────

class DecisionEngine:
    """
    Classifies intent, evaluates risk, and selects the appropriate action.

    This is a stateless class — all context is passed in as arguments.
    The JarvisCore passes in conversation history and preferences.
    """

    def decide(
        self,
        text:        str,
        history:     list[dict] | None = None,
        preferences: dict | None       = None,
        session_state: dict | None     = None,
    ) -> DecisionResult:
        """
        Analyse input and return a full decision.

        Args:
            text:          Raw user input (text, transcription, or gesture ID).
            history:       Recent conversation turns [{role, content}].
            preferences:   User preference dict from memory.
            session_state: Current system state dict.

        Returns:
            DecisionResult
        """
        stripped = text.strip()

        # ── 1. Gesture shortcut ────────────────────────────────────────
        if stripped.lower() in _GESTURE_IDS:
            return self._decide_gesture(stripped.lower())

        # ── 2. Intent classification ───────────────────────────────────
        intent     = self._classify_intent(stripped)
        risk       = self._evaluate_risk(stripped, intent)
        action     = None
        decision   = ""
        proactive  = None
        confidence = 0.88

        # ── 3. Action selection ────────────────────────────────────────
        if intent == COMMAND:
            action, decision = self._select_command_action(stripped)
            if action is None:
                # Couldn't parse a specific command → fall back to conversational
                intent    = CONVERSATIONAL
                decision  = "Respond conversationally"
                confidence = 0.65

        elif intent == QUERY:
            decision  = "Answer with LLM knowledge"
            confidence = 0.90

        elif intent == SYSTEM_CONTROL:
            action, decision = self._select_system_action(stripped)

        else:  # CONVERSATIONAL
            decision  = "Generate conversational response"
            confidence = 0.88
            proactive  = self._proactive_suggestion(stripped, history, preferences)

        # ── 4. Risk gate ───────────────────────────────────────────────
        requires_confirm = (risk == RISK_MEDIUM)
        if risk == RISK_HIGH:
            return DecisionResult(
                intent=intent,
                decision="Blocked: high-risk operation requires explicit authorisation",
                action=None,
                risk=RISK_HIGH,
                confidence=0.99,
                requires_confirmation=True,
            )

        return DecisionResult(
            intent=intent,
            decision=decision,
            action=action,
            risk=risk,
            confidence=confidence,
            proactive=proactive,
            requires_confirmation=requires_confirm,
        )

    # ── Intent classification ──────────────────────────────────────────

    def _classify_intent(self, text: str) -> str:
        scores = {
            COMMAND:        bool(_CMD_PATTERNS.search(text)) * 2,
            QUERY:          bool(_QUERY_PATTERNS.search(text)) * 2,
            SYSTEM_CONTROL: bool(_SYS_PATTERNS.search(text)) * 2,
            CONVERSATIONAL: 1,  # default
        }
        # Short phrases with action verbs → command
        words = text.lower().split()
        if len(words) <= 4 and _CMD_PATTERNS.search(text):
            scores[COMMAND] += 1

        return max(scores, key=scores.__getitem__)

    # ── Risk evaluation ────────────────────────────────────────────────

    def _evaluate_risk(self, text: str, intent: str) -> str:
        if _HIGH_RISK.search(text):
            return RISK_HIGH
        if _MEDIUM_RISK.search(text):
            return RISK_MEDIUM
        if intent == SYSTEM_CONTROL:
            return RISK_MEDIUM
        return RISK_LOW

    # ── Command action selector ────────────────────────────────────────

    def _select_command_action(self, text: str) -> tuple[ActionDecision | None, str]:
        lower = text.lower().strip()

        # open <site>
        m = re.match(r"^(?:open|launch|go to|navigate to|show)\s+(.+?)\.?\s*$", lower)
        if m:
            target = m.group(1).strip()
            url    = _URL_MAP.get(target)
            if url:
                return (
                    ActionDecision("browser", "open_url", {"url": url, "site": target}),
                    f"Open {target}",
                )
            # Might be a raw domain
            if "." in target and " " not in target:
                return (
                    ActionDecision("browser", "open_url", {"url": f"https://{target}"}),
                    f"Open {target}",
                )
            # Unknown → search
            q = target.replace(" ", "+")
            return (
                ActionDecision("browser", "open_url", {"url": f"https://www.google.com/search?q={q}", "query": target}),
                f"Search for '{target}'",
            )

        # search [on <engine>] <query>
        m = re.match(r"^search(?:\s+for)?\s+(.+?)(?:\s+on\s+(\w+))?$", lower)
        if m:
            query  = m.group(1).strip()
            engine = (m.group(2) or "google").strip()
            base   = _SEARCH_ENGINES.get(engine, _SEARCH_ENGINES["google"])
            return (
                ActionDecision("browser", "search_query",
                               {"query": query, "engine": engine, "url": base + query.replace(" ", "+")}),
                f"Search '{query}' on {engine}",
            )

        # play <query>
        m = re.match(r"^play\s+(.+)$", lower)
        if m:
            q = m.group(1).strip()
            return (
                ActionDecision("browser", "open_url",
                               {"url": f"https://www.youtube.com/results?search_query={q.replace(' ','+')}", "site": "youtube", "query": q}),
                f"Play '{q}' on YouTube",
            )

        # volume <level>
        m = re.match(r"^(?:set\s+)?volume(?:\s+to)?\s+(\d+)", lower)
        if m:
            return (
                ActionDecision("media", "set_volume", {"level": int(m.group(1))}),
                f"Set volume to {m.group(1)}%",
            )

        # next / previous track
        if re.match(r"^(?:next|skip)", lower):
            return ActionDecision("media", "next_track", {}), "Next track"
        if re.match(r"^(?:previous|back|last)", lower):
            return ActionDecision("media", "previous_track", {}), "Previous track"

        # pause / resume / stop
        if re.match(r"^pause", lower):
            return ActionDecision("media", "pause", {}), "Pause"
        if re.match(r"^(?:resume|continue)", lower):
            return ActionDecision("media", "play", {}), "Resume"
        if re.match(r"^stop", lower):
            return ActionDecision("media", "stop", {}), "Stop"

        # screenshot
        if "screenshot" in lower:
            return ActionDecision("system", "screenshot", {}), "Take screenshot"

        # fullscreen
        if "fullscreen" in lower or "full screen" in lower:
            return ActionDecision("system", "fullscreen", {}), "Toggle fullscreen"

        return None, ""

    # ── System action selector ─────────────────────────────────────────

    def _select_system_action(self, text: str) -> tuple[ActionDecision | None, str]:
        lower = text.lower()
        if "dark mode" in lower:
            return ActionDecision("ui", "set_theme", {"theme": "dark"}), "Enable dark mode"
        if "light mode" in lower:
            return ActionDecision("ui", "set_theme", {"theme": "light"}), "Enable light mode"
        if "voice mode" in lower:
            return ActionDecision("system", "set_mode", {"mode": "voice"}), "Activate voice mode"
        if "gesture mode" in lower:
            return ActionDecision("system", "set_mode", {"mode": "gesture"}), "Activate gesture mode"
        if "clear" in lower and ("memory" in lower or "history" in lower):
            return ActionDecision("system", "clear_memory", {}), "Clear conversation memory"
        if "status" in lower or "health" in lower:
            return ActionDecision("api", "get_status", {}), "Get system status"
        return ActionDecision("system", "system_action", {"instruction": text}), "Execute system action"

    # ── Gesture routing ────────────────────────────────────────────────

    def _decide_gesture(self, gesture_id: str) -> DecisionResult:
        _GESTURE_MAP = {
            "swipe_right":           ("ui", "next_screen",       "Navigate right"),
            "swipe_left":            ("ui", "previous_screen",   "Navigate left"),
            "swipe_up":              ("ui", "scroll_top",        "Scroll up"),
            "swipe_down":            ("ui", "scroll_bottom",     "Scroll down"),
            "open_hand":             ("ui", "open_menu",         "Open menu"),
            "fist":                  ("ui", "close_menu",        "Close menu"),
            "pinch":                 ("ui", "select_object",     "Select object"),
            "grab":                  ("ui", "grab_element",      "Grab element"),
            "release":               ("ui", "release_element",   "Release element"),
            "zoom_in":               ("ui", "zoom_in",           "Zoom in"),
            "zoom_out":              ("ui", "zoom_out",          "Zoom out"),
            "thumbs_up":             ("ui", "confirm",           "Confirmed"),
            "thumbs_down":           ("ui", "reject",            "Rejected"),
            "wave":                  ("system", "wake",          "JARVIS activated"),
            "circle_clockwise":      ("ui", "rotate_right",      "Rotate right"),
            "circle_counterclockwise":("ui","rotate_left",       "Rotate left"),
        }
        entry = _GESTURE_MAP.get(gesture_id, ("ui", "unknown_gesture", "Unknown gesture"))
        typ, cmd, label = entry
        return DecisionResult(
            intent=GESTURE,
            decision=label,
            action=ActionDecision(typ, cmd, {"gesture_id": gesture_id}),
            risk=RISK_LOW,
            confidence=0.98,
        )

    # ── Proactive suggestions ──────────────────────────────────────────

    def _proactive_suggestion(
        self,
        text:        str,
        history:     list[dict] | None,
        preferences: dict | None,
    ) -> str | None:
        """Generate a proactive suggestion based on content analysis."""
        lower = text.lower()

        if any(w in lower for w in ["study", "learn", "research", "homework"]):
            return "Shall I open YouTube for tutorials or a notes application?"
        if any(w in lower for w in ["music", "relax", "chill", "focus"]):
            return "Shall I open Spotify or YouTube Music?"
        if any(w in lower for w in ["news", "update", "current events", "headlines", "show me the news"]):
            return "Shall I open Google News or a news website?"
        if any(w in lower for w in ["code", "programming", "debug", "develop"]):
            return "Shall I open GitHub, Stack Overflow, or your code editor?"
        if any(w in lower for w in ["tired", "bored", "entertain"]):
            return "Shall I open Netflix, YouTube, or Twitch?"
        return None


# Module singleton
decision_engine = DecisionEngine()
