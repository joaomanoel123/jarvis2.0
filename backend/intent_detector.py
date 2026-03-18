"""
backend/intent_detector.py
Classifies user input into one of four intents:
  conversational | command | gesture_action | system_control
"""

import re
from dataclasses import dataclass

# ── Intent constants ──────────────────────────────────────────────────
CONVERSATIONAL  = "conversational"
COMMAND         = "command"
GESTURE_ACTION  = "gesture_action"
SYSTEM_CONTROL  = "system_control"

# ── Detection rule tables ─────────────────────────────────────────────
_GESTURE_IDS = {
    "swipe_right", "swipe_left", "swipe_up", "swipe_down",
    "open_hand", "fist", "pinch", "grab", "release",
    "zoom_in", "zoom_out", "thumbs_up", "thumbs_down",
    "wave", "point", "circle_clockwise", "circle_counterclockwise",
}

_SYSTEM_PATTERNS = re.compile(
    r"\b(activate|deactivate|enable|disable|switch to|turn on|turn off|"
    r"voice mode|gesture mode|dark mode|light mode|restart|shutdown|"
    r"reset|clear (memory|history|session))\b",
    re.IGNORECASE,
)

_COMMAND_PATTERNS = re.compile(
    r"\b(open|launch|go to|navigate to|show|start|play|search|find|"
    r"look up|run|execute|close|stop|pause|next|previous|volume|"
    r"mute|screenshot|download|send|create|make|set|get)\b",
    re.IGNORECASE,
)

_CONVERSATIONAL_PATTERNS = re.compile(
    r"\b(what|who|why|how|when|where|explain|tell me|describe|"
    r"define|meaning of|what is|what are|can you|could you|would you|"
    r"I want to know|understand|help me understand)\b",
    re.IGNORECASE,
)

# ── Command parameter extractors ───────────────────────────────────────
_URL_MAP = {
    "youtube":   "https://www.youtube.com",
    "google":    "https://www.google.com",
    "github":    "https://github.com",
    "gmail":     "https://mail.google.com",
    "netflix":   "https://www.netflix.com",
    "spotify":   "https://open.spotify.com",
    "twitter":   "https://twitter.com",
    "instagram": "https://www.instagram.com",
    "reddit":    "https://www.reddit.com",
    "linkedin":  "https://www.linkedin.com",
    "amazon":    "https://www.amazon.com",
    "twitch":    "https://www.twitch.tv",
    "discord":   "https://discord.com",
    "notion":    "https://notion.so",
    "slack":     "https://slack.com",
    "chatgpt":   "https://chat.openai.com",
    "claude":    "https://claude.ai",
}


@dataclass
class DetectionResult:
    intent:     str
    confidence: float
    command:    str | None = None
    parameters: dict       = None

    def __post_init__(self):
        if self.parameters is None:
            self.parameters = {}


def detect(text: str) -> DetectionResult:
    """
    Classify the input text and extract any command parameters.

    Args:
        text: Raw user input string (may be a gesture JSON key or plain text).

    Returns:
        DetectionResult with intent, confidence, and optional command/params.
    """
    stripped = text.strip().lower()

    # ── 1. Gesture detection ─────────────────────────────────────────
    if stripped in _GESTURE_IDS:
        return DetectionResult(
            intent=GESTURE_ACTION,
            confidence=0.98,
            command="gesture_execution",
            parameters={"gesture": stripped},
        )

    # ── 2. System control ────────────────────────────────────────────
    if _SYSTEM_PATTERNS.search(text):
        return DetectionResult(
            intent=SYSTEM_CONTROL,
            confidence=0.92,
            command="system_action",
            parameters={"instruction": text.strip()},
        )

    # ── 3. Command detection ─────────────────────────────────────────
    if _COMMAND_PATTERNS.search(text):
        cmd, params = _extract_command(stripped)
        if cmd:
            return DetectionResult(
                intent=COMMAND,
                confidence=0.90,
                command=cmd,
                parameters=params,
            )

    # ── 4. Conversational ────────────────────────────────────────────
    if _CONVERSATIONAL_PATTERNS.search(text) or len(stripped.split()) > 4:
        return DetectionResult(
            intent=CONVERSATIONAL,
            confidence=0.85,
        )

    # ── 5. Fallback: short unknown input → conversational ────────────
    return DetectionResult(
        intent=CONVERSATIONAL,
        confidence=0.60,
    )


def _extract_command(text: str) -> tuple[str | None, dict]:
    """
    Extract a specific command name and parameters from the lowercased input.
    Returns (command_name, parameters_dict).
    """

    # open <site>
    m = re.match(r"^(?:open|launch|go to|show|navigate to)\s+(.+)$", text)
    if m:
        target = m.group(1).strip().rstrip(".")
        url    = _URL_MAP.get(target)
        if url:
            return "open_url", {"url": url, "site": target}
        # Looks like a real URL
        if "." in target and " " not in target:
            return "open_url", {"url": f"https://{target}", "site": target}
        # Unknown target — fall back to search
        return "open_url", {
            "url":  f"https://www.google.com/search?q={target.replace(' ', '+')}",
            "site": target,
        }

    # search / find / look up <query>
    m = re.match(r"^(?:search(?: for)?|find|look up|look for)\s+(.+)$", text)
    if m:
        return "search", {"query": m.group(1).strip(), "engine": "google"}

    # play <target>
    m = re.match(r"^play\s+(.+)$", text)
    if m:
        target = m.group(1).strip()
        return "search", {
            "query":  target,
            "engine": "youtube",
        }

    # volume <level>
    m = re.match(r"^(?:set )?volume(?: to)?\s+(\d+)", text)
    if m:
        return "set_volume", {"level": int(m.group(1))}

    return None, {}
