"""
command_parser.py
=================
CommandParser — converts raw transcribed text into structured VoiceCommand objects.

Architecture
────────────
Two parsing strategies run in order:

1. Rule-based parser (always active, zero latency)
   Pattern-matching on the cleaned transcript using a priority-ordered
   intent table. Each intent has one or more regex patterns and an
   optional entity extractor.

2. LLM-based parser (optional, higher accuracy)
   Sends the transcript to the JARVIS backend /voice/parse endpoint.
   Falls back to rule-based if the API is unavailable.

Supported intents
──────────────────
  open_url        → open youtube / google / spotify / github / …
  web_search      → search for <query> / look up <query>
  launch_app      → open vscode / open terminal / open notepad
  media_control   → play music / pause / skip / next / previous
  system_control  → shutdown / restart / sleep / lock
  volume_control  → volume up / down / mute / set volume 50
  send_to_jarvis  → anything else → forward to JARVIS AI agent

Output
──────
    VoiceCommand(intent, action, entities, raw_text, confidence)
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger("jarvis.voice.parser")


# ── Data types ─────────────────────────────────────────────────────────────────

@dataclass
class VoiceCommand:
    """
    Structured representation of a parsed voice command.

    Attributes:
        intent:     High-level category (open_url, web_search, launch_app, …).
        action:     Specific action within the intent (e.g. "navigate", "search").
        entities:   Extracted parameters (url, query, app_name, …).
        raw_text:   Original transcribed text.
        confidence: Parser confidence [0.0–1.0].
        backend:    "rule" | "llm".
    """
    intent:     str
    action:     str
    entities:   dict[str, Any]
    raw_text:   str
    confidence: float          = 1.0
    backend:    str            = "rule"

    def to_dict(self) -> dict:
        return {
            "intent":     self.intent,
            "action":     self.action,
            "entities":   self.entities,
            "raw_text":   self.raw_text,
            "confidence": self.confidence,
        }

    def __str__(self) -> str:
        ent = ", ".join(f"{k}={v!r}" for k, v in self.entities.items())
        return f"[{self.intent}] {self.action}  {{{ent}}}"


# ── URL / app catalogue ────────────────────────────────────────────────────────

_URL_MAP: dict[str, str] = {
    # Search / news
    "youtube":        "https://www.youtube.com",
    "google":         "https://www.google.com",
    "gmail":          "https://mail.google.com",
    "google drive":   "https://drive.google.com",
    "google maps":    "https://maps.google.com",
    "google calendar":"https://calendar.google.com",
    # Social / media
    "twitter":        "https://twitter.com",
    "x":              "https://x.com",
    "instagram":      "https://www.instagram.com",
    "facebook":       "https://www.facebook.com",
    "linkedin":       "https://www.linkedin.com",
    "reddit":         "https://www.reddit.com",
    "twitch":         "https://www.twitch.tv",
    # Entertainment
    "netflix":        "https://www.netflix.com",
    "spotify":        "https://open.spotify.com",
    "prime":          "https://www.primevideo.com",
    "prime video":    "https://www.primevideo.com",
    "disney plus":    "https://www.disneyplus.com",
    "disney+":        "https://www.disneyplus.com",
    # Dev / work
    "github":         "https://github.com",
    "stack overflow": "https://stackoverflow.com",
    "stackoverflow":  "https://stackoverflow.com",
    "notion":         "https://notion.so",
    "trello":         "https://trello.com",
    "jira":           "https://jira.atlassian.com",
    # News
    "bbc":            "https://www.bbc.com",
    "cnn":            "https://cnn.com",
    "hacker news":    "https://news.ycombinator.com",
    # AI
    "openai":         "https://openai.com",
    "anthropic":      "https://anthropic.com",
    "hugging face":   "https://huggingface.co",
    "huggingface":    "https://huggingface.co",
    "claude":         "https://claude.ai",
    "chatgpt":        "https://chat.openai.com",
}

_APP_MAP: dict[str, list[str]] = {
    # Editors / IDEs
    "vscode":          ["code"],
    "visual studio code": ["code"],
    "sublime":         ["subl"],
    "vim":             ["vim"],
    "neovim":          ["nvim"],
    "jetbrains":       ["idea"],
    # Terminals
    "terminal":        ["x-terminal-emulator", "gnome-terminal", "xterm"],
    "cmd":             ["cmd.exe"],
    "powershell":      ["powershell.exe"],
    "iterm":           ["open", "-a", "iTerm"],
    # Browsers
    "chrome":          ["google-chrome", "chromium-browser"],
    "firefox":         ["firefox"],
    "safari":          ["open", "-a", "Safari"],
    "brave":           ["brave-browser"],
    # Office
    "word":            ["libreoffice", "--writer"],
    "excel":           ["libreoffice", "--calc"],
    "powerpoint":      ["libreoffice", "--impress"],
    # Communication
    "slack":           ["slack"],
    "discord":         ["discord"],
    "zoom":            ["zoom"],
    "teams":           ["teams"],
    # System
    "file manager":    ["nautilus", "explorer.exe", "Finder"],
    "files":           ["nautilus", "explorer.exe"],
    "calculator":      ["gnome-calculator", "calc.exe"],
    "settings":        ["gnome-control-center", "ms-settings:"],
}

# Search engines
_SEARCH_ENGINES: dict[str, str] = {
    "youtube":    "https://www.youtube.com/results?search_query=",
    "google":     "https://www.google.com/search?q=",
    "bing":       "https://www.bing.com/search?q=",
    "duckduckgo": "https://duckduckgo.com/?q=",
    "reddit":     "https://www.reddit.com/search/?q=",
    "github":     "https://github.com/search?q=",
    "amazon":     "https://www.amazon.com/s?k=",
}


# ── Intent pattern table ───────────────────────────────────────────────────────
# Each entry: (intent, action, [regex patterns], entity_extractor_fn | None)

def _extract_target(m: re.Match) -> dict:
    return {"target": m.group("target").strip()}


def _extract_query(m: re.Match) -> dict:
    return {"query": m.group("query").strip()}


def _extract_on_engine(m: re.Match) -> dict:
    return {
        "query":  m.group("query").strip(),
        "engine": (m.groupdict().get("engine") or "google").strip().lower(),
    }


def _extract_volume(m: re.Match) -> dict:
    level = m.groupdict().get("level")
    return {"level": int(level) if level and level.isdigit() else None}


_INTENT_TABLE: list[tuple[str, str, list[str], Any]] = [
    # ── Open URL ─────────────────────────────────────────────────────────────
    ("open_url", "navigate", [
        r"open (?P<target>.+)",
        r"go to (?P<target>.+)",
        r"launch (?P<target>.+)",
        r"navigate to (?P<target>.+)",
        r"take me to (?P<target>.+)",
        r"show me (?P<target>.+)",
        r"load (?P<target>.+)",
    ], _extract_target),

    # ── Web search ────────────────────────────────────────────────────────────
    ("web_search", "search", [
        r"search(?: for)? (?P<query>.+?) on (?P<engine>youtube|google|bing|duckduckgo|reddit|github|amazon)",
        r"(?P<engine>youtube|google|bing|duckduckgo|reddit|github|amazon) search (?P<query>.+)",
        r"find (?P<query>.+?) on (?P<engine>youtube|google|reddit|github)",
    ], _extract_on_engine),

    ("web_search", "search", [
        r"search(?: for)? (?P<query>.+)",
        r"look up (?P<query>.+)",
        r"find (?P<query>.+)",
        r"what is (?P<query>.+)",
        r"who is (?P<query>.+)",
        r"how (?:do i|to) (?P<query>.+)",
        r"tell me about (?P<query>.+)",
    ], _extract_query),

    # ── Media ─────────────────────────────────────────────────────────────────
    ("media_control", "play",  [r"play(?: music| song| audio)?$",
                                 r"play (?P<target>.+)"], _extract_target),
    ("media_control", "pause",  [r"pause(?: music| playback)?"], None),
    ("media_control", "resume", [r"resume(?: music| playback)?",
                                  r"continue(?: music| playback)?"], None),
    ("media_control", "next",   [r"next(?: track| song)?",
                                  r"skip(?: track| song)?"], None),
    ("media_control", "previous", [r"previous(?: track| song)?",
                                    r"go back",
                                    r"last song"], None),
    ("media_control", "stop",  [r"stop(?: music| playback)?"], None),

    # ── Volume ────────────────────────────────────────────────────────────────
    ("volume_control", "up",   [r"volume up",  r"louder"], None),
    ("volume_control", "down", [r"volume down", r"quieter", r"softer"], None),
    ("volume_control", "mute", [r"mute(?: volume)?"], None),
    ("volume_control", "unmute", [r"unmute(?: volume)?"], None),
    ("volume_control", "set",  [r"set volume(?: to)? (?P<level>\d+)",
                                  r"volume (?P<level>\d+)"], _extract_volume),

    # ── System ────────────────────────────────────────────────────────────────
    ("system_control", "shutdown",   [r"shut (?:down|off)(?: computer| pc)?",
                                       r"turn off(?: computer| pc)?"], None),
    ("system_control", "restart",    [r"restart(?: computer| pc)?",
                                       r"reboot(?: computer| pc)?"], None),
    ("system_control", "sleep",      [r"sleep(?: mode)?",
                                       r"put to sleep"], None),
    ("system_control", "lock",       [r"lock(?: screen| computer)?"], None),
    ("system_control", "screenshot", [r"take (?:a )?screenshot",
                                       r"screenshot"], None),

    # ── JARVIS commands ───────────────────────────────────────────────────────
    ("jarvis_command", "ask", [
        r"(?P<query>.+)",   # catch-all — forward to JARVIS AI
    ], _extract_query),
]


# ── CommandParser ──────────────────────────────────────────────────────────────

class CommandParser:
    """
    Converts clean transcribed text into a structured VoiceCommand.

    Args:
        use_llm_fallback: Call JARVIS backend /voice/parse for unmatched intents.
        api_url:          JARVIS backend URL for LLM fallback.
        min_confidence:   Discard results below this confidence threshold.
    """

    def __init__(
        self,
        use_llm_fallback: bool  = False,
        api_url:          str   = "http://localhost:8000",
        min_confidence:   float = 0.3,
    ) -> None:
        self._use_llm   = use_llm_fallback
        self._api_url   = api_url
        self._min_conf  = min_confidence

        # Pre-compile all patterns for performance
        self._compiled: list[tuple[str, str, list[re.Pattern], Any]] = []
        for intent, action, patterns, extractor in _INTENT_TABLE:
            compiled = [re.compile(p, re.IGNORECASE) for p in patterns]
            self._compiled.append((intent, action, compiled, extractor))

        # Metrics
        self._total_parsed = 0
        self._rule_hits    = 0
        self._llm_hits     = 0

    # ── Public API ─────────────────────────────────────────────────────────────

    def parse(self, text: str) -> VoiceCommand | None:
        """
        Parse cleaned transcribed text into a VoiceCommand.

        Args:
            text: Cleaned, lowercase text with wake word stripped.

        Returns:
            VoiceCommand, or None if below min_confidence.
        """
        if not text.strip():
            return None

        self._total_parsed += 1

        # Rule-based first
        cmd = self._rule_parse(text)

        if cmd and cmd.confidence >= self._min_conf:
            self._rule_hits += 1
            log.debug("Parsed: %s", cmd)
            return cmd

        # LLM fallback
        if self._use_llm:
            cmd = self._llm_parse(text)
            if cmd and cmd.confidence >= self._min_conf:
                self._llm_hits += 1
                return cmd

        # Last resort: treat as JARVIS AI query
        log.debug("No intent matched — treating as JARVIS query: %r", text)
        return VoiceCommand(
            intent="jarvis_command",
            action="ask",
            entities={"query": text},
            raw_text=text,
            confidence=0.5,
        )

    # ── Rule-based parser ──────────────────────────────────────────────────────

    def _rule_parse(self, text: str) -> VoiceCommand | None:
        clean = text.lower().strip()

        for intent, action, patterns, extractor in self._compiled:
            # Skip the catch-all rule (last entry) — only use it as last resort
            if intent == "jarvis_command":
                continue

            for pat in patterns:
                m = pat.fullmatch(clean) or pat.match(clean)
                if m:
                    entities = {}
                    if extractor:
                        try:
                            entities = extractor(m) or {}
                        except IndexError:
                            entities = {}

                    # Resolve target to URL or app
                    entities = self._enrich_entities(intent, action, entities)

                    return VoiceCommand(
                        intent=intent,
                        action=action,
                        entities=entities,
                        raw_text=text,
                        confidence=0.92,
                        backend="rule",
                    )
        return None

    def _enrich_entities(self, intent: str, action: str, entities: dict) -> dict:
        """
        Resolve friendly names to concrete URLs or app commands.

        "youtube" → "https://www.youtube.com"
        "vscode"  → ["code"]
        """
        if intent == "open_url" and "target" in entities:
            target = entities["target"].lower().strip()

            # Exact URL map hit
            url = _URL_MAP.get(target)
            if url:
                entities["url"]       = url
                entities["site_name"] = target
                return entities

            # App map hit
            app = _APP_MAP.get(target)
            if app:
                entities["app_cmd"]  = app
                entities["app_name"] = target
                return entities

            # Looks like a real URL?
            if re.match(r"https?://", target):
                entities["url"] = target
                return entities

            # Try to guess as a domain
            if "." in target and " " not in target:
                entities["url"] = f"https://{target}"
                return entities

            # Unknown target — search for it
            log.debug("Unknown target %r — falling back to Google search", target)
            import urllib.parse
            entities["url"]   = _SEARCH_ENGINES["google"] + urllib.parse.quote_plus(target)
            entities["query"] = target

        if intent == "web_search" and "query" in entities:
            import urllib.parse
            engine = entities.get("engine", "google")
            base   = _SEARCH_ENGINES.get(engine, _SEARCH_ENGINES["google"])
            entities["search_url"] = base + urllib.parse.quote_plus(entities["query"])
            entities["engine"]     = engine

        if intent == "media_control" and action == "play" and "target" in entities:
            target = entities["target"].lower()
            # If target is a known URL (spotify, youtube), resolve it
            url = _URL_MAP.get(target)
            if url:
                entities["url"] = url

        return entities

    # ── LLM fallback ──────────────────────────────────────────────────────────

    def _llm_parse(self, text: str) -> VoiceCommand | None:
        """Call JARVIS backend /voice/parse for NLU-based intent parsing."""
        import requests
        try:
            resp = requests.post(
                f"{self._api_url}/voice/parse",
                json={"text": text},
                timeout=3.0,
            )
            if resp.status_code == 200:
                data = resp.json()
                return VoiceCommand(
                    intent=data.get("intent",  "jarvis_command"),
                    action=data.get("action",  "ask"),
                    entities=data.get("entities", {"query": text}),
                    raw_text=text,
                    confidence=data.get("confidence", 0.75),
                    backend="llm",
                )
        except Exception as exc:
            log.debug("LLM parse failed: %s", exc)
        return None

    # ── URL / app helpers (public for executor) ────────────────────────────────

    @staticmethod
    def resolve_url(name: str) -> str | None:
        """Return URL for a known site name, or None."""
        return _URL_MAP.get(name.lower().strip())

    @staticmethod
    def resolve_app(name: str) -> list[str] | None:
        """Return app command list for a known app name, or None."""
        return _APP_MAP.get(name.lower().strip())

    # ── Diagnostics ───────────────────────────────────────────────────────────

    def stats(self) -> dict:
        return {
            "total_parsed": self._total_parsed,
            "rule_hits":    self._rule_hits,
            "llm_hits":     self._llm_hits,
        }

    def supported_intents(self) -> list[str]:
        return sorted({intent for intent, _, _, _ in _INTENT_TABLE})

    def supported_sites(self) -> list[str]:
        return sorted(_URL_MAP.keys())

    def supported_apps(self) -> list[str]:
        return sorted(_APP_MAP.keys())
