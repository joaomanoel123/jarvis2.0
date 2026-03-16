"""
wake_word_detector.py
=====================
WakeWordDetector — listens for "Jarvis" before passing audio to STT.

Strategies (in priority order)
────────────────────────────────
1. Porcupine (pvporcupine) — Picovoice's on-device wake word engine.
   Low latency, low CPU, built-in "Jarvis" model available.
   Requires a free Picovoice access key.

2. Text-match fallback — runs STT on every utterance and checks whether
   the transcription starts with (or contains) the wake word.
   Works without any extra model. Slightly higher latency but zero setup.

The fallback is always available; Porcupine is used when available.

Wake word flow
──────────────
  ┌─────────────────┐
  │  audio chunk    │
  └────────┬────────┘
           │
           ▼
  Porcupine (if loaded)  ──detected──►  WakeEvent emitted
           │
        not heard
           │
           ▼
     (text fallback)
   STT on short buffer
   contains "jarvis"? ──yes──►  WakeEvent emitted
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

log = logging.getLogger("jarvis.voice.wake")

WAKE_WORDS        = ["jarvis", "hey jarvis", "ok jarvis"]
WAKE_TIMEOUT_S    = 10.0   # ignore duplicate wakes within this window
ACTIVATION_WINDOW = 5.0    # seconds to stay "awake" after wake word


# ── Result type ────────────────────────────────────────────────────────────────

@dataclass
class WakeEvent:
    """Emitted when the wake word is detected."""
    trigger:    str            # exact phrase that triggered ("jarvis")
    strategy:   str            # "porcupine" | "text_match"
    confidence: float          # [0–1]; 1.0 for Porcupine (binary)
    timestamp:  float = field(default_factory=time.time)


WakeCallback = Callable[[WakeEvent], None]


# ── WakeWordDetector ───────────────────────────────────────────────────────────

class WakeWordDetector:
    """
    Detects the "Jarvis" wake word and gates the command pipeline.

    Args:
        access_key:         Picovoice access key for Porcupine.
                            None = skip Porcupine, use text fallback only.
        wake_words:         List of trigger phrases (lowercase).
        require_wake_word:  If False, every utterance is treated as a command.
                            Useful for testing.
        activation_window:  Seconds to remain "active" after wake word.
        stt_engine:         STT engine used by the text-match fallback.
    """

    def __init__(
        self,
        access_key:        str | None = None,
        wake_words:        list[str] = WAKE_WORDS,
        require_wake_word: bool = True,
        activation_window: float = ACTIVATION_WINDOW,
        stt_engine=None,               # SpeechToText instance (injected)
    ) -> None:
        self._access_key      = access_key
        self._wake_words      = [w.lower().strip() for w in wake_words]
        self._require         = require_wake_word
        self._active_window   = activation_window
        self._stt             = stt_engine

        # Porcupine handle
        self._porcupine = None

        # State
        self._last_wake_ts: float = 0.0
        self._active:       bool  = False
        self._active_until: float = 0.0

        # Callbacks
        self._callbacks: list[WakeCallback] = []

        # Metrics
        self._detections = 0

    # ── Lifecycle ──────────────────────────────────────────────────────────────

    def load(self) -> "WakeWordDetector":
        """Attempt to load Porcupine; fall back to text-match if unavailable."""
        if self._access_key:
            try:
                self._load_porcupine()
                log.info("WakeWordDetector: Porcupine engine loaded")
            except Exception as exc:
                log.warning("Porcupine unavailable (%s) — using text-match fallback", exc)
                self._porcupine = None
        else:
            log.info("WakeWordDetector: no access key — text-match fallback active")

        strategy = "porcupine" if self._porcupine else "text_match"
        log.info("WakeWordDetector ready  strategy=%s  words=%s", strategy, self._wake_words)
        return self

    def _load_porcupine(self) -> None:
        """Load the Porcupine wake word engine."""
        import pvporcupine

        # Try to use the built-in "jarvis" keyword
        try:
            self._porcupine = pvporcupine.create(
                access_key=self._access_key,
                keywords=["jarvis"],
            )
            log.debug("Porcupine built-in 'jarvis' keyword loaded")
        except pvporcupine.PorcupineInvalidArgumentError:
            # jarvis may not be in the free tier — fall back to "hey google" or similar
            log.warning("Built-in 'jarvis' keyword not available — using 'computer'")
            self._porcupine = pvporcupine.create(
                access_key=self._access_key,
                keywords=["computer"],
            )

    def stop(self) -> None:
        if self._porcupine:
            self._porcupine.delete()
            self._porcupine = None

    # ── Callback registration ──────────────────────────────────────────────────

    def on_wake(self, callback: WakeCallback) -> None:
        """Register a function called when the wake word is detected."""
        self._callbacks.append(callback)

    # ── Detection methods ──────────────────────────────────────────────────────

    def check_audio_chunk(self, pcm_int16: np.ndarray) -> WakeEvent | None:
        """
        Process a raw PCM int16 chunk with Porcupine.

        Args:
            pcm_int16: int16 array of exactly porcupine.frame_length samples.

        Returns:
            WakeEvent if detected, else None.
        """
        if not self._porcupine:
            return None

        # Porcupine expects a Python list of int16 values
        frame = pcm_int16[:self._porcupine.frame_length].tolist()
        idx   = self._porcupine.process(frame)

        if idx >= 0:
            return self._emit_wake("jarvis", "porcupine", 1.0)
        return None

    def check_text(self, text: str) -> WakeEvent | None:
        """
        Check whether the transcribed text contains a wake word.

        Args:
            text: Lowercased, cleaned transcript from STT.

        Returns:
            WakeEvent if a wake word is found, else None.
        """
        t = text.lower().strip()

        # Check each wake word
        for word in self._wake_words:
            pattern = r"\b" + re.escape(word) + r"\b"
            if re.search(pattern, t):
                log.debug("Wake word '%s' found in: %r", word, t)
                return self._emit_wake(word, "text_match", 0.9)

        return None

    def strip_wake_word(self, text: str) -> str:
        """
        Remove the wake word from the beginning of a transcribed text.

        "jarvis open youtube"  →  "open youtube"
        "hey jarvis search AI" →  "search AI"
        """
        t = text.strip()
        for word in sorted(self._wake_words, key=len, reverse=True):
            # Match at start of string (case-insensitive)
            pattern = r"^" + re.escape(word) + r"[\s,!?.]*"
            cleaned = re.sub(pattern, "", t, flags=re.IGNORECASE).strip()
            if cleaned != t:
                return cleaned
        return t

    # ── Activation state ───────────────────────────────────────────────────────

    def is_active(self) -> bool:
        """
        Return True if JARVIS is currently in the active/listening state.

        The active window starts when a wake word is detected and lasts for
        `activation_window` seconds, or until a command is consumed.
        """
        if not self._require:
            return True
        return time.monotonic() < self._active_until

    def activate(self, duration: float | None = None) -> None:
        """Manually activate the listening window (e.g. after wake word)."""
        self._active       = True
        self._active_until = time.monotonic() + (duration or self._active_window)
        log.debug("Listening window open for %.1f s", duration or self._active_window)

    def deactivate(self) -> None:
        """Close the listening window."""
        self._active       = False
        self._active_until = 0.0

    def consume(self) -> None:
        """
        Mark that a command was consumed.
        Resets the active window — user must say wake word again.
        """
        self.deactivate()
        log.debug("Command consumed — wake word required again")

    # ── Internal ───────────────────────────────────────────────────────────────

    def _emit_wake(self, trigger: str, strategy: str, confidence: float) -> WakeEvent:
        now = time.time()

        # Suppress duplicate wakes within the timeout window
        if (now - self._last_wake_ts) < WAKE_TIMEOUT_S and self._active:
            log.debug("Wake word suppressed (active window)")
            return None

        self._last_wake_ts = now
        self._detections  += 1
        self.activate()

        event = WakeEvent(trigger=trigger, strategy=strategy, confidence=confidence)
        log.info("🎙  Wake word detected: %r  [%s  %.0f%%]",
                 trigger, strategy, confidence * 100)

        for cb in self._callbacks:
            try:
                cb(event)
            except Exception as exc:
                log.exception("Wake callback raised: %s", exc)

        return event

    # ── Diagnostics ────────────────────────────────────────────────────────────

    def stats(self) -> dict:
        strategy = "porcupine" if self._porcupine else "text_match"
        return {
            "strategy":   strategy,
            "detections": self._detections,
            "is_active":  self.is_active(),
            "active_until": round(max(0.0, self._active_until - time.monotonic()), 1),
            "wake_words": self._wake_words,
        }

    def __repr__(self) -> str:
        strategy = "porcupine" if self._porcupine else "text_match"
        return f"<WakeWordDetector strategy={strategy} active={self.is_active()}>"
