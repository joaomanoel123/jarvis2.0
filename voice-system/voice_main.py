"""
voice_main.py
=============
JARVIS 2.0 Voice Command System — main pipeline controller.

Pipeline
────────
  Microphone (SpeechListener)
       ↓  Utterance
  Speech-to-Text (SpeechToText)
       ↓  STTResult
  Wake Word Detection (WakeWordDetector)
       ↓  confirmed text (wake word stripped)
  Command Parsing (CommandParser)
       ↓  VoiceCommand
  Command Execution (CommandExecutor)
       ↓  ExecutionResult
  TTS / Audio Feedback (pyttsx3 / espeak)

Usage
─────
  # Default run
  python voice_main.py

  # Custom backend
  python voice_main.py --api http://192.168.1.10:8000

  # Test without backend
  python voice_main.py --no-backend --dry-run

  # With Porcupine wake word
  python voice_main.py --porcupine-key YOUR_KEY

  # Disable wake word requirement (always listening)
  python voice_main.py --always-on

  # Verbose logging
  python voice_main.py --verbose

Keyboard shortcut: Ctrl+C to stop.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import signal
import sys
import time
from pathlib import Path

from speech_listener  import SpeechListener, Utterance
from speech_to_text   import SpeechToText, STTResult
from wake_word_detector import WakeWordDetector, WakeEvent
from command_parser   import CommandParser, VoiceCommand
from command_executor import CommandExecutor, ExecutionResult

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("jarvis.voice.main")


# ── TTS feedback ───────────────────────────────────────────────────────────────

class TTSFeedback:
    """
    Text-to-speech feedback using pyttsx3 or espeak fallback.

    Speaks short acknowledgements so the user knows JARVIS heard them.
    """

    def __init__(self, rate: int = 170, volume: float = 0.9) -> None:
        self._engine = None
        self._rate   = rate
        self._volume = volume

        try:
            import pyttsx3
            self._engine = pyttsx3.init()
            self._engine.setProperty("rate",   self._rate)
            self._engine.setProperty("volume", self._volume)
            log.debug("TTS: pyttsx3 ready")
        except Exception as exc:
            log.debug("pyttsx3 not available (%s) — using espeak fallback", exc)

    def speak(self, text: str) -> None:
        """Speak text asynchronously."""
        if not text:
            return
        if self._engine:
            try:
                self._engine.say(text)
                self._engine.runAndWait()
                return
            except Exception as exc:
                log.debug("pyttsx3 speak error: %s", exc)

        # espeak fallback
        try:
            import subprocess
            subprocess.Popen(
                ["espeak", "-s", str(self._rate), text],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except FileNotFoundError:
            # No TTS at all — just print
            print(f"[JARVIS] {text}")

    def acknowledge(self) -> None:
        self.speak("Yes?")

    def say_heard(self, command: str) -> None:
        """Give brief confirmation of what was heard."""
        if len(command) > 40:
            command = command[:40] + "…"
        self.speak(f"Got it — {command}")

    def say_result(self, result: ExecutionResult) -> None:
        """Speak the execution result or error."""
        if result.success:
            msg = result.message
            # Don't read long API responses aloud — just say "done"
            if len(msg) > 120:
                self.speak("Done.")
            elif msg:
                self.speak(msg)
        else:
            err = result.error or "something went wrong"
            self.speak(f"Sorry, {err.replace('_', ' ')}")


# ── VoiceSystem ────────────────────────────────────────────────────────────────

class VoiceSystem:
    """
    Full voice command pipeline controller.

    Wires SpeechListener → STT → WakeWordDetector → CommandParser
    → CommandExecutor together and manages the async event loop.

    Args:
        config: Namespace from argparse (or custom config dict).
    """

    def __init__(self, config) -> None:
        self._cfg     = config
        self._running = False
        self._stats: dict = {
            "utterances":      0,
            "wake_detections": 0,
            "commands_parsed": 0,
            "commands_executed": 0,
            "errors":          0,
            "start_time":      0.0,
        }

        # ── Initialise components ──────────────────────────────────────────────
        self._listener = SpeechListener(
            device=getattr(config, "mic_device", None),
            vad_threshold=getattr(config, "vad_threshold", 0.018),
            verbose=getattr(config, "verbose", False),
        )

        self._stt = SpeechToText(
            backend=getattr(config, "stt_backend", "auto"),
            model_size=getattr(config, "whisper_model", "base"),
            language=getattr(config, "language", "en"),
        )

        self._wake = WakeWordDetector(
            access_key=getattr(config, "porcupine_key", None),
            require_wake_word=not getattr(config, "always_on", False),
            stt_engine=self._stt,
        )

        self._parser = CommandParser(
            use_llm_fallback=not getattr(config, "no_backend", False),
            api_url=getattr(config, "api", "http://localhost:8000"),
        )

        self._executor = CommandExecutor(
            api_url=getattr(config, "api", "http://localhost:8000"),
            dry_run=getattr(config, "dry_run", False),
            session_id=None,
        )

        self._tts = TTSFeedback()

    # ── Lifecycle ──────────────────────────────────────────────────────────────

    def start(self) -> None:
        """Initialise all components and start listening."""
        log.info("═══ JARVIS 2.0 Voice System starting ═══")

        # Load STT model
        log.info("Loading STT engine …")
        try:
            self._stt.load()
        except Exception as exc:
            log.error("STT load failed: %s", exc)
            sys.exit(1)

        # Load wake word detector
        self._wake.load()

        # Register callbacks
        self._wake.on_wake(self._on_wake_word)

        # Metrics
        self._stats["start_time"] = time.time()

        # Start microphone
        self._listener.start()
        self._running = True

        log.info("═══ JARVIS ready — say 'Jarvis' to activate ═══")
        if getattr(self._cfg, "always_on", False):
            log.info("  Mode: ALWAYS-ON (no wake word required)")
        log.info("  STT backend: %s", self._stt.backend)
        log.info("  API: %s", getattr(self._cfg, "api", "http://localhost:8000"))
        log.info("  Dry run: %s", getattr(self._cfg, "dry_run", False))

    def stop(self) -> None:
        """Stop all components gracefully."""
        log.info("Shutting down voice system …")
        self._running = False
        self._listener.stop()
        self._wake.stop()
        self._print_stats()
        log.info("Voice system stopped")

    # ── Main loop ──────────────────────────────────────────────────────────────

    def run(self) -> None:
        """
        Synchronous main loop.

        Reads utterances from the listener queue and processes them through
        the full pipeline. Blocks until stop() is called.
        """
        self.start()
        try:
            for utterance in self._listener.utterances():
                if not self._running:
                    break
                self._process_utterance(utterance)
        except KeyboardInterrupt:
            log.info("Interrupted by user")
        finally:
            self.stop()

    async def run_async(self) -> None:
        """Async entry point for integration with existing async codebases."""
        self.start()
        try:
            async for utterance in self._listener.async_utterances():
                if not self._running:
                    break
                await asyncio.get_running_loop().run_in_executor(
                    None, self._process_utterance, utterance
                )
        except asyncio.CancelledError:
            pass
        finally:
            self.stop()

    # ── Pipeline ───────────────────────────────────────────────────────────────

    def _process_utterance(self, utterance: Utterance) -> None:
        """Full pipeline: Utterance → STT → Wake → Parse → Execute."""
        self._stats["utterances"] += 1

        # ── Step 1: Speech to text ─────────────────────────────────────────────
        try:
            stt_result = self._stt.transcribe(utterance)
        except Exception as exc:
            log.error("STT error: %s", exc)
            self._stats["errors"] += 1
            return

        text = stt_result.clean_text
        if not text:
            log.debug("Empty transcription — skipping")
            return

        log.info("🎤  Heard: %r  (%.2f s, conf=%.0f%%)",
                 text, utterance.duration_s, stt_result.confidence * 100)

        # ── Step 2: Wake word detection ────────────────────────────────────────
        # Porcupine check runs on raw audio chunks (see speech_listener callback).
        # Here we do the text-match fallback.

        if self._wake.is_active():
            # Already in active window — process as command directly
            command_text = text
        else:
            # Check if this utterance contains the wake word
            wake_event = self._wake.check_text(text)
            if wake_event is None:
                log.debug("Wake word not detected — ignoring utterance")
                return

            self._stats["wake_detections"] += 1
            # Strip the wake word from the text
            command_text = self._wake.strip_wake_word(text)
            log.debug("Wake word stripped → %r", command_text)

            # If only the wake word was spoken, prompt for a command
            if not command_text.strip():
                self._tts.acknowledge()
                log.info("🔔  Wake word only — waiting for command")
                return

        # ── Step 3: Command parsing ────────────────────────────────────────────
        cmd = self._parser.parse(command_text)
        if cmd is None:
            log.debug("Command parse returned None for %r", command_text)
            self._wake.deactivate()
            return

        self._stats["commands_parsed"] += 1
        log.info("📋  Parsed: %s", cmd)

        # Brief acknowledgement
        self._tts.say_heard(command_text)

        # ── Step 4: Execution ──────────────────────────────────────────────────
        result = self._executor.execute(cmd)
        self._stats["commands_executed"] += 1

        if result.success:
            log.info("✅  %s", result)
        else:
            log.warning("❌  %s", result)
            self._stats["errors"] += 1

        # Speak result
        self._tts.say_result(result)

        # Close the wake window — user must say "Jarvis" again
        self._wake.consume()

    # ── Wake word callback (for Porcupine audio-chunk path) ───────────────────

    def _on_wake_word(self, event: WakeEvent) -> None:
        """Called by WakeWordDetector when wake word is detected via audio chunk."""
        self._stats["wake_detections"] += 1
        log.info("🔔  Wake word detected [%s  %.0f%%]",
                 event.strategy, event.confidence * 100)
        self._tts.acknowledge()

    # ── Stats ──────────────────────────────────────────────────────────────────

    def _print_stats(self) -> None:
        s   = self._stats
        run = round(time.time() - s["start_time"], 1) if s["start_time"] else 0
        log.info("─── Session Stats ───")
        log.info("  Runtime:           %.0f s", run)
        log.info("  Utterances heard:  %d", s["utterances"])
        log.info("  Wake detections:   %d", s["wake_detections"])
        log.info("  Commands parsed:   %d", s["commands_parsed"])
        log.info("  Commands executed: %d", s["commands_executed"])
        log.info("  Errors:            %d", s["errors"])
        log.info("─────────────────────")

        # Also print component stats
        log.info("Listener: %s", self._listener.stats())
        log.info("STT:      %s", self._stt.stats())
        log.info("Wake:     %s", self._wake.stats())
        log.info("Parser:   %s", self._parser.stats())
        log.info("Executor: %s", self._executor.stats())


# ── FastAPI /voice-command endpoint (add to jarvis-v2) ────────────────────────

def create_voice_router(executor: CommandExecutor, parser: CommandParser):
    """
    Returns a FastAPI router with /voice-command endpoint.
    Add to jarvis-v2/app/main.py:

        from voice_main import create_voice_router
        app.include_router(create_voice_router(executor, parser))
    """
    from fastapi import APIRouter
    from pydantic import BaseModel

    router = APIRouter(tags=["Voice"])

    class VoiceCommandRequest(BaseModel):
        text:       str
        session_id: str | None = None
        source:     str        = "voice"
        confidence: float      = 1.0

    class VoiceCommandResponse(BaseModel):
        success:   bool
        message:   str
        intent:    str
        action:    str
        data:      dict = {}
        raw_text:  str  = ""

    @router.post("/voice-command", response_model=VoiceCommandResponse)
    async def voice_command(req: VoiceCommandRequest):
        """Receive a transcribed voice command and execute it."""
        cmd = parser.parse(req.text)
        if cmd is None:
            return VoiceCommandResponse(
                success=False, message="Could not parse command",
                intent="unknown", action="none", raw_text=req.text,
            )
        result = executor.execute(cmd)
        return VoiceCommandResponse(
            success=result.success,
            message=result.message,
            intent=result.intent,
            action=result.action,
            data=result.data,
            raw_text=req.text,
        )

    @router.get("/voice/status")
    async def voice_status():
        return {
            "executor": executor.stats(),
            "parser":   parser.stats(),
        }

    @router.get("/voice/commands")
    async def list_commands():
        return {
            "intents": parser.supported_intents(),
            "sites":   parser.supported_sites(),
            "apps":    parser.supported_apps(),
        }

    return router


# ── Argument parser ────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="JARVIS 2.0 Voice Command System",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Audio
    p.add_argument("--mic-device",   type=int,   default=None,    help="Microphone device index")
    p.add_argument("--vad-threshold",type=float, default=0.018,   help="VAD RMS energy threshold")
    p.add_argument("--list-mics",    action="store_true",          help="List available microphones")

    # STT
    p.add_argument("--stt-backend",  default="auto",
                   choices=["auto", "whisper_local", "whisper_api", "google", "vosk"],
                   help="Speech-to-text backend")
    p.add_argument("--whisper-model",default="base",
                   choices=["tiny", "base", "small", "medium", "large"],
                   help="Whisper model size (whisper_local only)")
    p.add_argument("--language",     default="en",                help="Language code (en, pt, es, …)")

    # Wake word
    p.add_argument("--porcupine-key",default=None,                help="Picovoice access key for Porcupine")
    p.add_argument("--always-on",    action="store_true",          help="No wake word required")

    # Backend
    p.add_argument("--api",          default="http://localhost:8000", help="JARVIS backend URL")
    p.add_argument("--no-backend",   action="store_true",          help="Disable backend API calls")
    p.add_argument("--dry-run",      action="store_true",          help="Parse but don't execute commands")

    # Misc
    p.add_argument("--verbose",      action="store_true",          help="Debug logging")
    return p.parse_args()


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # List microphones and exit
    if args.list_mics:
        listener = SpeechListener()
        devices  = listener.list_devices()
        print("\nAvailable microphones:")
        for d in devices:
            print(f"  [{d['index']}] {d['name']}  ({d['channels']} ch)")
        return

    voice = VoiceSystem(args)

    # Graceful shutdown on SIGINT / SIGTERM
    def _handle_signal(sig, frame):
        log.info("Signal %d received — stopping", sig)
        voice.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT,  _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    voice.run()


if __name__ == "__main__":
    main()
