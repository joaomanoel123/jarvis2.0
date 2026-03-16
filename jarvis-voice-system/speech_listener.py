"""
speech_listener.py
==================
SpeechListener — real-time microphone capture with Voice Activity Detection.

Responsibilities
────────────────
• Open the system microphone via sounddevice (fallback: PyAudio).
• Continuously read audio in small chunks (100 ms default).
• Apply simple energy-based Voice Activity Detection (VAD) to avoid
  processing silence frames and wasting CPU / STT quota.
• Accumulate voiced frames into utterance buffers and emit them as
  complete numpy arrays for downstream processing.
• Expose an asyncio-friendly async generator AND a synchronous
  callback interface so it works in both sync and async pipelines.

VAD algorithm
─────────────
Energy gate: RMS amplitude of each chunk is compared to a configurable
threshold. A speech segment starts when energy exceeds the threshold
for VOICE_START_CHUNKS consecutive chunks, and ends after VOICE_END_CHUNKS
below-threshold chunks of silence. The complete utterance (pre-roll
included) is then emitted as a single numpy array.

Performance target: < 5 ms overhead per 100 ms chunk on CPU.
"""

from __future__ import annotations

import asyncio
import logging
import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Generator, Iterator

import numpy as np

log = logging.getLogger("jarvis.voice.listener")

# ── Audio constants ────────────────────────────────────────────────────────────
SAMPLE_RATE   = 16_000     # Hz  — Whisper and Vosk both prefer 16 kHz
CHANNELS      = 1          # Mono
DTYPE         = np.float32 # [-1.0, 1.0]  — normalised float
CHUNK_FRAMES  = 1_600      # 100 ms @ 16 kHz

# VAD thresholds
VAD_RMS_THRESHOLD   = 0.018   # RMS energy gate (tune per environment)
VOICE_START_CHUNKS  = 2       # consecutive loud chunks to start utterance
VOICE_END_CHUNKS    = 12      # consecutive quiet chunks to end utterance (1.2 s)
PRE_ROLL_CHUNKS     = 3       # chunks kept before speech starts (300 ms lead-in)
MAX_UTTERANCE_SECS  = 15      # maximum utterance length before forced flush

AudioCallback = Callable[[np.ndarray], None]


# ── Data types ─────────────────────────────────────────────────────────────────

@dataclass
class AudioChunk:
    """A single processed audio frame with energy metadata."""
    data:        np.ndarray          # shape (CHUNK_FRAMES,) float32
    rms:         float               # RMS energy
    is_speech:   bool                # VAD verdict
    timestamp:   float = field(default_factory=time.monotonic)


@dataclass
class Utterance:
    """A complete speech utterance ready for STT."""
    audio:       np.ndarray          # concatenated float32 array
    duration_s:  float               # seconds
    peak_rms:    float               # loudest chunk RMS
    timestamp:   float = field(default_factory=time.time)

    def to_int16(self) -> np.ndarray:
        """Convert to int16 PCM for libraries that require it (Vosk, PyAudio)."""
        clipped = np.clip(self.audio, -1.0, 1.0)
        return (clipped * 32767).astype(np.int16)


# ── Backend detection ──────────────────────────────────────────────────────────

def _detect_backend() -> str:
    """Return "sounddevice" or "pyaudio" depending on what's installed."""
    try:
        import sounddevice  # noqa: F401
        return "sounddevice"
    except ImportError:
        pass
    try:
        import pyaudio  # noqa: F401
        return "pyaudio"
    except ImportError:
        pass
    raise ImportError(
        "No audio backend found. Install one of:\n"
        "  pip install sounddevice\n"
        "  pip install pyaudio"
    )


# ── SpeechListener ─────────────────────────────────────────────────────────────

class SpeechListener:
    """
    Real-time microphone capture with energy-based VAD.

    Usage — callback style (sync):
        listener = SpeechListener()
        listener.on_utterance(lambda u: process(u))
        listener.start()
        # ... later ...
        listener.stop()

    Usage — generator style (sync):
        for utterance in listener.utterances():
            process(utterance)

    Usage — async generator:
        async for utterance in listener.async_utterances():
            await process(utterance)

    Args:
        device:        Microphone device index (None = system default).
        sample_rate:   Target sample rate in Hz.
        chunk_frames:  Frames per audio chunk.
        vad_threshold: RMS energy threshold for voice detection.
        verbose:       Log every chunk's RMS energy (debug).
    """

    def __init__(
        self,
        device:        int | None = None,
        sample_rate:   int   = SAMPLE_RATE,
        chunk_frames:  int   = CHUNK_FRAMES,
        vad_threshold: float = VAD_RMS_THRESHOLD,
        verbose:       bool  = False,
    ) -> None:
        self._device        = device
        self._sample_rate   = sample_rate
        self._chunk_frames  = chunk_frames
        self._vad_threshold = vad_threshold
        self._verbose       = verbose
        self._backend       = _detect_backend()

        # State
        self._running   = False
        self._stream    = None
        self._thread:   threading.Thread | None = None

        # Utterance queue — holds complete Utterance objects
        self._utt_queue: queue.Queue[Utterance] = queue.Queue(maxsize=8)

        # Registered callbacks
        self._callbacks: list[AudioCallback] = []

        # VAD state
        self._vad_state:       str        = "silence"  # "silence" | "voice"
        self._loud_count:      int        = 0
        self._quiet_count:     int        = 0
        self._current_frames:  list[np.ndarray] = []
        self._pre_roll:        list[np.ndarray] = []
        self._peak_rms:        float      = 0.0

        # Metrics
        self._chunks_read      = 0
        self._utterances_emitted = 0

    # ── Lifecycle ──────────────────────────────────────────────────────────────

    def start(self) -> "SpeechListener":
        """Open the microphone and start the capture thread."""
        if self._running:
            log.warning("SpeechListener already running")
            return self
        self._running = True
        self._thread  = threading.Thread(
            target=self._capture_loop,
            name="speech-listener",
            daemon=True,
        )
        self._thread.start()
        log.info("SpeechListener started  backend=%s  sr=%d  device=%s",
                 self._backend, self._sample_rate, self._device)
        return self

    def stop(self) -> None:
        """Stop capturing and close the microphone."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=3.0)
            self._thread = None
        log.info("SpeechListener stopped  utterances=%d", self._utterances_emitted)

    # ── Callback registration ──────────────────────────────────────────────────

    def on_utterance(self, callback: Callable[[Utterance], None]) -> None:
        """Register a function called with each complete utterance."""
        self._callbacks.append(callback)

    # ── Synchronous generator interface ───────────────────────────────────────

    def utterances(self, timeout: float = 0.1) -> Generator[Utterance, None, None]:
        """
        Yield complete utterances as they arrive from the microphone.

        Blocks the calling thread. Call stop() from another thread to exit.

        Args:
            timeout: Queue poll timeout in seconds.
        """
        while self._running:
            try:
                utt = self._utt_queue.get(timeout=timeout)
                yield utt
            except queue.Empty:
                continue

    # ── Async generator interface ──────────────────────────────────────────────

    async def async_utterances(self, poll_interval: float = 0.05) -> "AsyncGenerator[Utterance]":
        """
        Async generator that yields utterances without blocking the event loop.

        Usage:
            async for utterance in listener.async_utterances():
                await handle(utterance)
        """
        while self._running:
            try:
                utt = self._utt_queue.get_nowait()
                yield utt
            except queue.Empty:
                await asyncio.sleep(poll_interval)

    # ── Capture loops ──────────────────────────────────────────────────────────

    def _capture_loop(self) -> None:
        """Main capture thread. Dispatches to the correct backend."""
        try:
            if self._backend == "sounddevice":
                self._capture_sounddevice()
            else:
                self._capture_pyaudio()
        except Exception as exc:
            log.exception("Capture loop crashed: %s", exc)
        finally:
            self._running = False

    def _capture_sounddevice(self) -> None:
        import sounddevice as sd

        def _sd_callback(indata: np.ndarray, frames: int, time_info, status) -> None:
            if status:
                log.debug("sounddevice status: %s", status)
            chunk = indata[:, 0].copy().astype(np.float32)
            self._process_chunk(chunk)

        with sd.InputStream(
            device=self._device,
            samplerate=self._sample_rate,
            channels=CHANNELS,
            dtype="float32",
            blocksize=self._chunk_frames,
            callback=_sd_callback,
        ):
            log.debug("sounddevice stream open")
            while self._running:
                sd.sleep(100)

    def _capture_pyaudio(self) -> None:
        import pyaudio

        pa = pyaudio.PyAudio()
        stream = pa.open(
            format=pyaudio.paFloat32,
            channels=CHANNELS,
            rate=self._sample_rate,
            input=True,
            input_device_index=self._device,
            frames_per_buffer=self._chunk_frames,
        )
        log.debug("PyAudio stream open")
        try:
            while self._running:
                raw = stream.read(self._chunk_frames, exception_on_overflow=False)
                chunk = np.frombuffer(raw, dtype=np.float32).copy()
                self._process_chunk(chunk)
        finally:
            stream.stop_stream()
            stream.close()
            pa.terminate()

    # ── VAD processing ─────────────────────────────────────────────────────────

    def _process_chunk(self, chunk: np.ndarray) -> None:
        """Apply VAD to one audio chunk and accumulate utterance frames."""
        self._chunks_read += 1
        rms = float(np.sqrt(np.mean(chunk ** 2)))

        if self._verbose:
            bar = "█" * int(rms * 200)
            log.debug("RMS %.4f  %s", rms, bar[:40])

        is_speech = rms >= self._vad_threshold

        # Pre-roll buffer (always keep last N chunks for lead-in)
        self._pre_roll.append(chunk)
        if len(self._pre_roll) > PRE_ROLL_CHUNKS:
            self._pre_roll.pop(0)

        if self._vad_state == "silence":
            if is_speech:
                self._loud_count += 1
                if self._loud_count >= VOICE_START_CHUNKS:
                    self._vad_state      = "voice"
                    self._quiet_count    = 0
                    self._peak_rms       = rms
                    # Include pre-roll
                    self._current_frames = list(self._pre_roll)
                    log.debug("VAD: speech START  rms=%.4f", rms)
            else:
                self._loud_count = 0

        elif self._vad_state == "voice":
            self._current_frames.append(chunk)
            self._peak_rms = max(self._peak_rms, rms)

            if not is_speech:
                self._quiet_count += 1
                if self._quiet_count >= VOICE_END_CHUNKS:
                    self._flush_utterance()
            else:
                self._quiet_count = 0

            # Force flush if utterance is too long
            max_frames = int(MAX_UTTERANCE_SECS * self._sample_rate / self._chunk_frames)
            if len(self._current_frames) >= max_frames:
                log.debug("VAD: max length reached — force flush")
                self._flush_utterance()

    def _flush_utterance(self) -> None:
        """Emit the accumulated frames as an Utterance object."""
        if not self._current_frames:
            self._vad_state   = "silence"
            self._loud_count  = 0
            self._quiet_count = 0
            return

        audio    = np.concatenate(self._current_frames)
        duration = len(audio) / self._sample_rate

        utt = Utterance(
            audio=audio,
            duration_s=duration,
            peak_rms=self._peak_rms,
        )

        log.debug("VAD: utterance END  dur=%.2f s  peak=%.4f", duration, self._peak_rms)

        # Reset VAD state
        self._vad_state      = "silence"
        self._loud_count     = 0
        self._quiet_count    = 0
        self._current_frames = []
        self._pre_roll       = []
        self._peak_rms       = 0.0
        self._utterances_emitted += 1

        # Dispatch
        self._dispatch(utt)

    def _dispatch(self, utt: Utterance) -> None:
        """Send utterance to queue and registered callbacks."""
        try:
            self._utt_queue.put_nowait(utt)
        except queue.Full:
            log.warning("Utterance queue full — dropping utterance")

        for cb in self._callbacks:
            try:
                cb(utt)
            except Exception as exc:
                log.exception("Utterance callback raised: %s", exc)

    # ── Diagnostics ────────────────────────────────────────────────────────────

    @property
    def is_running(self) -> bool:
        return self._running

    def stats(self) -> dict:
        return {
            "backend":             self._backend,
            "chunks_read":         self._chunks_read,
            "utterances_emitted":  self._utterances_emitted,
            "vad_state":           self._vad_state,
            "queue_size":          self._utt_queue.qsize(),
        }

    def list_devices(self) -> list[dict]:
        """Return available input devices for the active backend."""
        if self._backend == "sounddevice":
            import sounddevice as sd
            devices = sd.query_devices()
            return [
                {"index": i, "name": d["name"], "channels": d["max_input_channels"]}
                for i, d in enumerate(devices)
                if d["max_input_channels"] > 0
            ]
        return []

    # ── Context manager ────────────────────────────────────────────────────────

    def __enter__(self) -> "SpeechListener":
        return self.start()

    def __exit__(self, *_) -> None:
        self.stop()
