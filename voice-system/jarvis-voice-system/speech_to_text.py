"""
speech_to_text.py
=================
SpeechToText — multi-backend speech recognition engine.

Supported backends (in order of preference)
────────────────────────────────────────────
  1. whisper_local  — OpenAI Whisper running on-device (GPU or CPU).
                      Best accuracy, works offline.
                      Model sizes: tiny, base, small, medium, large.

  2. whisper_api    — OpenAI Whisper via API call (requires API key).
                      Accurate, cloud-hosted, small binary.

  3. google         — Google Speech Recognition via SpeechRecognition lib.
                      Good accuracy, free tier, requires internet.

  4. vosk           — Vosk offline recogniser.
                      Fully offline, lower accuracy, fast on CPU.

Backend is selected at construction time with automatic fallback:
  If "whisper_local" fails to load, falls back to "google".

Output
──────
    STTResult(text, confidence, backend, duration_s, language)
"""

from __future__ import annotations

import io
import logging
import time
import tempfile
import os
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from speech_listener import Utterance, SAMPLE_RATE

log = logging.getLogger("jarvis.voice.stt")

# Supported language codes (ISO 639-1)
DEFAULT_LANGUAGE = "en"

# Whisper model sizes in increasing accuracy/cost order
WHISPER_MODELS = ["tiny", "base", "small", "medium", "large"]


# ── Result type ────────────────────────────────────────────────────────────────

@dataclass
class STTResult:
    """Output of the speech-to-text engine."""
    text:        str
    confidence:  float          # [0.0–1.0]; 1.0 when backend doesn't report it
    backend:     str
    duration_s:  float
    language:    str = DEFAULT_LANGUAGE
    raw:         Any = field(default=None, repr=False)

    @property
    def clean_text(self) -> str:
        """Lowercased, stripped text without punctuation artifacts."""
        import re
        t = self.text.strip().lower()
        t = re.sub(r"[^\w\s'-]", "", t)
        return t.strip()

    def is_empty(self) -> bool:
        return not self.clean_text


# ── SpeechToText ───────────────────────────────────────────────────────────────

class SpeechToText:
    """
    Converts a speech Utterance into text using one of several backends.

    Args:
        backend:       "whisper_local" | "whisper_api" | "google" | "vosk" | "auto"
                       "auto" tries backends in the preference order above.
        model_size:    Whisper model size (tiny, base, small, medium, large).
        language:      BCP-47 language code hint (e.g. "en", "pt", "es").
        device:        Whisper device ("cpu", "cuda", "mps"). None = auto.
        vosk_model:    Path to Vosk model directory.
        openai_api_key: API key for whisper_api backend.
    """

    def __init__(
        self,
        backend:        str = "auto",
        model_size:     str = "base",
        language:       str = DEFAULT_LANGUAGE,
        device:         str | None = None,
        vosk_model:     str | None = None,
        openai_api_key: str | None = None,
    ) -> None:
        self._requested_backend = backend
        self._model_size   = model_size
        self._language     = language
        self._device       = device
        self._vosk_model   = vosk_model
        self._openai_key   = openai_api_key or os.getenv("OPENAI_API_KEY", "")

        self._backend:    str | None = None
        self._model:      Any | None = None
        self._recognizer: Any | None = None

        # Metrics
        self._calls   = 0
        self._errors  = 0
        self._total_s = 0.0

    # ── Initialisation ─────────────────────────────────────────────────────────

    def load(self) -> "SpeechToText":
        """Load the STT model / backend. Call once at startup."""
        if self._requested_backend == "auto":
            for backend in ["whisper_local", "google", "vosk"]:
                if self._try_load(backend):
                    break
        else:
            if not self._try_load(self._requested_backend):
                # Fallback
                log.warning("Requested backend '%s' unavailable — falling back to google",
                            self._requested_backend)
                self._try_load("google")

        if self._backend is None:
            raise RuntimeError("No STT backend could be initialised")

        log.info("SpeechToText ready  backend=%s  language=%s", self._backend, self._language)
        return self

    def _try_load(self, backend: str) -> bool:
        try:
            if backend == "whisper_local":
                return self._load_whisper_local()
            elif backend == "whisper_api":
                return self._load_whisper_api()
            elif backend == "google":
                return self._load_google()
            elif backend == "vosk":
                return self._load_vosk()
        except Exception as exc:
            log.debug("Backend '%s' load failed: %s", backend, exc)
        return False

    def _load_whisper_local(self) -> bool:
        import whisper
        device = self._device or ("cuda" if _cuda_available() else "cpu")
        log.info("Loading Whisper '%s' on %s …", self._model_size, device)
        self._model   = whisper.load_model(self._model_size, device=device)
        self._backend = "whisper_local"
        log.info("Whisper '%s' loaded", self._model_size)
        return True

    def _load_whisper_api(self) -> bool:
        if not self._openai_key:
            log.debug("No OPENAI_API_KEY — skipping whisper_api")
            return False
        import openai  # noqa: F401
        self._backend = "whisper_api"
        log.info("Whisper API backend ready")
        return True

    def _load_google(self) -> bool:
        import speech_recognition as sr
        self._recognizer = sr.Recognizer()
        self._backend    = "google"
        log.info("Google STT backend ready")
        return True

    def _load_vosk(self) -> bool:
        from vosk import Model, KaldiRecognizer
        model_path = self._vosk_model or f"vosk-model-{self._language}"
        if not os.path.isdir(model_path):
            log.debug("Vosk model not found at '%s'", model_path)
            return False
        self._model   = Model(model_path)
        self._backend = "vosk"
        log.info("Vosk model loaded from %s", model_path)
        return True

    # ── Transcription ──────────────────────────────────────────────────────────

    def transcribe(self, utterance: Utterance) -> STTResult:
        """
        Convert an Utterance to text.

        Args:
            utterance: Complete speech utterance from SpeechListener.

        Returns:
            STTResult with the transcribed text.
        """
        t0 = time.perf_counter()
        self._calls += 1

        try:
            if self._backend == "whisper_local":
                result = self._transcribe_whisper(utterance)
            elif self._backend == "whisper_api":
                result = self._transcribe_whisper_api(utterance)
            elif self._backend == "google":
                result = self._transcribe_google(utterance)
            elif self._backend == "vosk":
                result = self._transcribe_vosk(utterance)
            else:
                raise RuntimeError(f"Unknown backend: {self._backend}")

            elapsed = time.perf_counter() - t0
            self._total_s += elapsed
            log.debug("STT: %r  (%.2f s, backend=%s)", result.text, elapsed, self._backend)
            return result

        except Exception as exc:
            self._errors += 1
            log.error("STT error (%s): %s", self._backend, exc)
            return STTResult(
                text="", confidence=0.0,
                backend=self._backend or "unknown",
                duration_s=utterance.duration_s,
            )

    # ── Backend implementations ────────────────────────────────────────────────

    def _transcribe_whisper(self, utt: Utterance) -> STTResult:
        """Transcribe using local Whisper model."""
        import whisper

        # Whisper expects float32 mono at 16kHz — already in that format
        audio = utt.audio.astype(np.float32)

        # Pad or trim to 30 s (Whisper's window)
        audio = whisper.pad_or_trim(audio)

        options = whisper.DecodingOptions(
            language=self._language if self._language != "auto" else None,
            without_timestamps=True,
            fp16=False,
        )
        mel    = whisper.log_mel_spectrogram(audio).to(self._model.device)
        result = whisper.decode(self._model, mel, options)

        # Confidence: average log-prob ≈ exp(avg_logprob)
        import math
        conf = round(math.exp(max(result.avg_logprob, -5)), 3)

        return STTResult(
            text=result.text,
            confidence=conf,
            backend="whisper_local",
            duration_s=utt.duration_s,
            language=result.language or self._language,
            raw=result,
        )

    def _transcribe_whisper_api(self, utt: Utterance) -> STTResult:
        """Transcribe via OpenAI Whisper API."""
        import openai

        client = openai.OpenAI(api_key=self._openai_key)

        # Write audio to a temporary WAV file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            _write_wav(tmp.name, utt.to_int16(), SAMPLE_RATE)
            tmp_path = tmp.name

        try:
            with open(tmp_path, "rb") as f:
                response = client.audio.transcriptions.create(
                    model="whisper-1",
                    file=f,
                    language=self._language if self._language != "auto" else None,
                )
            text = response.text
        finally:
            os.unlink(tmp_path)

        return STTResult(
            text=text,
            confidence=0.9,  # API doesn't expose confidence
            backend="whisper_api",
            duration_s=utt.duration_s,
            raw=response,
        )

    def _transcribe_google(self, utt: Utterance) -> STTResult:
        """Transcribe using Google Speech Recognition."""
        import speech_recognition as sr

        audio_data = sr.AudioData(
            utt.to_int16().tobytes(),
            sample_rate=SAMPLE_RATE,
            sample_width=2,  # int16 = 2 bytes
        )

        lang_map = {"en": "en-US", "pt": "pt-BR", "es": "es-ES", "fr": "fr-FR"}
        lang = lang_map.get(self._language, f"{self._language}-{self._language.upper()}")

        try:
            text = self._recognizer.recognize_google(
                audio_data,
                language=lang,
                show_all=False,
            )
            confidence = 0.85  # Google basic API doesn't return confidence
        except sr.UnknownValueError:
            text       = ""
            confidence = 0.0
        except sr.RequestError as exc:
            log.warning("Google STT request failed: %s", exc)
            text       = ""
            confidence = 0.0

        return STTResult(
            text=text,
            confidence=confidence,
            backend="google",
            duration_s=utt.duration_s,
        )

    def _transcribe_vosk(self, utt: Utterance) -> STTResult:
        """Transcribe using local Vosk model."""
        import json
        from vosk import KaldiRecognizer

        rec = KaldiRecognizer(self._model, SAMPLE_RATE)
        pcm = utt.to_int16().tobytes()

        # Feed audio in chunks
        chunk_size = 4000
        for i in range(0, len(pcm), chunk_size):
            rec.AcceptWaveform(pcm[i:i + chunk_size])

        result  = json.loads(rec.FinalResult())
        text    = result.get("text", "")
        conf    = result.get("confidence", 0.8) if text else 0.0

        return STTResult(
            text=text,
            confidence=conf,
            backend="vosk",
            duration_s=utt.duration_s,
            raw=result,
        )

    # ── Convenience: transcribe from numpy array directly ──────────────────────

    def transcribe_array(self, audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> STTResult:
        """
        Transcribe a raw numpy audio array without a full Utterance wrapper.

        Useful for testing or one-shot transcription.
        """
        utt = Utterance(
            audio=audio.astype(np.float32),
            duration_s=len(audio) / sample_rate,
            peak_rms=float(np.sqrt(np.mean(audio ** 2))),
        )
        return self.transcribe(utt)

    # ── Diagnostics ────────────────────────────────────────────────────────────

    @property
    def backend(self) -> str | None:
        return self._backend

    def stats(self) -> dict:
        avg = round(self._total_s / self._calls, 3) if self._calls else 0.0
        return {
            "backend":      self._backend,
            "model_size":   self._model_size,
            "language":     self._language,
            "calls":        self._calls,
            "errors":       self._errors,
            "avg_latency_s": avg,
        }


# ── Helpers ────────────────────────────────────────────────────────────────────

def _cuda_available() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def _write_wav(path: str, pcm_int16: np.ndarray, sample_rate: int) -> None:
    """Write int16 PCM data to a WAV file."""
    import struct
    import wave
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)   # 16-bit
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_int16.tobytes())
