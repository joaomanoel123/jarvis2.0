"""
backend/main.py
JARVIS 2.0 — FastAPI backend server.

Endpoints:
    POST /chat             — main text/voice/gesture entry point
    POST /voice-command    — voice-specific entry point
    POST /gesture-command  — gesture-specific entry point
    GET  /health           — liveness probe
    GET  /status           — system diagnostics

Start:
    uvicorn main:app --host 0.0.0.0 --port 8000 --reload
"""

import asyncio
import logging
import os
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("jarvis.main")

ALLOWED_ORIGINS = os.getenv(
    "ALLOWED_ORIGINS",
    "http://localhost:3000,http://localhost:5500,http://127.0.0.1:5500,"
    "https://joaomanoel123.github.io",
).split(",")


# ── Lifespan ───────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    from llm_service  import llm_service
    from jarvis_core  import jarvis_core

    log.info("═══ JARVIS 2.0 starting ═══")
    loop = asyncio.get_running_loop()

    # Load LLM in background thread (non-blocking)
    await loop.run_in_executor(None, llm_service.load)
    log.info("═══ JARVIS ready ═══")

    yield

    log.info("JARVIS shutting down")


# ── App factory ────────────────────────────────────────────────────────

app = FastAPI(
    title="JARVIS 2.0 API",
    version="2.0.0",
    description="Multi-agent AI assistant backend",
    lifespan=lifespan,
    docs_url="/docs",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
)


# ── Schemas ────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    message:    str = Field(..., min_length=1, max_length=8000)
    session_id: str | None = None

    @field_validator("message")
    @classmethod
    def not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("message must not be blank")
        return v.strip()


class VoiceRequest(BaseModel):
    text:       str = Field(..., min_length=1, max_length=4000)
    session_id: str | None = None
    confidence: float = 1.0


class GestureRequest(BaseModel):
    gesture_id: str = Field(..., min_length=1, max_length=64)
    session_id: str | None = None
    confidence: float = 1.0

    @field_validator("gesture_id")
    @classmethod
    def safe_id(cls, v: str) -> str:
        import re
        if not re.match(r"^[a-zA-Z0-9_\-]+$", v):
            raise ValueError("gesture_id must be alphanumeric with _ or -")
        return v.lower()


# ── Routes ─────────────────────────────────────────────────────────────

@app.get("/health", tags=["Infrastructure"])
async def health():
    """Liveness probe — instant, no model call."""
    from llm_service import llm_service
    info = llm_service.info()
    return {
        "status":       "ok",
        "model_loaded": info["loaded"],
        "version":      "2.0.0",
        "uptime_s":     round(time.time(), 1),
    }


@app.get("/status", tags=["Infrastructure"])
async def status():
    """Full system diagnostics."""
    from llm_service import llm_service
    from jarvis_core import jarvis_core
    return {
        "status":   "operational",
        "model":    llm_service.info(),
        "sessions": jarvis_core.session_count(),
    }


@app.post("/chat", tags=["Agents"])
async def chat(req: ChatRequest):
    """
    Main entry point. Accepts text, classifies intent,
    routes to LLM or command executor, returns structured response.
    """
    from jarvis_core import jarvis_core
    try:
        resp = await jarvis_core.process(
            text=req.message,
            session_id=req.session_id,
            source="text",
        )
        return JSONResponse(resp.to_dict())
    except Exception as exc:
        log.exception("Chat error: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/voice-command", tags=["Voice"])
async def voice_command(req: VoiceRequest):
    """
    Voice-transcribed command. Treated identically to /chat
    but tagged as voice source.
    """
    from jarvis_core import jarvis_core
    try:
        resp = await jarvis_core.process(
            text=req.text,
            session_id=req.session_id,
            source="voice",
        )
        return JSONResponse(resp.to_dict())
    except Exception as exc:
        log.exception("Voice error: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/gesture-command", tags=["Gesture"])
async def gesture_command(req: GestureRequest):
    """
    MediaPipe gesture event — maps gesture_id to an action.
    """
    from jarvis_core import jarvis_core
    try:
        resp = await jarvis_core.process(
            text=req.gesture_id,   # gesture IDs are matched by intent_detector
            session_id=req.session_id,
            source="gesture",
        )
        return JSONResponse(resp.to_dict())
    except Exception as exc:
        log.exception("Gesture error: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/voice/commands", tags=["Voice"])
async def voice_commands():
    """List supported voice commands."""
    from intent_detector import _URL_MAP
    return {
        "sites":   sorted(_URL_MAP.keys()),
        "actions": ["open", "search", "play", "set volume"],
    }
