"""
app/api/routes.py
All REST endpoints for JARVIS 2.0.

POST /chat              — primary text input
POST /voice-command     — voice-transcribed command
POST /gesture-command   — MediaPipe gesture event
GET  /health            — liveness probe
GET  /status            — full system diagnostics
GET  /memory/history    — conversation history for a session
GET  /memory/commands   — recent command log
POST /memory/preference — set a user preference
GET  /metrics           — observer metrics
"""

from __future__ import annotations

import re
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

router = APIRouter()


# ── Schemas ────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    message:    str = Field(..., min_length=1, max_length=8000)
    session_id: str | None = None
    metadata:   dict = Field(default_factory=dict)

    @field_validator("message")
    @classmethod
    def not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("message must not be blank")
        return v.strip()


class VoiceRequest(BaseModel):
    text:       str   = Field(..., min_length=1, max_length=4000)
    session_id: str | None = None
    confidence: float = Field(1.0, ge=0.0, le=1.0)
    source:     str   = "voice"


class GestureRequest(BaseModel):
    gesture_id: str = Field(..., min_length=1, max_length=64)
    session_id: str | None = None
    confidence: float = Field(1.0, ge=0.0, le=1.0)
    landmarks:  list[dict] = Field(default_factory=list)
    metadata:   dict = Field(default_factory=dict)

    @field_validator("gesture_id")
    @classmethod
    def safe_id(cls, v: str) -> str:
        if not re.match(r"^[a-zA-Z0-9_\-]+$", v):
            raise ValueError("gesture_id must be alphanumeric + _ or -")
        return v.lower()


class PreferenceRequest(BaseModel):
    key:   str
    value: object


# ── Dependency ─────────────────────────────────────────────────────────

def _get_core():
    from app.core.jarvis_core import jarvis
    if jarvis is None:
        raise HTTPException(status_code=503, detail="JARVIS core initialising — retry shortly")
    return jarvis


def _get_db():
    from app.memory.database import db
    return db


# ── Endpoints ──────────────────────────────────────────────────────────

@router.get("/health")
async def health():
    """Instant liveness probe — no model call."""
    try:
        core = _get_core()
        h = await core.health()
        return h
    except HTTPException:
        return {"status": "starting", "model_loaded": False}


@router.get("/status")
async def status():
    """Full system diagnostics."""
    core = _get_core()
    return await core.status()


@router.post("/chat")
async def chat(req: ChatRequest):
    """Primary text input — runs the full Observe→Think→Decide→Act→Learn loop."""
    core = _get_core()
    resp = await core.process(
        text=req.message,
        session_id=req.session_id,
        source="text",
        metadata=req.metadata,
    )
    return JSONResponse(resp.to_dict())


@router.post("/voice-command")
async def voice_command(req: VoiceRequest):
    """Voice-transcribed command — same pipeline as /chat, tagged as voice."""
    core = _get_core()
    resp = await core.process(
        text=req.text,
        session_id=req.session_id,
        source="voice",
        metadata={"confidence": req.confidence, "source": req.source},
    )
    return JSONResponse(resp.to_dict())


@router.post("/gesture-command")
async def gesture_command(req: GestureRequest):
    """MediaPipe gesture event — dispatches to gesture pipeline."""
    core = _get_core()
    resp = await core.process(
        text=req.gesture_id,
        session_id=req.session_id,
        source="gesture",
        metadata={
            "confidence": req.confidence,
            "landmarks":  req.landmarks,
            **req.metadata,
        },
    )
    return JSONResponse(resp.to_dict())


@router.get("/memory/history")
async def memory_history(session_id: str, limit: int = 20):
    """Return recent conversation history for a session."""
    db = _get_db()
    history = await db.get_history(session_id, limit=limit)
    return {"session_id": session_id, "history": history, "count": len(history)}


@router.get("/memory/commands")
async def memory_commands(session_id: str, limit: int = 10):
    """Return recent command log for a session."""
    db = _get_db()
    commands = await db.get_recent_commands(session_id, limit=limit)
    return {"session_id": session_id, "commands": commands}


@router.post("/memory/preference")
async def set_preference(req: PreferenceRequest):
    """Set a persistent user preference."""
    db = _get_db()
    await db.set_preference(req.key, req.value)
    return {"status": "saved", "key": req.key}


@router.get("/metrics")
async def metrics():
    """Observer agent performance metrics."""
    try:
        from app.core.jarvis_core import jarvis
        if jarvis and jarvis._router._observer:
            return jarvis._router._observer.metrics()
    except Exception:
        pass
    return {"status": "metrics unavailable"}
