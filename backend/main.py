"""
backend/main.py
===============
Unified FastAPI application for JARVIS 2.0.

Endpoints
─────────
  POST  /chat                — text conversation
  POST  /voice-command       — pre-transcribed voice command
  POST  /gesture-command     — MediaPipe gesture event
  GET   /health              — liveness probe (no model call)
  GET   /status              — full system diagnostics
  WS    /ws/gestures         — real-time gesture stream (from gesture-system)
  WS    /ws/ui               — holographic UI notifications (to frontend)
  GET   /voice/commands      — list supported voice commands
  GET   /voice/status        — voice system statistics

Startup sequence (lifespan)
────────────────────────────
  1. Load LLM model into ThreadPoolExecutor.
  2. Load vector memory (SentenceTransformer + FAISS).
  3. Initialise JarvisCore (wires all agents + memory).
  4. Register tools with ToolService.
  5. Start session eviction background task.
  6. Register WebSocket connection manager.

Deploy on Render
────────────────
  Start command: uvicorn main:app --host 0.0.0.0 --port 10000
  Health check:  /health
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.DEBUG if os.getenv("DEBUG", "false").lower() == "true" else logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("jarvis.main")

APP_VERSION  = "2.0.0"
APP_NAME     = os.getenv("APP_NAME", "JARVIS")
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000,https://joaomanoel123.github.io").split(",")

# Module-level singletons — populated during lifespan startup
_core         = None   # JarvisCore
_ws_manager   = None   # WebSocket connection manager

# ── Lifespan ───────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup / shutdown."""
    global _core, _ws_manager
    t0 = time.perf_counter()
    log.info("═══ %s %s starting ═══", APP_NAME, APP_VERSION)

    # 1 — Load LLM
    loop = asyncio.get_running_loop()
    try:
        from services.llm_service import llm_service
        log.info("Loading LLM model …")
        await loop.run_in_executor(None, llm_service.load)
        log.info("✓ LLM loaded (%.1f s)", time.perf_counter() - t0)
    except Exception as exc:
        log.error("LLM load failed: %s — API will return 503", exc)

    # 2 — Load vector memory
    try:
        from memory.vector_memory import vector_memory
        await vector_memory.load()
        log.info("✓ Vector memory loaded")
    except Exception as exc:
        log.warning("Vector memory unavailable: %s", exc)

    # 3 — Build JarvisCore
    try:
        from jarvis_core import JarvisCore
        _core = JarvisCore.build()
        log.info("✓ JarvisCore ready")
    except Exception as exc:
        log.error("JarvisCore build failed: %s", exc)

    # 4 — Register tools
    try:
        from services.tool_service import tool_service
        tool_service.register_all_defaults()
        log.info("✓ Tools registered: %s", tool_service.list_allowed())
    except Exception as exc:
        log.warning("Tool registration failed: %s", exc)

    # 5 — Session eviction background task
    eviction_task = asyncio.create_task(_eviction_loop(), name="session-eviction")

    # 6 — WebSocket manager
    from _ws_manager_impl import ConnectionManager
    _ws_manager = ConnectionManager()

    log.info("═══ %s ready (%.1f s) ═══", APP_NAME, time.perf_counter() - t0)

    yield   # ← FastAPI serves requests

    # Shutdown
    log.info("Shutting down …")
    eviction_task.cancel()
    try:
        await eviction_task
    except asyncio.CancelledError:
        pass
    log.info("Stopped cleanly")


async def _eviction_loop(interval: int = 300) -> None:
    """Evict expired sessions every 5 minutes."""
    while True:
        await asyncio.sleep(interval)
        try:
            from memory.conversation_memory import conversation_memory
            n = await conversation_memory.evict_expired()
            if n:
                log.info("Evicted %d expired sessions", n)
        except Exception:
            pass


# ── Application factory ────────────────────────────────────────────────────────

def create_app() -> FastAPI:
    app = FastAPI(
        title=APP_NAME,
        version=APP_VERSION,
        description=(
            f"{APP_NAME} 2.0 — Multi-Agent AI Backend.\n"
            "Endpoints: /chat  /voice-command  /gesture-command  /ws/gestures  /ws/ui"
        ),
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=ALLOWED_ORIGINS,
        allow_credentials=True,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type", "Authorization", "X-Session-ID"],
    )

    @app.middleware("http")
    async def timing_middleware(request: Request, call_next):
        t0   = time.perf_counter()
        resp = await call_next(request)
        ms   = round((time.perf_counter() - t0) * 1000, 1)
        resp.headers["X-Process-Time-Ms"] = str(ms)
        return resp

    @app.exception_handler(Exception)
    async def global_error(request: Request, exc: Exception):
        log.exception("Unhandled: %s %s", request.method, request.url)
        debug = os.getenv("DEBUG", "false").lower() == "true"
        return JSONResponse(
            status_code=500,
            content={"error": "Internal server error",
                     "detail": str(exc) if debug else "An unexpected error occurred."},
        )

    app.include_router(_build_router())
    return app


# ── Pydantic schemas ───────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    message:    str = Field(..., min_length=1, max_length=8000)
    session_id: str | None = None
    metadata:   dict[str, Any] = Field(default_factory=dict)

    @field_validator("message")
    @classmethod
    def not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("message must not be blank")
        return v.strip()


class ChatResponse(BaseModel):
    session_id: str
    text:       str
    intent:     str
    agent_path: list[str]
    tool_calls: list[dict]
    steps:      list[str]
    latency_ms: float
    model_id:   str | None = None
    success:    bool
    error:      str | None = None


class VoiceCommandRequest(BaseModel):
    text:       str = Field(..., min_length=1, max_length=4000)
    session_id: str | None = None
    confidence: float = Field(1.0, ge=0.0, le=1.0)
    source:     str   = "voice"


class GestureCommandRequest(BaseModel):
    gesture_id: str = Field(..., min_length=1, max_length=64)
    session_id: str | None = None
    confidence: float = Field(1.0, ge=0.0, le=1.0)
    landmarks:  list[dict] = Field(default_factory=list)
    active_widget: str | None = None
    coordinates:   dict | None = None
    metadata:      dict = Field(default_factory=dict)

    @field_validator("gesture_id")
    @classmethod
    def safe_gesture_id(cls, v: str) -> str:
        import re
        if not re.match(r"^[a-zA-Z0-9_\-]+$", v):
            raise ValueError("gesture_id must be alphanumeric with _ or -")
        return v.lower()


class HealthResponse(BaseModel):
    status:       str
    model_loaded: bool
    uptime_s:     float
    version:      str


# ── Routes ─────────────────────────────────────────────────────────────────────

def _core_or_503():
    if _core is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="JARVIS core not initialised — check startup logs.",
        )
    return _core


def _build_router():
    from fastapi import APIRouter
    router = APIRouter()

    # ── Health ──────────────────────────────────────────────────────────────────
    @router.get("/health", response_model=HealthResponse, tags=["Infrastructure"])
    async def health():
        """Liveness probe — instant, no model call."""
        if _core:
            h = await _core.health()
        else:
            try:
                from services.llm_service import llm_service
                loaded = llm_service.info().get("loaded", False)
            except Exception:
                loaded = False
            h = {"status": "ok", "model_loaded": loaded, "uptime_s": 0.0}
        return HealthResponse(status=h["status"], model_loaded=h["model_loaded"],
                               uptime_s=h["uptime_s"], version=APP_VERSION)

    # ── Status ──────────────────────────────────────────────────────────────────
    @router.get("/status", tags=["Infrastructure"])
    async def full_status():
        """Full system diagnostics."""
        core = _core_or_503()
        return await core.status()

    # ── Chat ────────────────────────────────────────────────────────────────────
    @router.post("/chat", response_model=ChatResponse, tags=["Agents"])
    async def chat(req: ChatRequest):
        """
        Send a text message to JARVIS.

        Returns the agent response with full pipeline metadata.
        Pass the returned `session_id` in subsequent requests to maintain context.
        """
        core = _core_or_503()
        resp = await core.chat(
            text=req.message,
            session_id=req.session_id,
            metadata=req.metadata,
        )
        return ChatResponse(
            session_id=resp.session_id,
            text=resp.text,
            intent=resp.intent,
            agent_path=resp.agent_path,
            tool_calls=resp.tool_calls,
            steps=resp.steps,
            latency_ms=resp.latency_ms,
            model_id=resp.metadata.get("model_id"),
            success=resp.success,
            error=resp.error,
        )

    # ── Voice command ───────────────────────────────────────────────────────────
    @router.post("/voice-command", tags=["Voice"])
    async def voice_command(req: VoiceCommandRequest):
        """
        Process a voice-transcribed command.

        The voice-system (speech_listener + speech_to_text + wake_word_detector)
        transcribes audio and strips the wake word, then sends it here.
        """
        core = _core_or_503()
        resp = await core.voice_command(
            text=req.text,
            session_id=req.session_id,
            confidence=req.confidence,
            source=req.source,
        )
        return resp.to_dict()

    # ── Gesture command ─────────────────────────────────────────────────────────
    @router.post("/gesture-command", tags=["Gesture"])
    async def gesture_command(req: GestureCommandRequest):
        """
        Process a MediaPipe gesture event.

        The gesture-system (camera → MediaPipe → classifier → interpreter)
        sends the final confirmed gesture here.
        """
        core = _core_or_503()
        ctx = {}
        if req.active_widget:
            ctx["active_widget"] = req.active_widget
        if req.coordinates:
            ctx["coordinates"] = req.coordinates
        ctx.update(req.metadata)

        resp = await core.gesture_command(
            gesture_id=req.gesture_id,
            session_id=req.session_id,
            confidence=req.confidence,
            landmarks=req.landmarks,
            context=ctx,
        )
        return resp.to_dict()

    # ── Voice info ──────────────────────────────────────────────────────────────
    @router.get("/voice/commands", tags=["Voice"])
    async def voice_commands():
        """List all supported voice command intents, sites, and apps."""
        try:
            import sys, os
            vs_path = os.path.join(os.path.dirname(__file__), "..", "voice-system")
            if vs_path not in sys.path:
                sys.path.insert(0, vs_path)
            from command_parser import CommandParser, _URL_MAP, _APP_MAP, _INTENT_TABLE
            return {
                "intents": sorted(set(x[0] for x in _INTENT_TABLE)),
                "sites":   sorted(_URL_MAP.keys()),
                "apps":    sorted(_APP_MAP.keys()),
            }
        except ImportError:
            return {"error": "voice-system not installed"}

    @router.get("/voice/status", tags=["Voice"])
    async def voice_status():
        return {"voice_system": "available", "version": APP_VERSION}

    # ── WebSocket ───────────────────────────────────────────────────────────────
    @router.websocket("/ws/gestures")
    async def ws_gestures(ws: WebSocket):
        """
        Real-time WebSocket for the gesture-system client.
        Receives gesture events and routes them through JarvisCore.
        """
        import json
        await ws.accept()
        client_id = f"gesture-{id(ws)}"
        log.info("WS gesture client connected: %s", client_id)
        try:
            while True:
                raw = await ws.receive_text()
                try:
                    msg = json.loads(raw)
                except json.JSONDecodeError:
                    await ws.send_json({"type": "error", "message": "Invalid JSON"})
                    continue

                if msg.get("type") == "gesture" and _core:
                    resp = await _core.gesture_command(
                        gesture_id=msg.get("gesture", ""),
                        session_id=msg.get("session_id"),
                        confidence=float(msg.get("confidence", 1.0)),
                        landmarks=msg.get("landmarks", []),
                    )
                    await ws.send_json({
                        "type":       "gesture_response",
                        "gesture":    msg.get("gesture"),
                        "response":   resp.text,
                        "intent":     resp.intent,
                        "session_id": resp.session_id,
                        "latency_ms": resp.latency_ms,
                    })
                    await _broadcast_ui(resp)

                elif msg.get("type") == "pong":
                    continue
                else:
                    await ws.send_json({"type": "ack", "ts": time.time()})

        except WebSocketDisconnect:
            log.info("WS gesture client disconnected: %s", client_id)

    @router.websocket("/ws/ui")
    async def ws_ui(ws: WebSocket):
        """
        WebSocket for the Three.js holographic interface.
        Receives gesture_response broadcasts and status updates.
        """
        await ws.accept()
        if _ws_manager:
            _ws_manager.add_ui(ws)
        log.info("WS UI client connected")
        try:
            await ws.send_json({
                "type":    "connected",
                "message": f"{APP_NAME} {APP_VERSION} — holographic interface connected",
                "ts":      time.time(),
            })
            while True:
                try:
                    raw = await asyncio.wait_for(ws.receive_text(), timeout=30.0)
                    import json
                    msg = json.loads(raw)
                    if msg.get("type") == "pong":
                        continue
                    if msg.get("type") == "status" and _core:
                        s = await _core.status()
                        await ws.send_json({"type": "status_response", **s})
                except asyncio.TimeoutError:
                    await ws.send_json({"type": "ping", "ts": time.time()})
                except Exception:
                    break
        except WebSocketDisconnect:
            log.info("WS UI client disconnected")
        finally:
            if _ws_manager:
                _ws_manager.remove_ui(ws)

    return router


async def _broadcast_ui(resp) -> None:
    """Broadcast a CoreResponse to all connected UI WebSocket clients."""
    if _ws_manager:
        await _ws_manager.broadcast({
            "type":       "jarvis_update",
            "text":       resp.text,
            "intent":     resp.intent,
            "agent_path": resp.agent_path,
            "session_id": resp.session_id,
            "ts":         time.time(),
        })


# ── Simple in-process WS manager ──────────────────────────────────────────────

class _WSManagerImpl:
    """Minimal in-process WebSocket manager (no Redis pub/sub needed)."""

    def __init__(self) -> None:
        self._ui_clients: set = set()

    def add_ui(self, ws: WebSocket) -> None:
        self._ui_clients.add(ws)

    def remove_ui(self, ws: WebSocket) -> None:
        self._ui_clients.discard(ws)

    async def broadcast(self, payload: dict) -> None:
        dead = set()
        for ws in list(self._ui_clients):
            try:
                await ws.send_json(payload)
            except Exception:
                dead.add(ws)
        self._ui_clients -= dead


# Inject into the module namespace so lifespan can import it
import sys as _sys
import types as _types
_mod = _types.ModuleType("_ws_manager_impl")
_mod.ConnectionManager = _WSManagerImpl
_sys.modules["_ws_manager_impl"] = _mod


# Module-level app — imported by uvicorn
app = create_app()
