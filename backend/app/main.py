"""
app/main.py
===========
FastAPI application factory for JARVIS v2.

Startup sequence (via lifespan context manager)
────────────────────────────────────────────────
  1. Load HuggingFace model in a thread (non-blocking).
  2. Bootstrap the tool registry (register all tools).
  3. Start the session eviction background task.

The `app` variable at module level is what uvicorn imports.
"""

from __future__ import annotations

import asyncio
import logging
import logging.config
import time
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.routes import router
from config.settings import get_settings
from memory.memory_manager import memory
from services.llm_service import llm_service
from tools import bootstrap_tools  # noqa: bootstrap_tools = register_all_tools

cfg = get_settings()

# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.DEBUG if cfg.DEBUG else logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("jarvis.main")


# ── Background tasks ──────────────────────────────────────────────────────────

async def _eviction_loop(interval: int = 300) -> None:
    """Evict expired sessions every `interval` seconds."""
    while True:
        await asyncio.sleep(interval)
        evicted = await memory.evict_expired_sessions()
        if evicted:
            log.info("Session eviction: removed %d expired session(s)", evicted)


# ── Lifespan ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Runs once at startup and once at shutdown.

    Startup
    ───────
    1. Load model in the LLM thread pool (blocks that pool, not the event loop).
    2. Register all tools into the ToolRegistry.
    3. Launch the background session eviction task.

    Shutdown
    ────────
    • Cancel the eviction task gracefully.
    """
    t0 = time.perf_counter()
    log.info("═══ JARVIS %s starting ═══", cfg.APP_VERSION)

    # 1 — Model load (offloaded to the LLM executor so the event loop stays free)
    loop = asyncio.get_running_loop()
    log.info("Loading model: %s …", cfg.LLM_MODEL_ID)
    try:
        await loop.run_in_executor(None, llm_service.load)
        log.info("✓ Model loaded (%.1f s)", time.perf_counter() - t0)
    except Exception as exc:
        log.error("Model load FAILED: %s — API will return 503 until resolved", exc)

    # 2 — Tool registry
    bootstrap_tools()
    log.info("✓ Tools registered")

    # 3 — Background session eviction
    eviction_task = asyncio.create_task(_eviction_loop(), name="session-eviction")
    log.info("✓ Session eviction task started")

    log.info("═══ JARVIS ready (%.1f s total) ═══", time.perf_counter() - t0)

    yield  # ← FastAPI serves requests here

    # Shutdown
    log.info("JARVIS shutting down …")
    eviction_task.cancel()
    try:
        await eviction_task
    except asyncio.CancelledError:
        pass
    log.info("JARVIS stopped cleanly")


# ── Application factory ───────────────────────────────────────────────────────

def create_app() -> FastAPI:
    """
    Build and configure the FastAPI application.

    Separated from module-level instantiation so the factory can be
    called in tests with a different Settings instance.
    """
    application = FastAPI(
        title=cfg.APP_NAME,
        version=cfg.APP_VERSION,
        description=(
            "JARVIS — Multi-Agent AI Backend.\n\n"
            "Powered by HuggingFace Transformers (Mistral, LLaMA, Phi, …) "
            "with autonomous agent orchestration via JarvisBrain."
        ),
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan,
    )

    # ── CORS ──────────────────────────────────────────────────────────────────
    application.add_middleware(
        CORSMiddleware,
        allow_origins=cfg.ALLOWED_ORIGINS,
        allow_credentials=True,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type", "Authorization", "X-Session-ID"],
    )

    # ── Request ID / timing middleware ────────────────────────────────────────
    @application.middleware("http")
    async def add_request_timing(request: Request, call_next):
        t0  = time.perf_counter()
        resp = await call_next(request)
        ms   = round((time.perf_counter() - t0) * 1000, 1)
        resp.headers["X-Process-Time-Ms"] = str(ms)
        return resp

    # ── Global exception handler ──────────────────────────────────────────────
    @application.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        log.exception("Unhandled exception on %s %s", request.method, request.url)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error": "Internal server error",
                "detail": str(exc) if cfg.DEBUG else "An unexpected error occurred.",
            },
        )

    # ── Routes ────────────────────────────────────────────────────────────────
    application.include_router(router)

    return application


# Module-level app — imported by uvicorn
app = create_app()
