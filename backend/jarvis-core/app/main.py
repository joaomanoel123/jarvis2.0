"""
app/main.py
JARVIS 2.0 — Production FastAPI Application

Start:
    uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

Render:
    uvicorn app.main:app --host 0.0.0.0 --port 10000
"""

from __future__ import annotations

import asyncio
import os
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.utils.logger import get_logger

log = get_logger("jarvis.main")

ALLOWED_ORIGINS = os.getenv(
    "ALLOWED_ORIGINS",
    "http://localhost:3000,http://localhost:5500,http://127.0.0.1:5500,"
    "https://joaomanoel123.github.io",
).split(",")


# ── Lifespan ────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    import app.core.jarvis_core as jc_module

    log.info("═══ JARVIS 2.0 AUTONOMOUS ENGINE STARTING ═══")
    t0 = time.perf_counter()

    # 1. Initialise database
    from app.memory.database import db
    await db.init()
    log.info("✓ Database ready")

    # 2. Load LLM in thread pool (non-blocking)
    from app.services.llm_service import llm_service
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, llm_service.load)
    log.info("✓ LLM service ready")

    # 3. Build JarvisCore (wires all agents)
    from app.core.jarvis_core import JarvisCore
    jc_module.jarvis = JarvisCore.build()
    log.info("✓ JarvisCore ready")

    # 4. Background session pruner
    async def _pruner():
        while True:
            await asyncio.sleep(300)
            from app.core.state_manager import state_manager
            await state_manager.prune_idle(ttl_s=3600)

    prune_task = asyncio.create_task(_pruner(), name="session-pruner")
    log.info("═══ JARVIS 2.0 READY (%.1f s) ═══", time.perf_counter() - t0)

    yield   # serve requests

    # Shutdown
    prune_task.cancel()
    await db.close()
    log.info("JARVIS 2.0 stopped cleanly")


# ── App factory ──────────────────────────────────────────────────────────

def create_app() -> FastAPI:
    app = FastAPI(
        title="JARVIS 2.0 — Autonomous Intelligence Engine",
        version="2.0.0",
        description=(
            "Multi-agent autonomous AI platform.\n\n"
            "Pipeline: **Observe → Think → Decide → Act → Learn**"
        ),
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=ALLOWED_ORIGINS,
        allow_credentials=True,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Content-Type", "Authorization", "X-Session-ID"],
    )

    # Timing header
    @app.middleware("http")
    async def timing(request: Request, call_next):
        t0   = time.perf_counter()
        resp = await call_next(request)
        ms   = round((time.perf_counter() - t0) * 1000, 1)
        resp.headers["X-Process-Time-Ms"] = str(ms)
        return resp

    # Global error handler
    @app.exception_handler(Exception)
    async def _err(request: Request, exc: Exception):
        log.exception("Unhandled: %s %s", request.method, request.url)
        debug = os.getenv("DEBUG", "false").lower() == "true"
        return JSONResponse(
            status_code=500,
            content={
                "error":  "Internal server error",
                "detail": str(exc) if debug else "An unexpected error occurred.",
            },
        )

    # Mount routers
    from app.api.routes    import router
    from app.api.websocket import ws_router
    app.include_router(router)
    app.include_router(ws_router)

    @app.get("/", include_in_schema=False)
    async def root():
        return {
            "name":    "JARVIS 2.0",
            "version": "2.0.0",
            "status":  "operational",
            "docs":    "/docs",
        }

    return app


app = create_app()
