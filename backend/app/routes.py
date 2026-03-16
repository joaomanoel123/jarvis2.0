"""
app/routes.py
=============
FastAPI route handlers — the thin HTTP layer between clients and JarvisBrain.

Every handler:
  1. Validates input via Pydantic (automatic, via schema annotation).
  2. Delegates ALL business logic to brain.process() or brain.process_gesture().
  3. Converts the BrainResponse into a typed API schema.
  4. Never imports LLMService, agents, or tools directly.
"""

from __future__ import annotations

import time

from fastapi import APIRouter, HTTPException, Request, status

from app.schemas import (
    ChatRequest,
    ChatResponse,
    GestureRequest,
    GestureResponse,
    HealthResponse,
    MemoryStats,
    ModelInfo,
    StatusResponse,
    ToolCall,
)
from brain.jarvis_brain import brain
from config.settings import get_settings
from services.llm_service import llm_service

cfg = get_settings()
router = APIRouter()

_START_TIME = time.time()


# ─────────────────────────────────────────────────────────────────────────────
# GET /health   — liveness probe (no model call, instant)
# ─────────────────────────────────────────────────────────────────────────────

@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Liveness probe",
    tags=["Infrastructure"],
)
async def health() -> HealthResponse:
    """
    Returns 200 immediately as long as the process is alive.
    Used by Docker HEALTHCHECK and Render's liveness probe.
    Does NOT wait for the model to finish loading.
    """
    return HealthResponse(
        status="ok",
        model_loaded=llm_service.info().get("loaded", False),
        version=cfg.APP_VERSION,
    )


# ─────────────────────────────────────────────────────────────────────────────
# GET /status   — full diagnostic snapshot
# ─────────────────────────────────────────────────────────────────────────────

@router.get(
    "/status",
    response_model=StatusResponse,
    summary="Full system status",
    tags=["Infrastructure"],
)
async def system_status() -> StatusResponse:
    """
    Returns model info, memory stats, agent list, and uptime.
    Slightly heavier than /health — intended for dashboards and monitoring.
    """
    info = await brain.status()

    model_raw = info.get("model", {})
    mem_raw   = info.get("memory", {})

    model = ModelInfo(
        loaded=model_raw.get("loaded", False),
        model_id=model_raw.get("model_id"),
        parameters_b=model_raw.get("parameters_b"),
        device=model_raw.get("device"),
        dtype=model_raw.get("dtype"),
        quantized_4bit=model_raw.get("4bit", False),
    )
    mem = MemoryStats(
        active_sessions=mem_raw.get("active_sessions", 0),
        total_messages=mem_raw.get("total_messages", 0),
        total_short_term=mem_raw.get("total_short_term", 0),
    )

    overall = "operational" if model.loaded else "degraded"

    return StatusResponse(
        status=overall,
        version=cfg.APP_VERSION,
        brain=info.get("brain", "unknown"),
        model=model,
        memory=mem,
        agents_available=info.get("agents", []),
        intents_supported=info.get("intents_supported", []),
        uptime_seconds=round(time.time() - _START_TIME, 1),
    )


# ─────────────────────────────────────────────────────────────────────────────
# POST /chat   — main conversation endpoint
# ─────────────────────────────────────────────────────────────────────────────

@router.post(
    "/chat",
    response_model=ChatResponse,
    summary="Send a chat message",
    tags=["Agents"],
    status_code=status.HTTP_200_OK,
)
async def chat(request: ChatRequest) -> ChatResponse:
    """
    Send a text message to JARVIS.

    JarvisBrain classifies the intent, selects the appropriate agent
    pipeline (Planner → Executor, KnowledgeAgent, direct chat…),
    invokes tools as needed, and returns the final response.

    A new session is created when `session_id` is omitted.
    Pass the returned `session_id` in subsequent requests to maintain
    conversation context.
    """
    if not llm_service.info().get("loaded"):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is still loading — please retry in a few seconds.",
        )

    response = await brain.process(
        user_input=request.message,
        session_id=request.session_id,
        metadata=request.metadata,
    )

    return ChatResponse(
        session_id=response.session_id,
        text=response.text,
        intent=response.intent,
        agent_path=response.agent_path,
        tool_calls=[
            ToolCall(
                tool=tc.get("tool", "unknown"),
                result=tc.get("result"),
                success=tc.get("success", False),
            )
            for tc in response.tool_calls
        ],
        steps=response.steps,
        latency_ms=response.latency_ms,
        model_id=response.metadata.get("model_id"),
        success=response.success,
        error=response.error,
    )


# ─────────────────────────────────────────────────────────────────────────────
# POST /gesture   — MediaPipe gesture input
# ─────────────────────────────────────────────────────────────────────────────

@router.post(
    "/gesture",
    response_model=GestureResponse,
    summary="Process a MediaPipe gesture event",
    tags=["Agents"],
    status_code=status.HTTP_200_OK,
)
async def gesture(request: GestureRequest) -> GestureResponse:
    """
    Receive a gesture event from a MediaPipe client.

    The GestureAgent maps the `gesture_id` to a natural-language intent,
    applies geometric refinement when `landmarks` are supplied, and
    generates a conversational response aligned with the gesture's meaning.

    Gestures below the confidence threshold are rejected with a 422.
    """
    if not llm_service.info().get("loaded"):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is still loading — please retry in a few seconds.",
        )

    context: dict = {}
    if request.active_widget:
        context["active_widget"] = request.active_widget
    if request.coordinates:
        context["coordinates"] = request.coordinates
    context.update(request.metadata)

    response = await brain.process_gesture(
        gesture_id=request.gesture_id,
        session_id=request.session_id,
        confidence=request.confidence,
        landmarks=[lm.model_dump() for lm in request.landmarks],
        context=context,
    )

    gesture_meta = response.metadata
    return GestureResponse(
        session_id=response.session_id,
        text=response.text,
        gesture_id=gesture_meta.get("gesture", request.gesture_id),
        intent=gesture_meta.get("intent", "unknown"),
        confidence=gesture_meta.get("confidence", request.confidence),
        landmarks_used=gesture_meta.get("landmarks", 0),
        latency_ms=response.latency_ms,
        success=response.success,
        error=response.error,
    )
