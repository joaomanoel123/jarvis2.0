"""
app/schemas.py
==============
Pydantic v2 request and response schemas for every JARVIS endpoint.

Every field has an explicit type, description, and example so that the
auto-generated OpenAPI docs are immediately useful to frontend developers.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator


# ─────────────────────────────────────────────────────────────────────────────
# Shared primitives
# ─────────────────────────────────────────────────────────────────────────────

class ToolCall(BaseModel):
    """Record of a single tool invocation."""
    tool:    str  = Field(..., description="Tool name that was called")
    result:  Any  = Field(..., description="Raw tool output")
    success: bool = Field(..., description="Whether the tool succeeded")


# ─────────────────────────────────────────────────────────────────────────────
# POST /chat
# ─────────────────────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    """Incoming chat message."""
    message:    str = Field(
        ...,
        min_length=1,
        max_length=8_000,
        description="User message text",
        examples=["Write a Python function that reverses a string"],
    )
    session_id: str | None = Field(
        None,
        description="Existing session UUID. Omit to start a new conversation.",
        examples=["3fa85f64-5717-4562-b3fc-2c963f66afa6"],
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Optional extra context passed to the agent (client_id, locale…)",
    )

    @field_validator("message")
    @classmethod
    def message_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("message must not be blank")
        return v.strip()


class ChatResponse(BaseModel):
    """Response to a chat message."""
    session_id: str  = Field(..., description="Active session UUID")
    text:       str  = Field(..., description="Assistant response text")
    intent:     str  = Field(..., description="Classified intent (chat/code/search…)")
    agent_path: list[str] = Field(
        default_factory=list,
        description="Ordered list of agents that handled this request",
    )
    tool_calls: list[ToolCall] = Field(
        default_factory=list,
        description="Tool invocations that occurred during processing",
    )
    steps:      list[str] = Field(
        default_factory=list,
        description="Plan steps (populated for code/analyse/plan intents)",
    )
    latency_ms: float = Field(..., description="Total processing time in milliseconds")
    model_id:   str | None = Field(None, description="HuggingFace model that generated the response")
    success:    bool = Field(True,  description="False if the request failed")
    error:      str | None = Field(None, description="Error message when success=False")


# ─────────────────────────────────────────────────────────────────────────────
# POST /gesture
# ─────────────────────────────────────────────────────────────────────────────

class LandmarkPoint(BaseModel):
    """Single 3-D hand landmark from MediaPipe Hands."""
    x: float = Field(..., description="Normalised x coordinate [0–1]")
    y: float = Field(..., description="Normalised y coordinate [0–1]")
    z: float = Field(0.0, description="Normalised depth (relative to wrist)")


class GestureRequest(BaseModel):
    """Gesture event from the MediaPipe client."""
    gesture_id: str = Field(
        ...,
        min_length=1,
        max_length=64,
        description="Gesture identifier (e.g. 'swipe_right', 'pinch', 'open_palm')",
        examples=["swipe_right"],
    )
    session_id: str | None = Field(
        None,
        description="Existing session UUID. Omit to start a new session.",
    )
    confidence: float = Field(
        1.0,
        ge=0.0,
        le=1.0,
        description="Classifier confidence from MediaPipe [0–1]",
    )
    landmarks: list[LandmarkPoint] = Field(
        default_factory=list,
        description="21 MediaPipe hand landmarks for geometric refinement",
    )
    active_widget: str | None = Field(
        None,
        description="UI element under the gesture (for context)",
    )
    coordinates: dict[str, float] | None = Field(
        None,
        description="Screen coordinates {x, y} of the gesture origin",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Any additional context from the client",
    )

    @field_validator("gesture_id")
    @classmethod
    def gesture_id_safe(cls, v: str) -> str:
        import re
        if not re.match(r"^[a-zA-Z0-9_\-]+$", v):
            raise ValueError("gesture_id must contain only letters, digits, underscores, hyphens")
        return v.lower()


class GestureResponse(BaseModel):
    """Response to a gesture event."""
    session_id: str  = Field(..., description="Active session UUID")
    text:       str  = Field(..., description="Assistant response to the gesture")
    gesture_id: str  = Field(..., description="Interpreted gesture name")
    intent:     str  = Field(..., description="Intent mapped to the gesture")
    confidence: float = Field(..., description="Classifier confidence used")
    landmarks_used: int = Field(0, description="Number of landmarks processed")
    latency_ms: float = Field(..., description="Total processing time in milliseconds")
    success:    bool  = Field(True)
    error:      str | None = Field(None)


# ─────────────────────────────────────────────────────────────────────────────
# GET /status
# ─────────────────────────────────────────────────────────────────────────────

class ModelInfo(BaseModel):
    loaded:       bool
    model_id:     str | None = None
    parameters_b: float | None = None
    device:       str | None = None
    dtype:        str | None = None
    quantized_4bit: bool = False


class MemoryStats(BaseModel):
    active_sessions:  int
    total_messages:   int
    total_short_term: int


class StatusResponse(BaseModel):
    """Full system status snapshot."""
    status:            str  = Field(..., description="'operational' | 'degraded' | 'error'")
    version:           str  = Field(..., description="JARVIS version string")
    brain:             str  = Field(..., description="Brain status")
    model:             ModelInfo
    memory:            MemoryStats
    agents_available:  list[str] = Field(..., description="Registered agent names")
    intents_supported: list[str] = Field(..., description="Intent categories")
    uptime_seconds:    float


# ─────────────────────────────────────────────────────────────────────────────
# GET /health
# ─────────────────────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    """Lightweight liveness probe — no model call, instant response."""
    status:    str   = Field("ok", description="Always 'ok' when the process is alive")
    model_loaded: bool = Field(..., description="Whether the LLM has finished loading")
    version:   str   = Field(..., description="JARVIS version string")
