"""
tests/test_api.py
=================
FastAPI endpoint tests using httpx AsyncClient.

These tests mock JarvisBrain so no real LLM is needed.
They verify:
  • HTTP status codes
  • Response schema conformance
  • Validation error handling (422)
  • 503 when model not loaded
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

# Patch LLM loading before importing the app
with patch("services.llm_service.LLMService.load", return_value=None):
    from app.main import app


@pytest.fixture
def mock_brain_response():
    resp = MagicMock()
    resp.session_id  = "test-session-abc"
    resp.text        = "Hello! I can help you with that."
    resp.intent      = "chat"
    resp.agent_path  = ["executor"]
    resp.tool_calls  = []
    resp.steps       = []
    resp.latency_ms  = 42.0
    resp.metadata    = {"model_id": "tinyllama"}
    resp.success     = True
    resp.error       = None
    return resp


@pytest.fixture
def mock_gesture_response():
    resp = MagicMock()
    resp.session_id  = "test-session-abc"
    resp.text        = "Navigating to next item."
    resp.intent      = "gesture"
    resp.agent_path  = ["gesture"]
    resp.tool_calls  = []
    resp.steps       = []
    resp.latency_ms  = 15.0
    resp.metadata    = {
        "gesture":    "swipe_right",
        "intent":     "Navigate to the next item",
        "confidence": 0.95,
        "landmarks":  21,
        "model_id":   "tinyllama",
    }
    resp.success     = True
    resp.error       = None
    return resp


@pytest.fixture
async def client():
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as c:
        yield c


# ─────────────────────────────────────────────────────────────────────────────
# GET /health
# ─────────────────────────────────────────────────────────────────────────────

class TestHealth:

    @pytest.mark.asyncio
    async def test_health_returns_200(self, client: AsyncClient) -> None:
        with patch("app.routes.llm_service") as mock_llm:
            mock_llm.info.return_value = {"loaded": True}
            r = await client.get("/health")
        assert r.status_code == 200

    @pytest.mark.asyncio
    async def test_health_schema(self, client: AsyncClient) -> None:
        with patch("app.routes.llm_service") as mock_llm:
            mock_llm.info.return_value = {"loaded": True}
            r = await client.get("/health")
        body = r.json()
        assert body["status"] == "ok"
        assert "model_loaded" in body
        assert "version" in body

    @pytest.mark.asyncio
    async def test_health_model_not_loaded(self, client: AsyncClient) -> None:
        with patch("app.routes.llm_service") as mock_llm:
            mock_llm.info.return_value = {"loaded": False}
            r = await client.get("/health")
        assert r.status_code == 200
        assert r.json()["model_loaded"] is False


# ─────────────────────────────────────────────────────────────────────────────
# GET /status
# ─────────────────────────────────────────────────────────────────────────────

class TestStatus:

    @pytest.mark.asyncio
    async def test_status_200(self, client: AsyncClient) -> None:
        mock_status = {
            "brain": "operational",
            "model": {"loaded": True, "model_id": "tinyllama", "parameters_b": 1.1,
                      "device": "cpu", "dtype": "float32", "4bit": False},
            "memory": {"active_sessions": 0, "total_messages": 0, "total_short_term": 0},
            "agents": ["planner", "executor", "knowledge", "gesture"],
            "intents_supported": ["chat", "code", "search", "analyse"],
        }
        with patch("app.routes.brain") as mock_brain:
            mock_brain.status = AsyncMock(return_value=mock_status)
            r = await client.get("/status")
        assert r.status_code == 200
        body = r.json()
        assert "status" in body
        assert "model" in body
        assert "memory" in body
        assert "agents_available" in body

    @pytest.mark.asyncio
    async def test_status_degraded_when_model_not_loaded(self, client: AsyncClient) -> None:
        mock_status = {
            "brain": "operational",
            "model": {"loaded": False},
            "memory": {"active_sessions": 0, "total_messages": 0, "total_short_term": 0},
            "agents": [],
            "intents_supported": [],
        }
        with patch("app.routes.brain") as mock_brain:
            mock_brain.status = AsyncMock(return_value=mock_status)
            r = await client.get("/status")
        assert r.status_code == 200
        assert r.json()["status"] == "degraded"


# ─────────────────────────────────────────────────────────────────────────────
# POST /chat
# ─────────────────────────────────────────────────────────────────────────────

class TestChat:

    @pytest.mark.asyncio
    async def test_chat_success(
        self, client: AsyncClient, mock_brain_response: MagicMock
    ) -> None:
        with patch("app.routes.llm_service") as mock_llm, \
             patch("app.routes.brain") as mock_brain:
            mock_llm.info.return_value = {"loaded": True}
            mock_brain.process = AsyncMock(return_value=mock_brain_response)
            r = await client.post("/chat", json={"message": "Hello JARVIS"})
        assert r.status_code == 200
        body = r.json()
        assert body["text"] == "Hello! I can help you with that."
        assert body["session_id"] == "test-session-abc"
        assert body["intent"] == "chat"
        assert isinstance(body["agent_path"], list)
        assert isinstance(body["latency_ms"], float)

    @pytest.mark.asyncio
    async def test_chat_session_id_passed_through(
        self, client: AsyncClient, mock_brain_response: MagicMock
    ) -> None:
        sid = "my-existing-session"
        with patch("app.routes.llm_service") as mock_llm, \
             patch("app.routes.brain") as mock_brain:
            mock_llm.info.return_value = {"loaded": True}
            mock_brain.process = AsyncMock(return_value=mock_brain_response)
            r = await client.post("/chat", json={
                "message": "remember me",
                "session_id": sid,
            })
        assert r.status_code == 200
        # Brain was called with the correct session_id
        call_kwargs = mock_brain.process.call_args[1]
        assert call_kwargs.get("session_id") == sid or \
               mock_brain.process.call_args[0][1] == sid

    @pytest.mark.asyncio
    async def test_chat_503_when_model_not_loaded(self, client: AsyncClient) -> None:
        with patch("app.routes.llm_service") as mock_llm:
            mock_llm.info.return_value = {"loaded": False}
            r = await client.post("/chat", json={"message": "hello"})
        assert r.status_code == 503
        assert "loading" in r.json()["detail"].lower()

    @pytest.mark.asyncio
    async def test_chat_validation_empty_message(self, client: AsyncClient) -> None:
        with patch("app.routes.llm_service") as mock_llm:
            mock_llm.info.return_value = {"loaded": True}
            r = await client.post("/chat", json={"message": ""})
        assert r.status_code == 422

    @pytest.mark.asyncio
    async def test_chat_validation_message_too_long(self, client: AsyncClient) -> None:
        with patch("app.routes.llm_service") as mock_llm:
            mock_llm.info.return_value = {"loaded": True}
            r = await client.post("/chat", json={"message": "x" * 9000})
        assert r.status_code == 422

    @pytest.mark.asyncio
    async def test_chat_missing_message_field(self, client: AsyncClient) -> None:
        with patch("app.routes.llm_service") as mock_llm:
            mock_llm.info.return_value = {"loaded": True}
            r = await client.post("/chat", json={})
        assert r.status_code == 422


# ─────────────────────────────────────────────────────────────────────────────
# POST /gesture
# ─────────────────────────────────────────────────────────────────────────────

class TestGesture:

    @pytest.mark.asyncio
    async def test_gesture_success(
        self, client: AsyncClient, mock_gesture_response: MagicMock
    ) -> None:
        with patch("app.routes.llm_service") as mock_llm, \
             patch("app.routes.brain") as mock_brain:
            mock_llm.info.return_value = {"loaded": True}
            mock_brain.process_gesture = AsyncMock(return_value=mock_gesture_response)
            r = await client.post("/gesture", json={
                "gesture_id": "swipe_right",
                "confidence": 0.95,
            })
        assert r.status_code == 200
        body = r.json()
        assert body["gesture_id"] == "swipe_right"
        assert "text" in body
        assert body["success"] is True

    @pytest.mark.asyncio
    async def test_gesture_with_landmarks(
        self, client: AsyncClient, mock_gesture_response: MagicMock
    ) -> None:
        landmarks = [{"x": 0.5, "y": 0.5, "z": 0.0}] * 21
        with patch("app.routes.llm_service") as mock_llm, \
             patch("app.routes.brain") as mock_brain:
            mock_llm.info.return_value = {"loaded": True}
            mock_brain.process_gesture = AsyncMock(return_value=mock_gesture_response)
            r = await client.post("/gesture", json={
                "gesture_id": "open_palm",
                "confidence": 0.88,
                "landmarks": landmarks,
            })
        assert r.status_code == 200

    @pytest.mark.asyncio
    async def test_gesture_invalid_id_rejected(self, client: AsyncClient) -> None:
        with patch("app.routes.llm_service") as mock_llm:
            mock_llm.info.return_value = {"loaded": True}
            r = await client.post("/gesture", json={
                "gesture_id": "../../etc/passwd",  # path traversal attempt
                "confidence": 0.99,
            })
        assert r.status_code == 422

    @pytest.mark.asyncio
    async def test_gesture_503_when_model_not_loaded(self, client: AsyncClient) -> None:
        with patch("app.routes.llm_service") as mock_llm:
            mock_llm.info.return_value = {"loaded": False}
            r = await client.post("/gesture", json={
                "gesture_id": "swipe_right",
                "confidence": 0.9,
            })
        assert r.status_code == 503

    @pytest.mark.asyncio
    async def test_gesture_confidence_validation(self, client: AsyncClient) -> None:
        with patch("app.routes.llm_service") as mock_llm:
            mock_llm.info.return_value = {"loaded": True}
            r = await client.post("/gesture", json={
                "gesture_id": "swipe_right",
                "confidence": 1.5,   # out of range
            })
        assert r.status_code == 422
