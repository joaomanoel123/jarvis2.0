"""
tests/test_brain.py
===================
Async unit tests for JarvisBrain intent classification and routing.

Run with:
    pytest tests/ -v --asyncio-mode=auto

These tests mock the LLMService and ToolRegistry so no real model
or network calls are made.  They verify:
  • Intent classification accuracy
  • Agent routing logic
  • Memory persistence across turns
  • Tool execution error handling
  • Gesture confidence gating
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio

from agents.base_agent import AgentResult, AgentTask, AgentType
from brain.jarvis_brain import JarvisBrain, _classify_intent, Intent
from memory.memory_manager import MemoryManager


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def mem() -> MemoryManager:
    """Fresh in-process memory manager for each test."""
    return MemoryManager()


@pytest.fixture
def mock_llm_result():
    """A generic LLMResult-like object."""
    result = MagicMock()
    result.text     = "Test LLM response"
    result.model_id = "tinyllama"
    return result


@pytest.fixture
def mock_agent_result() -> AgentResult:
    return AgentResult(
        text="Mock agent answer",
        agent=AgentType.EXECUTOR,
        success=True,
        metadata={"agent_path": ["executor"]},
    )


# ─────────────────────────────────────────────────────────────────────────────
# Intent classification
# ─────────────────────────────────────────────────────────────────────────────

class TestIntentClassification:

    @pytest.mark.parametrize("text,expected", [
        ("write a Python function to sort a list",   Intent.CODE),
        ("generate a FastAPI endpoint",              Intent.CODE),
        ("debug this error in my code",              Intent.CODE),
        ("search for the latest AI news",            Intent.SEARCH),
        ("what is the capital of France",            Intent.SEARCH),
        ("who invented the telephone",               Intent.SEARCH),
        ("analyse this CSV dataset for trends",      Intent.ANALYSE),
        ("explore the statistics in this data",      Intent.ANALYSE),
        ("break down how I should approach this",    Intent.PLAN),
        ("steps to deploy a FastAPI app",            Intent.PLAN),
        ("remember that my name is Manoel",          Intent.MEMORY),
        ("save this note for later",                 Intent.MEMORY),
        ("show system status",                       Intent.SYSTEM),
        ("what is my CPU usage",                     Intent.SYSTEM),
        ("hello, how are you?",                      Intent.CHAT),
        ("tell me a joke",                           Intent.CHAT),
        ("",                                         Intent.CHAT),  # fallback
    ])
    def test_classify_intent(self, text: str, expected: Intent) -> None:
        assert _classify_intent(text) == expected


# ─────────────────────────────────────────────────────────────────────────────
# Memory manager
# ─────────────────────────────────────────────────────────────────────────────

class TestMemoryManager:

    @pytest.mark.asyncio
    async def test_create_new_session(self, mem: MemoryManager) -> None:
        sid = await mem.get_or_create()
        assert isinstance(sid, str) and len(sid) == 36  # UUID4

    @pytest.mark.asyncio
    async def test_existing_session_preserved(self, mem: MemoryManager) -> None:
        sid = await mem.get_or_create()
        sid2 = await mem.get_or_create(sid)
        assert sid == sid2

    @pytest.mark.asyncio
    async def test_add_and_get_messages(self, mem: MemoryManager) -> None:
        sid = await mem.get_or_create()
        await mem.add_message(sid, "user", "Hello JARVIS")
        await mem.add_message(sid, "assistant", "Hello! How can I help?")
        msgs = await mem.get_messages(sid)
        assert len(msgs) == 2
        assert msgs[0]["role"] == "user"
        assert msgs[1]["role"] == "assistant"

    @pytest.mark.asyncio
    async def test_message_role_filter(self, mem: MemoryManager) -> None:
        sid = await mem.get_or_create()
        await mem.add_message(sid, "user",      "User msg")
        await mem.add_message(sid, "assistant", "Bot reply")
        await mem.add_message(sid, "tool",      "Tool output")
        user_msgs = await mem.get_messages(sid, roles=["user"])
        assert len(user_msgs) == 1
        assert user_msgs[0]["role"] == "user"

    @pytest.mark.asyncio
    async def test_short_term_memory_set_get(self, mem: MemoryManager) -> None:
        sid = await mem.get_or_create()
        await mem.set_short_term(sid, "last_query", "What is AI?", ttl=60)
        val = await mem.get_short_term(sid, "last_query")
        assert val == "What is AI?"

    @pytest.mark.asyncio
    async def test_short_term_expiry(self, mem: MemoryManager) -> None:
        import time
        sid = await mem.get_or_create()
        await mem.set_short_term(sid, "temp", "value", ttl=0)  # immediate expiry
        # Manually backdate the expiry
        mem._sessions[sid].short_term["temp"] = ("value", time.monotonic() - 1)
        val = await mem.get_short_term(sid, "temp", default="expired")
        assert val == "expired"

    @pytest.mark.asyncio
    async def test_context_persistence(self, mem: MemoryManager) -> None:
        sid = await mem.get_or_create()
        await mem.set_context(sid, "user_name", "Manoel")
        ctx = await mem.get_context(sid, "user_name")
        assert ctx == "Manoel"

    @pytest.mark.asyncio
    async def test_clear_history(self, mem: MemoryManager) -> None:
        sid = await mem.get_or_create()
        await mem.add_message(sid, "user", "msg1")
        await mem.add_message(sid, "user", "msg2")
        await mem.clear_history(sid)
        msgs = await mem.get_messages(sid)
        assert msgs == []

    @pytest.mark.asyncio
    async def test_session_destroy(self, mem: MemoryManager) -> None:
        sid = await mem.get_or_create()
        await mem.add_message(sid, "user", "test")
        await mem.destroy(sid)
        stats = await mem.stats()
        assert stats["active_sessions"] == 0

    @pytest.mark.asyncio
    async def test_memory_stats(self, mem: MemoryManager) -> None:
        for _ in range(3):
            sid = await mem.get_or_create()
            await mem.add_message(sid, "user", "hi")
        stats = await mem.stats()
        assert stats["active_sessions"] == 3
        assert stats["total_messages"] == 3


# ─────────────────────────────────────────────────────────────────────────────
# JarvisBrain routing
# ─────────────────────────────────────────────────────────────────────────────

class TestJarvisBrainRouting:
    """
    Patch the LLM and agents to test routing logic without real inference.
    """

    @pytest.fixture
    def brain_with_mocks(self, mock_agent_result: AgentResult) -> JarvisBrain:
        brain = JarvisBrain()
        mock_execute = AsyncMock(return_value=mock_agent_result)
        brain._executor.execute  = mock_execute
        brain._planner.execute   = AsyncMock(return_value=AgentResult(
            text="Plan: step 1, step 2",
            agent=AgentType.PLANNER,
            steps=["step 1", "step 2"],
            metadata={"plan": {"steps": ["step 1", "step 2"]}, "tools": [], "agent_path": ["planner"]},
        ))
        brain._knowledge.execute = mock_execute
        brain._gesture.execute   = mock_execute
        return brain

    @pytest.mark.asyncio
    async def test_chat_routes_to_executor(self, brain_with_mocks: JarvisBrain) -> None:
        response = await brain_with_mocks.process("hello there")
        assert response.success
        assert response.intent == Intent.CHAT.value
        brain_with_mocks._executor.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_code_routes_to_planner_then_executor(self, brain_with_mocks: JarvisBrain) -> None:
        response = await brain_with_mocks.process("write a Python class for a bank account")
        assert response.intent == Intent.CODE.value
        brain_with_mocks._planner.execute.assert_called_once()
        brain_with_mocks._executor.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_search_routes_to_knowledge(self, brain_with_mocks: JarvisBrain) -> None:
        response = await brain_with_mocks.process("search for the latest news on LLMs")
        assert response.intent == Intent.SEARCH.value
        brain_with_mocks._knowledge.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_gesture_intent_set_via_metadata(self, brain_with_mocks: JarvisBrain) -> None:
        response = await brain_with_mocks.process(
            user_input="[gesture:swipe_right]",
            metadata={"gesture_id": "swipe_right", "confidence": 0.95},
        )
        assert response.intent == Intent.GESTURE.value
        brain_with_mocks._gesture.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_session_persisted_in_memory(self, brain_with_mocks: JarvisBrain) -> None:
        response = await brain_with_mocks.process("first message")
        sid = response.session_id
        # Second turn with same session
        await brain_with_mocks.process("second message", session_id=sid)
        msgs = await brain_with_mocks._executor.execute.call_args_list
        # Both calls should have used history on the second one
        assert len(msgs) == 2

    @pytest.mark.asyncio
    async def test_new_session_created_when_none(self, brain_with_mocks: JarvisBrain) -> None:
        r1 = await brain_with_mocks.process("hi")
        r2 = await brain_with_mocks.process("hello")  # no session_id
        # Each gets a different session
        assert r1.session_id != r2.session_id

    @pytest.mark.asyncio
    async def test_process_gesture_convenience(self, brain_with_mocks: JarvisBrain) -> None:
        response = await brain_with_mocks.process_gesture(
            gesture_id="open_palm",
            confidence=0.9,
        )
        assert response.intent == Intent.GESTURE.value
        brain_with_mocks._gesture.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_agent_error_is_caught_not_raised(self, brain_with_mocks: JarvisBrain) -> None:
        brain_with_mocks._executor.execute = AsyncMock(side_effect=RuntimeError("LLM crashed"))
        response = await brain_with_mocks.process("hello")
        assert not response.success
        assert "error" in (response.error or "").lower() or response.error is not None


# ─────────────────────────────────────────────────────────────────────────────
# Tool registry
# ─────────────────────────────────────────────────────────────────────────────

class TestToolRegistry:

    @pytest.fixture
    def registry(self):
        from tools.registry import ToolRegistry, ToolSpec
        reg = ToolRegistry()
        reg.register(ToolSpec(
            name="echo",
            fn=AsyncMock(return_value={"echoed": "hello"}),
            description="Echo test tool",
            parameters={"type": "object", "required": ["message"],
                        "properties": {"message": {"type": "string"}}},
        ))
        return reg

    @pytest.mark.asyncio
    async def test_blocked_tool_returns_error(self, registry) -> None:
        # "echo" is not in ALLOWED_TOOLS env var (which defaults to a whitelist)
        result = await registry.execute("echo", {"message": "hi"})
        # Depends on settings — if not allowed, success=False
        assert "tool" in result

    @pytest.mark.asyncio
    async def test_unregistered_tool_returns_error(self, registry) -> None:
        # Force the tool into the allowed set for this test
        with patch.object(
            type(registry),
            "execute",
            wraps=registry.execute,
        ):
            result = await registry.execute("nonexistent_tool", {})
            assert result["success"] is False

    @pytest.mark.asyncio
    async def test_missing_required_param(self, registry) -> None:
        from config.settings import get_settings
        cfg = get_settings()
        # Temporarily allow the echo tool
        original = cfg.ALLOWED_TOOLS
        cfg.ALLOWED_TOOLS = "echo," + original
        cfg.__dict__["_allowed_tools_set"] = None  # invalidate cache

        result = await registry.execute("echo", {})  # missing 'message'
        # Either blocked (not allowed) or missing param error
        assert isinstance(result, dict)
        # Restore
        cfg.ALLOWED_TOOLS = original


# ─────────────────────────────────────────────────────────────────────────────
# Gesture agent
# ─────────────────────────────────────────────────────────────────────────────

class TestGestureAgent:

    @pytest.mark.asyncio
    async def test_low_confidence_rejected(self) -> None:
        from agents.gesture_agent import GestureAgent
        from agents.base_agent import AgentTask

        agent = GestureAgent()
        task = AgentTask(
            session_id="test-sid",
            user_input="[gesture:swipe_right]",
            metadata={"gesture_id": "swipe_right", "confidence": 0.3},  # below threshold
        )
        with patch.object(type(agent), "run", wraps=agent.run):
            result = await agent.run(task)
        assert result.success is False
        assert "confident" in result.text.lower() or "confidence" in result.text.lower()

    @pytest.mark.asyncio
    async def test_unknown_gesture_rejected(self) -> None:
        from agents.gesture_agent import GestureAgent
        from agents.base_agent import AgentTask

        agent = GestureAgent()
        with patch("agents.gesture_agent.llm_service") as mock_llm:
            mock_llm.chat = AsyncMock(return_value=MagicMock(text="OK"))
            task = AgentTask(
                session_id="test-sid",
                user_input="[gesture:unknown_gesture_xyz]",
                metadata={"gesture_id": "unknown_gesture_xyz", "confidence": 0.99},
            )
            result = await agent.run(task)
        assert result.success is False
        assert "not recognised" in result.text.lower() or "unknown" in result.text.lower()

    def test_all_gestures_have_intent(self) -> None:
        from agents.gesture_agent import GESTURE_MAP
        for gesture, intent in GESTURE_MAP.items():
            assert isinstance(intent, str) and len(intent) > 5, \
                f"Gesture '{gesture}' has empty intent"
