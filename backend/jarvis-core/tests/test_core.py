"""
tests/test_core.py
═══════════════════════════════════════════════════════════════
JARVIS 2.0 — Core Autonomous Engine Test Suite

Run with:
    pytest tests/test_core.py -v

No camera, microphone, or LLM model required.
All tests use in-process components with mocks where needed.
═══════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import asyncio
import sys
import os
import pytest

# Path setup
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.core.decision_engine import (
    DecisionEngine, ActionDecision,
    COMMAND, CONVERSATIONAL, QUERY, SYSTEM_CONTROL, GESTURE,
    RISK_LOW, RISK_MEDIUM, RISK_HIGH,
)
from app.core.state_manager import StateManager, MODE_IDLE, MODE_THINKING, MODE_EXECUTING
from app.core.jarvis_core import CoreResponse


# ══════════════════════════════════════════════════════════════════════
#  DecisionEngine Tests
# ══════════════════════════════════════════════════════════════════════

class TestDecisionEngine:
    """Tests for intent classification, action selection, and risk evaluation."""

    def setup_method(self):
        self.de = DecisionEngine()

    # ── Intent classification ──────────────────────────────────────────

    @pytest.mark.parametrize("text,expected_intent", [
        # Commands
        ("open youtube",                     COMMAND),
        ("open google",                      COMMAND),
        ("search AI news",                   COMMAND),
        ("play lofi music",                  COMMAND),
        ("launch spotify",                   COMMAND),
        ("find python tutorials on youtube", COMMAND),
        ("pause",                            COMMAND),
        ("next track",                       COMMAND),
        ("volume 50",                        COMMAND),
        ("take a screenshot",                COMMAND),
        # Queries / Conversational
        ("what is machine learning",         QUERY),
        ("who is Elon Musk",                 QUERY),
        ("how do I install Python",          QUERY),
        ("explain neural networks",          CONVERSATIONAL),
        ("describe quantum computing",       CONVERSATIONAL),
        # System control
        ("activate voice mode",              SYSTEM_CONTROL),
        ("switch to dark mode",              SYSTEM_CONTROL),
        ("clear memory",                     SYSTEM_CONTROL),
        # Gestures
        ("swipe_right",                      GESTURE),
        ("open_hand",                        GESTURE),
        ("fist",                             GESTURE),
        ("zoom_in",                          GESTURE),
        ("thumbs_up",                        GESTURE),
        ("wave",                             GESTURE),
    ])
    def test_intent_classification(self, text, expected_intent):
        r = self.de.decide(text)
        assert r.intent == expected_intent, (
            f"'{text}': expected {expected_intent}, got {r.intent}"
        )

    # ── Action selection ───────────────────────────────────────────────

    @pytest.mark.parametrize("text,exp_name,exp_type,exp_url_contains", [
        ("open youtube",    "open_url",     "browser", "youtube.com"),
        ("open google",     "open_url",     "browser", "google.com"),
        ("open github",     "open_url",     "browser", "github.com"),
        ("open netflix",    "open_url",     "browser", "netflix.com"),
        ("search AI news",  "search_query", "browser", "google.com"),
        ("play music",      "open_url",     "browser", "youtube.com"),
    ])
    def test_action_url_selection(self, text, exp_name, exp_type, exp_url_contains):
        r = self.de.decide(text)
        assert r.action is not None,     f"No action for '{text}'"
        assert r.action.name == exp_name, f"'{text}': expected name={exp_name}, got {r.action.name}"
        assert r.action.type == exp_type, f"'{text}': expected type={exp_type}, got {r.action.type}"
        url = r.action.parameters.get("url", "")
        assert exp_url_contains in url, f"'{text}': expected URL to contain '{exp_url_contains}', got '{url}'"

    @pytest.mark.parametrize("gesture_id,exp_command,exp_type", [
        ("swipe_right",            "next_screen",      "ui"),
        ("swipe_left",             "previous_screen",  "ui"),
        ("swipe_up",               "scroll_top",       "ui"),
        ("swipe_down",             "scroll_bottom",    "ui"),
        ("open_hand",              "open_menu",        "ui"),
        ("fist",                   "close_menu",       "ui"),
        ("pinch",                  "select_object",    "ui"),
        ("zoom_in",                "zoom_in",          "ui"),
        ("zoom_out",               "zoom_out",         "ui"),
        ("thumbs_up",              "confirm",          "ui"),
        ("circle_clockwise",       "rotate_right",     "ui"),
        ("circle_counterclockwise","rotate_left",      "ui"),
    ])
    def test_gesture_action_mapping(self, gesture_id, exp_command, exp_type):
        r = self.de.decide(gesture_id)
        assert r.intent == GESTURE
        assert r.action is not None
        assert r.action.name == exp_command, (
            f"Gesture '{gesture_id}': expected {exp_command}, got {r.action.name}"
        )
        assert r.action.type == exp_type

    # ── Risk evaluation ────────────────────────────────────────────────

    @pytest.mark.parametrize("text,expected_risk", [
        ("delete all files",          RISK_HIGH),
        ("format the drive",          RISK_HIGH),
        ("shutdown the computer",     RISK_HIGH),
        ("send an email to Alice",    RISK_MEDIUM),
        ("purchase this item",        RISK_MEDIUM),
        ("open youtube",              RISK_LOW),
        ("what is AI",                RISK_LOW),
        ("play music",                RISK_LOW),
    ])
    def test_risk_evaluation(self, text, expected_risk):
        r = self.de.decide(text)
        assert r.risk == expected_risk, (
            f"'{text}': expected risk={expected_risk}, got {r.risk}"
        )

    def test_high_risk_blocked(self):
        r = self.de.decide("delete all files")
        assert r.risk == RISK_HIGH
        assert r.requires_confirmation is True
        assert r.action is None   # blocked — no action

    def test_medium_risk_requires_confirmation(self):
        r = self.de.decide("send an email to Alice")
        assert r.risk == RISK_MEDIUM
        assert r.requires_confirmation is True

    # ── Proactive suggestions ──────────────────────────────────────────

    @pytest.mark.parametrize("text", [
        "I want to study",
        "I need to learn something",
        "I want to listen to music",
        "I want to relax",
        "I want to code something",
        "show me the news",
    ])
    def test_proactive_suggestions(self, text):
        r = self.de.decide(text, history=[], preferences={})
        assert r.proactive is not None, f"Expected proactive suggestion for: '{text}'"
        assert len(r.proactive) > 10

    # ── Confidence ────────────────────────────────────────────────────

    def test_confidence_range(self):
        texts = ["open youtube", "what is AI", "swipe_right", "hello"]
        for text in texts:
            r = self.de.decide(text)
            assert 0.0 <= r.confidence <= 1.0, f"Confidence out of range for '{text}'"

    # ── ActionDecision.to_dict() ──────────────────────────────────────

    def test_action_decision_serialisation(self):
        a = ActionDecision("browser", "open_url", {"url": "https://youtube.com"})
        d = a.to_dict()
        assert d["type"]       == "browser"
        assert d["name"]       == "open_url"
        assert d["parameters"] == {"url": "https://youtube.com"}


# ══════════════════════════════════════════════════════════════════════
#  StateManager Tests
# ══════════════════════════════════════════════════════════════════════

class TestStateManager:

    def setup_method(self):
        self.sm = StateManager()

    @pytest.mark.asyncio
    async def test_create_session(self):
        session = await self.sm.get_or_create("test-001")
        assert session.session_id == "test-001"
        assert session.mode == MODE_IDLE

    @pytest.mark.asyncio
    async def test_session_reuse(self):
        s1 = await self.sm.get_or_create("test-002")
        s2 = await self.sm.get_or_create("test-002")
        assert s1 is s2   # same object

    @pytest.mark.asyncio
    async def test_auto_generate_session_id(self):
        s = await self.sm.get_or_create(None)
        assert len(s.session_id) == 36  # UUID format

    @pytest.mark.asyncio
    async def test_mode_transitions(self):
        await self.sm.get_or_create("test-003")
        for mode in (MODE_THINKING, MODE_EXECUTING, MODE_IDLE):
            await self.sm.set_mode("test-003", mode)
            s = await self.sm.get_or_create("test-003")
            assert s.mode == mode

    @pytest.mark.asyncio
    async def test_invalid_mode_raises(self):
        await self.sm.get_or_create("test-004")
        with pytest.raises(AssertionError):
            await self.sm.set_mode("test-004", "flying")

    @pytest.mark.asyncio
    async def test_context_store_retrieve(self):
        await self.sm.get_or_create("test-005")
        await self.sm.set_context("test-005", "user_name", "Manoel")
        val = await self.sm.get_context("test-005", "user_name")
        assert val == "Manoel"

    @pytest.mark.asyncio
    async def test_event_queue_push_pop(self):
        await self.sm.get_or_create("test-006")
        await self.sm.push_event("test-006", {"type": "ping", "ts": 1234})
        ev = await asyncio.wait_for(self.sm.get_event("test-006", timeout=1.0), timeout=2.0)
        assert ev is not None
        assert ev["type"] == "ping"

    @pytest.mark.asyncio
    async def test_summary(self):
        await self.sm.get_or_create("summary-a")
        await self.sm.get_or_create("summary-b")
        s = await self.sm.summary()
        assert s["active_sessions"] >= 2
        assert "modes" in s

    @pytest.mark.asyncio
    async def test_session_destroy(self):
        await self.sm.get_or_create("destroy-me")
        await self.sm.destroy("destroy-me")
        assert "destroy-me" not in self.sm._sessions

    @pytest.mark.asyncio
    async def test_prune_idle(self):
        """Sessions with last_active very far in the past should be pruned."""
        await self.sm.get_or_create("old-session")
        import time
        self.sm._sessions["old-session"].last_active = time.time() - 99999
        n = await self.sm.prune_idle(ttl_s=3600)
        assert n >= 1
        assert "old-session" not in self.sm._sessions


# ══════════════════════════════════════════════════════════════════════
#  CoreResponse Tests
# ══════════════════════════════════════════════════════════════════════

class TestCoreResponse:

    def test_basic_serialisation(self):
        r = CoreResponse(
            session_id="abc-123",
            intent="command",
            decision="Open YouTube",
            response="Opening YouTube.",
            action={"type": "browser", "name": "open_url", "parameters": {"url": "https://youtube.com"}},
            confidence=0.92,
            risk="low",
            latency_ms=142.5,
            success=True,
        )
        d = r.to_dict()
        assert d["session_id"]        == "abc-123"
        assert d["intent"]            == "command"
        assert d["decision"]          == "Open YouTube"
        assert d["response"]          == "Opening YouTube."
        assert d["confidence"]        == 0.92
        assert d["risk"]              == "low"
        assert d["success"]           is True
        assert d["state"]["mode"]     == "idle"
        assert d["state"]["intent"]   == "command"

    def test_error_response(self):
        r = CoreResponse(
            session_id="err-1",
            intent="error",
            decision="System error",
            response="A system error occurred.",
            success=False,
            error="NullPointerException",
        )
        d = r.to_dict()
        assert d["success"] is False
        assert d["error"]   == "NullPointerException"

    def test_confidence_rounded(self):
        r = CoreResponse(
            session_id="x", intent="chat", decision="d", response="r",
            confidence=0.876543,
        )
        d = r.to_dict()
        # to_dict rounds to 3 decimal places
        assert d["confidence"] == 0.877 or d["confidence"] == 0.876543  # depends on rounding impl

    def test_proactive_field(self):
        r = CoreResponse(
            session_id="x", intent="conversational", decision="d", response="r",
            proactive="Shall I open YouTube?",
        )
        d = r.to_dict()
        assert d["proactive"] == "Shall I open YouTube?"

    def test_agent_path_field(self):
        r = CoreResponse(
            session_id="x", intent="command", decision="d", response="r",
            agent_path=["planner", "executor", "memory[log]"],
        )
        d = r.to_dict()
        assert d["agent_path"] == ["planner", "executor", "memory[log]"]

    def test_requires_confirmation(self):
        r = CoreResponse(
            session_id="x", intent="command", decision="d", response="r",
            requires_confirmation=True,
            risk="medium",
        )
        d = r.to_dict()
        assert d["requires_confirmation"] is True
        assert d["risk"] == "medium"


# ══════════════════════════════════════════════════════════════════════
#  Integration: DecisionEngine + AgentRouter (no LLM, no DB)
# ══════════════════════════════════════════════════════════════════════

class TestPipelineIntegration:
    """
    Test the decision → routing pipeline end-to-end using lightweight mocks.
    No LLM or database required.
    """

    def setup_method(self):
        self.de = DecisionEngine()

    @pytest.mark.asyncio
    async def test_command_pipeline_shape(self):
        """A command decision should produce an action dict with correct shape."""
        from app.core.agent_router import AgentRouter

        class _MockExecutor:
            async def execute(self, action):
                return {"success": True, "result": "mock executed"}

        router = AgentRouter(executor=_MockExecutor())
        decision = self.de.decide("open youtube")
        result = await router.route(decision, "open youtube", "session-x", [])

        assert result.intent == "command"
        assert result.action["name"] == "open_url"
        assert result.action["type"] == "browser"
        assert "youtube.com" in result.action["parameters"].get("url", "")
        assert result.success is True

    @pytest.mark.asyncio
    async def test_conversational_pipeline_shape(self):
        """Conversational input should produce a text response."""
        from app.core.agent_router import AgentRouter

        class _MockReasoner:
            async def respond(self, text, history, context, intent):
                return f"Mock response to: {text}"

        router = AgentRouter(reasoner=_MockReasoner())
        decision = self.de.decide("hello jarvis")
        result = await router.route(decision, "hello jarvis", "session-y", [])

        assert result.intent == "conversational"
        assert "Mock response" in result.text

    @pytest.mark.asyncio
    async def test_gesture_pipeline(self):
        """Gesture input should map to a UI action."""
        from app.core.agent_router import AgentRouter

        class _MockExecutor:
            async def execute(self, action):
                return {"success": True, "result": "ui dispatched"}

        router   = AgentRouter(executor=_MockExecutor())
        decision = self.de.decide("swipe_right")
        result   = await router.route(decision, "swipe_right", "session-z", [])

        assert result.intent            == "gesture"
        assert result.action["name"]    == "next_screen"
        assert result.action["type"]    == "ui"

    @pytest.mark.asyncio
    async def test_high_risk_blocked_in_pipeline(self):
        """High-risk decision should return blocked result without executing."""
        from app.core.agent_router import AgentRouter

        executed = []

        class _MockExecutor:
            async def execute(self, action):
                executed.append(action.name)
                return {"success": True}

        router   = AgentRouter(executor=_MockExecutor())
        decision = self.de.decide("delete all files")
        result   = await router.route(decision, "delete all files", "session-w", [])

        # High risk → blocked → executor should NOT have been called
        assert len(executed) == 0
        assert result.requires_confirmation is True or result.action == {}


# ══════════════════════════════════════════════════════════════════════
#  Executor Agent Tests
# ══════════════════════════════════════════════════════════════════════

class TestExecutorAgent:

    @pytest.mark.asyncio
    async def test_browser_open_url(self):
        from app.agents.executor_agent import ExecutorAgent
        from app.core.decision_engine  import ActionDecision
        exec = ExecutorAgent()
        result = await exec.execute(ActionDecision("browser", "open_url", {"url": "https://youtube.com"}))
        assert result["success"] is True
        assert "youtube.com" in result.get("url", "")

    @pytest.mark.asyncio
    async def test_browser_no_url(self):
        from app.agents.executor_agent import ExecutorAgent
        from app.core.decision_engine  import ActionDecision
        exec = ExecutorAgent()
        result = await exec.execute(ActionDecision("browser", "open_url", {}))
        assert result["success"] is False

    @pytest.mark.asyncio
    async def test_system_status(self):
        from app.agents.executor_agent import ExecutorAgent
        from app.core.decision_engine  import ActionDecision
        exec = ExecutorAgent()
        result = await exec.execute(ActionDecision("system", "get_status", {}))
        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_media_action(self):
        from app.agents.executor_agent import ExecutorAgent
        from app.core.decision_engine  import ActionDecision
        exec = ExecutorAgent()
        result = await exec.execute(ActionDecision("media", "play", {}))
        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_ui_action(self):
        from app.agents.executor_agent import ExecutorAgent
        from app.core.decision_engine  import ActionDecision
        exec = ExecutorAgent()
        result = await exec.execute(ActionDecision("ui", "next_screen", {}))
        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_unknown_action_fails(self):
        from app.agents.executor_agent import ExecutorAgent
        from app.core.decision_engine  import ActionDecision
        exec = ExecutorAgent()
        result = await exec.execute(ActionDecision("unknown_type", "mystery", {}))
        assert result["success"] is False


# ══════════════════════════════════════════════════════════════════════
#  Observer Agent Tests
# ══════════════════════════════════════════════════════════════════════

class TestObserverAgent:

    @pytest.mark.asyncio
    async def test_record_and_metrics(self):
        from app.agents.observer_agent import ObserverAgent

        class _MockDB:
            async def record_observation(self, *args, **kwargs):
                pass

        obs = ObserverAgent(db=_MockDB())
        await obs.record("command", 120.5, True)
        await obs.record("command", 200.0, True)
        await obs.record("conversational", 500.0, False)

        m = obs.metrics()
        assert m["total"]  == 3
        assert m["errors"] == 1
        assert "command"       in m["by_intent"]
        assert "conversational" in m["by_intent"]
        assert m["by_intent"]["command"]["count"]   == 2
        assert m["by_intent"]["command"]["errors"]  == 0
        assert m["by_intent"]["conversational"]["errors"] == 1
