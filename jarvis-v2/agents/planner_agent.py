"""
agents/planner_agent.py
=======================
PlannerAgent — decomposes a complex user request into an ordered plan.

Responsibilities
────────────────
1. Receives an AgentTask from JarvisBrain.
2. Calls the LLM with a planning system prompt to produce a numbered
   list of steps.
3. Parses the steps into a structured Plan.
4. Returns an AgentResult whose `steps` list and `metadata["plan"]`
   the ExecutorAgent will consume.

The planner never executes tools itself — it only decides what to do.
"""

from __future__ import annotations

import json
import logging
import re

from agents.base_agent import AgentTask, AgentResult, AgentType, BaseAgent
from services.llm_service import llm_service
from tools import tool_registry

log = logging.getLogger(__name__)

# ── System prompt ──────────────────────────────────────────────────────────────
_PLANNER_SYSTEM = """\
You are the Planner component of JARVIS, an advanced AI assistant.

Your ONLY job is to break down the user's request into a numbered list of
clear, concrete, and executable steps.

Rules:
  • Output ONLY valid JSON — no prose, no markdown fences.
  • Schema: {"steps": ["step 1 …", "step 2 …", …], "requires_tools": [<tool names>]}
  • Keep steps concise (one action per step).
  • List only the tools actually needed from: {tools}.
  • If the task needs no tools, set "requires_tools" to [].
  • Maximum 7 steps.
"""


class PlannerAgent(BaseAgent):
    """
    Decomposes tasks into ordered execution plans.

    Output (in AgentResult):
        steps              — ordered list of step strings
        metadata["plan"]   — full JSON plan dict from the LLM
        metadata["tools"]  — list of tool names the plan requires
    """

    agent_type = AgentType.PLANNER

    async def run(self, task: AgentTask) -> AgentResult:
        tools_available = tool_registry.list_allowed()
        system = _PLANNER_SYSTEM.format(tools=", ".join(tools_available))

        # Build a concise planning prompt
        history_text = _format_history(task.history, max_turns=4)
        prompt = (
            f"{history_text}\n\n"
            f"User request: {task.user_input}\n\n"
            "Produce a step-by-step JSON plan to fulfil this request."
        )

        result = await llm_service.generate(
            prompt=prompt,
            system=system,
            max_new_tokens=512,
            temperature=0.2,    # deterministic for planning
        )

        plan = _parse_plan(result.text)
        steps = plan.get("steps", [f"Respond directly to: {task.user_input}"])
        requires_tools = plan.get("requires_tools", [])

        log.info(
            "PlannerAgent: %d step(s), tools needed: %s",
            len(steps), requires_tools,
        )

        summary = (
            f"Plan ({len(steps)} step{'s' if len(steps) != 1 else ''}):\n"
            + "\n".join(f"  {i+1}. {s}" for i, s in enumerate(steps))
        )

        return AgentResult(
            text=summary,
            agent=self.agent_type,
            steps=steps,
            metadata={
                "plan":   plan,
                "tools":  requires_tools,
                "intent": task.intent,
            },
        )


# ── Helpers ────────────────────────────────────────────────────────────────────

def _parse_plan(raw: str) -> dict:
    """Extract and parse the JSON plan from the LLM output."""
    # Strip markdown fences
    cleaned = re.sub(r"```(?:json)?\s*|```", "", raw, flags=re.IGNORECASE).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Regex fallback — extract first {...} block
    m = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass

    log.warning("PlannerAgent: could not parse LLM plan — using fallback")
    return {"steps": [raw.strip()], "requires_tools": []}


def _format_history(history: list[dict], max_turns: int = 4) -> str:
    """Render the last N message turns as plain text for the prompt."""
    recent = history[-max_turns * 2:] if history else []
    return "\n".join(
        f"{m['role'].capitalize()}: {m['content']}" for m in recent
    )
