"""
agents/executor_agent.py
========================
ExecutorAgent — carries a plan through to completion.

Responsibilities
────────────────
1. Receives an AgentTask that may already contain a plan in metadata["plan"].
2. Selects which tools to call (LLM decides via JSON tool-routing prompt).
3. Executes each tool via the ToolRegistry (sandboxed, timeout-bounded).
4. Calls the LLM to narrate tool results in natural language.
5. Returns a final AgentResult.

Security
────────
All tool calls are routed through ToolRegistry.execute() which enforces
the allowlist, schema validation, and timeout.  The executor never calls
tool functions directly.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from agents.base_agent import AgentTask, AgentResult, AgentType, BaseAgent
from services.llm_service import llm_service
from tools import tool_registry

log = logging.getLogger(__name__)

# ── System prompts ─────────────────────────────────────────────────────────────
_ROUTING_SYSTEM = """\
You are the tool-routing component of JARVIS.
Decide whether the user's next step requires a tool call.

Available tools:
{tools}

Respond with ONLY valid JSON (no markdown fences):
  If a tool is needed:  {{"tool_name": "<name>", "tool_input": {{<args>}}}}
  If no tool is needed: {{"tool_name": null, "tool_input": {{}}}}
"""

_NARRATE_SYSTEM = """\
You are JARVIS, an advanced AI assistant.
Synthesise the tool result into a clear, helpful, conversational response.
Never expose raw JSON.  Be concise and precise.
"""

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


class ExecutorAgent(BaseAgent):
    """
    Executes tool-based and direct-answer plans.

    The agent loops through each step of the plan, invokes the appropriate
    tool, and assembles a final narrated response.
    """

    agent_type = AgentType.EXECUTOR

    async def run(self, task: AgentTask) -> AgentResult:
        plan   = task.metadata.get("plan", {})
        steps  = task.metadata.get("steps", [task.user_input])
        all_tool_calls: list[dict] = []

        # If no steps were passed, derive from plan or treat as single step
        if not steps and isinstance(plan, dict):
            steps = plan.get("steps", [task.user_input])

        final_response = ""

        for i, step in enumerate(steps):
            log.info("ExecutorAgent: step %d/%d — %s", i + 1, len(steps), step[:80])

            # Route: does this step need a tool?
            tool_decision = await self._route_tool(step, task.history)

            if tool_decision["tool_name"]:
                tool_result = await tool_registry.execute(
                    tool_decision["tool_name"],
                    tool_decision.get("tool_input", {}),
                )
                all_tool_calls.append(tool_result)

                # Narrate the tool result
                narration = await self._narrate(step, tool_result, task.history)
                final_response = narration
            else:
                # Direct LLM answer for this step
                direct = await llm_service.chat(
                    messages=task.history + [{"role": "user", "content": step}],
                    system=_NARRATE_SYSTEM,
                    temperature=0.7,
                )
                final_response = direct.text

        if not final_response:
            final_response = "I completed the requested steps but had no specific output to report."

        return AgentResult(
            text=final_response,
            agent=self.agent_type,
            tool_calls=all_tool_calls,
            steps=steps,
            metadata={"step_count": len(steps)},
        )

    # ── Internal helpers ───────────────────────────────────────────────────────

    async def _route_tool(self, step: str, history: list[dict]) -> dict:
        """Ask the LLM which tool (if any) is needed for this step."""
        tools_schema = tool_registry.schema_for_llm()
        system = _ROUTING_SYSTEM.format(tools=json.dumps(tools_schema, indent=2))

        result = await llm_service.generate(
            prompt=f"Step to execute: {step}",
            system=system,
            max_new_tokens=256,
            temperature=0.1,   # near-deterministic routing
        )

        decision = _parse_json(result.text)
        if decision is None:
            log.debug("ExecutorAgent: tool routing parse failed — no tool")
            return {"tool_name": None, "tool_input": {}}

        return decision

    async def _narrate(self, step: str, tool_result: dict, history: list[dict]) -> str:
        """Convert a raw tool result into a natural-language response."""
        prompt = (
            f"The step was: {step}\n\n"
            f"Tool used: {tool_result.get('tool')}\n"
            f"Result:\n{json.dumps(tool_result.get('result', {}), indent=2)}\n\n"
            "Synthesise this into a clear, helpful response."
        )
        llm_result = await llm_service.chat(
            messages=history + [{"role": "user", "content": prompt}],
            system=_NARRATE_SYSTEM,
            temperature=0.5,
        )
        return llm_result.text


def _parse_json(raw: str) -> dict | None:
    cleaned = re.sub(r"```(?:json)?\s*|```", "", raw, flags=re.IGNORECASE).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        m = _JSON_RE.search(cleaned)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                pass
    return None
