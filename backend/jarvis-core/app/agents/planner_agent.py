"""
app/agents/planner_agent.py
Breaks complex actions into an ordered execution plan.
For simple single-step commands this is a pass-through.
"""

from __future__ import annotations

from app.core.decision_engine import ActionDecision
from app.utils.logger import get_logger

log = get_logger("jarvis.agent.planner")


class PlannerAgent:
    """
    Decomposes an ActionDecision into a sequence of steps.
    Currently a lightweight single-pass planner; upgrade with LLM for
    multi-step plans (open X, then search Y, then paste result into Z).
    """

    async def plan(self, action: ActionDecision, user_text: str) -> list[dict]:
        """
        Decompose an action into ordered steps.

        Args:
            action:    The primary action decision.
            user_text: Original user input for context.

        Returns:
            List of step dicts: [{step, name, parameters}]
        """
        steps = []

        # Multi-site open command — open multiple tabs
        if action.name == "open_url" and "," in user_text:
            sites = [s.strip() for s in user_text.split(",")]
            for i, site in enumerate(sites[:5]):   # max 5 tabs
                steps.append({
                    "step": i + 1,
                    "name": "open_url",
                    "parameters": {"url": f"https://www.google.com/search?q={site.replace(' ','+')}"},
                })
        else:
            steps.append({
                "step":       1,
                "name":       action.name,
                "parameters": action.parameters,
            })

        log.debug("Plan: %d step(s) for '%s'", len(steps), action.name)
        return steps
