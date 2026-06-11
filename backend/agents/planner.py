import logging

logger = logging.getLogger(__name__)


class PlannerAgent:
    def __init__(self, llm_manager):
        self.llm_manager = llm_manager

    async def plan(self, user_input: str) -> list:
        logger.info("PlannerAgent: planejando para '%s'", user_input[:50])
        return [user_input]
