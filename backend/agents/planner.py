# === ARQUIVO: backend/agents/planner.py ===
"""
PlannerAgent — decompõe a intenção do usuário em subtarefas.
Stub funcional: retorna uma única tarefa com a mensagem original.
"""
import logging

logger = logging.getLogger(__name__)


class PlannerAgent:
    def __init__(self, llm_manager):
        self.llm_manager = llm_manager

    async def plan(self, user_input: str) -> list[str]:
        """
        Analisa a mensagem e retorna lista de subtarefas.
        Versão atual: passa a mensagem diretamente como tarefa única.
        """
        logger.info("PlannerAgent: planejando para '%s'", user_input[:50])
        return [user_input]