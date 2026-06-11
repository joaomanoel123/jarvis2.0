# === ARQUIVO: backend/core/orchestrator.py ===
"""
Orquestrador de agentes do JARVIS 2.0.
Coordena PlannerAgent, ExecutorAgent, KnowledgeAgent e MemoryAgent.
"""

import asyncio
import logging
from agents.planner import PlannerAgent
from agents.executor import ExecutorAgent
from agents.knowledge import KnowledgeAgent
from agents.memory import MemoryAgent
from core.llm_manager import LLMManager

logger = logging.getLogger(__name__)


class Orchestrator:
    """
    Orquestra o fluxo de trabalho entre os diferentes agentes.
    Recebe mensagem do usuário → planeja → executa → sintetiza → retorna.
    """

    def __init__(self):
        self.llm_manager = LLMManager()
        self.planner_agent = PlannerAgent(self.llm_manager)
        self.executor_agent = ExecutorAgent()
        self.knowledge_agent = KnowledgeAgent()
        self.memory_agent = MemoryAgent()
        self.task_queue = asyncio.Queue()

    async def handle_message(self, user_input: str) -> str:
        """
        Processa a mensagem do usuário e retorna a resposta do assistente.
        """
        self.memory_agent.add_message("user", user_input)

        plan = await self.planner_agent.plan(user_input)

        results = []
        for task in plan:
            try:
                result = await self.executor_agent.execute(task)
                results.append(result)
            except Exception as e:
                logger.warning("ExecutorAgent falhou na task '%s': %s", task, e)
                results.append(None)

        final_response = await self._generate_final_response(user_input, results)
        self.memory_agent.add_message("assistant", final_response)
        return final_response

    async def _generate_final_response(self, user_input: str, execution_results: list) -> str:
        """
        Gera a resposta final com base nos resultados dos agentes.
        """
        context = "\n".join(str(r) for r in execution_results if r is not None)
        conversation_history = self.memory_agent.get_conversation_history()

        prompt = (
            "Based on the following context and conversation history, "
            "provide a comprehensive answer to the user's request.\n\n"
            "Conversation History:\n"
            + conversation_history
            + "\n\nExecution Results:\n"
            + (context if context else "No additional context available.")
            + "\n\nUser Request: "
            + user_input
            + "\n\nAnswer:"
        )

        try:
            response = await self.llm_manager.query(prompt)
            return response
        except Exception as e:
            logger.error("LLMManager.query falhou: %s", e, exc_info=True)
            return (
                "JARVIS está temporariamente indisponível. "
                "Tente novamente em alguns instantes."
            )
