
import asyncio
from agents.planner import PlannerAgent
from agents.executor import ExecutorAgent
from agents.knowledge import KnowledgeAgent
from agents.memory import MemoryAgent
from core.llm_manager import LLMManager

class Orchestrator:
    """
    Orquestra o fluxo de trabalho entre os diferentes agentes.
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

        Args:
            user_input: A mensagem enviada pelo usuário.

        Returns:
            A resposta do assistente.
        """
        self.memory_agent.add_message("user", user_input)

        plan = await self.planner_agent.plan(user_input)

        results = []
        for task in plan:
            result = await self.executor_agent.execute(task)
            results.append(result)

        # Agrega os resultados e gera uma resposta final
        final_response = await self._generate_final_response(user_input, results)
        
        self.memory_agent.add_message("assistant", final_response)
        return final_response

    async def _generate_final_response(self, user_input: str, execution_results: list) -> str:
        """
        Gera a resposta final para o usuário com base nos resultados da execução.
        """
        context = "\n".join(execution_results)
        conversation_history = self.memory_agent.get_conversation_history()

        prompt = f\"""Based on the following context and conversation history, provide a comprehensive answer to the user's request.

        Conversation History:
        {conversation_history}

        Execution Results:
        {context}

        User Request: {user_input}

        Answer:"""\n
        response = await self.llm_manager.query(prompt)
        return response
