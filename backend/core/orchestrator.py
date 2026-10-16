# === ARQUIVO: backend/core/orchestrator.py ===
"""
Orquestrador do JARVIS 2.0.
Pipeline: Planner → RAG → Web Search → LLM → Memory Firestore.
"""
import asyncio
import logging
from agents.planner   import PlannerAgent
from agents.executor  import ExecutorAgent
from agents.knowledge import KnowledgeAgent
from agents.memory    import MemoryAgent
from core.llm_manager import LLMManager

logger = logging.getLogger(__name__)


class Orchestrator:
    def __init__(self):
        self.llm_manager    = LLMManager()
        self.planner_agent  = PlannerAgent(self.llm_manager)
        self.executor_agent = ExecutorAgent()
        self.knowledge_agent = KnowledgeAgent()
        self.memory_agent   = MemoryAgent()

    async def handle_message(self, user_input: str, session_id: str = "default") -> str:
        """
        Pipeline completo:
        1. Carrega historico da sessao (Firestore)
        2. Planner decide uso de web/RAG
        3. KnowledgeAgent busca chunks relevantes (RAG)
        4. ExecutorAgent busca web (se necessario)
        5. LLM sintetiza resposta com todo o contexto
        6. MemoryAgent persiste no Firestore
        """
        # 1. Carregar historico da sessao
        await asyncio.to_thread(self.memory_agent.load_session, session_id)

        # 2. Planner
        plan = await self.planner_agent.plan(user_input)

        # 3. RAG — busca documentos relevantes
        rag_context = ""
        if plan.get("use_rag") or self.knowledge_agent.document_count() > 0:
            results = await self.knowledge_agent.search(user_input, k=3)
            if results:
                rag_context = "Contexto dos documentos:\n" + "\n".join(
                    f"[{i+1}] {r}" for i, r in enumerate(results)
                )
                logger.info("RAG: %d chunks relevantes encontrados.", len(results))

        # 4. Web search — se planner detectou necessidade
        web_context = ""
        if plan.get("use_web"):
            web_context = await self.executor_agent.web_search(user_input)
            logger.info("Web search executado.")

        # 5. Montar prompt final com todo o contexto
        response = await self._synthesize(user_input, rag_context, web_context)

        # 6. Persistir no Firestore
        self.memory_agent.add_message("user",      user_input, session_id)
        self.memory_agent.add_message("assistant", response,   session_id)

        return response

    async def _synthesize(
        self,
        user_input:  str,
        rag_context: str,
        web_context: str,
    ) -> str:
        """Monta o prompt com todo o contexto e chama o LLM."""
        history = self.memory_agent.get_conversation_history()

        parts = ["Voce e o JARVIS, assistente de IA do projeto ANTIGRAVITY. Responda em portugues."]

        if history and history != "(sem historico)":
            parts.append("Historico da conversa:\n" + history)

        if rag_context:
            parts.append(rag_context)

        if web_context:
            parts.append(web_context)

        parts.append("Pergunta do usuario: " + user_input)
        parts.append("Resposta:")

        prompt = "\n\n".join(parts)

        try:
            return await self.llm_manager.query(prompt)
        except Exception as e:
            logger.error("LLM falhou: %s", e)
            return "JARVIS esta temporariamente indisponivel. Tente novamente."
