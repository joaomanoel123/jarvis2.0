# === ARQUIVO: backend/agents/planner.py ===
"""
PlannerAgent — analisa a intencao do usuario e define quais agentes acionar.
Detecta automaticamente quando busca web ou RAG sao necessarios.
"""
import logging
import re

logger = logging.getLogger(__name__)

# Palavras-chave que indicam necessidade de busca web
WEB_SEARCH_TRIGGERS = [
    "hoje", "agora", "atual", "atualmente", "recente", "recentemente",
    "noticia", "notícia", "novidade", "ultimo", "último", "2024", "2025", "2026",
    "preco", "preço", "cotacao", "cotação", "clima", "tempo", "temperatura",
    "quem ganhou", "resultado", "aconteceu", "lançamento", "lancamento",
]

# Palavras-chave que indicam necessidade de busca em documentos (RAG)
RAG_TRIGGERS = [
    "documento", "arquivo", "pdf", "texto", "referencia", "referência",
    "segundo o documento", "no arquivo", "baseado em", "de acordo com",
    "no contexto", "encontrei", "upload", "enviou",
]


class PlannerAgent:
    def __init__(self, llm_manager):
        self.llm_manager = llm_manager

    async def plan(self, user_input: str) -> dict:
        """
        Analisa a mensagem e retorna plano de execucao.

        Returns:
            dict com keys: tasks (list), use_web (bool), use_rag (bool)
        """
        text_lower = user_input.lower()

        use_web = any(kw in text_lower for kw in WEB_SEARCH_TRIGGERS)
        use_rag = any(kw in text_lower for kw in RAG_TRIGGERS)

        logger.info(
            "PlannerAgent: mensagem='%s...' use_web=%s use_rag=%s",
            user_input[:40], use_web, use_rag
        )

        return {
            "tasks":   [user_input],
            "use_web": use_web,
            "use_rag": use_rag,
        }
