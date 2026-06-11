# === ARQUIVO: backend/agents/knowledge.py ===
"""
KnowledgeAgent — consulta memoria vetorial.
Stub funcional: retorna lista vazia sem dependencias pesadas.
"""
import logging

logger = logging.getLogger(__name__)


class KnowledgeAgent:
    def __init__(self):
        self._texts = []

    async def search(self, query: str, k: int = 3) -> list:
        """Busca textos similares — retorna vazio nesta versao stub."""
        return []

    async def add(self, text: str):
        """Adiciona texto ao indice — no-op nesta versao stub."""
        self._texts.append(text)
