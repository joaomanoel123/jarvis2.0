# === ARQUIVO: backend/agents/memory.py ===
"""
MemoryAgent — gerencia historico de conversa por sessao.
Curto prazo: deque(maxlen=20) in-memory.
"""
import logging
from collections import deque

logger = logging.getLogger(__name__)


class MemoryAgent:
    def __init__(self):
        self._history = deque(maxlen=20)

    def add_message(self, role: str, content: str):
        """Adiciona mensagem ao historico de curto prazo."""
        self._history.append({"role": role, "content": content})

    def get_conversation_history(self) -> str:
        """Retorna historico formatado como string para o prompt do LLM."""
        if not self._history:
            return "(sem historico)"
        return "\n".join(
            f"{m['role'].upper()}: {m['content']}" for m in self._history
        )

    def get_short_term_memory(self) -> list:
        """Retorna historico como lista de dicts."""
        return list(self._history)

    def clear(self):
        """Limpa o historico."""
        self._history.clear()
