# === ARQUIVO: backend/agents/memory.py ===
"""
MemoryAgent — gerencia histórico de conversa por sessão.
Curto prazo: deque(maxlen=20) in-memory.
Sem persistência entre restarts — esperado no Render free tier.
"""
import logging
from collections import deque

logger = logging.getLogger(__name__)


class MemoryAgent:
    def __init__(self):
        self._history: deque = deque(maxlen=20)

    def add_message(self, role: str, content: str):
        """Adiciona mensagem ao histórico de curto prazo."""
        self._history.append({"role": role, "content": content})

    def get_conversation_history(self) -> str:
        """Retorna histórico formatado como string para o prompt do LLM."""
        if not self._history:
            return "(sem histórico)"
        return "\n".join(
            f"{m['role'].upper()}: {m['content']}" for m in self._history
        )

    def get_short_term_memory(self) -> list[dict]:
        """Retorna histórico como lista de dicts — usado pelo endpoint /memory."""
        return list(self._history)

    def clear(self):
        """Limpa o histórico."""
        self._history.clear()