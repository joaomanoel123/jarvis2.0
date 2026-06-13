# === ARQUIVO: backend/agents/memory.py ===
"""
MemoryAgent — gerencia historico de conversa com persistencia no Firestore.
Curto prazo: deque in-memory (rapido).
Longo prazo: Firestore (persiste entre sessoes e restarts).
"""
import logging
import json
import os
from collections import deque
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

# Firestore client singleton — lazy init
_db = None

def get_db():
    """Retorna cliente Firestore. Inicializa na primeira chamada."""
    global _db
    if _db is not None:
        return _db
    try:
        import firebase_admin
        from firebase_admin import credentials, firestore

        cred_json = os.getenv("FIREBASE_CREDENTIALS")
        if not cred_json:
            logger.warning("FIREBASE_CREDENTIALS nao configurada — usando apenas memoria local.")
            return None

        cred_data = json.loads(cred_json)

        if not firebase_admin._apps:
            cred = credentials.Certificate(cred_data)
            firebase_admin.initialize_app(cred)

        _db = firestore.client()
        logger.info("Firestore conectado com sucesso.")
        return _db
    except Exception as e:
        logger.warning("Firestore nao disponivel: %s — usando memoria local.", e)
        return None


class MemoryAgent:
    def __init__(self):
        self._local: deque = deque(maxlen=20)

    def add_message(self, role: str, content: str, session_id: str = "default"):
        """Adiciona mensagem ao historico local e persiste no Firestore."""
        entry = {
            "role":      role,
            "content":   content,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "session_id": session_id,
        }
        self._local.append(entry)
        self._persist(entry, session_id)

    def _persist(self, entry: dict, session_id: str):
        """Salva mensagem no Firestore de forma assíncrona (fire-and-forget)."""
        db = get_db()
        if db is None:
            return
        try:
            db.collection("jarvis_memory") \
              .document(session_id) \
              .collection("messages") \
              .add(entry)
        except Exception as e:
            logger.warning("Falha ao persistir no Firestore: %s", e)

    def load_session(self, session_id: str, limit: int = 20):
        """Carrega historico do Firestore para a sessao especificada."""
        db = get_db()
        if db is None:
            return
        try:
            docs = (
                db.collection("jarvis_memory")
                  .document(session_id)
                  .collection("messages")
                  .order_by("timestamp")
                  .limit_to_last(limit)
                  .stream()
            )
            self._local.clear()
            for doc in docs:
                self._local.append(doc.to_dict())
            logger.info("Sessao '%s' carregada com %d mensagens.", session_id, len(self._local))
        except Exception as e:
            logger.warning("Falha ao carregar sessao do Firestore: %s", e)

    def get_conversation_history(self) -> str:
        """Retorna historico formatado para o prompt do LLM."""
        if not self._local:
            return "(sem historico)"
        return "\n".join(
            f"{m['role'].upper()}: {m['content']}" for m in self._local
        )

    def get_short_term_memory(self) -> list:
        """Retorna historico como lista de dicts — usado pelo endpoint /memory."""
        return list(self._local)

    def clear(self):
        """Limpa historico local."""
        self._local.clear()
