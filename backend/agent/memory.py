# === ARQUIVO: backend/agents/memory.py ===
"""
MemoryAgent — historico de conversa com persistencia no Firestore.
"""
import logging
import json
import os
from collections import deque
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

_db = None

def get_db():
    global _db
    if _db is not None:
        return _db
    try:
        import firebase_admin
        from firebase_admin import credentials, firestore

        cred_json = os.getenv("FIREBASE_CREDENTIALS")
        if not cred_json:
            logger.warning("FIREBASE_CREDENTIALS nao configurada.")
            return None

        cred_data = json.loads(cred_json)

        if not firebase_admin._apps:
            cred = credentials.Certificate(cred_data)
            firebase_admin.initialize_app(cred)

        _db = firestore.client()
        logger.info("Firestore conectado com sucesso.")
        return _db
    except Exception as e:
        logger.warning("Firestore nao disponivel: %s", e)
        return None


class MemoryAgent:
    def __init__(self):
        self._local: deque = deque(maxlen=20)

    def add_message(self, role: str, content: str, session_id: str = "default"):
        entry = {
            "role":       role,
            "content":    content,
            "timestamp":  datetime.now(timezone.utc).isoformat(),
            "session_id": session_id,
        }
        self._local.append(entry)
        self._persist(entry, session_id)

    def _persist(self, entry: dict, session_id: str):
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
        """
        Carrega historico do Firestore.
        FIX: usa .get() em vez de .stream() para evitar erro com limit_to_last().
        """
        db = get_db()
        if db is None:
            return
        try:
            # FIX: order_by + limit normal + get() em vez de limit_to_last() + stream()
            docs = (
                db.collection("jarvis_memory")
                  .document(session_id)
                  .collection("messages")
                  .order_by("timestamp")
                  .limit(limit)
                  .get()
            )
            self._local.clear()
            for doc in docs:
                self._local.append(doc.to_dict())
            logger.info("Sessao '%s': %d mensagens carregadas.", session_id, len(self._local))
        except Exception as e:
            logger.warning("Falha ao carregar sessao: %s", e)

    def get_conversation_history(self) -> str:
        if not self._local:
            return "(sem historico)"
        return "\n".join(
            f"{m['role'].upper()}: {m['content']}" for m in self._local
        )

    def get_short_term_memory(self) -> list:
        return list(self._local)

    def clear(self):
        self._local.clear()
