# === ARQUIVO: backend/agents/knowledge.py ===
"""
KnowledgeAgent — consulta memória vetorial (FAISS in-memory).
Usa lazy loading de SentenceTransformers para não estourar RAM no Render free tier.
"""
import logging

logger = logging.getLogger(__name__)


class KnowledgeAgent:
    def __init__(self):
        # Lazy load — não importar SentenceTransformers no __init__
        # Evita OOM no cold start do Render (512MB RAM)
        self._index = None
        self._model = None
        self._texts: list[str] = []

    def _load_models(self):
        """Carrega SentenceTransformers e FAISS na primeira chamada."""
        if self._model is None:
            try:
                import importlib
                st = importlib.import_module("sentence_transformers")
                faiss = importlib.import_module("faiss")
                self._model = st.SentenceTransformer("all-MiniLM-L6-v2")
                self._index = faiss.IndexFlatL2(384)
                logger.info("KnowledgeAgent: modelos carregados com sucesso")
            except Exception as e:
                logger.warning("KnowledgeAgent: falha ao carregar modelos: %s", e)

    async def search(self, query: str, k: int = 3) -> list[str]:
        """Busca os k textos mais próximos semanticamente."""
        try:
            self._load_models()
            if self._model is None or self._index is None or len(self._texts) == 0:
                return []
            import numpy as np
            embedding = self._model.encode([query])
            _, indices = self._index.search(np.array(embedding, dtype="float32"), k)
            return [self._texts[i] for i in indices[0] if i < len(self._texts)]
        except Exception as e:
            logger.warning("KnowledgeAgent.search falhou: %s", e)
            return []

    async def add(self, text: str):
        """Adiciona texto à memória vetorial."""
        try:
            self._load_models()
            if self._model is None or self._index is None:
                return
            import numpy as np
            embedding = self._model.encode([text])
            self._index.add(np.array(embedding, dtype="float32"))
            self._texts.append(text)
        except Exception as e:
            logger.warning("KnowledgeAgent.add falhou: %s", e)