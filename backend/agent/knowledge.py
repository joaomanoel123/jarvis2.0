# === ARQUIVO: backend/agents/knowledge.py ===
"""
KnowledgeAgent — RAG com FAISS in-memory + embeddings via OpenRouter.
Lazy loading para nao estourar RAM no Render free tier.
"""
import logging
import asyncio
from typing import Optional

logger = logging.getLogger(__name__)

_model  = None
_index  = None
_texts  = []
_loaded = False


def _load_models():
    """Carrega SentenceTransformers e FAISS na primeira chamada (lazy)."""
    global _model, _index, _loaded
    if _loaded:
        return _model is not None

    _loaded = True
    try:
        import importlib
        st    = importlib.import_module("sentence_transformers")
        faiss = importlib.import_module("faiss")
        _model = st.SentenceTransformer("all-MiniLM-L6-v2")
        _index = faiss.IndexFlatL2(384)
        logger.info("KnowledgeAgent: modelos carregados.")
        return True
    except Exception as e:
        logger.warning("KnowledgeAgent: modelos nao disponíveis: %s", e)
        return False


class KnowledgeAgent:

    async def add_document(self, text: str, metadata: Optional[dict] = None):
        """
        Adiciona documento ao indice vetorial.
        Divide em chunks de ~500 caracteres com overlap de 50.
        """
        chunks = self._chunk(text)
        ok = await asyncio.to_thread(_load_models)
        if not ok:
            logger.warning("KnowledgeAgent.add_document: modelos indisponíveis.")
            return 0

        import numpy as np
        added = 0
        for chunk in chunks:
            try:
                emb = _model.encode([chunk])
                _index.add(np.array(emb, dtype="float32"))
                _texts.append(chunk)
                added += 1
            except Exception as e:
                logger.warning("KnowledgeAgent.add_document chunk error: %s", e)

        logger.info("KnowledgeAgent: %d chunks indexados.", added)
        return added

    async def search(self, query: str, k: int = 3) -> list[str]:
        """Busca os k chunks mais relevantes para a query."""
        if not _texts:
            return []

        ok = await asyncio.to_thread(_load_models)
        if not ok or _index is None:
            return []

        try:
            import numpy as np
            emb = _model.encode([query])
            distances, indices = _index.search(
                np.array(emb, dtype="float32"), min(k, len(_texts))
            )
            results = []
            for i, idx in enumerate(indices[0]):
                if idx < len(_texts) and distances[0][i] < 2.0:
                    results.append(_texts[idx])
            return results
        except Exception as e:
            logger.warning("KnowledgeAgent.search error: %s", e)
            return []

    def _chunk(self, text: str, size: int = 500, overlap: int = 50) -> list[str]:
        """Divide texto em chunks com overlap."""
        chunks = []
        start  = 0
        while start < len(text):
            end = start + size
            chunks.append(text[start:end])
            start = end - overlap
        return [c for c in chunks if len(c.strip()) > 20]

    def document_count(self) -> int:
        return len(_texts)
