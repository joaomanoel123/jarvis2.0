"""
backend/memory/vector_memory.py
=================================
VectorMemory — semantic retrieval for long-term knowledge storage.

Architecture
────────────
• Embedding model: sentence-transformers (all-MiniLM-L6-v2, 384-dim).
  Runs on CPU, ~80 MB, no GPU required.
• Index backend: FAISS flat inner-product index.
  In-process, no external service required.
• Persistence: index saved to disk on every write (FAISS .index + JSON metadata).
• Upgrade path: replace _encode() and _search() with a Chroma / Qdrant / Pinecone
  client — the public async API is identical.

Entry schema
────────────
    {
        "id":         str (UUID),
        "text":       str,
        "session_id": str | None,
        "source":     str ("conversation" | "tool" | "document" | "user"),
        "metadata":   dict,
        "ts":         float,
    }

Usage
─────
    vm = VectorMemory()
    await vm.load()
    await vm.add("Python uses indentation for code blocks", source="fact")
    results = await vm.search("How does Python handle code blocks?", top_k=3)
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

log = logging.getLogger("jarvis.memory.vector")

# Storage paths
VECTOR_DIR       = Path(os.getenv("VECTOR_MEMORY_DIR", "./data/vector_memory"))
INDEX_FILE       = VECTOR_DIR / "jarvis.index"
METADATA_FILE    = VECTOR_DIR / "metadata.json"

# Model config
EMBED_MODEL      = "all-MiniLM-L6-v2"
EMBED_DIM        = 384
MAX_ENTRIES      = 10_000   # maximum stored vectors before LRU eviction


# ── Result type ────────────────────────────────────────────────────────────────

@dataclass
class VectorSearchResult:
    id:         str
    text:       str
    score:      float           # cosine similarity [0–1]
    session_id: str | None      = None
    source:     str             = "unknown"
    metadata:   dict            = field(default_factory=dict)
    ts:         float           = 0.0


# ── VectorMemory ───────────────────────────────────────────────────────────────

class VectorMemory:
    """
    Async semantic memory with FAISS + sentence-transformers.

    All public methods are async and safe to call from FastAPI handlers.
    CPU-bound embedding and index operations are offloaded to a thread pool.
    """

    def __init__(self, model_name: str = EMBED_MODEL) -> None:
        self._model_name  = model_name
        self._model       = None     # SentenceTransformer
        self._index       = None     # faiss.IndexFlatIP
        self._metadata:   list[dict] = []
        self._loaded      = False
        self._lock        = asyncio.Lock()
        self._executor    = None     # ThreadPoolExecutor

    # ── Lifecycle ──────────────────────────────────────────────────────────────

    async def load(self) -> "VectorMemory":
        """Load the embedding model and restore the index from disk if it exists."""
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._load_sync)
        self._loaded = True
        log.info("VectorMemory loaded  model=%s  entries=%d",
                 self._model_name, len(self._metadata))
        return self

    def _load_sync(self) -> None:
        """Synchronous init — runs in thread pool."""
        from concurrent.futures import ThreadPoolExecutor
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="vector")

        # Load embedding model
        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self._model_name)
            log.debug("SentenceTransformer '%s' loaded", self._model_name)
        except ImportError:
            log.warning(
                "sentence-transformers not installed — vector memory disabled.\n"
                "Install: pip install sentence-transformers"
            )
            self._model = None
            return

        # Init FAISS index
        try:
            import faiss
            if INDEX_FILE.exists():
                self._index = faiss.read_index(str(INDEX_FILE))
                log.debug("FAISS index restored from %s", INDEX_FILE)
            else:
                self._index = faiss.IndexFlatIP(EMBED_DIM)
                log.debug("FAISS index created fresh")
        except ImportError:
            log.warning(
                "faiss-cpu not installed — falling back to numpy brute-force search.\n"
                "Install: pip install faiss-cpu"
            )
            self._index = None

        # Restore metadata
        if METADATA_FILE.exists():
            with open(METADATA_FILE) as f:
                self._metadata = json.load(f)
        else:
            self._metadata = []

    # ── Write ──────────────────────────────────────────────────────────────────

    async def add(
        self,
        text:       str,
        session_id: str | None = None,
        source:     str        = "conversation",
        metadata:   dict | None = None,
    ) -> str:
        """
        Embed and store a text entry.

        Returns:
            Entry UUID.
        """
        if not self._loaded or self._model is None:
            return ""

        entry_id = str(uuid.uuid4())
        entry = {
            "id":         entry_id,
            "text":       text[:2000],   # cap entry size
            "session_id": session_id,
            "source":     source,
            "metadata":   metadata or {},
            "ts":         time.time(),
        }

        loop = asyncio.get_running_loop()
        await loop.run_in_executor(self._executor, self._add_sync, entry)
        return entry_id

    def _add_sync(self, entry: dict) -> None:
        vec = self._encode(entry["text"])
        async_lock = asyncio.new_event_loop()

        if self._index is not None:
            import faiss
            vec_f32 = vec.astype(np.float32).reshape(1, -1)
            faiss.normalize_L2(vec_f32)
            self._index.add(vec_f32)
        else:
            # Numpy fallback: store in metadata
            entry["_vec"] = vec.tolist()

        self._metadata.append(entry)

        # LRU eviction
        if len(self._metadata) > MAX_ENTRIES:
            n_remove = len(self._metadata) - MAX_ENTRIES
            self._metadata = self._metadata[n_remove:]
            if self._index is not None:
                # Rebuild index on eviction (simple but correct)
                self._rebuild_index()

        self._save()

    # ── Search ─────────────────────────────────────────────────────────────────

    async def search(
        self,
        query:      str,
        top_k:      int          = 5,
        session_id: str | None   = None,
        min_score:  float        = 0.3,
    ) -> list[VectorSearchResult]:
        """
        Semantic search for entries similar to query.

        Args:
            query:      Natural language search string.
            top_k:      Maximum results to return.
            session_id: Filter to one session (None = search all).
            min_score:  Minimum cosine similarity threshold.

        Returns:
            List of VectorSearchResult sorted by score descending.
        """
        if not self._loaded or self._model is None or not self._metadata:
            return []

        loop = asyncio.get_running_loop()
        results = await loop.run_in_executor(
            self._executor,
            self._search_sync, query, top_k, session_id, min_score,
        )
        return results

    def _search_sync(
        self,
        query:      str,
        top_k:      int,
        session_id: str | None,
        min_score:  float,
    ) -> list[VectorSearchResult]:
        vec = self._encode(query)

        if self._index is not None and self._index.ntotal > 0:
            import faiss
            vec_f32 = vec.astype(np.float32).reshape(1, -1)
            faiss.normalize_L2(vec_f32)
            scores, indices = self._index.search(vec_f32, min(top_k * 3, self._index.ntotal))
            candidates = [
                (float(scores[0][i]), self._metadata[indices[0][i]])
                for i in range(len(indices[0]))
                if indices[0][i] >= 0 and indices[0][i] < len(self._metadata)
            ]
        else:
            # Numpy brute-force fallback
            candidates = self._numpy_search(vec, top_k * 3)

        # Filter by session and score
        results = []
        for score, entry in candidates:
            if session_id and entry.get("session_id") != session_id:
                continue
            if score < min_score:
                continue
            results.append(VectorSearchResult(
                id=entry["id"],
                text=entry["text"],
                score=round(score, 4),
                session_id=entry.get("session_id"),
                source=entry.get("source", "unknown"),
                metadata=entry.get("metadata", {}),
                ts=entry.get("ts", 0.0),
            ))

        results.sort(key=lambda r: r.score, reverse=True)
        return results[:top_k]

    def _numpy_search(self, query_vec: np.ndarray, top_k: int) -> list[tuple[float, dict]]:
        """Brute-force cosine search (no FAISS)."""
        if not self._metadata:
            return []
        vecs = np.array([
            e.get("_vec", [0.0] * EMBED_DIM)
            for e in self._metadata
        ], dtype=np.float32)

        # Normalise
        q = query_vec / (np.linalg.norm(query_vec) + 1e-8)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-8
        vecs_normed = vecs / norms
        scores = vecs_normed @ q

        top_idx = np.argsort(scores)[::-1][:top_k]
        return [(float(scores[i]), self._metadata[i]) for i in top_idx]

    # ── Persistence ────────────────────────────────────────────────────────────

    def _save(self) -> None:
        VECTOR_DIR.mkdir(parents=True, exist_ok=True)
        if self._index is not None:
            try:
                import faiss
                faiss.write_index(self._index, str(INDEX_FILE))
            except Exception as exc:
                log.debug("FAISS save error: %s", exc)
        # Save metadata (without embedded vectors to keep file small)
        meta_clean = [{k: v for k, v in e.items() if k != "_vec"}
                      for e in self._metadata]
        with open(METADATA_FILE, "w") as f:
            json.dump(meta_clean, f)

    def _rebuild_index(self) -> None:
        """Rebuild FAISS index from current metadata after eviction."""
        try:
            import faiss
            self._index = faiss.IndexFlatIP(EMBED_DIM)
            for entry in self._metadata:
                text = entry.get("text", "")
                if text:
                    vec = self._encode(text).astype(np.float32).reshape(1, -1)
                    faiss.normalize_L2(vec)
                    self._index.add(vec)
        except Exception as exc:
            log.warning("Index rebuild failed: %s", exc)

    # ── Embedding ──────────────────────────────────────────────────────────────

    def _encode(self, text: str) -> np.ndarray:
        """Embed a text string using the sentence transformer."""
        return self._model.encode(text, normalize_embeddings=True)

    # ── Diagnostics ────────────────────────────────────────────────────────────

    async def stats(self) -> dict:
        index_size = self._index.ntotal if self._index else len(self._metadata)
        return {
            "loaded":       self._loaded,
            "model":        self._model_name,
            "entries":      len(self._metadata),
            "index_size":   index_size,
            "index_file":   str(INDEX_FILE),
        }

    async def delete(self, entry_id: str) -> bool:
        """Remove an entry by ID (marks as deleted; rebuilds index on next write)."""
        async with self._lock:
            before = len(self._metadata)
            self._metadata = [e for e in self._metadata if e["id"] != entry_id]
            if len(self._metadata) < before:
                loop = asyncio.get_running_loop()
                await loop.run_in_executor(self._executor, self._rebuild_index)
                await loop.run_in_executor(self._executor, self._save)
                return True
        return False


# Module singleton
vector_memory = VectorMemory()
