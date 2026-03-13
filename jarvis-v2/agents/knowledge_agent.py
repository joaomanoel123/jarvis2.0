"""
agents/knowledge_agent.py
=========================
KnowledgeAgent — retrieves and synthesises knowledge.

Current implementation
──────────────────────
• Performs a semantic web search (web_search tool) to ground answers.
• Calls the LLM to synthesise search results + conversation history into
  a coherent, cited knowledge response.

Embeddings / vector DB upgrade path
─────────────────────────────────────
The `_vector_search()` stub below is the insertion point for a local
embedding store (FAISS, Chroma, Qdrant).  Replace the stub with a real
vector retrieval call and the rest of the agent is unchanged.

Example upgrade:
    from sentence_transformers import SentenceTransformer
    from chromadb import Client

    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    vdb = Client()  # or PersistentClient(path="./chroma_db")

    async def _vector_search(query: str) -> list[str]:
        vec = embedder.encode(query).tolist()
        results = vdb.get_collection("jarvis_kb").query(
            query_embeddings=[vec], n_results=5
        )
        return results["documents"][0]
"""

from __future__ import annotations

import logging

from agents.base_agent import AgentTask, AgentResult, AgentType, BaseAgent
from services.llm_service import llm_service
from tools import tool_registry

log = logging.getLogger(__name__)

_KNOWLEDGE_SYSTEM = """\
You are JARVIS Knowledge Engine.

You have access to real-time search results and conversation context.
Your job is to produce a clear, accurate, well-structured answer.

Guidelines:
  • Cite sources when you reference search results.
  • If search results are not relevant, rely on your training knowledge.
  • Prefer concise factual answers over speculation.
  • Use bullet points for multi-part answers.
"""


class KnowledgeAgent(BaseAgent):
    """
    Retrieves information from web search and/or a local vector store,
    then synthesises a grounded knowledge response.
    """

    agent_type = AgentType.KNOWLEDGE

    async def run(self, task: AgentTask) -> AgentResult:
        query = task.user_input
        log.info("KnowledgeAgent: query=%r", query[:80])

        # Phase 1 — gather evidence
        search_snippets: list[str] = []
        tool_calls: list[dict]     = []

        # (a) Web search
        web_result = await tool_registry.execute("web_search", {"query": query})
        tool_calls.append(web_result)
        if web_result["success"]:
            data  = web_result["result"]
            answer = data.get("answer") or ""
            links  = data.get("results", [])
            if answer:
                search_snippets.append(f"Direct answer: {answer}")
            for r in links[:3]:
                if r.get("title"):
                    search_snippets.append(f"• {r['title']} ({r.get('url', '')})")

        # (b) Vector store (stub — returns [] until upgraded)
        vector_hits = await _vector_search(query)
        search_snippets.extend(vector_hits)

        # Phase 2 — synthesise
        context_block = (
            "Search evidence:\n" + "\n".join(search_snippets)
            if search_snippets
            else "No search results available."
        )

        synthesis_prompt = (
            f"User question: {query}\n\n"
            f"{context_block}\n\n"
            "Produce a comprehensive answer based on the evidence above."
        )

        synthesis = await llm_service.chat(
            messages=task.history + [{"role": "user", "content": synthesis_prompt}],
            system=_KNOWLEDGE_SYSTEM,
            temperature=0.4,
        )

        return AgentResult(
            text=synthesis.text,
            agent=self.agent_type,
            tool_calls=tool_calls,
            metadata={
                "sources_found": len(search_snippets),
                "vector_hits":   len(vector_hits),
            },
        )


# ── Vector search stub ─────────────────────────────────────────────────────────

async def _vector_search(query: str) -> list[str]:
    """
    Placeholder for a local vector database retrieval.

    Replace this function body with your embedding + ANN search logic.
    Returns a list of text chunks relevant to the query.
    """
    # Example upgrade:
    # loop = asyncio.get_running_loop()
    # hits = await loop.run_in_executor(None, _sync_vector_search, query)
    # return hits
    return []
