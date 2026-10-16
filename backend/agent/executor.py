# === ARQUIVO: backend/agents/executor.py ===
"""
ExecutorAgent — executa acoes externas.
Suporta: busca web via DuckDuckGo (sem API key necessaria).
"""
import logging
import httpx

logger = logging.getLogger(__name__)

DDGO_URL = "https://api.duckduckgo.com/"


class ExecutorAgent:

    async def execute(self, task: str) -> str:
        """Executa a task — por padrao retorna a task sem processamento."""
        return task

    async def web_search(self, query: str, max_results: int = 4) -> str:
        """
        Busca informacoes na web via DuckDuckGo Instant Answer API.
        Nao requer API key — gratuito e sem rate limit agressivo.
        """
        logger.info("ExecutorAgent web_search: '%s'", query[:60])
        try:
            async with httpx.AsyncClient(timeout=8.0) as client:
                resp = await client.get(DDGO_URL, params={
                    "q": query,
                    "format": "json",
                    "no_redirect": "1",
                    "no_html": "1",
                    "skip_disambig": "1",
                })
                data = resp.json()

            results = []

            # Abstract (resposta direta)
            if data.get("Abstract"):
                results.append(f"Resumo: {data['Abstract']}")

            # Answer (resposta curta)
            if data.get("Answer"):
                results.append(f"Resposta: {data['Answer']}")

            # Related topics
            for topic in data.get("RelatedTopics", [])[:max_results]:
                if isinstance(topic, dict) and topic.get("Text"):
                    results.append(topic["Text"])

            if results:
                return "Resultados da busca web:\n" + "\n".join(f"- {r}" for r in results)
            else:
                return f"Nenhum resultado encontrado para: {query}"

        except Exception as e:
            logger.warning("ExecutorAgent web_search falhou: %s", e)
            return f"Falha na busca web para: {query}"
