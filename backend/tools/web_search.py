"""
tools/web_search.py
===================
web_search tool — async HTTP query via DuckDuckGo Instant Answer API.

No API key required.  Returns a concise answer + top organic results.
Production upgrade: swap the DDG call for SerpAPI / Brave Search API.
"""

from __future__ import annotations

import logging

import httpx

from config.settings import get_settings

log = logging.getLogger(__name__)
cfg = get_settings()

DDG_URL = "https://api.duckduckgo.com/"


async def web_search(query: str, max_results: int = 5) -> dict:
    """
    Search the web for a query and return a summary + top results.

    Args:
        query:       Natural-language search string.
        max_results: Maximum number of related results to include.

    Returns:
        {
            "query":      str,
            "answer":     str | None,    # Instant Answer if available
            "results":    list[dict],    # [{title, url, snippet}]
            "source":     "duckduckgo",
        }
    """
    log.info("web_search: %r", query)

    params = {
        "q":       query,
        "format":  "json",
        "no_html": 1,
        "skip_disambig": 1,
    }

    try:
        async with httpx.AsyncClient(timeout=cfg.TOOL_TIMEOUT_SECONDS) as client:
            resp = await client.get(DDG_URL, params=params)
            resp.raise_for_status()
            data = resp.json()
    except httpx.HTTPError as exc:
        raise RuntimeError(f"web_search HTTP error: {exc}") from exc

    answer  = data.get("AbstractText") or data.get("Answer") or None
    results = [
        {"title": r.get("Text", ""), "url": r.get("FirstURL", ""), "snippet": ""}
        for r in data.get("RelatedTopics", [])[:max_results]
        if isinstance(r, dict) and "FirstURL" in r
    ]

    return {
        "query":   query,
        "answer":  answer,
        "results": results,
        "source":  "duckduckgo",
    }
