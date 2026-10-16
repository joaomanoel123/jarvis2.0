# === ARQUIVO: backend/agent/policy.py ===
"""
Politicas do agente JARVIS 3.0.

- PROTECTED_PREFIXES: o agente NUNCA propoe mudancas no proprio codigo
  (evita que ele enfraqueca o Verifier/politicas sem revisao humana).
- pick_entrypoints: escolhe arquivos de entrada (main/app/server/index) para
  dar ao Planner contexto real de onde a API/aplicacao e definida.
"""

import re

PROTECTED_PREFIXES = ("backend/agent/",)

_ENTRY_RE = re.compile(r"(^|/)(main|app|server|index)\.(py|js|ts)$")


def is_protected(path: str) -> bool:
    p = path.strip().lstrip("/")
    return any(p.startswith(prefix) for prefix in PROTECTED_PREFIXES)


def pick_entrypoints(tree: list[str], limit: int = 2) -> list[str]:
    """Arquivos de entrada mais rasos da arvore (ex.: backend/main.py)."""
    candidates = [p for p in tree if _ENTRY_RE.search(p)]
    candidates.sort(key=lambda p: (p.count("/"), p))
    return candidates[:limit]
