# === ARQUIVO: backend/agent/github_reader.py ===
"""
RepoReader — JARVIS 3.0 Fase 1.5 (SOMENTE LEITURA)

Le a arvore de arquivos e o conteudo de arquivos de um repo no GitHub
via API REST. Usa apenas requisicoes GET — nao existe nenhum metodo de escrita.

Env vars (todas opcionais):
  GITHUB_TOKEN   token com permissao SOMENTE LEITURA (Contents: read-only).
                 Sem token funciona para repo publico, com limite de 60 req/h.
  GITHUB_REPO    default "joaomanoel123/jarvis2.0"
  GITHUB_BRANCH  default "main"
"""

import logging
import os
import time
from urllib.parse import quote

import httpx

logger = logging.getLogger(__name__)

API = "https://api.github.com"

IGNORED_PREFIXES = (
    ".git/", ".agents/", ".claude/", ".cursor/",
    "node_modules/", "__pycache__/", "default/icons/",
)
IGNORED_EXT = (
    ".png", ".jpg", ".jpeg", ".gif", ".ico", ".svg", ".woff", ".woff2",
    ".ttf", ".zip", ".pdf", ".mp3", ".mp4", ".lock", ".pyc",
)
SENSITIVE_HINTS = (
    ".env", "credentials", "service-account", "serviceaccount",
    "id_rsa", ".pem", ".key", "secret",
)

MAX_TREE_FILES = 500
MAX_FILE_CHARS = 20000
CACHE_TTL_SECONDS = 300

_tree_cache: dict[tuple[str, str], tuple[float, list[str]]] = {}


class RepoReaderError(Exception):
    """Falha ao ler o repositorio (mensagem ja pronta para o usuario)."""


def is_blocked(path: str) -> bool:
    """Caminhos que o agente nunca le (segredos, chaves, traversal)."""
    p = path.strip().lstrip("/")
    if not p or ".." in p.split("/"):
        return True
    low = p.lower()
    return any(h in low for h in SENSITIVE_HINTS)


class RepoReader:
    def __init__(self, repo: str | None = None, branch: str | None = None,
                 token: str | None = None):
        self.repo = repo or os.getenv("GITHUB_REPO", "joaomanoel123/jarvis2.0")
        self.branch = branch or os.getenv("GITHUB_BRANCH", "main")
        self.token = token if token is not None else os.getenv("GITHUB_TOKEN", "")

    @property
    def token_configured(self) -> bool:
        return bool(self.token)

    def _headers(self, accept: str = "application/vnd.github+json") -> dict:
        headers = {
            "Accept": accept,
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "jarvis-agent-readonly",
        }
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        return headers

    async def get_tree(self) -> list[str]:
        """Lista de caminhos de arquivos do repo (filtrada, com cache de 5 min)."""
        key = (self.repo, self.branch)
        cached = _tree_cache.get(key)
        if cached and time.time() - cached[0] < CACHE_TTL_SECONDS:
            return cached[1]

        url = f"{API}/repos/{self.repo}/git/trees/{quote(self.branch)}"
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                resp = await client.get(url, params={"recursive": "1"},
                                        headers=self._headers())
        except httpx.HTTPError as e:
            raise RepoReaderError(f"Falha de rede ao acessar o GitHub: {e}") from e

        if resp.status_code == 404:
            raise RepoReaderError(
                f"Repo/branch nao encontrado ({self.repo}@{self.branch}) ou sem acesso. "
                "Se o repo e privado, configure GITHUB_TOKEN (somente leitura)."
            )
        if resp.status_code in (401, 403):
            raise RepoReaderError(
                f"GitHub recusou o acesso (HTTP {resp.status_code}). Token invalido/sem "
                "permissao de leitura, ou limite de requisicoes atingido."
            )
        if resp.status_code != 200:
            raise RepoReaderError(f"GitHub respondeu HTTP {resp.status_code}.")

        paths = []
        for item in resp.json().get("tree", []):
            if item.get("type") != "blob":
                continue
            path = item.get("path", "")
            if path.startswith(IGNORED_PREFIXES) or path.lower().endswith(IGNORED_EXT):
                continue
            if is_blocked(path):
                continue
            paths.append(path)

        paths = sorted(paths)[:MAX_TREE_FILES]
        _tree_cache[key] = (time.time(), paths)
        logger.info("RepoReader: %d arquivos em %s@%s", len(paths), self.repo, self.branch)
        return paths

    async def read_file(self, path: str) -> str | None:
        """Conteudo (texto) de um arquivo, ou None se nao existir/bloqueado."""
        if is_blocked(path):
            logger.warning("RepoReader: leitura bloqueada para '%s'", path)
            return None

        url = f"{API}/repos/{self.repo}/contents/{quote(path.lstrip('/'))}"
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                resp = await client.get(
                    url, params={"ref": self.branch},
                    headers=self._headers("application/vnd.github.raw+json"),
                )
        except httpx.HTTPError as e:
            logger.warning("RepoReader: erro ao ler '%s': %s", path, e)
            return None

        if resp.status_code != 200:
            return None
        return resp.text[:MAX_FILE_CHARS]
