# === ARQUIVO: backend/agent/router.py ===
"""
Router FastAPI do JARVIS 3.0 (montado em /agent/* no main.py).

  GET  /agent/repo     -> diagnostico da leitura do repo (sem LLM, nao expoe token)
  POST /agent/plan     -> plano (Planner, com arvore real do repo)
  POST /agent/propose  -> plano + diffs reais (Planner + Executor)
  POST /agent/review   -> plano + diffs + verificacao estatica

Nenhum endpoint escreve em disco, executa comandos, usa git ou faz commit.
A leitura do GitHub e SOMENTE LEITURA (GET).
"""

import logging

from fastapi import APIRouter
from pydantic import BaseModel

from agent.executor import AgentExecutor
from agent.github_reader import RepoReader, RepoReaderError
from agent.planner import AgentPlanner
from agent.verifier import AgentVerifier
from core.llm_manager import LLMManager

logger = logging.getLogger(__name__)

agent_router = APIRouter(prefix="/agent", tags=["agent"])

_llm = LLMManager()
_planner = AgentPlanner(_llm)
_executor = AgentExecutor(_llm)
_verifier = AgentVerifier()

STACK_FILES = ("requirements.txt", "package.json")
STACK_MAX_CHARS = 1500


class TaskRequest(BaseModel):
    task: str
    repo_hint: str = ""  # contexto opcional em texto livre


async def _load_repo_context(reader: RepoReader):
    """Devolve (tree, stack_context). Levanta RepoReaderError se o repo nao abrir."""
    tree = await reader.get_tree()

    stack_parts = []
    for path in [p for p in tree if p.endswith(STACK_FILES)][:3]:
        content = await reader.read_file(path)
        if content:
            stack_parts.append(f"### {path}\n{content[:STACK_MAX_CHARS]}")
    return tree, "\n\n".join(stack_parts)


def _task_text(request: TaskRequest) -> str:
    if request.repo_hint:
        return f"[Contexto: {request.repo_hint}]\n{request.task}"
    return request.task


@agent_router.get("/repo")
async def repo_status():
    """Testa a leitura do GitHub sem gastar chamada de LLM."""
    reader = RepoReader()
    info = {
        "repo": reader.repo,
        "branch": reader.branch,
        "token_configured": reader.token_configured,
    }
    try:
        tree = await reader.get_tree()
    except RepoReaderError as e:
        return {"status": "error", "error": str(e), **info}
    return {"status": "success", "file_count": len(tree), "sample": tree[:40], **info}


@agent_router.post("/plan")
async def create_plan(request: TaskRequest):
    if not request.task.strip():
        return {"error": "Tarefa vazia.", "status": "error"}

    reader = RepoReader()
    repo_note = "ok"
    tree, stack = None, ""
    try:
        tree, stack = await _load_repo_context(reader)
    except RepoReaderError as e:
        repo_note = f"repo indisponivel, plano gerado sem contexto: {e}"

    plan = await _planner.plan(_task_text(request), tree, stack)
    return {"status": "success", "repo_context": repo_note, "plan": plan}


async def _run_pipeline(request: TaskRequest):
    """Planner + Executor com contexto real. Retorna (erro|None, dados)."""
    reader = RepoReader()
    try:
        tree, stack = await _load_repo_context(reader)
    except RepoReaderError as e:
        return {"status": "error", "error": str(e)}, None

    plan = await _planner.plan(_task_text(request), tree, stack)
    if plan.get("error") or not plan.get("steps"):
        return {"status": "error", "error": "Nao foi possivel gerar um plano valido.",
                "plan": plan}, None

    proposals = await _executor.propose_plan_changes(plan, reader, stack)
    return None, {"plan": plan, "proposals": proposals, "tree": tree, "stack": stack}


@agent_router.post("/propose")
async def propose_changes(request: TaskRequest):
    if not request.task.strip():
        return {"error": "Tarefa vazia.", "status": "error"}

    err, data = await _run_pipeline(request)
    if err:
        return err

    for p in data["proposals"]:
        p.pop("resulting_content", None)
    return {"status": "success", "plan": data["plan"], "proposals": data["proposals"]}


@agent_router.post("/review")
async def full_review(request: TaskRequest):
    """Pipeline completo desta fase. Nada e aplicado."""
    if not request.task.strip():
        return {"error": "Tarefa vazia.", "status": "error"}

    err, data = await _run_pipeline(request)
    if err:
        return err

    verification = _verifier.verify_all(
        data["plan"], data["proposals"], data["tree"], data["stack"]
    )
    for p in data["proposals"]:
        p.pop("resulting_content", None)

    return {
        "status": "success",
        "plan": data["plan"],
        "proposals": data["proposals"],
        "verification": verification,
        "note": "Nenhuma alteracao foi aplicada ao repositorio. Aprovacao humana obrigatoria para prosseguir.",
    }
