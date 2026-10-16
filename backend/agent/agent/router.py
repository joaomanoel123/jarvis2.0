# === ARQUIVO: backend/agent/router.py ===
"""
Router FastAPI do JARVIS 3.0 (montado em /agent/* no main.py).

  GET  /agent/repo     -> diagnostico da leitura do repo (sem LLM, nao expoe token)
  POST /agent/plan     -> plano (Planner, com arvore e arquivos de contexto reais)
  POST /agent/propose  -> plano + diffs reais (Planner + Executor)
  POST /agent/review   -> plano + diffs + verificacao estatica

Nenhum endpoint escreve em disco, executa comandos, usa git ou faz commit.
A leitura do GitHub e SOMENTE LEITURA (GET). O codigo do proprio agente
(backend/agent/) e protegido: nao aparece para o Planner e nao e proposto.
"""

import logging

from fastapi import APIRouter
from pydantic import BaseModel

from agent.executor import AgentExecutor
from agent.github_reader import RepoReader, RepoReaderError
from agent.planner import AgentPlanner
from agent.policy import is_protected, pick_entrypoints
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
CODE_CONTEXT_MAX_CHARS = 3000
MAX_FOCUS_FILES = 4


class TaskRequest(BaseModel):
    task: str
    repo_hint: str = ""  # contexto opcional em texto livre
    # arquivos que voce sabe serem relevantes; o Planner recebe o conteudo deles.
    # Se vazio, o agente usa os arquivos de entrada (ex.: backend/main.py).
    focus_files: list[str] = []


async def _load_repo_context(reader: RepoReader, focus_files: list[str]) -> dict:
    """Arvore, stack e arquivos de contexto. Levanta RepoReaderError se o repo nao abrir."""
    tree = await reader.get_tree()
    visible = [p for p in tree if not is_protected(p)]

    stack_parts = []
    for path in [p for p in visible if p.endswith(STACK_FILES)][:3]:
        content = await reader.read_file(path)
        if content:
            stack_parts.append(f"### {path}\n{content[:STACK_MAX_CHARS]}")

    if focus_files:
        chosen = [p for p in focus_files if p in visible][:MAX_FOCUS_FILES]
    else:
        chosen = pick_entrypoints(visible)

    code_parts, used = [], []
    for path in chosen:
        content = await reader.read_file(path)
        if content:
            code_parts.append(f"### {path}\n{content[:CODE_CONTEXT_MAX_CHARS]}")
            used.append(path)

    return {
        "tree": tree,
        "visible": visible,
        "stack": "\n\n".join(stack_parts),
        "code": "\n\n".join(code_parts),
        "context_files": used,
    }


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
    repo_note, ctx = "ok", None
    try:
        ctx = await _load_repo_context(reader, request.focus_files)
    except RepoReaderError as e:
        repo_note = f"repo indisponivel, plano gerado sem contexto: {e}"

    plan = await _planner.plan(
        _task_text(request),
        ctx["visible"] if ctx else None,
        ctx["stack"] if ctx else "",
        ctx["code"] if ctx else "",
    )
    return {
        "status": "success",
        "repo_context": repo_note,
        "context_files": ctx["context_files"] if ctx else [],
        "plan": plan,
    }


async def _run_pipeline(request: TaskRequest):
    """Planner + Executor com contexto real. Retorna (erro|None, dados)."""
    reader = RepoReader()
    try:
        ctx = await _load_repo_context(reader, request.focus_files)
    except RepoReaderError as e:
        return {"status": "error", "error": str(e)}, None

    plan = await _planner.plan(_task_text(request), ctx["visible"], ctx["stack"], ctx["code"])
    if plan.get("error") or not plan.get("steps"):
        return {"status": "error", "error": "Nao foi possivel gerar um plano valido.",
                "plan": plan}, None

    proposals = await _executor.propose_plan_changes(plan, reader, ctx["stack"])
    return None, {"plan": plan, "proposals": proposals, "ctx": ctx}


@agent_router.post("/propose")
async def propose_changes(request: TaskRequest):
    if not request.task.strip():
        return {"error": "Tarefa vazia.", "status": "error"}

    err, data = await _run_pipeline(request)
    if err:
        return err

    for p in data["proposals"]:
        p.pop("resulting_content", None)
    return {
        "status": "success",
        "context_files": data["ctx"]["context_files"],
        "plan": data["plan"],
        "proposals": data["proposals"],
    }


@agent_router.post("/review")
async def full_review(request: TaskRequest):
    """Pipeline completo desta fase. Nada e aplicado."""
    if not request.task.strip():
        return {"error": "Tarefa vazia.", "status": "error"}

    err, data = await _run_pipeline(request)
    if err:
        return err

    ctx = data["ctx"]
    verification = _verifier.verify_all(data["plan"], data["proposals"], ctx["tree"], ctx["stack"])
    for p in data["proposals"]:
        p.pop("resulting_content", None)

    return {
        "status": "success",
        "context_files": ctx["context_files"],
        "plan": data["plan"],
        "proposals": data["proposals"],
        "verification": verification,
        "note": "Nenhuma alteracao foi aplicada ao repositorio. Aprovacao humana obrigatoria para prosseguir.",
    }
