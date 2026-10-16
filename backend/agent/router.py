# === ARQUIVO: backend/agent/router.py ===
"""
Router FastAPI do JARVIS 3.0 — Fase 1 (Agent Orchestrator).

Monta em /agent/* no mesmo backend do JARVIS 2.0 (main.py: app.include_router(agent_router)).

Fluxo desta fase:
  POST /agent/plan     -> gera plano estruturado (Planner)
  POST /agent/propose  -> gera plano + propostas de diff (Planner + Executor)
  POST /agent/review    -> gera plano + diffs + verificacao estatica (Planner + Executor + Verifier)

IMPORTANTE: nenhum endpoint aqui escreve em disco, executa comandos,
roda git ou faz commit. Tudo e proposta, para revisao humana.
"""

import logging

from fastapi import APIRouter
from pydantic import BaseModel

from agent.executor import AgentExecutor
from agent.planner import AgentPlanner
from agent.verifier import AgentVerifier
from core.llm_manager import LLMManager

logger = logging.getLogger(__name__)

agent_router = APIRouter(prefix="/agent", tags=["agent"])

_llm = LLMManager()
_planner = AgentPlanner(_llm)
_executor = AgentExecutor(_llm)
_verifier = AgentVerifier()


class TaskRequest(BaseModel):
    task: str
    repo_hint: str = ""  # nome/descricao opcional do repo-alvo, so para contexto no prompt


@agent_router.post("/plan")
async def create_plan(request: TaskRequest):
    if not request.task.strip():
        return {"error": "Tarefa vazia.", "status": "error"}

    task_text = request.task
    if request.repo_hint:
        task_text = f"[Repositorio alvo: {request.repo_hint}]\n{task_text}"

    plan = await _planner.plan(task_text)
    return {"status": "success", "plan": plan}


@agent_router.post("/propose")
async def propose_changes(request: TaskRequest):
    if not request.task.strip():
        return {"error": "Tarefa vazia.", "status": "error"}

    task_text = request.task
    if request.repo_hint:
        task_text = f"[Repositorio alvo: {request.repo_hint}]\n{task_text}"

    plan = await _planner.plan(task_text)
    if plan.get("error") or not plan.get("steps"):
        return {"status": "error", "error": "Nao foi possivel gerar um plano valido.", "plan": plan}

    proposals = await _executor.propose_plan_changes(plan)
    return {"status": "success", "plan": plan, "proposals": proposals}


@agent_router.post("/review")
async def full_review(request: TaskRequest):
    """Pipeline completo desta fase: Planner -> Executor -> Verifier. Nada e aplicado."""
    if not request.task.strip():
        return {"error": "Tarefa vazia.", "status": "error"}

    task_text = request.task
    if request.repo_hint:
        task_text = f"[Repositorio alvo: {request.repo_hint}]\n{task_text}"

    plan = await _planner.plan(task_text)
    if plan.get("error") or not plan.get("steps"):
        return {"status": "error", "error": "Nao foi possivel gerar um plano valido.", "plan": plan}

    proposals = await _executor.propose_plan_changes(plan)
    verification = _verifier.verify_all(plan, proposals)

    return {
        "status": "success",
        "plan": plan,
        "proposals": proposals,
        "verification": verification,
        "note": "Nenhuma alteracao foi aplicada ao repositorio. Aprovacao humana obrigatoria para prosseguir.",
    }
