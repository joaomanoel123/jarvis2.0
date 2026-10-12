# === ARQUIVO: backend/agent/planner.py ===
"""
AgentPlanner — JARVIS 3.0 Fase 1

Recebe uma tarefa de programacao em linguagem natural e, via OpenRouter
(reaproveitando o LLMManager ja usado pelo JARVIS 2.0), devolve um PLANO
ESTRUTURADO em JSON. Nao toca em arquivos, nao executa nada.
"""

import json
import logging
import re
from typing import Any

from core.llm_manager import LLMManager

logger = logging.getLogger(__name__)

PLANNER_SYSTEM_PROMPT = """Voce e o modulo PLANNER de um agente autonomo de programacao chamado JARVIS 3.0.

Sua UNICA funcao e transformar um pedido em linguagem natural em um plano estruturado.
Voce NUNCA escreve codigo final nem edita arquivos — apenas planeja.

Responda SEMPRE em JSON valido, com exatamente este formato:
{
  "goal": "resumo curto do objetivo",
  "steps": [
    {
      "id": 1,
      "description": "descricao clara do passo",
      "target_files": ["caminho/relativo/arquivo.py"],
      "action": "create | modify | delete",
      "risk": "low | medium | high"
    }
  ],
  "requires_new_dependencies": false,
  "notes": "observacoes relevantes, riscos, ambiguidades"
}

Regras:
- "target_files" deve conter caminhos RELATIVOS plausiveis dentro do repositorio.
- "risk" = "high" para qualquer passo que apague arquivos, mexa em autenticacao,
  segredos, configuracao de deploy ou infraestrutura.
- Nao invente tarefas fora do pedido do usuario.
- Nao inclua texto fora do JSON. Nao use markdown (sem ```)."""


class AgentPlanner:
    def __init__(self, llm_manager: LLMManager | None = None):
        self.llm = llm_manager or LLMManager()

    async def plan(self, task_description: str) -> dict[str, Any]:
        """
        Gera um plano estruturado para a tarefa descrita.
        Nunca levanta excecao para o chamador — em falha, devolve um plano
        de erro com steps vazios para o Orchestrator/endpoint tratar.
        """
        prompt = (
            f"{PLANNER_SYSTEM_PROMPT}\n\n"
            f"Tarefa solicitada pelo usuario:\n\"\"\"\n{task_description}\n\"\"\"\n\n"
            f"Gere o plano em JSON agora."
        )

        try:
            raw = await self.llm.query(prompt)
            plan = self._parse_json(raw)
            plan = self._validate_plan(plan)
            logger.info("Planner: plano gerado com %d passo(s).", len(plan.get("steps", [])))
            return plan
        except Exception as e:
            logger.error("Planner: falha ao gerar/parsear plano: %s", e, exc_info=True)
            return {
                "goal": task_description[:200],
                "steps": [],
                "requires_new_dependencies": False,
                "notes": f"Falha ao gerar plano automaticamente: {e}",
                "error": True,
            }

    @staticmethod
    def _parse_json(raw: str) -> dict[str, Any]:
        """Extrai o primeiro bloco JSON valido da resposta do LLM."""
        text = raw.strip()

        # Remove cercas de markdown, se o modelo ignorar a instrucao
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Fallback: extrai o maior trecho entre a primeira '{' e a ultima '}'
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise ValueError("Resposta do LLM nao contem JSON reconhecivel.")

        candidate = text[start : end + 1]
        return json.loads(candidate)

    @staticmethod
    def _validate_plan(plan: dict[str, Any]) -> dict[str, Any]:
        """Garante formato minimo esperado, preenchendo defaults seguros."""
        if not isinstance(plan, dict):
            raise ValueError("Plano nao e um objeto JSON.")

        plan.setdefault("goal", "")
        plan.setdefault("steps", [])
        plan.setdefault("requires_new_dependencies", False)
        plan.setdefault("notes", "")

        if not isinstance(plan["steps"], list):
            raise ValueError("Campo 'steps' deve ser uma lista.")

        normalized_steps = []
        for i, step in enumerate(plan["steps"], start=1):
            if not isinstance(step, dict):
                continue
            normalized_steps.append(
                {
                    "id": step.get("id", i),
                    "description": str(step.get("description", "")).strip(),
                    "target_files": step.get("target_files", []) or [],
                    "action": step.get("action", "modify"),
                    "risk": step.get("risk", "medium"),
                }
            )
        plan["steps"] = normalized_steps
        return plan
