# === ARQUIVO: backend/agent/planner.py ===
"""
AgentPlanner — JARVIS 3.0

Transforma uma tarefa em linguagem natural em um PLANO ESTRUTURADO (JSON),
via OpenRouter (LLMManager do JARVIS 2.0). Recebe, opcionalmente, a arvore
real de arquivos do repo e o contexto de stack (requirements/package.json)
para planejar com caminhos e frameworks reais. Nao toca em arquivos.
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
- Cada passo deve ter EXATAMENTE 1 arquivo em "target_files".
- Para "modify" e "delete", use SOMENTE caminhos que existam na lista de arquivos do repositorio.
- Para "create", escolha um caminho coerente com a estrutura existente.
- Use o framework e as bibliotecas que o projeto ja usa (veja o contexto de stack). Nao introduza outro framework.
- Prefira o MENOR numero de passos possivel; nao divida em passos o que cabe em um so.
- "risk" = "high" para qualquer passo que apague arquivos, mexa em autenticacao,
  segredos, configuracao de deploy ou infraestrutura.
- Nao invente tarefas fora do pedido do usuario.
- Nao inclua texto fora do JSON. Nao use markdown (sem ```)."""


class AgentPlanner:
    def __init__(self, llm_manager: LLMManager | None = None):
        self.llm = llm_manager or LLMManager()

    async def plan(
        self,
        task_description: str,
        file_tree: list[str] | None = None,
        stack_context: str = "",
    ) -> dict[str, Any]:
        """
        Gera um plano estruturado. Nunca levanta excecao: em falha devolve
        um plano com steps vazios e "error": True.
        """
        context = ""
        if file_tree:
            context += "Arquivos existentes no repositorio:\n" + "\n".join(file_tree) + "\n\n"
        if stack_context:
            context += f"Contexto de stack (dependencias do projeto):\n{stack_context}\n\n"

        prompt = (
            f"{PLANNER_SYSTEM_PROMPT}\n\n"
            f"{context}"
            f"Tarefa solicitada pelo usuario:\n\"\"\"\n{task_description}\n\"\"\"\n\n"
            f"Gere o plano em JSON agora."
        )

        try:
            raw = await self.llm.query(prompt)
            plan = self._parse_json(raw)
            plan = self._validate_plan(plan)
            logger.info("Planner: plano com %d passo(s).", len(plan.get("steps", [])))
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
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise ValueError("Resposta do LLM nao contem JSON reconhecivel.")
        return json.loads(text[start : end + 1])

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
            targets = step.get("target_files", []) or []
            if isinstance(targets, str):
                targets = [targets]
            normalized_steps.append(
                {
                    "id": step.get("id", i),
                    "description": str(step.get("description", "")).strip(),
                    "target_files": targets,
                    "action": step.get("action", "modify"),
                    "risk": step.get("risk", "medium"),
                }
            )
        plan["steps"] = normalized_steps
        return plan
