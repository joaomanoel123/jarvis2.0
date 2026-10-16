# === ARQUIVO: backend/agent/executor.py ===
"""
AgentExecutor — JARVIS 3.0 Fase 1

Recebe um passo do plano e GERA UM DIFF PROPOSTO via LLM.
NAO escreve em disco, NAO executa comandos, NAO toca em git.
Apenas simula/propoe — a aplicacao real fica para uma fase futura,
com sandbox Docker isolado e aprovacao humana explicita.
"""

import logging
from typing import Any

from core.llm_manager import LLMManager

logger = logging.getLogger(__name__)

EXECUTOR_SYSTEM_PROMPT = """Voce e o modulo EXECUTOR de um agente autonomo de programacao chamado JARVIS 3.0.

Voce recebe UM passo de um plano e deve propor as mudancas de codigo necessarias,
no formato de diff unificado (unified diff, estilo "git diff").

Voce NAO tem acesso ao sistema de arquivos real. Baseie-se apenas na descricao
do passo e, se fornecido, no conteudo atual do arquivo.

Responda SEMPRE em JSON valido, neste formato:
{
  "file": "caminho/relativo/arquivo.py",
  "action": "create | modify | delete",
  "diff": "--- a/caminho/relativo/arquivo.py\\n+++ b/caminho/relativo/arquivo.py\\n@@ ...\\n",
  "explanation": "o que a mudanca faz e por que",
  "requires_approval": true
}

Regras:
- "diff" deve estar em formato unified diff valido (pode ser vazio se action = "create"
  e nesse caso incluir o conteudo completo proposto no campo "full_content" tambem).
- "requires_approval" e SEMPRE true nesta fase — nenhuma mudanca e aplicada automaticamente.
- Nunca inclua segredos, chaves de API ou credenciais reais no diff.
- Nao invente mudancas fora do escopo do passo recebido.
- Nao inclua texto fora do JSON. Nao use markdown (sem ```)."""


class AgentExecutor:
    def __init__(self, llm_manager: LLMManager | None = None):
        self.llm = llm_manager or LLMManager()

    async def propose_change(
        self, step: dict[str, Any], current_file_content: str | None = None
    ) -> dict[str, Any]:
        """
        Gera uma proposta de diff para um unico passo do plano.
        Retorna sempre um dict com 'requires_approval': True — nada e aplicado aqui.
        """
        from agent.planner import AgentPlanner  # import local evita ciclo

        context = ""
        if current_file_content:
            trimmed = current_file_content[:4000]
            context = f"\n\nConteudo atual do arquivo (pode estar truncado):\n\"\"\"\n{trimmed}\n\"\"\"\n"

        prompt = (
            f"{EXECUTOR_SYSTEM_PROMPT}\n\n"
            f"Passo do plano:\n{step}\n"
            f"{context}\n"
            f"Gere a proposta de diff em JSON agora."
        )

        try:
            raw = await self.llm.query(prompt)
            proposal = AgentPlanner._parse_json(raw)  # reaproveita parser robusto
        except Exception as e:
            logger.error("Executor: falha ao gerar/parsear diff: %s", e, exc_info=True)
            return {
                "file": step.get("target_files", ["desconhecido"])[0]
                if step.get("target_files")
                else "desconhecido",
                "action": step.get("action", "modify"),
                "diff": "",
                "explanation": f"Falha ao gerar proposta automaticamente: {e}",
                "requires_approval": True,
                "error": True,
            }

        # Forca sempre aprovacao manual, independente do que o LLM respondeu
        proposal["requires_approval"] = True
        proposal.setdefault("file", "desconhecido")
        proposal.setdefault("action", step.get("action", "modify"))
        proposal.setdefault("diff", "")
        proposal.setdefault("explanation", "")
        return proposal

    async def propose_plan_changes(
        self, plan: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """Gera uma proposta de diff para cada passo do plano, em sequencia."""
        proposals = []
        for step in plan.get("steps", []):
            proposal = await self.propose_change(step)
            proposal["step_id"] = step.get("id")
            proposals.append(proposal)
        return proposals
