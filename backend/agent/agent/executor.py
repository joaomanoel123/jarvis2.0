# === ARQUIVO: backend/agent/executor.py ===
"""
AgentExecutor — JARVIS 3.0

Propoe mudancas de codigo para cada passo do plano. NAO escreve em disco,
NAO roda comandos, NAO usa git.

Em vez de pedir ao modelo um diff (que ele costuma inventar), pedimos
edicoes "find/replace" sobre o conteudo REAL do arquivo. O proprio codigo:
  1. confere que cada trecho 'find' existe exatamente uma vez;
  2. aplica as edicoes em memoria;
  3. confere a sintaxe Python do resultado;
  4. gera o unified diff verdadeiro com difflib.
Se algo falha, o modelo recebe o erro e tem UMA nova tentativa.
Codigo do proprio agente (backend/agent/) e protegido e nunca e proposto.
"""

import ast
import difflib
import json
import logging
from typing import Any

from agent.planner import AgentPlanner
from agent.policy import is_protected
from core.llm_manager import LLMManager

logger = logging.getLogger(__name__)

MAX_CONTEXT_CHARS = 12000
MAX_ATTEMPTS = 2  # 1 tentativa + 1 retry com o erro como feedback

EXECUTOR_SYSTEM_PROMPT = """Voce e o modulo EXECUTOR de um agente autonomo de programacao chamado JARVIS 3.0.

Voce recebe UM passo de um plano e deve propor a mudanca de codigo desse passo.
Voce NAO tem acesso ao sistema de arquivos; baseie-se no conteudo atual fornecido.

Responda SEMPRE em JSON valido, em UM dos formatos abaixo.

1) action "modify" (arquivo existente):
{
  "file": "caminho/arquivo.py",
  "action": "modify",
  "edits": [
    {"find": "trecho EXATO copiado do arquivo atual", "replace": "novo trecho"}
  ],
  "explanation": "o que muda e por que"
}

2) action "create" (arquivo novo):
{
  "file": "caminho/novo.py",
  "action": "create",
  "full_content": "conteudo completo do arquivo",
  "explanation": "..."
}

3) action "delete":
{"file": "caminho/arquivo.py", "action": "delete", "explanation": "..."}

Regras para "modify":
- "find" deve ser copiado LITERALMENTE do conteudo atual (mesma indentacao e espacos).
- Cada "find" deve aparecer UMA unica vez no arquivo; inclua linhas vizinhas se precisar torna-lo unico.
- Use trechos curtos (poucas linhas). Para inserir codigo, use como "find" uma linha ancora existente e repita-a INTEGRALMENTE no "replace" junto com o codigo novo (nunca altere nem quebre a linha ancora).
- O arquivo resultante deve continuar sintaticamente valido.
- Use o framework e as bibliotecas do projeto (contexto de stack). Nunca troque de framework.
- Nunca inclua segredos, chaves de API ou credenciais.
- Nao invente mudancas fora do escopo do passo.
- Nao inclua texto fora do JSON. Nao use markdown (sem ```)."""


class AgentExecutor:
    def __init__(self, llm_manager: LLMManager | None = None):
        self.llm = llm_manager or LLMManager()

    # ------------------------------------------------------------------
    async def propose_change(
        self,
        step: dict[str, Any],
        current_file_content: str | None = None,
        stack_context: str = "",
    ) -> dict[str, Any]:
        targets = step.get("target_files") or []
        path = targets[0] if targets else "desconhecido"
        action = step.get("action", "modify")

        base = {
            "file": path,
            "action": action,
            "diff": "",
            "explanation": "",
            "requires_approval": True,  # SEMPRE true nesta fase
        }

        if is_protected(path):
            base["apply_error"] = (
                f"'{path}' e codigo do proprio agente (protegido); passo nao proposto."
            )
            return base

        # modify/delete exigem o arquivo real
        if action in ("modify", "delete") and current_file_content is None:
            base["apply_error"] = (
                f"Arquivo '{path}' nao encontrado (ou ilegivel) no repositorio; "
                "passo nao pode ser proposto."
            )
            return base

        if action == "delete":
            base["explanation"] = step.get("description", "")
            base["diff"] = self._make_diff(path, current_file_content or "", "")
            return base

        feedback = ""
        for attempt in range(1, MAX_ATTEMPTS + 1):
            last = attempt == MAX_ATTEMPTS
            prompt = self._build_prompt(step, current_file_content, stack_context, feedback)

            try:
                raw = await self.llm.query(prompt)
                data = AgentPlanner._parse_json(raw)
            except Exception as e:
                logger.error("Executor: falha ao gerar/parsear proposta: %s", e, exc_info=True)
                base["apply_error"] = f"Falha ao gerar proposta automaticamente: {e}"
                return base

            base["explanation"] = str(data.get("explanation", ""))
            for key in ("apply_error", "edits", "full_content", "resulting_content"):
                base.pop(key, None)

            try:
                if action == "create":
                    content = data.get("full_content")
                    if not isinstance(content, str) or not content.strip():
                        raise ValueError("Modelo nao devolveu 'full_content' para o arquivo novo.")
                    new_content, old_content = content, ""
                    base["full_content"] = content
                else:  # modify
                    edits = data.get("edits")
                    if not isinstance(edits, list) or not edits:
                        raise ValueError("Modelo nao devolveu 'edits' para o arquivo existente.")
                    old_content = current_file_content or ""
                    new_content = self._apply_edits(old_content, edits)
                    base["edits"] = edits

                base["resulting_content"] = new_content
                base["diff"] = self._make_diff(path, old_content, new_content)

                syntax_error = self._syntax_error(path, new_content)
                if syntax_error and not last:
                    feedback = f"o arquivo resultante teria erro de sintaxe: {syntax_error}"
                    logger.info("Executor: retry de '%s' (%s)", path, feedback)
                    continue
                return base  # se ainda houver erro de sintaxe, o Verifier reprova

            except ValueError as e:
                base["apply_error"] = str(e)
                base["diff"] = ""
                if last:
                    return base
                feedback = str(e)
                logger.info("Executor: retry de '%s' (%s)", path, feedback)

        return base

    # ------------------------------------------------------------------
    async def propose_plan_changes(
        self,
        plan: dict[str, Any],
        reader=None,
        stack_context: str = "",
    ) -> list[dict[str, Any]]:
        """Uma proposta por passo. 'reader' (RepoReader) fornece o conteudo real."""
        proposals = []
        for step in plan.get("steps", []):
            targets = step.get("target_files") or []
            path = targets[0] if targets else None
            content = None
            if (reader and path and not is_protected(path)
                    and step.get("action") in ("modify", "delete")):
                try:
                    content = await reader.read_file(path)
                except Exception as e:  # leitura nunca derruba o pipeline
                    logger.warning("Executor: erro ao ler '%s': %s", path, e)

            proposal = await self.propose_change(step, content, stack_context)
            proposal["step_id"] = step.get("id")
            proposals.append(proposal)
        return proposals

    # ------------------------------------------------------------------
    @staticmethod
    def _build_prompt(step, current_content, stack_context, feedback: str = "") -> str:
        parts = [EXECUTOR_SYSTEM_PROMPT, ""]
        if stack_context:
            parts.append(f"Contexto de stack (dependencias do projeto):\n{stack_context}\n")
        parts.append("Passo do plano:\n" + json.dumps(step, ensure_ascii=False, indent=2))
        if current_content is not None:
            shown = current_content[:MAX_CONTEXT_CHARS]
            parts.append(
                f"\nConteudo ATUAL do arquivo '{(step.get('target_files') or ['?'])[0]}':\n"
                f"<<<ARQUIVO\n{shown}\nARQUIVO>>>"
            )
        if feedback:
            parts.append(
                f"\nATENCAO: sua tentativa anterior falhou: {feedback}. "
                "Corrija e responda novamente em JSON."
            )
        parts.append("\nGere a proposta em JSON agora.")
        return "\n".join(parts)

    @staticmethod
    def _syntax_error(path: str, content: str) -> str | None:
        if not path.endswith(".py"):
            return None
        try:
            ast.parse(content)
        except SyntaxError as e:
            return f"{e.msg} (linha {e.lineno})"
        return None

    @staticmethod
    def _apply_edits(original: str, edits: list[dict[str, Any]]) -> str:
        content = original
        for i, edit in enumerate(edits, start=1):
            if not isinstance(edit, dict):
                raise ValueError(f"edit #{i}: formato invalido.")
            find = edit.get("find", "")
            replace = edit.get("replace", "")
            if not isinstance(find, str) or not find:
                raise ValueError(f"edit #{i}: campo 'find' vazio.")
            if not isinstance(replace, str):
                raise ValueError(f"edit #{i}: campo 'replace' invalido.")
            count = content.count(find)
            if count == 0:
                raise ValueError(f"edit #{i}: trecho 'find' nao existe no arquivo real.")
            if count > 1:
                raise ValueError(f"edit #{i}: trecho 'find' aparece {count} vezes (ambiguo).")
            content = content.replace(find, replace, 1)
        return content

    @staticmethod
    def _make_diff(path: str, old: str, new: str) -> str:
        from_file = f"a/{path}" if old else "/dev/null"
        to_file = f"b/{path}" if new else "/dev/null"
        lines = difflib.unified_diff(
            old.splitlines(keepends=True),
            new.splitlines(keepends=True),
            fromfile=from_file,
            tofile=to_file,
        )
        return "".join(lines)
