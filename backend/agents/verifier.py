# === ARQUIVO: backend/agent/verifier.py ===
"""
AgentVerifier — JARVIS 3.0 Fase 1

Roda checagens ESTATICAS (sem executar nada) sobre um diff/proposta
gerada pelo Executor, antes de qualquer coisa chegar perto de um commit real.

Nesta fase: validacao de sintaxe Python (quando aplicavel), deteccao de
segredos/hardcoded secrets, deteccao de exclusoes fora de escopo, e
checagem de risco declarado pelo Planner.
"""

import ast
import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

# Padroes simples de deteccao de segredos hardcoded
SECRET_PATTERNS = [
    re.compile(r"(?i)api[_-]?key\s*=\s*['\"][a-zA-Z0-9_\-]{16,}['\"]"),
    re.compile(r"(?i)secret\s*=\s*['\"][a-zA-Z0-9_\-]{16,}['\"]"),
    re.compile(r"(?i)password\s*=\s*['\"].+['\"]"),
    re.compile(r"sk-[a-zA-Z0-9]{20,}"),           # OpenAI/OpenRouter-style
    re.compile(r"ghp_[a-zA-Z0-9]{30,}"),          # GitHub token
    re.compile(r"AIza[0-9A-Za-z\-_]{30,}"),       # Google API key
]

DANGEROUS_PATH_HINTS = [
    ".env",
    "credentials",
    "service-account",
    ".git/",
    "id_rsa",
]


class AgentVerifier:
    def verify_proposal(self, proposal: dict[str, Any]) -> dict[str, Any]:
        """
        Roda todas as checagens estaticas sobre uma unica proposta de diff.
        Retorna um relatorio com 'passed': bool e lista de 'issues'.
        """
        issues: list[str] = []

        file_path = proposal.get("file", "")
        diff_text = proposal.get("diff", "") or ""
        full_content = proposal.get("full_content", "") or ""
        action = proposal.get("action", "modify")

        # 1. Caminho sensivel
        if any(hint in file_path.lower() for hint in DANGEROUS_PATH_HINTS):
            issues.append(f"Arquivo-alvo parece sensivel/fora de escopo: '{file_path}'.")

        # 2. Delecao fora de escopo (sem justificativa clara)
        if action == "delete":
            issues.append(
                f"Acao 'delete' em '{file_path}' requer revisao manual obrigatoria."
            )

        # 3. Segredos hardcoded no diff ou no conteudo completo proposto
        combined_text = f"{diff_text}\n{full_content}"
        for pattern in SECRET_PATTERNS:
            if pattern.search(combined_text):
                issues.append(
                    f"Possivel segredo/credencial hardcoded detectado no conteudo proposto para '{file_path}'."
                )
                break

        # 4. Sintaxe Python valida (quando for arquivo .py e tivermos conteudo completo)
        if file_path.endswith(".py") and full_content:
            try:
                ast.parse(full_content)
            except SyntaxError as e:
                issues.append(f"Erro de sintaxe Python em '{file_path}': {e}")

        passed = len(issues) == 0
        return {
            "file": file_path,
            "passed": passed,
            "issues": issues,
            "requires_human_approval": True,  # SEMPRE true nesta fase
        }

    def verify_plan_risk(self, plan: dict[str, Any]) -> dict[str, Any]:
        """Avalia o risco agregado do plano com base no 'risk' de cada passo."""
        steps = plan.get("steps", [])
        high_risk_steps = [s for s in steps if s.get("risk") == "high"]

        return {
            "total_steps": len(steps),
            "high_risk_count": len(high_risk_steps),
            "high_risk_steps": [s.get("id") for s in high_risk_steps],
            "overall_risk": "high" if high_risk_steps else (
                "medium" if steps else "low"
            ),
        }

    def verify_all(
        self, plan: dict[str, Any], proposals: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Roda a verificacao completa: risco do plano + cada proposta de diff."""
        proposal_reports = [self.verify_proposal(p) for p in proposals]
        plan_risk = self.verify_plan_risk(plan)

        all_passed = all(r["passed"] for r in proposal_reports)

        return {
            "plan_risk": plan_risk,
            "proposal_reports": proposal_reports,
            "all_checks_passed": all_passed,
            "ready_for_human_review": True,
            "note": (
                "Nenhuma mudanca foi aplicada. Esta e apenas uma analise estatica "
                "— revisao e aprovacao humana sao obrigatorias antes de qualquer commit."
            ),
        }
