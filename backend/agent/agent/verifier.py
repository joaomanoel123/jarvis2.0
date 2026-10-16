# === ARQUIVO: backend/agent/verifier.py ===
"""
AgentVerifier — JARVIS 3.0

Checagens ESTATICAS sobre as propostas do Executor (nada e executado):
  - proposta aplicavel (find/replace bateu com o arquivo real)
  - arquivo existe (modify/delete) / nao existe (create)
  - sintaxe Python do arquivo RESULTANTE
  - imports nao declarados nas dependencias do projeto (ex.: Flask em projeto FastAPI)
  - segredos hardcoded nas linhas ADICIONADAS
  - caminhos sensiveis e delecoes (revisao manual obrigatoria)
"""

import ast
import logging
import re
import sys
from typing import Any

from agent.astcheck import decorator_changes
from agent.policy import is_protected

logger = logging.getLogger(__name__)

SECRET_PATTERNS = [
    re.compile(r"(?i)api[_-]?key\s*=\s*['\"][a-zA-Z0-9_\-]{16,}['\"]"),
    re.compile(r"(?i)secret\s*=\s*['\"][a-zA-Z0-9_\-]{16,}['\"]"),
    re.compile(r"(?i)password\s*=\s*['\"].+['\"]"),
    re.compile(r"sk-[a-zA-Z0-9]{20,}"),
    re.compile(r"ghp_[a-zA-Z0-9]{30,}"),
    re.compile(r"github_pat_[a-zA-Z0-9_]{30,}"),
    re.compile(r"AIza[0-9A-Za-z\-_]{30,}"),
]

DANGEROUS_PATH_HINTS = [".env", "credentials", "service-account", ".git/", "id_rsa"]

# import name -> nome normalizado do pacote no requirements
IMPORT_ALIASES = {
    "dotenv": "python_dotenv",
    "yaml": "pyyaml",
    "pil": "pillow",
    "cv2": "opencv_python",
    "sklearn": "scikit_learn",
    "bs4": "beautifulsoup4",
    "multipart": "python_multipart",
    "jose": "python_jose",
    "jwt": "pyjwt",
}


def _norm(name: str) -> str:
    return re.sub(r"[-.]", "_", name.strip().lower())


def _requirement_names(stack_context: str) -> set[str]:
    names = set()
    for line in stack_context.splitlines():
        m = re.match(r"^\s*([A-Za-z0-9][A-Za-z0-9_.\-]*)", line)
        if m:
            names.add(_norm(m.group(1)))
    return names


def _local_modules(repo_files: list[str]) -> set[str]:
    local = set()
    for path in repo_files:
        parts = path.split("/")
        local.update(_norm(p) for p in parts[:-1])
        if parts[-1].endswith(".py"):
            local.add(_norm(parts[-1][:-3]))
    return local


def _added_lines(diff_text: str) -> str:
    return "\n".join(
        line[1:] for line in diff_text.splitlines()
        if line.startswith("+") and not line.startswith("+++")
    )


class AgentVerifier:
    def verify_proposal(
        self,
        proposal: dict[str, Any],
        repo_files: list[str] | None = None,
        stack_context: str = "",
        allow_new_deps: bool = False,
    ) -> dict[str, Any]:
        issues: list[str] = []
        warnings: list[str] = []

        path = proposal.get("file", "")
        action = proposal.get("action", "modify")
        diff_text = proposal.get("diff", "") or ""
        resulting = proposal.get("resulting_content") or proposal.get("full_content") or ""

        # 1. Proposta aplicavel
        if proposal.get("apply_error"):
            issues.append(f"Proposta nao aplicavel: {proposal['apply_error']}")

        # 2. Existencia do arquivo no repo
        if repo_files is not None:
            exists = path in repo_files
            if action in ("modify", "delete") and not exists:
                issues.append(f"Arquivo '{path}' nao existe no repositorio.")
            if action == "create" and exists:
                issues.append(f"Arquivo '{path}' ja existe; 'create' sobrescreveria o original.")

        # 3. Caminho protegido / sensivel / delecao
        if is_protected(path):
            issues.append(
                f"'{path}' e codigo do proprio agente (protegido); mudancas exigem edicao manual."
            )
        if any(h in path.lower() for h in DANGEROUS_PATH_HINTS):
            issues.append(f"Arquivo-alvo sensivel/fora de escopo: '{path}'.")
        if action == "delete":
            issues.append(f"Acao 'delete' em '{path}' requer revisao manual obrigatoria.")

        # 4. Segredos nas linhas adicionadas
        added = _added_lines(diff_text) or (proposal.get("full_content") or "")
        if any(p.search(added) for p in SECRET_PATTERNS):
            issues.append(f"Possivel segredo/credencial hardcoded adicionado em '{path}'.")

        # 5. Sintaxe Python do arquivo resultante
        tree = None
        if path.endswith(".py") and resulting:
            try:
                tree = ast.parse(resulting)
            except SyntaxError as e:
                issues.append(f"Erro de sintaxe Python em '{path}' apos a mudanca: {e}")

        # 5b. Decoradores de funcoes existentes nao podem mudar (ex.: rota que perde o @app.get)
        original = proposal.get("original_content")
        if tree is not None and action == "modify" and original:
            for msg in decorator_changes(original, resulting):
                issues.append(f"'{path}': {msg} (a mudanca pode quebrar uma rota/funcao existente).")

        # 6. Imports nao declarados (so se temos contexto de dependencias)
        if tree is not None and stack_context.strip():
            for mod in self._undeclared_imports(tree, stack_context, repo_files or []):
                msg = (
                    f"'{path}' importa '{mod}', que nao aparece nas dependencias do "
                    "projeto (dependencia nova ou framework errado?)."
                )
                (warnings if allow_new_deps else issues).append(msg)

        return {
            "file": path,
            "passed": not issues,
            "issues": issues,
            "warnings": warnings,
            "requires_human_approval": True,  # SEMPRE true nesta fase
        }

    @staticmethod
    def _undeclared_imports(tree: ast.AST, stack_context: str, repo_files: list[str]) -> list[str]:
        declared = _requirement_names(stack_context)
        local = _local_modules(repo_files)
        stdlib = set(sys.stdlib_module_names)

        found: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                tops = [a.name.split(".")[0] for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                if node.level or not node.module:
                    continue
                tops = [node.module.split(".")[0]]
            else:
                continue
            for top in tops:
                n = _norm(top)
                if top in stdlib or n in local:
                    continue
                if n in declared or IMPORT_ALIASES.get(n) in declared:
                    continue
                if n == "google" and any(d.startswith("google") for d in declared):
                    continue
                if top not in found:
                    found.append(top)
        return found

    def verify_plan_risk(self, plan: dict[str, Any]) -> dict[str, Any]:
        steps = plan.get("steps", [])
        high = [s for s in steps if s.get("risk") == "high"]
        return {
            "total_steps": len(steps),
            "high_risk_count": len(high),
            "high_risk_steps": [s.get("id") for s in high],
            "overall_risk": "high" if high else ("medium" if steps else "low"),
        }

    def verify_all(
        self,
        plan: dict[str, Any],
        proposals: list[dict[str, Any]],
        repo_files: list[str] | None = None,
        stack_context: str = "",
    ) -> dict[str, Any]:
        allow_new = bool(plan.get("requires_new_dependencies"))
        reports = [
            self.verify_proposal(p, repo_files, stack_context, allow_new)
            for p in proposals
        ]
        return {
            "plan_risk": self.verify_plan_risk(plan),
            "proposal_reports": reports,
            "all_checks_passed": bool(reports) and all(r["passed"] for r in reports),
            "ready_for_human_review": True,
            "note": (
                "Checagem estatica: sintaxe, existencia de arquivos, imports e segredos. "
                "Nao garante que a mudanca esteja correta. Nenhuma mudanca foi aplicada; "
                "aprovacao humana e obrigatoria antes de qualquer commit."
            ),
        }
