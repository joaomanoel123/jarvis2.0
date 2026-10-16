# === ARQUIVO: backend/agent/astcheck.py ===
"""
Checagens estruturais (AST) sobre mudancas em codigo Python.

decorator_changes: detecta funcoes/classes JA EXISTENTES cujos decoradores mudaram.
Pega o erro classico de inserir codigo logo apos uma linha '@app.get("/")',
o que separa o decorador da sua funcao e quebra a rota original — algo que
passa na checagem de sintaxe.
"""

import ast


def _decorators_by_name(tree: ast.AST) -> dict[str, list[tuple[str, ...]]]:
    result: dict[str, list[tuple[str, ...]]] = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            decs = tuple(ast.unparse(d) for d in node.decorator_list)
            result.setdefault(node.name, []).append(decs)
    return result


def decorator_changes(old: str, new: str) -> list[str]:
    """Mensagens para cada funcao/classe existente cujos decoradores mudaram."""
    try:
        old_tree, new_tree = ast.parse(old), ast.parse(new)
    except SyntaxError:
        return []  # erro de sintaxe e reportado por outra checagem

    old_decs = _decorators_by_name(old_tree)
    new_decs = _decorators_by_name(new_tree)

    messages = []
    for name, before in old_decs.items():
        after = new_decs.get(name)
        if after is None:
            continue  # removida: tratado em revisao humana do diff
        if sorted(before) != sorted(after):
            messages.append(
                f"decoradores de '{name}' mudaram: {list(before[0])} -> {list(after[0])}"
            )
    return messages
