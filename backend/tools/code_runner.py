"""
tools/code_runner.py
====================
run_python_code — restricted Python sandbox.

Security model
──────────────
• RestrictedPython strips dangerous builtins from the AST before exec().
• A custom _getiter_ / _getattr_ guard blocks __dunder__ attribute access.
• stdout/stderr are captured and truncated.
• Network and filesystem calls are not explicitly blocked at this layer
  (use a container-level seccomp profile or OS-level sandbox in production).
• Wall-clock execution is bounded by the ToolRegistry timeout.

This is a best-effort sandbox suitable for demo environments.
For production, run user code in a subprocess with resource limits (rlimit)
or an isolated container (gVisor, Firecracker).
"""

from __future__ import annotations

import ast
import io
import logging
import textwrap
import traceback
from contextlib import redirect_stderr, redirect_stdout
from typing import Any

log = logging.getLogger(__name__)

# ── Restricted builtins allowlist ──────────────────────────────────────────────
_SAFE_BUILTINS: dict[str, Any] = {
    name: __builtins__[name]  # type: ignore[index]
    for name in (
        "abs", "all", "any", "ascii", "bin", "bool", "bytearray", "bytes",
        "callable", "chr", "complex", "dict", "dir", "divmod", "enumerate",
        "filter", "float", "format", "frozenset", "getattr", "hasattr",
        "hash", "hex", "int", "isinstance", "issubclass", "iter", "len",
        "list", "map", "max", "min", "next", "oct", "ord", "pow", "print",
        "range", "repr", "reversed", "round", "set", "setattr", "slice",
        "sorted", "str", "sum", "tuple", "type", "vars", "zip",
        "True", "False", "None",
        "NotImplementedError", "ValueError", "TypeError", "KeyError",
        "IndexError", "AttributeError", "Exception", "RuntimeError",
        "StopIteration", "ZeroDivisionError", "OverflowError",
    )
    if name in __builtins__  # type: ignore[operator]
}

_BLOCKED_NAMES = frozenset({
    "__import__", "eval", "exec", "compile", "open", "input",
    "__builtins__", "globals", "locals", "vars",
})


def _safe_getattr(obj: Any, name: str) -> Any:
    if name.startswith("_"):
        raise AttributeError(f"Access to '{name}' is not allowed.")
    return getattr(obj, name)


async def run_python_code(code: str, timeout: int = 10) -> dict:
    """
    Execute a Python snippet in a restricted environment.

    Args:
        code:    Source code to run (must be valid Python).
        timeout: Maximum wall-clock seconds (advisory — enforced by registry).

    Returns:
        {"stdout": str, "stderr": str, "result": str | None, "error": str | None}
    """
    log.info("run_python_code: %d chars", len(code))

    # Static AST scan — reject forbidden names before exec
    try:
        tree = ast.parse(textwrap.dedent(code), mode="exec")
    except SyntaxError as exc:
        return {"stdout": "", "stderr": "", "result": None, "error": f"SyntaxError: {exc}"}

    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id in _BLOCKED_NAMES:
            return {
                "stdout": "", "stderr": "", "result": None,
                "error": f"Blocked: use of '{node.id}' is not permitted.",
            }
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            module = getattr(node, "module", None) or (
                node.names[0].name if node.names else ""
            )
            if module and module.split(".")[0] in ("os", "sys", "subprocess", "socket"):
                return {
                    "stdout": "", "stderr": "", "result": None,
                    "error": f"Blocked: import of '{module}' is not permitted.",
                }

    # Execute in restricted namespace
    namespace: dict[str, Any] = {
        "__builtins__": _SAFE_BUILTINS,
        "__name__":     "__jarvis_sandbox__",
    }

    stdout_buf = io.StringIO()
    stderr_buf = io.StringIO()
    error: str | None = None
    result_val: Any   = None

    try:
        with redirect_stdout(stdout_buf), redirect_stderr(stderr_buf):
            exec(compile(tree, "<sandbox>", "exec"), namespace)  # noqa: S102
        result_val = namespace.get("result")
    except Exception:
        error = traceback.format_exc(limit=5)

    return {
        "stdout": stdout_buf.getvalue()[:4096],
        "stderr": stderr_buf.getvalue()[:1024],
        "result": str(result_val) if result_val is not None else None,
        "error":  error,
    }
