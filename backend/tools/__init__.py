"""
tools/__init__.py
=================
Auto-registers every built-in tool with the module-level ToolRegistry.

Import this module once at startup (main.py lifespan) and all tools
become available for agent use.  To add a new tool:

  1. Implement it in its own file under tools/.
  2. Import the async function here.
  3. Add a ToolSpec entry below.
"""

from .registry import ToolRegistry, ToolSpec, tool_registry
from .web_search import web_search
from .code_runner import run_python_code
from .file_reader import file_reader
from .system_status import system_status


def register_all_tools(registry: ToolRegistry | None = None) -> None:
    """Register every built-in tool. Call once during application startup."""
    reg = registry or tool_registry

    reg.register(ToolSpec(
        name="web_search",
        fn=web_search,
        description="Search the web for current information. Returns an answer and top result links.",
        parameters={
            "type": "object",
            "properties": {
                "query":       {"type": "string",  "description": "Search query."},
                "max_results": {"type": "integer", "description": "Max results (default 5)."},
            },
            "required": ["query"],
        },
        tags=["search", "information"],
    ))

    reg.register(ToolSpec(
        name="run_python_code",
        fn=run_python_code,
        description=(
            "Execute a Python code snippet in a sandboxed environment. "
            "Assign the final value to a variable named 'result' to capture it. "
            "Imports of os, sys, subprocess, and socket are blocked."
        ),
        parameters={
            "type": "object",
            "properties": {
                "code": {"type": "string", "description": "Python source code to run."},
            },
            "required": ["code"],
        },
        tags=["code", "computation"],
    ))

    reg.register(ToolSpec(
        name="file_reader",
        fn=file_reader,
        description=(
            "Read the contents of a text file from the sandboxed file directory. "
            "Only files inside /tmp/jarvis_files are accessible."
        ),
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path to the file to read."},
            },
            "required": ["path"],
        },
        tags=["file", "io"],
    ))

    reg.register(ToolSpec(
        name="system_status",
        fn=system_status,
        description="Return current CPU, RAM, disk, and uptime metrics for the JARVIS host.",
        parameters={
            "type": "object",
            "properties": {},
            "required": [],
        },
        tags=["system", "diagnostics"],
    ))


__all__ = [
    "ToolRegistry", "ToolSpec", "tool_registry",
    "web_search", "run_python_code", "file_reader", "system_status",
    "register_all_tools",
]


# Alias for backwards compatibility
bootstrap_tools = register_all_tools
