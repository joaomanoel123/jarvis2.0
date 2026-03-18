"""
tools/file_reader.py
====================
file_reader — reads plain-text files from a restricted base directory.

Security model
──────────────
• All paths are resolved with Path.resolve() to prevent traversal.
• Only files inside FILE_BASE_DIR (default: /tmp/jarvis_files) are readable.
• Maximum file size is capped to prevent memory exhaustion.
• Binary files are rejected.
"""

from __future__ import annotations

import logging
from pathlib import Path

from config.settings import get_settings

log = logging.getLogger(__name__)
cfg = get_settings()

# Only files inside this directory may be read
FILE_BASE_DIR = Path("/tmp/jarvis_files").resolve()
MAX_FILE_BYTES = 256_000   # 256 KB


async def file_reader(path: str) -> dict:
    """
    Read a text file from the sandboxed file directory.

    Args:
        path: Relative or absolute path.  Must resolve inside FILE_BASE_DIR.

    Returns:
        {"path": str, "content": str, "size_bytes": int, "lines": int}
    """
    log.info("file_reader: %r", path)

    requested = Path(path).resolve()

    # Traversal guard
    try:
        requested.relative_to(FILE_BASE_DIR)
    except ValueError:
        raise PermissionError(
            f"Access denied: '{path}' is outside the allowed directory."
        )

    if not requested.exists():
        raise FileNotFoundError(f"File not found: '{path}'")

    if not requested.is_file():
        raise ValueError(f"Not a file: '{path}'")

    size = requested.stat().st_size
    if size > MAX_FILE_BYTES:
        raise ValueError(
            f"File too large ({size:,} bytes > {MAX_FILE_BYTES:,} limit)."
        )

    # Try to read as UTF-8 text
    try:
        content = requested.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        raise ValueError("File appears to be binary — only text files are supported.")

    lines = content.count("\n") + (1 if content and not content.endswith("\n") else 0)

    return {
        "path":       str(requested),
        "content":    content,
        "size_bytes": size,
        "lines":      lines,
    }
