"""
config/settings.py
==================
Single source of truth for all configuration.
Every tunable and secret lives here – never hard-coded elsewhere.
"""

from __future__ import annotations

from functools import lru_cache

from pydantic import field_validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):

    # ── LLM / HuggingFace ─────────────────────────────────────────────────────
    LLM_MODEL_ID:       str   = "mistralai/Mistral-7B-Instruct-v0.3"
    LLM_DEVICE:         str   = "auto"   # auto | cpu | cuda | mps
    LLM_LOAD_IN_4BIT:   bool  = False
    LLM_MAX_NEW_TOKENS: int   = 1024
    LLM_TEMPERATURE:    float = 0.7
    LLM_THREAD_WORKERS: int   = 1        # one model per process – do not raise
    HF_TOKEN:           str   = ""

    # ── Tool sandbox ──────────────────────────────────────────────────────────
    TOOL_TIMEOUT_SECONDS:  int = 30
    TOOL_MAX_OUTPUT_BYTES: int = 65_536
    TOOL_THREAD_WORKERS:   int = 4
    # Allowlist of tool names the system may invoke
    ALLOWED_TOOLS: str = "web_search,run_python_code,file_reader,system_status"

    # ── Memory / session ──────────────────────────────────────────────────────
    SESSION_TTL_SECONDS:  int = 3_600   # 1 h
    MAX_HISTORY_MESSAGES: int = 40
    SHORT_TERM_TTL:       int = 300     # 5 min

    # ── Application ───────────────────────────────────────────────────────────
    APP_NAME:       str       = "JARVIS"
    APP_VERSION:    str       = "2.0.0"
    DEBUG:          bool      = False
    ALLOWED_ORIGINS: list[str] = ["*"]

    # ── Validators ────────────────────────────────────────────────────────────
    @field_validator("LLM_DEVICE")
    @classmethod
    def _device_valid(cls, v: str) -> str:
        ok = {"auto", "cpu", "cuda", "mps"}
        if v not in ok and not v.startswith("cuda:"):
            raise ValueError(f"LLM_DEVICE must be one of {ok}")
        return v

    @field_validator("LLM_TEMPERATURE")
    @classmethod
    def _temp_range(cls, v: float) -> float:
        if not 0.0 <= v <= 2.0:
            raise ValueError("LLM_TEMPERATURE must be in [0.0, 2.0]")
        return v

    @field_validator("LLM_MAX_NEW_TOKENS")
    @classmethod
    def _tokens_positive(cls, v: int) -> int:
        if v < 1:
            raise ValueError("LLM_MAX_NEW_TOKENS must be >= 1")
        return v

    @property
    def allowed_tools_set(self) -> frozenset[str]:
        return frozenset(t.strip() for t in self.ALLOWED_TOOLS.split(","))

    model_config = {
        "env_file":          ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive":    True,
    }


@lru_cache()
def get_settings() -> Settings:
    """Process-level singleton – safe to call from anywhere."""
    return Settings()
