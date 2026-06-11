import os
from dotenv import load_dotenv

load_dotenv()

OPENROUTER_API_KEY: str = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL: str = "https://openrouter.ai/api/v1"
OPENROUTER_MODEL: str = os.getenv("OPENROUTER_MODEL", "meta-llama/llama-3.3-70b-instruct:free")
OPENROUTER_FALLBACK_MODEL: str = os.getenv("OPENROUTER_FALLBACK_MODEL", "meta-llama/llama-3.1-8b-instruct:free")
LLM_MAX_TOKENS: int = int(os.getenv("LLM_MAX_TOKENS", "1024"))
LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0.7"))
LLM_TIMEOUT: int = int(os.getenv("LLM_TIMEOUT", "30"))
OPENROUTER_SITE_URL: str = os.getenv("OPENROUTER_SITE_URL", "https://jarvis-5e839.web.app")
OPENROUTER_SITE_NAME: str = os.getenv("OPENROUTER_SITE_NAME", "JARVIS 2.0 ANTIGRAVITY")

ENVIRONMENT: str = os.getenv("ENVIRONMENT", "development")
FRONTEND_ORIGIN: str = os.getenv("FRONTEND_ORIGIN", "*")


def validate_settings() -> list:
    errors = []
    if not OPENROUTER_API_KEY:
        errors.append("OPENROUTER_API_KEY nao configurada.")
    return errors
