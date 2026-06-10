# === ARQUIVO: backend/core/llm_manager.py ===
"""
Gerenciador de chamadas ao OpenRouter para o JARVIS 2.0.
OpenRouter é compatível com o openai SDK via base_url customizada.
Cadeia: modelo primário → modelo fallback → resposta amigável offline.
Nunca lança exceção para o caller — sempre retorna uma string.
"""

import asyncio
import logging
from openai import OpenAI, APIStatusError, APIConnectionError, APITimeoutError, OpenAIError
from settings import (
    OPENROUTER_API_KEY,
    OPENROUTER_BASE_URL,
    OPENROUTER_MODEL,
    OPENROUTER_FALLBACK_MODEL,
    LLM_MAX_TOKENS,
    LLM_TEMPERATURE,
    LLM_TIMEOUT,
    OPENROUTER_SITE_URL,
    OPENROUTER_SITE_NAME,
)

logger = logging.getLogger(__name__)

# Cliente OpenRouter singleton — inicializado uma vez
_client: OpenAI | None = None


def get_client() -> OpenAI:
    """
    Retorna cliente OpenRouter singleton.
    OpenRouter é compatível com openai.OpenAI via base_url + api_key customizados.
    """
    global _client
    if _client is None:
        _client = OpenAI(
            api_key=OPENROUTER_API_KEY,
            base_url=OPENROUTER_BASE_URL,
            timeout=LLM_TIMEOUT,
            default_headers={
                # Headers opcionais para analytics no dashboard do OpenRouter
                "HTTP-Referer": OPENROUTER_SITE_URL,
                "X-Title": OPENROUTER_SITE_NAME,
            },
        )
    return _client


def _call_openrouter(model: str, messages: list[dict], system: str) -> str:
    """
    Faz uma chamada síncrona ao OpenRouter com o modelo especificado.
    Lança exceção em caso de falha — o caller decide o que fazer.
    """
    full_messages = []
    if system:
        full_messages.append({"role": "system", "content": system})
    full_messages.extend(messages)

    response = get_client().chat.completions.create(
        model=model,
        messages=full_messages,
        max_tokens=LLM_MAX_TOKENS,
        temperature=LLM_TEMPERATURE,
    )
    return response.choices[0].message.content or ""


def complete(
    messages: list[dict],
    system: str = "Você é o JARVIS, assistente de IA do ANTIGRAVITY. Responda em português.",
) -> str:
    """
    Ponto de entrada síncrono.
    Tenta modelo primário → fallback → resposta amigável offline.
    Nunca lança exceção.
    """
    # ── Tentativa 1: modelo primário ─────────────────────────────────────────
    try:
        logger.info("Chamando OpenRouter — modelo primário: %s", OPENROUTER_MODEL)
        result = _call_openrouter(OPENROUTER_MODEL, messages, system)
        logger.info("OpenRouter respondeu (modelo: %s)", OPENROUTER_MODEL)
        return result
    except APIStatusError as e:
        logger.warning(
            "OpenRouter APIStatusError no modelo %s (status=%s): %s",
            OPENROUTER_MODEL, e.status_code, e.message,
        )
    except APIConnectionError as e:
        logger.warning("OpenRouter APIConnectionError no modelo %s: %s", OPENROUTER_MODEL, e)
    except APITimeoutError:
        logger.warning(
            "OpenRouter APITimeoutError no modelo %s (%ss)", OPENROUTER_MODEL, LLM_TIMEOUT
        )
    except OpenAIError as e:
        logger.warning("OpenAIError no modelo %s: %s", OPENROUTER_MODEL, e)

    # ── Tentativa 2: modelo fallback ─────────────────────────────────────────
    try:
        logger.info("Tentando fallback: %s", OPENROUTER_FALLBACK_MODEL)
        result = _call_openrouter(OPENROUTER_FALLBACK_MODEL, messages, system)
        logger.info("OpenRouter respondeu (fallback: %s)", OPENROUTER_FALLBACK_MODEL)
        return result
    except APIStatusError as e:
        logger.error(
            "OpenRouter APIStatusError no fallback %s (status=%s): %s",
            OPENROUTER_FALLBACK_MODEL, e.status_code, e.message,
        )
    except (APIConnectionError, APITimeoutError, OpenAIError) as e:
        logger.error("OpenRouter falhou no fallback %s: %s", OPENROUTER_FALLBACK_MODEL, e)

    # ── Nível 3: resposta amigável offline ───────────────────────────────────
    logger.error("Todos os modelos OpenRouter falharam. Retornando resposta offline.")
    return (
        "JARVIS está temporariamente indisponível. "
        "Os sistemas estão sendo reinicializados. "
        "Tente novamente em alguns instantes."
    )


async def complete_async(
    messages: list[dict],
    system: str = "Você é o JARVIS, assistente de IA do ANTIGRAVITY. Responda em português.",
) -> str:
    """
    Versão async — usa asyncio.to_thread para não bloquear o event loop do FastAPI.
    Mesma lógica de fallback. Nunca lança exceção.
    """
    return await asyncio.to_thread(complete, messages, system)


# ── Classe LLMManager — compatibilidade com Orchestrator ─────────────────────
class LLMManager:
    """
    Wrapper de classe sobre as funções complete/complete_async.
    O Orchestrator instancia esta classe e chama await self.llm_manager.query(prompt).
    """

    async def query(self, prompt: str) -> str:
        """
        Recebe um prompt em texto e retorna a resposta do OpenRouter.
        Alias de complete_async() para compatibilidade com o Orchestrator.
        """
        messages = [{"role": "user", "content": prompt}]
        return await complete_async(messages)