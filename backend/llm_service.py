"""
backend/llm_service.py
HuggingFace Transformers wrapper for JARVIS 2.0.
Loads the model once, runs inference in a thread pool to avoid blocking
the FastAPI event loop.
"""

import asyncio
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

log = logging.getLogger("jarvis.llm")

# ── Model registry ─────────────────────────────────────────────────────
_MODEL_REGISTRY = {
    "tinyllama":  "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "phi-2":      "microsoft/phi-2",
    "phi-3-mini": "microsoft/Phi-3-mini-4k-instruct",
    "mistral-7b": "mistralai/Mistral-7B-Instruct-v0.3",
    "llama-3":    "meta-llama/Meta-Llama-3-8B-Instruct",
}

# ── Config from environment ────────────────────────────────────────────
_MODEL_ALIAS   = os.getenv("LLM_MODEL_ID",       "tinyllama")
_DEVICE        = os.getenv("LLM_DEVICE",          "auto")
_LOAD_4BIT     = os.getenv("LLM_LOAD_IN_4BIT",   "false").lower() == "true"
_MAX_TOKENS    = int(os.getenv("LLM_MAX_NEW_TOKENS", "512"))
_TEMPERATURE   = float(os.getenv("LLM_TEMPERATURE",  "0.7"))

_JARVIS_SYSTEM_PROMPT = (
    "You are JARVIS 2.0, an advanced AI assistant inspired by Iron Man's J.A.R.V.I.S. "
    "You are intelligent, concise, and precise. You speak in a calm, slightly futuristic tone. "
    "You avoid unnecessary verbosity and casual slang. "
    "Answer the user's question directly and helpfully."
)


@dataclass
class LLMResult:
    text:       str
    model_id:   str
    latency_ms: float
    tokens:     int = 0


class LLMService:
    """
    Wraps a HuggingFace causal LM for async generation.

    Usage:
        svc = LLMService()
        svc.load()                          # call once at startup
        result = await svc.generate("hi")
    """

    def __init__(self) -> None:
        self._model       = None
        self._tokenizer   = None
        self._pipeline    = None
        self._model_id    = _MODEL_REGISTRY.get(_MODEL_ALIAS, _MODEL_ALIAS)
        self._loaded      = False
        self._executor    = ThreadPoolExecutor(max_workers=1, thread_name_prefix="llm")

    # ── Load ───────────────────────────────────────────────────────────

    def load(self) -> None:
        """Load the model synchronously. Call from the lifespan thread."""
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
            import torch

            log.info("Loading LLM: %s …", self._model_id)
            t0 = time.perf_counter()

            hf_token = os.getenv("HF_TOKEN") or None

            load_kwargs: dict[str, Any] = {
                "token":         hf_token,
                "low_cpu_mem_usage": True,
            }

            if _LOAD_4BIT:
                from transformers import BitsAndBytesConfig
                load_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16
                )
                load_kwargs["device_map"] = "auto"
            else:
                device = _DEVICE
                if device == "auto":
                    device = "cuda" if torch.cuda.is_available() else "cpu"
                load_kwargs["torch_dtype"] = torch.float32 if device == "cpu" else torch.float16

            self._tokenizer = AutoTokenizer.from_pretrained(self._model_id, **load_kwargs)
            self._model     = AutoModelForCausalLM.from_pretrained(self._model_id, **load_kwargs)

            if not _LOAD_4BIT and _DEVICE != "auto":
                device = "cuda" if torch.cuda.is_available() else "cpu"
                self._model = self._model.to(device)

            self._pipeline = pipeline(
                "text-generation",
                model=self._model,
                tokenizer=self._tokenizer,
                max_new_tokens=_MAX_TOKENS,
                temperature=_TEMPERATURE,
                do_sample=_TEMPERATURE > 0,
                repetition_penalty=1.1,
                pad_token_id=self._tokenizer.eos_token_id,
            )
            self._loaded = True
            log.info("LLM loaded in %.1f s", time.perf_counter() - t0)

        except Exception as exc:
            log.error("LLM load failed: %s — conversational replies will be stub text", exc)
            self._loaded = False

    # ── Generate ───────────────────────────────────────────────────────

    async def generate(
        self,
        user_message: str,
        system_prompt: str | None = None,
        history: list[dict] | None = None,
    ) -> LLMResult:
        """
        Generate a response for the given user message.

        Args:
            user_message:  The user's input text.
            system_prompt: Optional override for the system prompt.
            history:       List of {role, content} dicts for context.

        Returns:
            LLMResult with generated text.
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self._executor,
            self._generate_sync,
            user_message,
            system_prompt or _JARVIS_SYSTEM_PROMPT,
            history or [],
        )

    def _generate_sync(
        self,
        user_message: str,
        system_prompt: str,
        history: list[dict],
    ) -> LLMResult:
        t0 = time.perf_counter()

        # Stub when model is not loaded
        if not self._loaded or self._pipeline is None:
            stub = self._stub_response(user_message)
            return LLMResult(
                text=stub,
                model_id="stub",
                latency_ms=round((time.perf_counter() - t0) * 1000, 1),
            )

        # Build chat messages
        messages = [{"role": "system", "content": system_prompt}]
        for h in (history or []):
            messages.append(h)
        messages.append({"role": "user", "content": user_message})

        # Tokenizer chat template (if supported)
        try:
            prompt = self._tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            prompt = f"[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n{user_message} [/INST]"

        outputs = self._pipeline(prompt)
        generated = outputs[0]["generated_text"]

        # Strip the prompt prefix from the output
        if generated.startswith(prompt):
            generated = generated[len(prompt):]

        text = generated.strip()
        tokens = len(self._tokenizer.encode(text)) if self._tokenizer else 0
        latency = round((time.perf_counter() - t0) * 1000, 1)

        return LLMResult(text=text, model_id=self._model_id, latency_ms=latency, tokens=tokens)

    def _stub_response(self, text: str) -> str:
        """Fallback when no model is loaded — rule-based responses."""
        t = text.lower()
        if any(w in t for w in ["hello", "hi", "hey"]):
            return "JARVIS online. How can I assist you?"
        if "what is" in t or "explain" in t:
            return f"I can answer that once the language model is fully loaded. Query noted: '{text}'"
        if "thank" in t:
            return "At your service."
        return f"Understood. Processing: '{text}'. Language model is loading — please stand by."

    # ── Info ───────────────────────────────────────────────────────────

    def info(self) -> dict:
        return {
            "loaded":   self._loaded,
            "model_id": self._model_id,
            "alias":    _MODEL_ALIAS,
            "4bit":     _LOAD_4BIT,
        }


# Module singleton
llm_service = LLMService()
