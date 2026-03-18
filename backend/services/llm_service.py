"""
services/llm_service.py
=======================
LLMService — the single point of contact for all language model calls.

No other module imports transformers directly.  Route everything here.

Supported model families (swap via LLM_MODEL_ID env var)
──────────────────────────────────────────────────────────
  mistralai/Mistral-7B-Instruct-v0.3     ← default
  meta-llama/Meta-Llama-3-8B-Instruct
  microsoft/Phi-3-mini-4k-instruct
  microsoft/phi-2
  deepseek-ai/deepseek-coder-7b-instruct-v1.5
  TinyLlama/TinyLlama-1.1B-Chat-v1.0    ← CPU-safe dev model

Architecture
────────────
• Model loads once at startup (call load() from FastAPI lifespan).
• All generation runs inside a ThreadPoolExecutor so the asyncio event loop
  is never blocked by synchronous PyTorch forward passes.
• 4-bit NF4 quantisation (bitsandbytes) halves VRAM when enabled.
"""

from __future__ import annotations

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    pipeline,
)

from config.settings import get_settings

log = logging.getLogger(__name__)
cfg = get_settings()

# ── Trusted model registy ──────────────────────────────────────────────────────
# Only orgs in this set get trust_remote_code=True.
_TRUSTED_ORGS: frozenset[str] = frozenset({
    "mistralai", "meta-llama", "microsoft",
    "TinyLlama", "deepseek-ai", "google",
})

# Friendly alias → HuggingFace model ID
MODEL_REGISTRY: dict[str, str] = {
    "mistral-7b":      "mistralai/Mistral-7B-Instruct-v0.3",
    "llama-3":         "meta-llama/Meta-Llama-3-8B-Instruct",
    "phi-3-mini":      "microsoft/Phi-3-mini-4k-instruct",
    "phi-2":           "microsoft/phi-2",
    "deepseek-coder":  "deepseek-ai/deepseek-coder-7b-instruct-v1.5",
    "tinyllama":       "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
}


# ── Public result type ─────────────────────────────────────────────────────────

@dataclass
class LLMResult:
    text:       str
    model_id:   str
    tokens_out: int  = 0
    metadata:   dict = field(default_factory=dict)


# ── LLMService ─────────────────────────────────────────────────────────────────

class LLMService:
    """
    Async wrapper around a locally-loaded HuggingFace pipeline.

    Usage
    -----
    # In FastAPI lifespan:
    await llm_service.load()
    info = llm_service.info()

    # In a coroutine:
    result = await llm_service.generate("Tell me about Paris")
    result = await llm_service.chat([{"role": "user", "content": "Hello"}])
    """

    def __init__(self) -> None:
        self._tokenizer: Any | None = None
        self._model:     Any | None = None
        self._pipe:      Any | None = None
        self._model_id:  str | None = None
        self._executor   = ThreadPoolExecutor(
            max_workers=cfg.LLM_THREAD_WORKERS,
            thread_name_prefix="llm",
        )

    # ── Loading ────────────────────────────────────────────────────────────────

    def load(self, model_id: str | None = None) -> None:
        """
        Synchronous load – call from lifespan via run_in_executor.

        Args:
            model_id: HuggingFace model ID or a friendly alias from MODEL_REGISTRY.
                      Falls back to LLM_MODEL_ID env var.
        """
        raw     = model_id or cfg.LLM_MODEL_ID
        resolved = MODEL_REGISTRY.get(raw, raw)

        if self._pipe is not None and self._model_id == resolved:
            log.info("LLMService: %s already loaded — skipping", resolved)
            return

        log.info("LLMService: loading %s …", resolved)
        log.info("  device=%s  4bit=%s  max_tokens=%d",
                 cfg.LLM_DEVICE, cfg.LLM_LOAD_IN_4BIT, cfg.LLM_MAX_NEW_TOKENS)

        trust = self._is_trusted(resolved)
        if not trust:
            log.warning("LLMService: %s not in trusted org list → trust_remote_code=False", resolved)

        hf_token = cfg.HF_TOKEN or None

        # Quantisation
        quant = None
        if cfg.LLM_LOAD_IN_4BIT:
            quant = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )

        dtype = torch.float32 if cfg.LLM_DEVICE == "cpu" else (
            torch.bfloat16 if torch.cuda.is_available() else torch.float32
        )

        self._tokenizer = AutoTokenizer.from_pretrained(
            resolved,
            token=hf_token,
            trust_remote_code=trust,
        )
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        self._model = AutoModelForCausalLM.from_pretrained(
            resolved,
            token=hf_token,
            quantization_config=quant,
            device_map=cfg.LLM_DEVICE if not quant else "auto",
            torch_dtype=dtype,
            trust_remote_code=trust,
            low_cpu_mem_usage=True,
        )
        self._model.eval()

        self._pipe = pipeline(
            task="text-generation",
            model=self._model,
            tokenizer=self._tokenizer,
            return_full_text=False,
        )
        self._model_id = resolved
        log.info("LLMService: ✓ %s loaded", resolved)

    @staticmethod
    def _is_trusted(model_id: str) -> bool:
        org = model_id.split("/")[0] if "/" in model_id else ""
        return org in _TRUSTED_ORGS

    # ── Generation ─────────────────────────────────────────────────────────────

    def _sync_generate(
        self,
        prompt: str,
        max_new_tokens: int,
        temperature: float,
        system: str | None,
    ) -> str:
        """Blocking inference – runs inside the ThreadPoolExecutor."""
        assert self._pipe is not None, "Model not loaded"
        assert self._tokenizer is not None

        # Build chat-formatted prompt
        messages: list[dict] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        formatted = self._apply_template(messages)

        outputs = self._pipe(
            formatted,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
            top_p=0.95,
            repetition_penalty=1.1,
            pad_token_id=self._tokenizer.eos_token_id,
        )
        return outputs[0]["generated_text"].strip()

    def _sync_chat(
        self,
        messages: list[dict],
        max_new_tokens: int,
        temperature: float,
        system: str | None,
    ) -> str:
        """Multi-turn blocking inference."""
        assert self._pipe is not None, "Model not loaded"
        assert self._tokenizer is not None

        full: list[dict] = []
        if system:
            full.append({"role": "system", "content": system})
        full.extend(messages)
        formatted = self._apply_template(full)

        outputs = self._pipe(
            formatted,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
            top_p=0.95,
            repetition_penalty=1.1,
            pad_token_id=self._tokenizer.eos_token_id,
        )
        return outputs[0]["generated_text"].strip()

    def _apply_template(self, messages: list[dict]) -> str:
        """Apply tokenizer chat template; fall back to Mistral [INST] format."""
        assert self._tokenizer is not None
        if hasattr(self._tokenizer, "apply_chat_template") and self._tokenizer.chat_template:
            try:
                return self._tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except Exception as exc:
                log.debug("apply_chat_template failed (%s) — using fallback", exc)

        # Fallback: Mistral [INST] format
        parts: list[str] = []
        for msg in messages:
            role, content = msg["role"], msg["content"]
            if role == "system":
                parts.append(f"<<SYS>>\n{content}\n<</SYS>>\n\n")
            elif role == "user":
                parts.append(f"[INST] {content} [/INST]")
            elif role == "assistant":
                parts.append(f" {content} </s>")
        return "<s>" + "".join(parts)

    # ── Public async API ───────────────────────────────────────────────────────

    async def generate(
        self,
        prompt: str,
        system: str | None = None,
        max_new_tokens: int | None = None,
        temperature: float | None = None,
    ) -> LLMResult:
        """
        Single-turn text generation.

        Args:
            prompt:         User / instruction text.
            system:         Optional system prompt injected before the user turn.
            max_new_tokens: Override the default token budget.
            temperature:    Override the default sampling temperature.
        """
        self._check_loaded()
        n     = max_new_tokens or cfg.LLM_MAX_NEW_TOKENS
        temp  = temperature    or cfg.LLM_TEMPERATURE
        loop  = asyncio.get_running_loop()
        text  = await loop.run_in_executor(
            self._executor,
            self._sync_generate,
            prompt, n, temp, system,
        )
        return LLMResult(text=text, model_id=self._model_id or "unknown")

    async def chat(
        self,
        messages: list[dict],
        system: str | None = None,
        max_new_tokens: int | None = None,
        temperature: float | None = None,
    ) -> LLMResult:
        """
        Multi-turn chat completion.

        Args:
            messages: List of {role, content} dicts.
            system:   Optional system prompt.
        """
        self._check_loaded()
        n     = max_new_tokens or cfg.LLM_MAX_NEW_TOKENS
        temp  = temperature    or cfg.LLM_TEMPERATURE
        loop  = asyncio.get_running_loop()
        text  = await loop.run_in_executor(
            self._executor,
            self._sync_chat,
            messages, n, temp, system,
        )
        return LLMResult(text=text, model_id=self._model_id or "unknown")

    # ── Diagnostics ────────────────────────────────────────────────────────────

    def info(self) -> dict:
        if self._model is None:
            return {"loaded": False}
        try:
            device = str(next(self._model.parameters()).device)
            dtype  = str(next(self._model.parameters()).dtype)
            n_params = round(sum(p.numel() for p in self._model.parameters()) / 1e9, 2)
        except Exception:
            device, dtype, n_params = "unknown", "unknown", 0.0
        return {
            "loaded":       True,
            "model_id":     self._model_id,
            "parameters_b": n_params,
            "device":       device,
            "dtype":        dtype,
            "4bit":         cfg.LLM_LOAD_IN_4BIT,
        }

    def _check_loaded(self) -> None:
        if self._pipe is None:
            raise RuntimeError(
                "LLMService not initialised. Call load() during app startup."
            )


# Module-level singleton
llm_service = LLMService()
