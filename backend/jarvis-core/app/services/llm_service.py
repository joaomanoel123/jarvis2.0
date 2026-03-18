"""
app/services/llm_service.py
HuggingFace Transformers inference service.
Runs in a ThreadPoolExecutor to avoid blocking the event loop.
"""

from __future__ import annotations

import asyncio
import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from app.utils.logger import get_logger

log = get_logger("jarvis.llm")

_MODEL_REGISTRY = {
    "tinyllama":  "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "phi-2":      "microsoft/phi-2",
    "phi-3-mini": "microsoft/Phi-3-mini-4k-instruct",
    "mistral-7b": "mistralai/Mistral-7B-Instruct-v0.3",
    "llama-3":    "meta-llama/Meta-Llama-3-8B-Instruct",
    "deepseek":   "deepseek-ai/deepseek-coder-7b-instruct-v1.5",
}

_MODEL_ALIAS  = os.getenv("LLM_MODEL_ID",       "tinyllama")
_MAX_TOKENS   = int(os.getenv("LLM_MAX_NEW_TOKENS", "512"))
_TEMPERATURE  = float(os.getenv("LLM_TEMPERATURE",  "0.7"))
_LOAD_4BIT    = os.getenv("LLM_LOAD_IN_4BIT",   "false").lower() == "true"
_DEVICE       = os.getenv("LLM_DEVICE",          "auto")

_SYSTEM_PROMPT = (
    "You are JARVIS 2.0 — an advanced autonomous AI assistant. "
    "You are precise, intelligent, and slightly futuristic. "
    "Never verbose. Respond as a high-level AI system."
)


@dataclass
class LLMResult:
    text:       str
    model_id:   str
    latency_ms: float
    tokens:     int = 0


class LLMService:
    def __init__(self) -> None:
        self._model_id  = _MODEL_REGISTRY.get(_MODEL_ALIAS, _MODEL_ALIAS)
        self._pipeline  = None
        self._tokenizer = None
        self._loaded    = False
        self._executor  = ThreadPoolExecutor(max_workers=1, thread_name_prefix="llm")

    def load(self) -> None:
        try:
            import torch
            from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

            log.info("Loading LLM: %s …", self._model_id)
            t0  = time.perf_counter()
            tok = os.getenv("HF_TOKEN") or None

            kw = {"token": tok, "low_cpu_mem_usage": True}
            if _LOAD_4BIT:
                from transformers import BitsAndBytesConfig
                kw["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16
                )
                kw["device_map"] = "auto"
            else:
                device = _DEVICE if _DEVICE != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
                kw["torch_dtype"] = torch.float16 if "cuda" in device else torch.float32

            model = AutoModelForCausalLM.from_pretrained(self._model_id, **kw)
            self._tokenizer = AutoTokenizer.from_pretrained(self._model_id, token=tok)

            if not _LOAD_4BIT and _DEVICE == "auto":
                device = "cuda" if torch.cuda.is_available() else "cpu"
                model = model.to(device)

            self._pipeline = pipeline(
                "text-generation",
                model=model,
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
            log.error("LLM load failed: %s", exc)
            self._loaded = False

    def is_loaded(self) -> bool:
        return self._loaded

    async def generate(
        self,
        user_message: str,
        system_prompt: str | None = None,
        history: list[dict] | None = None,
    ) -> LLMResult:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self._executor,
            self._generate_sync,
            user_message,
            system_prompt or _SYSTEM_PROMPT,
            history or [],
        )

    def _generate_sync(self, user_message: str, system_prompt: str, history: list[dict]) -> LLMResult:
        t0 = time.perf_counter()
        if not self._loaded or not self._pipeline:
            return LLMResult("Model loading…", "stub", 0.0)

        messages = [{"role": "system", "content": system_prompt}]
        for h in history:
            messages.append(h)
        messages.append({"role": "user", "content": user_message})

        try:
            prompt = self._tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            prompt = f"[INST] {user_message} [/INST]"

        out  = self._pipeline(prompt)[0]["generated_text"]
        text = out[len(prompt):].strip() if out.startswith(prompt) else out.strip()
        toks = len(self._tokenizer.encode(text)) if self._tokenizer else 0
        return LLMResult(text=text, model_id=self._model_id,
                         latency_ms=round((time.perf_counter() - t0) * 1000, 1), tokens=toks)

    def info(self) -> dict:
        return {"loaded": self._loaded, "model_id": self._model_id, "alias": _MODEL_ALIAS}


llm_service = LLMService()
