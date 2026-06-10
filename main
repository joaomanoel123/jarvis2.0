# === ARQUIVO: backend/main.py ===
"""
Ponto de entrada da API JARVIS 2.0.
Comunicação: REST puro (POST /chat, GET /, GET /health, GET /memory).
WebSocket removido — frontend usa fetch() REST.
"""

import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

from core.orchestrator import Orchestrator
from settings import validate_settings, OPENROUTER_MODEL, FRONTEND_ORIGIN

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="JARVIS 2.0 API",
    description="Backend do ANTIGRAVITY — JARVIS 2.0",
    version="2.0.0",
)

# ── CORS ──────────────────────────────────────────────────────────────────────
# FIX ERRO #10: FRONTEND_ORIGIN pode ser string — garantir lista válida
origins = [FRONTEND_ORIGIN] if FRONTEND_ORIGIN and FRONTEND_ORIGIN != "*" else ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Orchestrator singleton ────────────────────────────────────────────────────
orchestrator = Orchestrator()


# ── Modelos Pydantic ──────────────────────────────────────────────────────────
# FIX ERRO #4: substituir request: dict por BaseModel
class ChatRequest(BaseModel):
    message: str


# ── Eventos de lifecycle ──────────────────────────────────────────────────────
@app.on_event("startup")
async def startup_event():
    """Valida configuração e loga status no boot."""
    errors = validate_settings()
    if errors:
        for err in errors:
            logger.warning("⚠️  CONFIG: %s", err)
    else:
        logger.info("✅ Configuração validada. Modelo OpenRouter: %s", OPENROUTER_MODEL)


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/")
async def root():
    """Health check principal — usado pelo frontend para verificar status."""
    return {
        "status": "online",
        "message": "J.A.R.V.I.S API está funcionando!",
        "endpoints": ["/", "/chat", "/health", "/memory"],
    }


@app.get("/health")
async def health_check():
    """
    Health check secundário.
    FIX ERRO #6: retorna 'online' em vez de 'ok' para consistência com o frontend.
    """
    return {"status": "online", "message": "JARVIS health check OK"}


@app.post("/chat")
async def chat(request: ChatRequest):
    """
    Endpoint principal de chat.
    Recebe mensagem do usuário, processa via Orchestrator e retorna resposta.
    FIX ERRO #4: usa Pydantic BaseModel em vez de dict.
    """
    if not request.message or not request.message.strip():
        return {"error": "Mensagem não pode estar vazia.", "status": "error"}

    try:
        response = await orchestrator.handle_message(request.message)
        return {"response": response, "status": "success"}
    except Exception as e:
        logger.error("Erro no endpoint /chat: %s", e, exc_info=True)
        return {
            "error": "Erro interno ao processar mensagem.",
            "status": "error",
        }


@app.get("/memory")
async def get_memory():
    """Inspeciona o histórico de curto prazo do MemoryAgent."""
    try:
        return orchestrator.memory_agent.get_short_term_memory()
    except Exception as e:
        logger.error("Erro no endpoint /memory: %s", e)
        return []


# ── Entry point local ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
