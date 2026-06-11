# === ARQUIVO: backend/main.py ===
import logging
import os
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

logger.info("=== JARVIS 2.0 iniciando boot ===")
logger.info("Python: %s", sys.version)
logger.info("CWD: %s", os.getcwd())

try:
    from fastapi import FastAPI
    logger.info("fastapi OK")
    from fastapi.middleware.cors import CORSMiddleware
    logger.info("cors OK")
    from pydantic import BaseModel
    logger.info("pydantic OK")
except Exception as e:
    logger.error("FALHA ao importar FastAPI/pydantic: %s", e)
    sys.exit(1)

try:
    from core.orchestrator import Orchestrator
    logger.info("Orchestrator OK")
except Exception as e:
    logger.error("FALHA ao importar Orchestrator: %s", e, exc_info=True)
    sys.exit(1)

try:
    from settings import validate_settings, OPENROUTER_MODEL, FRONTEND_ORIGIN
    logger.info("settings OK — modelo: %s", OPENROUTER_MODEL)
except Exception as e:
    logger.error("FALHA ao importar settings: %s", e, exc_info=True)
    sys.exit(1)

app = FastAPI(title="JARVIS 2.0 API", version="2.0.0")

origins = [FRONTEND_ORIGIN] if FRONTEND_ORIGIN and FRONTEND_ORIGIN != "*" else ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

orchestrator = Orchestrator()


class ChatRequest(BaseModel):
    message: str


@app.on_event("startup")
async def startup_event():
    errors = validate_settings()
    if errors:
        for err in errors:
            logger.warning("CONFIG: %s", err)
    else:
        logger.info("Configuracao validada. Modelo: %s", OPENROUTER_MODEL)


@app.get("/")
async def root():
    return {
        "status": "online",
        "message": "J.A.R.V.I.S API esta funcionando!",
        "endpoints": ["/", "/chat", "/health", "/memory"],
    }


@app.get("/health")
async def health_check():
    return {"status": "online"}


@app.post("/chat")
async def chat(request: ChatRequest):
    if not request.message or not request.message.strip():
        return {"error": "Mensagem vazia.", "status": "error"}
    try:
        response = await orchestrator.handle_message(request.message)
        return {"response": response, "status": "success"}
    except Exception as e:
        logger.error("Erro no /chat: %s", e, exc_info=True)
        return {"error": "Erro interno.", "status": "error"}


@app.get("/memory")
async def get_memory():
    try:
        return orchestrator.memory_agent.get_short_term_memory()
    except Exception as e:
        logger.error("Erro no /memory: %s", e)
        return []


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
