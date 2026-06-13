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

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from core.orchestrator import Orchestrator
from settings import validate_settings, OPENROUTER_MODEL, FRONTEND_ORIGIN

app = FastAPI(title="JARVIS 2.0 API", version="2.1.0")

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
    message:    str
    session_id: str = "default"


@app.on_event("startup")
async def startup_event():
    errors = validate_settings()
    if errors:
        for e in errors:
            logger.warning("CONFIG: %s", e)
    else:
        logger.info("Configuracao validada. Modelo: %s", OPENROUTER_MODEL)


@app.get("/")
async def root():
    return {
        "status":  "online",
        "message": "J.A.R.V.I.S API esta funcionando!",
        "endpoints": ["/", "/chat", "/upload", "/health", "/memory"],
    }


@app.get("/health")
async def health():
    return {"status": "online"}


@app.post("/chat")
async def chat(request: ChatRequest):
    if not request.message.strip():
        return {"error": "Mensagem vazia.", "status": "error"}
    try:
        response = await orchestrator.handle_message(
            request.message, request.session_id
        )
        return {"response": response, "status": "success"}
    except Exception as e:
        logger.error("Erro /chat: %s", e, exc_info=True)
        return {"error": "Erro interno.", "status": "error"}


@app.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
    session_id: str  = Form("default"),
):
    """Recebe documento (PDF ou TXT), extrai texto e indexa no RAG."""
    try:
        content = await file.read()
        text    = ""

        if file.filename.endswith(".pdf"):
            try:
                import io
                import pdfplumber
                with pdfplumber.open(io.BytesIO(content)) as pdf:
                    text = "\n".join(
                        page.extract_text() or "" for page in pdf.pages
                    )
            except ImportError:
                return {"error": "pdfplumber nao instalado.", "status": "error"}
        else:
            text = content.decode("utf-8", errors="ignore")

        if not text.strip():
            return {"error": "Documento vazio ou nao legivel.", "status": "error"}

        added = await orchestrator.knowledge_agent.add_document(text)
        return {
            "status":  "success",
            "message": f"{added} chunks indexados de '{file.filename}'.",
            "chunks":  added,
        }
    except Exception as e:
        logger.error("Erro /upload: %s", e, exc_info=True)
        return {"error": "Falha ao processar documento.", "status": "error"}


@app.get("/memory")
async def get_memory(session_id: str = "default"):
    return orchestrator.memory_agent.get_short_term_memory()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
