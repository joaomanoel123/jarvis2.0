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
from fastapi.responses import Response  # ← ADICIONADO
from pydantic import BaseModel
from core.orchestrator import Orchestrator
from core.tts_manager  import text_to_speech
from settings import validate_settings, OPENROUTER_MODEL, FRONTEND_ORIGIN

app = FastAPI(title="JARVIS 2.0 API", version="2.2.0")

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
    tts:        bool = False
    history:    list = []  # ← ADICIONADO (Fix 3 do jarvis.js)

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
        "endpoints": ["/", "/chat", "/upload", "/health", "/memory", "/config.js"],
    }

@app.get("/health")
async def health():
    return {"status": "online"}

# ── Config endpoint — injeta env vars no frontend ─────────────────────────────
@app.get("/config.js")
async def config_js():
    roboflow_key = os.environ.get("ROBOFLOW_API_KEY", "")
    api_url = os.environ.get("API_URL", "https://jarvis-tdgt.onrender.com")
    js = f"""window.__JARVIS_CONFIG__ = {{
  "apiUrl": "{api_url}",
  "roboflowKey": "{roboflow_key}"
}};"""
    return Response(content=js, media_type="application/javascript")

@app.post("/chat")
async def chat(request: ChatRequest):
    if not request.message.strip():
        return {"error": "Mensagem vazia.", "status": "error"}
    try:
        response = await orchestrator.handle_message(
            request.message, request.session_id
        )
        result = {"response": response, "status": "success"}
        if request.tts:
            audio_b64 = await text_to_speech(response)
            if audio_b64:
                result["audio"] = audio_b64
                result["audio_format"] = "mp3"
        return result
    except Exception as e:
        logger.error("Erro /chat: %s", e, exc_info=True)
        return {"error": "Erro interno.", "status": "error"}

@app.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
    session_id: str  = Form("default"),
):
    """Recebe PDF ou TXT e indexa no RAG."""
    try:
        content = await file.read()
        text    = ""
        if file.filename.endswith(".pdf"):
            try:
                import io, pdfplumber
                with pdfplumber.open(io.BytesIO(content)) as pdf:
                    text = "\n".join(p.extract_text() or "" for p in pdf.pages)
            except ImportError:
                return {"error": "pdfplumber nao instalado.", "status": "error"}
        else:
            text = content.decode("utf-8", errors="ignore")
        if not text.strip():
            return {"error": "Documento vazio.", "status": "error"}
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
