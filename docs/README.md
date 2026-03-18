# JARVIS 2.0 — Complete System Documentation

> Multi-agent AI assistant with voice control, gesture recognition, holographic UI, and cloud backend.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Architecture Diagram](#2-architecture-diagram)
3. [Repository Structure](#3-repository-structure)
4. [Module Reference](#4-module-reference)
5. [API Reference](#5-api-reference)
6. [Setup & Installation](#6-setup--installation)
7. [Deployment (Render)](#7-deployment-render)
8. [Configuration Reference](#8-configuration-reference)
9. [Data Flow Diagrams](#9-data-flow-diagrams)
10. [Roadmap](#10-roadmap)

---

## 1. Project Overview

JARVIS 2.0 is a modular AI assistant system inspired by Iron Man's J.A.R.V.I.S., composed of four independently deployable modules that communicate via REST API and WebSocket.

| Module | Purpose | Tech |
|---|---|---|
| `backend/` | FastAPI + multi-agent AI brain | Python, FastAPI, HuggingFace |
| `gesture-system/` | Real-time hand gesture recognition | OpenCV, MediaPipe, FAISS |
| `voice-system/` | Voice command pipeline | Whisper, SpeechRecognition, pyttsx3 |
| `frontend/` | Holographic 3D interface | Three.js, WebGL |

**Deployment topology:**
```
GitHub Pages (frontend)
       ↕  HTTPS / WSS
Render.com (backend API)   ← gesture-system and voice-system connect here
```

---

## 2. Architecture Diagram

```
╔══════════════════════════════════════════════════════════════════════╗
║                    JARVIS 2.0 — System Architecture                  ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                      ║
║  INPUT LAYER                                                         ║
║  ┌─────────────────────────────────────────────────────────────┐    ║
║  │  Text (Web UI)   Voice (Mic)   Gesture (Camera)              │    ║
║  └───────────┬──────────────┬────────────────┬────────────────-┘    ║
║              │              │                │                       ║
║  TRANSPORT LAYER            │                │                       ║
║  ┌───────────▼──────────────▼────────────────▼────────────────-┐    ║
║  │  POST /chat   POST /voice-command   POST /gesture-command    │    ║
║  │  WS /ws/gestures (gesture-system)   WS /ws/ui (frontend)    │    ║
║  └───────────────────────────┬────────────────────────────────-┘    ║
║                              │                                       ║
║  CORE LAYER                  │                                       ║
║  ┌───────────────────────────▼────────────────────────────────-┐    ║
║  │                      JarvisCore                              │    ║
║  │  1. Session resolution  2. Context assembly                  │    ║
║  │  3. Agent routing       4. Memory persistence               │    ║
║  └───────────────────────────┬────────────────────────────────-┘    ║
║                              │                                       ║
║  AGENT LAYER                 │                                       ║
║  ┌───────────────────────────▼────────────────────────────────-┐    ║
║  │                     AgentManager                             │    ║
║  │  ┌─────────────┬────────────┬───────────┬────────────────┐  │    ║
║  │  │ PlannerAgent│ExecutorAgt │KnowledgeAgt│ GestureAgent  │  │    ║
║  │  │ (decompose) │ (tools+LLM)│(search+RAG)│ (MediaPipe)   │  │    ║
║  │  └─────────────┴────────────┴───────────┴────────────────┘  │    ║
║  │                        MemoryAgent                           │    ║
║  └───────────────────────────┬────────────────────────────────-┘    ║
║                              │                                       ║
║  SERVICE LAYER               │                                       ║
║  ┌───────────────────────────▼────────────────────────────────-┐    ║
║  │   LLMService          ToolService       PromptManager        │    ║
║  │   (HuggingFace)       (4 tools)         (7 agent prompts)   │    ║
║  └───────────────────────────┬────────────────────────────────-┘    ║
║                              │                                       ║
║  MEMORY LAYER                │                                       ║
║  ┌───────────────────────────▼────────────────────────────────-┐    ║
║  │  ConversationMemory   VectorMemory (FAISS + MiniLM-L6-v2)   │    ║
║  │  (per-session history + context bag)   (semantic retrieval) │    ║
║  └─────────────────────────────────────────────────────────────┘    ║
╚══════════════════════════════════════════════════════════════════════╝
```

---

## 3. Repository Structure

```
jarvis2.0/
│
├── backend/                          ← FastAPI application
│   ├── main.py                       ← Application entry point & all routes
│   ├── jarvis_core.py                ← Central brain (text/voice/gesture routing)
│   ├── agent_manager.py              ← Multi-agent pipeline orchestrator
│   │
│   ├── agents/
│   │   ├── base_agent.py             ← AgentTask / AgentResult / BaseAgent ABC
│   │   ├── planner_agent.py          ← Decomposes tasks into JSON plans
│   │   ├── executor_agent.py         ← Executes plans via LLM + tool routing
│   │   ├── knowledge_agent.py        ← Web search + vector retrieval + synthesis
│   │   ├── gesture_agent.py          ← MediaPipe gesture → LLM response
│   │   └── memory_agent.py           ← store_turn() / retrieve_context()
│   │
│   ├── services/
│   │   ├── llm_service.py            ← HuggingFace Transformers wrapper
│   │   └── tool_service.py           ← Allowlisted tool execution
│   │
│   ├── memory/
│   │   ├── conversation_memory.py    ← Async session store (rolling window)
│   │   └── vector_memory.py          ← FAISS semantic search
│   │
│   ├── tools/
│   │   ├── web_search.py             ← DuckDuckGo instant answers
│   │   ├── code_runner.py            ← Sandboxed Python execution
│   │   ├── file_reader.py            ← Path-traversal-safe file access
│   │   └── system_status.py          ← CPU / RAM / disk metrics
│   │
│   ├── prompts/                      ← Agent prompt text files
│   │   ├── system_prompt.txt
│   │   ├── planner_agent_prompt.txt
│   │   ├── executor_agent_prompt.txt
│   │   ├── knowledge_agent_prompt.txt
│   │   ├── gesture_agent_prompt.txt
│   │   ├── conversation_agent_prompt.txt
│   │   └── memory_agent_prompt.txt
│   │
│   └── config/settings.py            ← Pydantic settings (all env vars)
│
├── gesture-system/
│   ├── main.py                       ← Standalone entry point (--backend flag)
│   ├── app_ws_routes.py              ← FastAPI WS endpoints to add to backend
│   └── gesture_system/
│       ├── camera.py                 ← Background thread camera capture + VAD
│       ├── hand_tracking.py          ← MediaPipe Tasks API (0.10+)
│       ├── feature_extraction.py     ← 42-dim geometric feature vector
│       ├── gesture_classifier.py     ← Rule-based + optional RandomForest
│       ├── gesture_interpreter.py    ← Temporal state machine (5-frame confirm)
│       ├── gesture_agent.py          ← Full pipeline + HUD renderer
│       ├── websocket_client.py       ← Reconnecting WS client
│       └── trainer.py                ← Dataset collection + model training
│
├── voice-system/
│   ├── voice_main.py                 ← Full pipeline controller + TTS feedback
│   ├── speech_listener.py            ← Sounddevice capture + energy VAD
│   ├── speech_to_text.py             ← Whisper / Google / Vosk backends
│   ├── wake_word_detector.py         ← Porcupine + text-match fallback
│   ├── command_parser.py             ← 47 regex patterns → VoiceCommand
│   └── command_executor.py           ← URL / app / media / API execution
│
├── frontend/                         ← Three.js holographic UI (GitHub Pages)
│
├── docs/                             ← This documentation
│   ├── README.md                     ← Master documentation (this file)
│   ├── API.md                        ← Full API endpoint reference
│   ├── ARCHITECTURE.md               ← Deep-dive architecture notes
│   └── DEPLOYMENT.md                 ← Step-by-step deployment guide
│
├── requirements.txt                  ← Unified Python dependencies
└── render.yaml                       ← Render.com deployment manifest
```

---

## 4. Module Reference

### `JarvisCore` (`backend/jarvis_core.py`)

Central brain — single entry point for all input types.

| Method | Description |
|---|---|
| `chat(text, session_id, metadata)` | Process a text message end-to-end |
| `voice_command(text, session_id, confidence)` | Process a voice-transcribed command |
| `gesture_command(gesture_id, session_id, confidence, landmarks)` | Process a MediaPipe gesture |
| `status()` | Return full system diagnostics dict |
| `health()` | Lightweight liveness check |
| `JarvisCore.build()` | Factory — wires all components together |

### `AgentManager` (`backend/agent_manager.py`)

Routes tasks to agent pipelines based on classified intent.

| Intent | Pipeline |
|---|---|
| `chat`, `voice` | ExecutorAgent (direct LLM) |
| `code`, `analyse`, `plan` | PlannerAgent → ExecutorAgent |
| `search` | KnowledgeAgent (web + vector) |
| `gesture` | GestureAgent (MediaPipe) |
| `system` | ExecutorAgent + system_status tool |
| `memory` | MemoryAgent → ExecutorAgent |

### `MemoryAgent` (`backend/agents/memory_agent.py`)

| Method | Description |
|---|---|
| `store_turn(session_id, user_text, bot_text, ...)` | Persist a conversation turn |
| `retrieve_context(session_id, query, history_n, vec_top_k)` | Assemble LLM context |
| `remember(session_id, key, value)` | Store an explicit fact |
| `recall(session_id, key)` | Retrieve a stored fact |
| `maybe_summarise(session_id)` | Summarise if history > 3000 tokens |

### `ConversationMemory` (`backend/memory/conversation_memory.py`)

Async session-scoped message store with asyncio.Lock per session.

- Rolling window: `MAX_MESSAGES = 60` (configurable)
- Session TTL: `SESSION_TTL_S = 7200` (2 hours idle)
- Redis upgrade: set `REDIS_URL` env var (drop-in swap)

### `VectorMemory` (`backend/memory/vector_memory.py`)

Semantic search with FAISS + sentence-transformers.

- Model: `all-MiniLM-L6-v2` (384-dim, ~80 MB, CPU-only)
- Index: FAISS `IndexFlatIP` (cosine similarity)
- Persistence: `data/vector_memory/jarvis.index` + metadata JSON
- Max entries: 10,000 (LRU eviction)
- Upgrade: replace `_encode()` with Chroma/Qdrant client

### `ToolService` (`backend/services/tool_service.py`)

| Tool | Description | Env toggle |
|---|---|---|
| `web_search` | DuckDuckGo instant answers + links | `ALLOWED_TOOLS` |
| `run_python_code` | Sandboxed Python (AST-gated) | `ALLOWED_TOOLS` |
| `file_reader` | Read files from `/tmp/jarvis_files` | `ALLOWED_TOOLS` |
| `system_status` | CPU / RAM / disk / uptime | `ALLOWED_TOOLS` |
| `open_url` | Open URL in default browser | `ALLOWED_TOOLS` |

### `LLMService` (`backend/services/llm_service.py`)

HuggingFace Transformers wrapper. Single `ThreadPoolExecutor` prevents OOM.

| Alias | Model ID |
|---|---|
| `tinyllama` | TinyLlama/TinyLlama-1.1B-Chat-v1.0 |
| `phi-2` | microsoft/phi-2 |
| `phi-3-mini` | microsoft/Phi-3-mini-4k-instruct |
| `mistral-7b` | mistralai/Mistral-7B-Instruct-v0.3 |
| `llama-3` | meta-llama/Meta-Llama-3-8B-Instruct |
| `deepseek-coder` | deepseek-ai/deepseek-coder-7b-instruct-v1.5 |

---

## 5. API Reference

### POST `/chat`

Send a text message to JARVIS.

**Request:**
```json
{
  "message": "Write a Python function that reverses a string",
  "session_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
  "metadata": {}
}
```

**Response:**
```json
{
  "session_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
  "text": "Here's a Python function that reverses a string...",
  "intent": "code",
  "agent_path": ["planner", "executor"],
  "tool_calls": [],
  "steps": ["Define function signature", "Implement using slicing"],
  "latency_ms": 1840.5,
  "model_id": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
  "success": true,
  "error": null
}
```

---

### POST `/voice-command`

Send a transcribed voice command (wake word already stripped).

**Request:**
```json
{
  "text": "open youtube",
  "session_id": null,
  "confidence": 0.93,
  "source": "voice"
}
```

**Response:** Same shape as `/chat` response.

---

### POST `/gesture-command`

Send a MediaPipe gesture event.

**Request:**
```json
{
  "gesture_id": "swipe_right",
  "session_id": null,
  "confidence": 0.95,
  "landmarks": [{"x": 0.5, "y": 0.5, "z": 0.0}],
  "active_widget": "main_panel",
  "coordinates": {"x": 640, "y": 360}
}
```

---

### GET `/health`

Liveness probe — instant, no model call.

```json
{
  "status": "ok",
  "model_loaded": true,
  "uptime_s": 3600.1,
  "version": "2.0.0"
}
```

---

### GET `/status`

Full system diagnostics.

```json
{
  "status": "operational",
  "uptime_s": 3600.1,
  "requests": 142,
  "errors": 0,
  "memory": {"active_sessions": 3, "total_messages": 87},
  "agents": {"planner": true, "executor": true, "knowledge": true},
  "model": {"loaded": true, "model_id": "TinyLlama/...", "device": "cpu"}
}
```

---

### WebSocket `/ws/gestures`

Real-time gesture stream from the gesture-system client.

**Send (gesture-system → backend):**
```json
{"type": "gesture", "gesture": "SWIPE_LEFT", "confidence": 0.9, "session_id": null}
```

**Receive (backend → gesture-system):**
```json
{"type": "gesture_response", "gesture": "SWIPE_LEFT", "response": "Navigating left...", "latency_ms": 420}
```

---

### WebSocket `/ws/ui`

Holographic UI notifications.

**Receive (backend → frontend):**
```json
{"type": "jarvis_update", "text": "Done.", "intent": "code", "agent_path": ["planner", "executor"]}
```

---

## 6. Setup & Installation

### Prerequisites

```bash
# System packages
sudo apt install ffmpeg espeak portaudio19-dev python3-dev

# Python 3.10+
python3 --version
```

### Install

```bash
git clone https://github.com/joaomanoel123/jarvis2.0
cd jarvis2.0
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Configure

```bash
cp .env.example .env
# Edit .env:
# LLM_MODEL_ID=tinyllama   (fast, CPU-safe for development)
# HF_TOKEN=your_token      (needed for gated models)
```

### Run (development)

```bash
# Backend
cd backend
uvicorn main:app --reload --port 8000

# Gesture system (separate terminal)
cd gesture-system
python main.py --backend ws://localhost:8000/ws/gestures

# Voice system (separate terminal)
cd voice-system
python voice_main.py --api http://localhost:8000 --stt-backend google --always-on
```

### Run (Docker)

```bash
docker build -t jarvis-backend .
docker run -p 8000:8000 \
  -e LLM_MODEL_ID=tinyllama \
  -e HF_TOKEN=your_token \
  jarvis-backend
```

---

## 7. Deployment (Render)

See [`DEPLOYMENT.md`](./DEPLOYMENT.md) for the full step-by-step guide.

**Quick start:**
1. Push repo to GitHub.
2. Go to [render.com](https://render.com) → New → Web Service → connect repo.
3. Render auto-detects `render.yaml`.
4. Add `HF_TOKEN` secret in the Render dashboard (Environment tab).
5. Deploy. First deploy downloads the model (~600 MB for TinyLlama).
6. Update `ALLOWED_ORIGINS` to include your GitHub Pages URL.

**Start command:**
```bash
uvicorn main:app --host 0.0.0.0 --port 10000
```

**Health check path:** `/health`

---

## 8. Configuration Reference

All values set via environment variables (`.env` locally, Render dashboard in production).

| Variable | Default | Description |
|---|---|---|
| `LLM_MODEL_ID` | `tinyllama` | Model alias or full HuggingFace ID |
| `HF_TOKEN` | _(empty)_ | HuggingFace token (required for gated models) |
| `LLM_DEVICE` | `auto` | `auto` \| `cpu` \| `cuda` \| `mps` |
| `LLM_LOAD_IN_4BIT` | `false` | 4-bit NF4 quantisation (GPU only) |
| `LLM_MAX_NEW_TOKENS` | `1024` | Max tokens per response |
| `LLM_TEMPERATURE` | `0.7` | Sampling temperature (0–2) |
| `ALLOWED_TOOLS` | `web_search,...` | Comma-separated allowed tool names |
| `TOOL_TIMEOUT_SECONDS` | `30` | Max seconds per tool call |
| `SESSION_TTL_SECONDS` | `7200` | Idle session expiry (seconds) |
| `VECTOR_MEMORY_DIR` | `./data/vector_memory` | FAISS index storage path |
| `ALLOWED_ORIGINS` | `http://localhost:3000` | Comma-separated CORS origins |
| `DEBUG` | `false` | Expose exception details in API errors |

---

## 9. Data Flow Diagrams

### Text chat flow

```
User types message
       │
  POST /chat
       │
  JarvisCore.chat()
       │
  ┌────┴──────────────────────────────────────────────────┐
  │ 1. ensure_session()     → create/restore session UUID │
  │ 2. retrieve_context()   → last 20 messages + top-3    │
  │                            vector hits                 │
  │ 3. AgentManager.route() → classify intent             │
  │                        → run agent pipeline           │
  │ 4. store_turn()         → save to ConversationMemory  │
  │                        → save to VectorMemory (if     │
  │                            "important" content)        │
  └────┬──────────────────────────────────────────────────┘
       │
  ChatResponse → frontend
       │
  WS /ws/ui broadcast → Three.js update hologram
```

### Voice command flow

```
User speaks "Jarvis open YouTube"
       │
  SpeechListener (sounddevice + energy VAD)
       │
  SpeechToText (Whisper local / Google fallback)
       │ "jarvis open youtube"
  WakeWordDetector (text-match "jarvis")
       │ "open youtube"  (wake word stripped)
  CommandParser (regex: open_url → youtube → https://youtube.com)
       │
  [Non-AI command?]─YES──→ CommandExecutor.execute()
       │                     webbrowser.open("https://youtube.com")
       NO
       │
  POST /voice-command
       │
  JarvisCore.voice_command()
       │ (same pipeline as /chat)
  JARVIS AI response
       │
  TTSFeedback.speak("Opening YouTube")
```

### Gesture flow

```
Camera frame
       │
  HandTracker (MediaPipe HandLandmarker)
       │ 21 landmarks
  GestureClassifier (rule-based + optional RandomForest)
       │ ClassificationResult
  GestureInterpreter (5-frame confirm + velocity check)
       │ GestureEvent (confirmed)
  WebSocketClient.send_gesture()
       │
  WS /ws/gestures (backend)
       │
  JarvisCore.gesture_command()
       │
  AgentManager → GestureAgent → LLM response
       │
  WS /ws/ui broadcast → Three.js hologram update
```

---

## 10. Roadmap

### Near-term (v2.1)

| Feature | Module | Effort |
|---|---|---|
| Streaming responses (SSE) | `backend/main.py` | Medium |
| Redis session store | `backend/memory/` | Small (1 file swap) |
| Chroma persistent vector store | `backend/memory/vector_memory.py` | Small (1 file swap) |
| User authentication (JWT) | `backend/main.py` | Medium |
| Audio upload endpoint (`POST /voice/transcribe`) | `backend/main.py` | Small |

### Medium-term (v2.2)

| Feature | Module | Effort |
|---|---|---|
| GPU deployment (Mistral-7B 4-bit) | `render.yaml` + `config/settings.py` | Small (config) |
| Multi-modal (image input) | `backend/agents/` | Large |
| Persistent user profiles | `backend/memory/` | Medium |
| Plugin system for custom tools | `backend/services/tool_service.py` | Medium |
| Dashboard UI (React) | new `dashboard/` | Large |

### Long-term (v3.0)

- Voice cloning for JARVIS TTS
- Real-time face recognition (Vision Agent)
- IoT device control (smart home integration)
- Memory export / import between sessions
- Multi-user support with session isolation
