# JARVIS 2.0 — Architecture Notes

Deep-dive technical documentation covering design decisions, patterns, and upgrade paths for each major component.

---

## 1. System Pipeline

Every user interaction flows through the same pipeline regardless of input type (text / voice / gesture):

```
Input
  └─ main.py (FastAPI route)
       └─ JarvisCore.chat() / .voice_command() / .gesture_command()
            ├─ ensure_session()        → ConversationMemory
            ├─ retrieve_context()      → ConversationMemory + VectorMemory
            ├─ AgentManager.route()    → agent pipeline
            │    ├─ PlannerAgent       (code/analyse/plan)
            │    ├─ ExecutorAgent      (all intents)
            │    ├─ KnowledgeAgent     (search)
            │    ├─ GestureAgent       (gesture)
            │    └─ MemoryAgent        (memory intent)
            ├─ store_turn()            → ConversationMemory + VectorMemory
            └─ CoreResponse            → API response + WS broadcast
```

---

## 2. JarvisCore

**File:** `backend/jarvis_core.py`  
**Pattern:** Facade + Factory

JarvisCore is a facade over all system components. It has no business logic of its own — it only orchestrates.

### Why a facade?

The FastAPI routes stay thin (schema validation only). All routing, memory, and agent decisions live in one place. Testing is easy: mock `AgentManager` and `MemoryAgent`, then test `JarvisCore` in isolation.

### Session lifecycle

```
Request arrives without session_id
  → ConversationMemory.get_or_create(None)
  → generates new UUID, creates session dict
  → returns UUID in response

Next request with session_id
  → ConversationMemory.get_or_create(sid)
  → restores existing session, updates last_active
  → returns same UUID
```

Sessions expire after `SESSION_TTL_SECONDS` (default: 7200 = 2 hours) of idle time. The background eviction loop runs every 5 minutes.

### Context assembly

Before every LLM call, `JarvisCore` assembles a context object:

```python
{
  "messages":    [last 20 {role, content} pairs],
  "relevant":    [top-3 vector search results],
  "context_bag": {user_name: "Manoel", ...}
}
```

The `relevant` list lets JARVIS answer questions about things said in previous sessions (semantic memory, not just rolling window).

---

## 3. AgentManager

**File:** `backend/agent_manager.py`  
**Pattern:** Strategy + Chain of Responsibility

### Intent classification

Classification uses a priority-ordered regex table (fast, deterministic, zero latency):

```python
_INTENT_RULES = [
  (re.compile(r"\b(write|generate|code|debug|...)\b"), "code"),
  (re.compile(r"\b(search|find|what is|...)\b"),       "search"),
  ...
]
```

The first matching pattern wins. If nothing matches, intent = `"chat"`.

### Pipeline selection

| Intent | Why this pipeline |
|---|---|
| `chat`, `voice` | ExecutorAgent only — single LLM call, fastest |
| `code`, `analyse`, `plan` | Planner first decomposes into steps; each step can call different tools |
| `search` | KnowledgeAgent does web search + vector retrieval then synthesises — optimised for factual accuracy |
| `gesture` | GestureAgent maps landmark geometry to intent then calls LLM — bypasses text NLU |
| `system` | Executor with `system_status` tool pre-selected — no LLM needed for plain metrics |
| `memory` | MemoryAgent extracts key/value then persists — no need for full agent pipeline |

---

## 4. Memory System

**Files:** `backend/memory/`, `backend/agents/memory_agent.py`

### Three-tier design

```
Tier 1: ConversationMemory  (rolling window, per session)
  → asyncio.Lock per session (no global lock bottleneck)
  → MAX_MESSAGES = 60 (oldest evicted)
  → Redis upgrade: change ConversationMemory backend to redis.asyncio

Tier 2: Context bag  (per session, no TTL)
  → Arbitrary key/value: user_name, last_intent, preferences
  → Persists for session lifetime
  → Cleared only when session is destroyed

Tier 3: VectorMemory  (cross-session, semantic)
  → FAISS IndexFlatIP (cosine similarity)
  → all-MiniLM-L6-v2 (384-dim, 80 MB, CPU)
  → Stores "important" content (messages matching _IMPORTANT_PATTERNS regex)
  → Upgrade: swap _encode()/_search() with Chroma or Qdrant client
```

### What gets stored in vector memory

The `MemoryAgent` stores to VectorMemory only when:
- The user message matches `_IMPORTANT_PATTERNS` (remember / note / important / prefer / etc.)
- A tool returned significant content (web_search, file_reader)

This prevents the vector index from growing with every trivial exchange.

### Concurrency model

Each session has its own `asyncio.Lock`. A meta-lock guards only the `_sessions` and `_locks` dicts (a tiny, fast operation). This allows N concurrent sessions without serialisation.

```
Session A ──lock_A──► read/write session A data
Session B ──lock_B──► read/write session B data
  (A and B run truly concurrently)
```

---

## 5. LLM Service

**File:** `backend/services/llm_service.py`  
**Pattern:** Thread pool offloading

### Why ThreadPoolExecutor?

PyTorch `forward()` is synchronous and CPU/GPU-bound. Calling it directly in an async handler blocks the entire event loop — all other requests freeze until inference completes.

Solution: `asyncio.get_running_loop().run_in_executor(executor, sync_fn, ...)` offloads the blocking call to a thread pool. The event loop stays free to handle WebSocket pings, health checks, etc.

```python
text = await loop.run_in_executor(
    self._executor,    # ThreadPoolExecutor(max_workers=1)
    self._sync_chat,   # blocking inference
    messages, n, temp, system,
)
```

`max_workers=1` prevents multiple model copies from loading into memory simultaneously (OOM).

### Model registry

```python
MODEL_REGISTRY = {
    "tinyllama":  "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "phi-2":      "microsoft/phi-2",
    "mistral-7b": "mistralai/Mistral-7B-Instruct-v0.3",
    ...
}
```

Use the alias in `LLM_MODEL_ID` env var. Swap models without touching code.

### trust_remote_code

Only models from trusted orgs get `trust_remote_code=True`:

```python
_TRUSTED_ORGS = {"mistralai", "meta-llama", "microsoft", "TinyLlama", "deepseek-ai"}
```

Unknown org → `trust_remote_code=False` + warning log. This prevents arbitrary code execution from malicious HuggingFace repos.

---

## 6. Tool System

**File:** `backend/services/tool_service.py`  
**Pattern:** Registry + Allowlist + Decorator

### Security layers

1. **Allowlist** — `ALLOWED_TOOLS` env var is the gatekeeper. Any tool not in the list is blocked before registration is checked.
2. **Schema validation** — required parameters checked before execution.
3. **Timeout** — `asyncio.wait_for(fn(), timeout=TOOL_TIMEOUT_SECONDS)` cancels hung tools.
4. **Output truncation** — results capped at `TOOL_MAX_OUTPUT_BYTES` (64 KB) to prevent memory exhaustion.
5. **Exception barrier** — any uncaught exception becomes a `ToolResult(success=False, error=...)` rather than propagating.

### Adding a custom tool

```python
from backend.services.tool_service import tool_service, ToolSpec

async def my_tool(query: str) -> dict:
    # ... your implementation
    return {"result": "..."}

tool_service.register(ToolSpec(
    name="my_tool",
    fn=my_tool,
    description="What this tool does",
    parameters={
        "type": "object",
        "required": ["query"],
        "properties": {"query": {"type": "string"}},
    },
))
```

Then add `my_tool` to the `ALLOWED_TOOLS` env var.

---

## 7. Gesture System

**Files:** `gesture-system/gesture_system/`

### Pipeline

```
Camera (sounddevice/OpenCV)
  └─ CameraManager (background thread, double-buffered frame slot)
       └─ HandTracker (MediaPipe HandLandmarker, 21 landmarks)
            └─ FeatureExtractor (42-dim geometric vector)
                 └─ GestureClassifier (rules → ML model if available)
                      └─ GestureInterpreter (5-frame confirm + velocity check)
                           └─ GestureAgent (HUD draw + WS emit)
                                └─ WebSocketClient → /ws/gestures
```

### Why 5-frame confirmation?

Single-frame classification is noisy — a hand passing through a gesture briefly fires false positives. Requiring 5 consecutive frames (~166 ms) before emitting an event eliminates noise without introducing perceptible latency.

### Velocity-based swipe confirmation

Swipes are confirmed only when wrist velocity exceeds `SWIPE_MIN_VX = 0.8 units/s`. This prevents accidental triggers when the hand drifts slowly.

### MediaPipe 0.10+ migration

MediaPipe 0.10 dropped `mp.solutions` entirely. The system uses the Tasks API:

```python
from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions
# Model: auto-downloaded to ~/.cache/jarvis/hand_landmarker.task
```

---

## 8. Voice System

**Files:** `voice-system/`

### Pipeline

```
Microphone
  └─ SpeechListener (sounddevice + energy VAD)
       └─ SpeechToText (Whisper / Google / Vosk)
            └─ WakeWordDetector (Porcupine / text-match)
                 └─ CommandParser (47 regex patterns → VoiceCommand)
                      ├─ [URL/app/media] CommandExecutor (local execution)
                      └─ [AI query]      POST /voice-command → JarvisCore
```

### VAD (Voice Activity Detection)

Energy-based gate: RMS amplitude of each 100 ms chunk compared to `VAD_RMS_THRESHOLD`. Speech segment:
- Starts: after 2 consecutive loud chunks
- Ends: after 12 consecutive quiet chunks (1.2 s silence)
- Pre-roll: 3 chunks kept before speech (300 ms lead-in to avoid clipping word starts)

This avoids sending silence to the (potentially slow) STT engine.

### STT backend fallback chain

```
whisper_local (best accuracy, offline)
  ↓ if unavailable
google (cloud, free tier)
  ↓ if unavailable
vosk (offline, lower accuracy)
  ↓ if unavailable
RuntimeError
```

Set with `--stt-backend` flag or `STT_BACKEND` env var.

### Command routing

`CommandParser` first checks if the command is a local action (URL, app, volume, etc.) using 47 compiled regex patterns. If it is, `CommandExecutor` handles it without an API call. Only "AI" queries (questions, code requests, etc.) go to the backend.

This means `"open youtube"` executes in <10 ms locally, while `"write me a haiku"` goes to the LLM.

---

## 9. Frontend → Backend Integration

### Session persistence

The frontend stores `session_id` in `localStorage`. This lets the user refresh the page without losing conversation context.

```javascript
let sessionId = localStorage.getItem("jarvis_session_id");

const resp = await fetch(`${BACKEND}/chat`, {
  method: "POST",
  headers: {"Content-Type": "application/json"},
  body: JSON.stringify({message, session_id: sessionId}),
});
const data = await resp.json();
sessionId = data.session_id;
localStorage.setItem("jarvis_session_id", sessionId);
```

### Real-time hologram updates (WebSocket)

The frontend connects to `/ws/ui` to receive real-time updates after every agent interaction. This decouples the UI from the REST polling loop.

```javascript
const ws = new WebSocket("wss://jarvis-api.onrender.com/ws/ui");
ws.onmessage = (e) => {
  const msg = JSON.parse(e.data);
  if (msg.type === "jarvis_update") {
    hologramEngine.activateCore();
    displayResponse(msg.text);
    setTimeout(() => hologramEngine.deactivateCore(), 2000);
  }
};
```

---

## 10. Upgrade Paths

### Redis session store (Tier 1 memory)

Replace `ConversationMemory` with a Redis backend:

```python
# In backend/memory/conversation_memory.py
import redis.asyncio as redis

class ConversationMemory:
    def __init__(self):
        self._redis = redis.from_url(os.getenv("REDIS_URL"))

    async def add_message(self, session_id, role, content, **meta):
        key = f"session:{session_id}:messages"
        msg = json.dumps({"role": role, "content": content, **meta})
        await self._redis.rpush(key, msg)
        await self._redis.ltrim(key, -MAX_MESSAGES, -1)
        await self._redis.expire(key, SESSION_TTL_S)
```

Public API is unchanged — no other files need modification.

### Chroma vector store (Tier 3 memory)

Replace `VectorMemory._index` with a Chroma collection:

```python
import chromadb

class VectorMemory:
    def _load_sync(self):
        client = chromadb.PersistentClient(path=str(VECTOR_DIR))
        self._collection = client.get_or_create_collection("jarvis")

    def _add_sync(self, entry):
        self._collection.add(
            ids=[entry["id"]],
            documents=[entry["text"]],
            metadatas=[entry],
        )

    def _search_sync(self, query, top_k, session_id, min_score):
        results = self._collection.query(
            query_texts=[query], n_results=top_k
        )
        # ... map to VectorSearchResult
```

### Streaming responses

Add a `/chat/stream` endpoint using Server-Sent Events:

```python
from fastapi.responses import StreamingResponse

@router.post("/chat/stream")
async def chat_stream(req: ChatRequest):
    async def generate():
        async for chunk in core.stream_chat(req.message, req.session_id):
            yield f"data: {json.dumps({'text': chunk})}\n\n"
    return StreamingResponse(generate(), media_type="text/event-stream")
```
