# JARVIS 2.0 — API Reference

Base URL (production): `https://jarvis-api.onrender.com`  
Base URL (development): `http://localhost:8000`

Interactive docs: `GET /docs` (Swagger UI) · `GET /redoc`

---

## Endpoints

| Method | Path | Description |
|---|---|---|
| POST | `/chat` | Text conversation |
| POST | `/voice-command` | Pre-transcribed voice command |
| POST | `/gesture-command` | MediaPipe gesture event |
| GET | `/health` | Liveness probe |
| GET | `/status` | Full system diagnostics |
| GET | `/voice/commands` | Supported voice commands list |
| GET | `/voice/status` | Voice system info |
| WS | `/ws/gestures` | Real-time gesture stream |
| WS | `/ws/ui` | Holographic UI updates |

---

## POST `/chat`

### Request

```http
POST /chat
Content-Type: application/json

{
  "message":    "Write a Python function that reverses a string",
  "session_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
  "metadata":   {}
}
```

| Field | Type | Required | Description |
|---|---|---|---|
| `message` | string | ✅ | User message (1–8000 chars) |
| `session_id` | string\|null | — | Session UUID. Omit to start a new session. |
| `metadata` | object | — | Extra context passed to agents |

### Response

```json
{
  "session_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
  "text":       "Here's a Python function that reverses a string:\n\n```python\ndef reverse_string(s: str) -> str:\n    return s[::-1]\n```",
  "intent":     "code",
  "agent_path": ["planner", "executor"],
  "tool_calls": [],
  "steps":      ["Define function signature", "Implement using slicing"],
  "latency_ms": 1840.5,
  "model_id":   "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
  "success":    true,
  "error":      null
}
```

| Field | Type | Description |
|---|---|---|
| `session_id` | string | Active session UUID — pass this in all subsequent requests |
| `text` | string | JARVIS response text |
| `intent` | string | Classified intent: `chat` \| `code` \| `search` \| `analyse` \| `gesture` \| `system` \| `memory` \| `plan` |
| `agent_path` | string[] | Agents invoked in order, e.g. `["planner", "executor"]` |
| `tool_calls` | object[] | Tool invocations: `{tool, result, success, latency_ms}` |
| `steps` | string[] | Plan steps (populated for `code`/`analyse`/`plan` intents) |
| `latency_ms` | number | Total processing time in milliseconds |
| `model_id` | string\|null | HuggingFace model that generated the response |
| `success` | boolean | `false` if an unrecoverable error occurred |
| `error` | string\|null | Error message when `success=false` |

### curl example

```bash
curl -X POST https://jarvis-api.onrender.com/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "search for AI news today"}'
```

### Intent routing

| Message contains | Intent | Agent pipeline |
|---|---|---|
| write / generate / code / debug | `code` | Planner → Executor |
| search / find / what is / who is | `search` | KnowledgeAgent |
| analyse / examine / dataset / csv | `analyse` | Planner → Executor |
| plan / steps to / strategy | `plan` | Planner → Executor |
| remember / save / recall | `memory` | MemoryAgent → Executor |
| system / status / cpu / ram | `system` | Executor (system_status tool) |
| anything else | `chat` | Executor (direct LLM) |

### Status codes

| Code | Meaning |
|---|---|
| 200 | Success |
| 422 | Validation error (message blank, too long, etc.) |
| 503 | Model not loaded — retry after a few seconds |
| 500 | Unexpected server error |

---

## POST `/voice-command`

Send a voice-transcribed command with the wake word already stripped.

### Request

```http
POST /voice-command
Content-Type: application/json

{
  "text":       "open youtube",
  "session_id": null,
  "confidence": 0.93,
  "source":     "voice"
}
```

| Field | Type | Required | Description |
|---|---|---|---|
| `text` | string | ✅ | Transcribed command text (wake word stripped) |
| `session_id` | string\|null | — | Session UUID |
| `confidence` | float | — | STT confidence [0–1] |
| `source` | string | — | Always `"voice"` |

### Response

Same shape as `/chat` response. For URL/app commands (`open youtube`, `volume up`, etc.) the response is handled by `CommandExecutor` directly without calling the LLM.

### curl example

```bash
# Simulate voice command
curl -X POST http://localhost:8000/voice-command \
  -H "Content-Type: application/json" \
  -d '{"text": "search for Python tutorials", "confidence": 0.95}'
```

---

## POST `/gesture-command`

Send a MediaPipe gesture event.

### Request

```http
POST /gesture-command
Content-Type: application/json

{
  "gesture_id":    "swipe_right",
  "session_id":    null,
  "confidence":    0.95,
  "landmarks":     [{"x": 0.5, "y": 0.5, "z": 0.0}],
  "active_widget": "main_panel",
  "coordinates":   {"x": 640, "y": 360},
  "metadata":      {}
}
```

| Field | Type | Required | Description |
|---|---|---|---|
| `gesture_id` | string | ✅ | Gesture name (alphanumeric, `_`, `-`) |
| `session_id` | string\|null | — | Session UUID |
| `confidence` | float | — | Classifier confidence [0–1] |
| `landmarks` | object[] | — | MediaPipe 21 hand landmarks `{x, y, z}` |
| `active_widget` | string\|null | — | UI element the gesture targeted |
| `coordinates` | object\|null | — | Screen coordinates `{x, y}` |

### Supported gestures

| gesture_id | Command | Description |
|---|---|---|
| `open_palm` | `open_menu` | Show available options |
| `fist` | `close_menu` | Close / cancel |
| `pinch` | `select_object` | Select item |
| `grab` | `grab_element` | Grab / hold |
| `release` | `release_element` | Release / drop |
| `swipe_right` | `navigate_right` | Next item |
| `swipe_left` | `navigate_left` | Previous item |
| `swipe_up` | `scroll_up` | Show more |
| `swipe_down` | `scroll_down` | Summary |
| `zoom_in` | `zoom_interface` | Zoom in |
| `zoom_out` | `shrink_interface` | Zoom out |
| `thumbs_up` | `confirm` | Confirm |
| `thumbs_down` | `reject` | Reject / retry |
| `wave` | `wake` | Greet / activate |
| `shake` | `reset` | Reset conversation |
| `point_up` | `scroll_top` | Top of list |

---

## GET `/health`

Liveness probe. Returns 200 as long as the process is alive.  
Used by Docker `HEALTHCHECK` and Render's liveness probe.

```bash
curl https://jarvis-api.onrender.com/health
```

```json
{
  "status":       "ok",
  "model_loaded": true,
  "uptime_s":     3600.1,
  "version":      "2.0.0"
}
```

> `model_loaded: false` means the server is alive but the LLM is still loading.  
> `/chat` will return 503 until `model_loaded` becomes `true`.

---

## GET `/status`

Full system diagnostics snapshot.

```json
{
  "status":    "operational",
  "uptime_s":  7245.3,
  "requests":  142,
  "errors":    0,
  "memory": {
    "active_sessions":  3,
    "total_messages":   87,
    "conversation": {"active_sessions": 3, "total_messages": 87},
    "vector":       {"loaded": true, "entries": 24, "model": "all-MiniLM-L6-v2"}
  },
  "agents": {
    "routes":  {"chat": 80, "code": 30, "search": 20, "gesture": 12},
    "agents":  {"planner": true, "executor": true, "knowledge": true, "gesture": true}
  },
  "model": {
    "loaded":       true,
    "model_id":     "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "parameters_b": 1.1,
    "device":       "cpu",
    "dtype":        "torch.float32",
    "4bit":         false
  }
}
```

---

## GET `/voice/commands`

List all supported voice commands.

```json
{
  "intents": ["jarvis_command", "launch_app", "media_control", "open_url", "system_control", "volume_control", "web_search"],
  "sites":   ["amazon", "anthropic", "bbc", "chatgpt", "claude", "cnn", "discord", "facebook", ...],
  "apps":    ["brave", "calculator", "chrome", "discord", "files", "firefox", "slack", "terminal", "vscode", ...]
}
```

---

## WebSocket `/ws/gestures`

Real-time bidirectional stream for the gesture-system client.

### Connect

```javascript
const ws = new WebSocket("wss://jarvis-api.onrender.com/ws/gestures");
```

### Send — gesture event

```json
{
  "type":       "gesture",
  "gesture":    "SWIPE_LEFT",
  "command":    "navigate_left",
  "confidence": 0.93,
  "velocity":   {"vx": -1.2, "vy": 0.0},
  "hand_count": 1,
  "timestamp":  1712345678.123,
  "session_id": null,
  "client_id":  "gesture-client-a3f8"
}
```

### Receive — acknowledgement

```json
{"type": "ack", "gesture": "SWIPE_LEFT", "ts": 1712345678.250}
```

### Receive — processed response

```json
{
  "type":       "gesture_response",
  "gesture":    "SWIPE_LEFT",
  "response":   "Navigating to the next item.",
  "intent":     "gesture",
  "session_id": "3fa85f64-...",
  "latency_ms": 420.5
}
```

### Receive — server ping (every 30s)

```json
{"type": "ping", "ts": 1712345700.0}
```

Send back `{"type": "pong"}` to keep the connection alive.

---

## WebSocket `/ws/ui`

Notification stream for the Three.js holographic interface.

### Connect

```javascript
const ws = new WebSocket("wss://jarvis-api.onrender.com/ws/ui");

ws.onmessage = (e) => {
  const msg = JSON.parse(e.data);

  if (msg.type === "connected") {
    console.log("JARVIS connected:", msg.message);
  }
  if (msg.type === "jarvis_update") {
    updateHologram(msg.text, msg.intent, msg.agent_path);
  }
  if (msg.type === "ping") {
    ws.send(JSON.stringify({type: "pong"}));
  }
};
```

### Receive — connection confirmation

```json
{
  "type":    "connected",
  "message": "JARVIS 2.0.0 — holographic interface connected",
  "ts":      1712345678.0
}
```

### Receive — JARVIS update (broadcast after every interaction)

```json
{
  "type":       "jarvis_update",
  "text":       "Search complete. Here are the top results...",
  "intent":     "search",
  "agent_path": ["knowledge"],
  "session_id": "3fa85f64-...",
  "ts":         1712345678.5
}
```

### Send — request status

```json
{"type": "status"}
```

### Receive — status response

Same shape as `GET /status`.

---

## Error Responses

All errors follow this envelope:

```json
{
  "detail": "Human-readable error message",
  "error":  "machine_readable_code"
}
```

| HTTP Code | Meaning |
|---|---|
| 400 | Bad request |
| 422 | Validation error — check `detail` for field errors |
| 503 | Service unavailable — model loading or agent init failed |
| 500 | Internal server error — check `/status` for diagnostics |

---

## Frontend Integration

### JavaScript fetch example

```javascript
const JARVIS_API = "https://jarvis-api.onrender.com";
let sessionId = localStorage.getItem("jarvis_session_id");

async function askJarvis(message) {
  const resp = await fetch(`${JARVIS_API}/chat`, {
    method:  "POST",
    headers: {"Content-Type": "application/json"},
    body:    JSON.stringify({message, session_id: sessionId}),
  });
  const data = await resp.json();
  // Persist session across page loads
  sessionId = data.session_id;
  localStorage.setItem("jarvis_session_id", sessionId);
  return data.text;
}

// WebSocket for real-time hologram updates
const uiWs = new WebSocket(`${JARVIS_API.replace("https", "wss")}/ws/ui`);
uiWs.onmessage = (e) => {
  const msg = JSON.parse(e.data);
  if (msg.type === "jarvis_update") {
    document.getElementById("response").textContent = msg.text;
    activateHologram(msg.intent);
  }
  if (msg.type === "ping") uiWs.send('{"type":"pong"}');
};
```
