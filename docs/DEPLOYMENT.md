# JARVIS 2.0 — Deployment Guide

## Overview

| Layer | Platform | URL |
|---|---|---|
| Backend API | Render.com | `https://jarvis-api.onrender.com` |
| Frontend | GitHub Pages | `https://joaomanoel123.github.io/jarvis2.0` |
| Gesture client | Local machine | connects to backend WS |
| Voice client | Local machine | connects to backend REST |

---

## 1. Render Deployment (Backend)

### Prerequisites

- GitHub account with the `jarvis2.0` repo
- Render.com account (free tier works for TinyLlama)

### Step-by-step

**1. Connect repo**

Go to [render.com](https://render.com) → **New** → **Web Service** → connect your GitHub repo.

**2. Configure service**

Render will detect `render.yaml` automatically. Verify these settings:

| Setting | Value |
|---|---|
| Name | `jarvis-v2-backend` |
| Environment | Python |
| Build Command | `pip install -r requirements.txt` |
| Start Command | `uvicorn main:app --host 0.0.0.0 --port 10000` |
| Plan | Starter (free) for TinyLlama |
| Health Check Path | `/health` |

**3. Add environment secrets**

In Render dashboard → **Environment** tab → add:

```
HF_TOKEN = hf_your_token_here
```

> Get your token at https://huggingface.co/settings/tokens  
> Required for: Mistral-7B, LLaMA-3 (not needed for TinyLlama/Phi-2)

**4. Set CORS origin**

Update `ALLOWED_ORIGINS` to your GitHub Pages URL:

```
ALLOWED_ORIGINS = https://joaomanoel123.github.io,http://localhost:3000
```

**5. Deploy**

Click **Manual Deploy** → **Deploy latest commit**.

First deploy downloads the model (~600 MB for TinyLlama). Watch the logs:

```
═══ JARVIS 2.0.0 starting ═══
Loading model: TinyLlama/TinyLlama-1.1B-Chat-v1.0 …
✓ LLM loaded (34.2 s)
✓ Vector memory loaded
✓ JarvisCore ready
✓ Tools registered: ['web_search', 'run_python_code', 'file_reader', 'system_status', 'open_url']
═══ JARVIS ready (36.1 s) ═══
```

**6. Verify**

```bash
curl https://jarvis-api.onrender.com/health
# → {"status":"ok","model_loaded":true,...}

curl -X POST https://jarvis-api.onrender.com/chat \
  -H "Content-Type: application/json" \
  -d '{"message":"hello jarvis"}'
```

---

## 2. Model Selection Guide

| Model | RAM | Speed | Quality | Plan |
|---|---|---|---|---|
| TinyLlama 1.1B | ~1.2 GB | ⚡⚡⚡ | ⭐⭐ | Starter (free) |
| Phi-2 2.7B | ~3 GB | ⚡⚡ | ⭐⭐⭐ | Standard |
| Phi-3-mini 3.8B | ~4 GB | ⚡⚡ | ⭐⭐⭐⭐ | Standard |
| Mistral-7B (4-bit) | ~5 GB GPU | ⚡⚡ | ⭐⭐⭐⭐⭐ | GPU plan |
| LLaMA-3 8B (4-bit) | ~6 GB GPU | ⚡ | ⭐⭐⭐⭐⭐ | GPU plan |

Change model by setting `LLM_MODEL_ID` env var in Render dashboard:

```
LLM_MODEL_ID = phi-2
```

Aliases: `tinyllama`, `phi-2`, `phi-3-mini`, `mistral-7b`, `llama-3`, `deepseek-coder`

---

## 3. Persistent Disk

The `render.yaml` configures a 10 GB persistent disk at `/app/data`.

This stores:
- HuggingFace model cache (`~/.cache/huggingface` → symlinked to `/app/data`)
- Vector memory FAISS index (`/app/data/vector_memory/`)

Without the persistent disk, the model re-downloads on every deploy (~600 MB).

To set up the cache symlink, add to your `main.py` startup:

```python
import os
os.makedirs("/app/data/hf_cache", exist_ok=True)
os.environ["HF_HOME"] = "/app/data/hf_cache"
```

---

## 4. GitHub Pages Deployment (Frontend)

```bash
# In your frontend directory
git checkout -b gh-pages
git add .
git commit -m "deploy frontend"
git push origin gh-pages
```

Go to GitHub repo → **Settings** → **Pages** → Source: `gh-pages` branch.

Your frontend will be at `https://joaomanoel123.github.io/jarvis2.0`.

Update the `BACKEND_URL` in your frontend JS:

```javascript
// In frontend/hologram.js or index.html
const BACKEND_URL = "https://jarvis-api.onrender.com";
```

---

## 5. Running Gesture System (Local)

The gesture system runs on your local machine and connects to the cloud backend.

```bash
cd gesture-system
pip install -r requirements.txt

# Run and point to your Render backend
python main.py --backend wss://jarvis-api.onrender.com/ws/gestures

# First run downloads the MediaPipe hand landmarker model (~4 MB)
# Cached at ~/.cache/jarvis/hand_landmarker.task
```

**Keyboard shortcuts while running:**
- `Q` — quit
- `R` — reset gesture state
- `S` — print stats to terminal
- `H` — toggle HUD overlay

---

## 6. Running Voice System (Local)

```bash
cd voice-system
pip install sounddevice SpeechRecognition pyttsx3 requests

# Development (Google STT, no wake word, no model download)
python voice_main.py \
  --stt-backend google \
  --always-on \
  --api https://jarvis-api.onrender.com

# Production (Whisper local, wake word "Jarvis")
python voice_main.py \
  --stt-backend whisper_local \
  --whisper-model base \
  --api https://jarvis-api.onrender.com
```

List your microphones:

```bash
python voice_main.py --list-mics
# [0] Built-in Microphone  (2 ch)
# [1] USB Audio Device     (1 ch)
```

Use a specific mic: `--mic-device 1`

---

## 7. Troubleshooting

### Backend won't start

**Problem:** `Model load FAILED`  
**Fix:** Check `HF_TOKEN` env var, verify model name in `LLM_MODEL_ID`

**Problem:** `503 JARVIS core not initialised`  
**Fix:** Model is still loading (can take 30–120 s on first start). Wait and retry.

**Problem:** `uvicorn: command not found`  
**Fix:** Ensure `uvicorn[standard]` is in `requirements.txt`

---

### CORS errors in frontend

**Problem:** `Access-Control-Allow-Origin` error in browser console  
**Fix:** Add your GitHub Pages domain to `ALLOWED_ORIGINS` env var in Render:
```
ALLOWED_ORIGINS = https://joaomanoel123.github.io,http://localhost:3000
```

---

### Gesture system can't connect

**Problem:** `WebSocket connection failed`  
**Fix 1:** Check backend URL — should use `wss://` not `ws://` in production  
**Fix 2:** Verify backend is running: `curl https://jarvis-api.onrender.com/health`

---

### Voice system not hearing commands

**Problem:** No transcription output  
**Fix 1:** Check mic: `python voice_main.py --list-mics`  
**Fix 2:** Increase VAD sensitivity: `--vad-threshold 0.01`  
**Fix 3:** Try always-on mode: `--always-on`

**Problem:** Wake word not detected  
**Fix:** Say "Jarvis" clearly at the start. Try `--always-on` to bypass wake word.

---

### Out of memory (OOM) on Render

**Problem:** Container killed during model loading  
**Fix:** Switch to a smaller model or upgrade the Render plan:

| Model | Minimum RAM |
|---|---|
| TinyLlama 1.1B | 1 GB |
| Phi-2 2.7B | 3 GB |
| Mistral-7B (4-bit) | 6 GB GPU |

Set `LLM_MODEL_ID=tinyllama` for the free starter plan.

---

## 8. Environment Variables Quick Reference

```bash
# Copy to .env for local development
LLM_MODEL_ID=tinyllama
HF_TOKEN=
LLM_DEVICE=cpu
LLM_LOAD_IN_4BIT=false
LLM_MAX_NEW_TOKENS=1024
LLM_TEMPERATURE=0.7
ALLOWED_TOOLS=web_search,run_python_code,file_reader,system_status,open_url
TOOL_TIMEOUT_SECONDS=30
SESSION_TTL_SECONDS=7200
VECTOR_MEMORY_DIR=./data/vector_memory
ALLOWED_ORIGINS=http://localhost:3000
DEBUG=false
```
