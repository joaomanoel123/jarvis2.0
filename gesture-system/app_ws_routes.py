"""
app/ws_routes.py
================
FastAPI WebSocket endpoint for real-time gesture streaming.

Add to jarvis-v2/app/main.py:
    from app.ws_routes import ws_router
    application.include_router(ws_router)

WebSocket URL:  ws://localhost:8000/ws/gestures

Protocol (inbound from gesture client)
───────────────────────────────────────
    {
        "type":       "gesture",
        "gesture":    "SWIPE_LEFT",
        "command":    "navigate_left",
        "confidence": 0.93,
        "velocity":   {"vx": -1.2, "vy": 0.0},
        "hand_count": 1,
        "timestamp":  1712345678.123,
        "client_id":  "gesture-client-a3f8"
    }

Protocol (outbound to gesture client)
──────────────────────────────────────
    {"type": "ack",   "gesture": "SWIPE_LEFT", "session_id": "..."}
    {"type": "error", "message": "..."}
    {"type": "pong",  "timestamp": ...}

Broadcast (outbound to UI clients)
────────────────────────────────────
    POST /gesture is called internally for each received gesture so that
    JarvisBrain processes it through the full agent pipeline. The result is
    then broadcast to all connected UI WebSocket clients.

Multiple clients
────────────────
• Gesture capture clients connect to /ws/gestures (write-heavy).
• UI clients (Three.js) connect to /ws/ui (read-heavy).
• Both share the same in-process ConnectionManager.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from typing import Callable

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from fastapi.websockets import WebSocketState

log = logging.getLogger("jarvis.ws")

ws_router = APIRouter()


# ── Connection manager ─────────────────────────────────────────────────────────

class ConnectionManager:
    """
    Manages all active WebSocket connections.

    Supports two channels:
      • "gesture"  — gesture capture clients
      • "ui"       — holographic UI clients (Three.js, web)
    """

    def __init__(self) -> None:
        self._gesture_clients: dict[str, WebSocket] = {}
        self._ui_clients:      dict[str, WebSocket] = {}
        self._lock = asyncio.Lock()

    async def connect_gesture(self, ws: WebSocket) -> str:
        await ws.accept()
        cid = f"gc-{uuid.uuid4().hex[:8]}"
        async with self._lock:
            self._gesture_clients[cid] = ws
        log.info("Gesture client connected: %s  (total: %d)", cid, len(self._gesture_clients))
        return cid

    async def connect_ui(self, ws: WebSocket) -> str:
        await ws.accept()
        cid = f"ui-{uuid.uuid4().hex[:8]}"
        async with self._lock:
            self._ui_clients[cid] = ws
        log.info("UI client connected: %s  (total: %d)", cid, len(self._ui_clients))
        return cid

    async def disconnect(self, cid: str) -> None:
        async with self._lock:
            self._gesture_clients.pop(cid, None)
            self._ui_clients.pop(cid, None)
        log.info("Client disconnected: %s", cid)

    async def send_to(self, cid: str, payload: dict) -> None:
        """Send to a specific client."""
        ws = self._gesture_clients.get(cid) or self._ui_clients.get(cid)
        if ws and ws.client_state == WebSocketState.CONNECTED:
            try:
                await ws.send_json(payload)
            except Exception as exc:
                log.debug("Send failed for %s: %s", cid, exc)

    async def broadcast_ui(self, payload: dict) -> None:
        """Broadcast a message to all UI clients."""
        dead = []
        for cid, ws in list(self._ui_clients.items()):
            if ws.client_state == WebSocketState.CONNECTED:
                try:
                    await ws.send_json(payload)
                except Exception:
                    dead.append(cid)
            else:
                dead.append(cid)
        async with self._lock:
            for cid in dead:
                self._ui_clients.pop(cid, None)

    @property
    def gesture_client_count(self) -> int:
        return len(self._gesture_clients)

    @property
    def ui_client_count(self) -> int:
        return len(self._ui_clients)


# Module-level singleton
manager = ConnectionManager()


# ── Gesture WebSocket endpoint ─────────────────────────────────────────────────

@ws_router.websocket("/ws/gestures")
async def gesture_websocket(ws: WebSocket):
    """
    WebSocket endpoint for the gesture recognition client.

    Each gesture event is:
      1. Acknowledged back to the sender.
      2. Forwarded to JarvisBrain via brain.process_gesture().
      3. Brain response broadcast to all UI clients.
    """
    cid = await manager.connect_gesture(ws)
    try:
        while True:
            try:
                raw = await asyncio.wait_for(ws.receive_text(), timeout=30.0)
            except asyncio.TimeoutError:
                # Send ping to keep connection alive
                await manager.send_to(cid, {"type": "ping", "timestamp": time.time()})
                continue

            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                await manager.send_to(cid, {"type": "error", "message": "Invalid JSON"})
                continue

            msg_type = msg.get("type", "gesture")

            if msg_type == "pong":
                continue

            if msg_type == "gesture":
                gesture_id = msg.get("gesture", "")
                confidence = float(msg.get("confidence", 1.0))
                landmarks  = msg.get("landmarks", [])
                session_id = msg.get("session_id")

                log.info("WS ← %s | gesture=%s conf=%.2f", cid, gesture_id, confidence)

                # Acknowledge immediately
                await manager.send_to(cid, {
                    "type":      "ack",
                    "gesture":   gesture_id,
                    "timestamp": time.time(),
                })

                # Process through JARVIS brain
                try:
                    from brain.jarvis_brain import brain
                    response = await brain.process_gesture(
                        gesture_id=gesture_id,
                        session_id=session_id,
                        confidence=confidence,
                        landmarks=landmarks,
                        context=msg.get("metadata", {}),
                    )

                    ui_payload = {
                        "type":       "gesture_response",
                        "gesture":    gesture_id,
                        "command":    msg.get("command", ""),
                        "response":   response.text,
                        "intent":     response.intent,
                        "session_id": response.session_id,
                        "latency_ms": response.latency_ms,
                        "timestamp":  time.time(),
                    }
                    await manager.broadcast_ui(ui_payload)

                except Exception as exc:
                    log.exception("Brain processing failed: %s", exc)
                    await manager.send_to(cid, {"type": "error", "message": str(exc)})

            elif msg_type == "hello":
                await manager.send_to(cid, {
                    "type":      "welcome",
                    "client_id": cid,
                    "timestamp": time.time(),
                })

    except WebSocketDisconnect:
        log.info("Gesture client disconnected: %s", cid)
    except Exception as exc:
        log.exception("Gesture WS error for %s: %s", cid, exc)
    finally:
        await manager.disconnect(cid)


# ── UI WebSocket endpoint ──────────────────────────────────────────────────────

@ws_router.websocket("/ws/ui")
async def ui_websocket(ws: WebSocket):
    """
    WebSocket endpoint for holographic UI clients (Three.js interface).

    Clients receive:
      • gesture_response  — JARVIS brain response to a gesture
      • ping              — keepalive
    """
    cid = await manager.connect_ui(ws)
    try:
        # Send initial connection confirmation
        await manager.send_to(cid, {
            "type":      "connected",
            "client_id": cid,
            "timestamp": time.time(),
            "message":   "JARVIS 2.0 Holographic Interface — connected",
        })

        while True:
            try:
                raw = await asyncio.wait_for(ws.receive_text(), timeout=30.0)
                msg = json.loads(raw)
                if msg.get("type") == "pong":
                    continue
                # UI clients can also query status
                if msg.get("type") == "status":
                    from brain.jarvis_brain import brain
                    status = await brain.status()
                    await manager.send_to(cid, {"type": "status_response", **status})
            except asyncio.TimeoutError:
                await manager.send_to(cid, {"type": "ping", "timestamp": time.time()})
            except json.JSONDecodeError:
                pass

    except WebSocketDisconnect:
        log.info("UI client disconnected: %s", cid)
    except Exception as exc:
        log.exception("UI WS error for %s: %s", cid, exc)
    finally:
        await manager.disconnect(cid)


# ── Status endpoint ────────────────────────────────────────────────────────────

@ws_router.get("/ws/status")
async def ws_status():
    return {
        "gesture_clients": manager.gesture_client_count,
        "ui_clients":      manager.ui_client_count,
    }
