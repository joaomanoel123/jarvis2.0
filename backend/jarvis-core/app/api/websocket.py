"""
app/api/websocket.py
Full-duplex WebSocket handler.

Connections:
  /ws/{session_id}   — main real-time channel (text, voice, gesture events)
  /ws/ui/{session_id} — UI broadcast only (read-only frontend channel)

Message format (inbound):
  { "type": "message", "text": "…", "source": "text|voice|gesture" }
  { "type": "gesture", "gesture_id": "swipe_right" }
  { "type": "pong" }

Message format (outbound):
  { "type": "state_change",   "mode": "thinking", "ts": … }
  { "type": "thinking",       "intent": "…" }
  { "type": "response_ready", "intent": "…", "response": "…", "action": {} }
  { "type": "ping",           "ts": … }
  { "type": "error",          "message": "…" }
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import TYPE_CHECKING

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.core.state_manager import state_manager
from app.utils.logger       import get_logger

if TYPE_CHECKING:
    from app.core.jarvis_core import JarvisCore

log = get_logger("jarvis.ws")

ws_router = APIRouter()


# ── Main real-time channel ─────────────────────────────────────────────

@ws_router.websocket("/ws/{session_id}")
async def ws_main(ws: WebSocket, session_id: str):
    """
    Main real-time channel.
    Receives user input, dispatches to JarvisCore, streams state events back.
    """
    from app.core.jarvis_core import jarvis

    await ws.accept()
    log.info("WS connected: session=%s", session_id[:8])

    # Prime the session
    await state_manager.get_or_create(session_id)
    await ws.send_json({"type": "connected", "session_id": session_id, "ts": time.time()})

    # Concurrently: receive messages + drain event queue
    recv_task  = asyncio.create_task(_receiver(ws, session_id, jarvis))
    drain_task = asyncio.create_task(_event_drainer(ws, session_id))

    try:
        done, pending = await asyncio.wait(
            [recv_task, drain_task],
            return_when=asyncio.FIRST_COMPLETED,
        )
        for task in pending:
            task.cancel()
    except WebSocketDisconnect:
        pass
    except Exception as exc:
        log.debug("WS session %s error: %s", session_id[:8], exc)
    finally:
        recv_task.cancel()
        drain_task.cancel()
        log.info("WS disconnected: session=%s", session_id[:8])


async def _receiver(ws: WebSocket, session_id: str, jarvis) -> None:
    """Read messages from the client and dispatch to JarvisCore."""
    while True:
        try:
            raw = await ws.receive_text()
        except WebSocketDisconnect:
            break

        try:
            msg = json.loads(raw)
        except json.JSONDecodeError:
            await ws.send_json({"type": "error", "message": "Invalid JSON"})
            continue

        msg_type = msg.get("type", "message")

        # Keepalive pong
        if msg_type == "pong":
            continue

        # User input (text, voice transcript, or gesture)
        if msg_type in ("message", "voice", "gesture"):
            text = (
                msg.get("text") or
                msg.get("gesture_id") or
                msg.get("gesture") or ""
            ).strip()

            if not text:
                continue

            source = msg.get("source", msg_type)
            meta   = {k: v for k, v in msg.items() if k not in ("type", "text", "source")}

            if jarvis is None:
                await ws.send_json({"type": "error", "message": "JARVIS core not initialised"})
                continue

            try:
                resp = await jarvis.process(text=text, session_id=session_id, source=source, metadata=meta)
                await ws.send_json({"type": "response", **resp.to_dict()})
            except Exception as exc:
                log.exception("WS dispatch error: %s", exc)
                await ws.send_json({"type": "error", "message": str(exc)})

        elif msg_type == "status":
            if jarvis:
                status = await jarvis.status()
                await ws.send_json({"type": "status_response", **status})


async def _event_drainer(ws: WebSocket, session_id: str) -> None:
    """Drain the session event queue and push events to the client."""
    while True:
        event = await state_manager.get_event(session_id, timeout=25.0)
        if event is None:
            # Timeout — send ping
            try:
                await ws.send_json({"type": "ping", "ts": time.time()})
            except Exception:
                break
        else:
            try:
                await ws.send_json(event)
            except Exception:
                break


# ── Read-only UI broadcast channel ────────────────────────────────────

@ws_router.websocket("/ws/ui/{session_id}")
async def ws_ui(ws: WebSocket, session_id: str):
    """
    Read-only WebSocket for the holographic UI.
    Streams state_change and response_ready events without accepting input.
    """
    await ws.accept()
    await state_manager.get_or_create(session_id)
    await ws.send_json({
        "type":      "connected",
        "session_id": session_id,
        "message":   "JARVIS 2.0 — UI stream connected",
        "ts":        time.time(),
    })

    try:
        while True:
            event = await state_manager.get_event(session_id, timeout=25.0)
            if event is None:
                await ws.send_json({"type": "ping", "ts": time.time()})
            else:
                try:
                    await ws.send_json(event)
                except Exception:
                    break
            # Drain pong from client
            try:
                raw = await asyncio.wait_for(ws.receive_text(), timeout=0.01)
            except (asyncio.TimeoutError, Exception):
                pass
    except WebSocketDisconnect:
        pass
    finally:
        log.info("UI WS disconnected: session=%s", session_id[:8])


# ── Legacy /ws/ui endpoint (no session) ──────────────────────────────

@ws_router.websocket("/ws/ui")
async def ws_ui_no_session(ws: WebSocket):
    """Backwards-compatible /ws/ui endpoint — auto-generates session."""
    import uuid
    sid = str(uuid.uuid4())
    await ws_ui(ws, sid)
