"""
gesture_system/websocket_client.py
====================================
WebSocketClient — async reconnecting client that streams gesture events
to the JARVIS FastAPI backend.

Transport
─────────
  ws://localhost:8000/ws/gestures   — default

Message format (outbound)
─────────────────────────
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

Messages from server (inbound)
───────────────────────────────
    {"type": "ack",   "gesture": "SWIPE_LEFT"}
    {"type": "error", "message": "..."}
    {"type": "pong"}

Reconnection
────────────
Exponential back-off starting at 1 s, capped at 30 s.
Queued events are held in memory and drained after reconnection.

Thread safety
─────────────
The client runs an asyncio event loop in a background daemon thread.
Call send_gesture(event) from any thread — events are queued safely.
"""

from __future__ import annotations

import asyncio
import json
import logging
import queue
import threading
import time
import uuid
from typing import Callable

from gesture_system.gesture_interpreter import GestureEvent

log = logging.getLogger("jarvis.gesture.websocket")

try:
    import websockets
    from websockets.exceptions import ConnectionClosed, WebSocketException
    HAS_WEBSOCKETS = True
except ImportError:
    HAS_WEBSOCKETS = False
    log.warning("'websockets' not installed — WebSocket client disabled")


class WebSocketClient:
    """
    Asynchronous reconnecting WebSocket client.

    Args:
        url:           WebSocket endpoint (ws:// or wss://).
        client_id:     Unique ID sent with every message. Auto-generated if None.
        reconnect_delay: Initial reconnect delay in seconds.
        max_delay:     Maximum reconnect delay.
        queue_size:    Maximum pending events in the send queue.
        on_message:    Optional callback for inbound messages from the server.
    """

    def __init__(
        self,
        url:              str = "ws://localhost:8000/ws/gestures",
        client_id:        str | None = None,
        reconnect_delay:  float = 1.0,
        max_delay:        float = 30.0,
        queue_size:       int   = 64,
        on_message:       Callable[[dict], None] | None = None,
    ) -> None:
        if not HAS_WEBSOCKETS:
            raise ImportError("Install websockets: pip install websockets")

        self._url           = url
        self._client_id     = client_id or f"gesture-client-{uuid.uuid4().hex[:8]}"
        self._base_delay    = reconnect_delay
        self._max_delay     = max_delay
        self._on_message    = on_message
        self._send_queue: asyncio.Queue = asyncio.Queue(maxsize=queue_size)

        # Thread-safe queue for cross-thread calls to send_gesture()
        self._thread_queue: queue.Queue = queue.Queue(maxsize=queue_size)

        self._connected = False
        self._running   = False
        self._loop:   asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None

        # Stats
        self._sent_count     = 0
        self._dropped_count  = 0
        self._reconnect_count = 0

    # ── Lifecycle ──────────────────────────────────────────────────────────────

    def start(self) -> "WebSocketClient":
        """Start the background WebSocket thread."""
        if self._running:
            return self
        self._running = True
        self._thread  = threading.Thread(
            target=self._run_event_loop,
            name="websocket-client",
            daemon=True,
        )
        self._thread.start()
        log.info("WebSocketClient started → %s  id=%s", self._url, self._client_id)
        return self

    def stop(self) -> None:
        """Stop the client and close the connection."""
        self._running = False
        if self._loop and not self._loop.is_closed():
            self._loop.call_soon_threadsafe(self._loop.stop)
        if self._thread:
            self._thread.join(timeout=3.0)
        log.info("WebSocketClient stopped  sent=%d  dropped=%d",
                 self._sent_count, self._dropped_count)

    # ── Public API ─────────────────────────────────────────────────────────────

    def send_gesture(self, event: GestureEvent) -> bool:
        """
        Queue a gesture event for sending (thread-safe).

        Args:
            event: GestureEvent to send.

        Returns:
            True if queued, False if the queue was full (event dropped).
        """
        payload = event.to_dict()
        payload["client_id"] = self._client_id

        if not self._connected:
            log.debug("Not connected — buffering gesture event")

        try:
            self._thread_queue.put_nowait(payload)
            return True
        except queue.Full:
            self._dropped_count += 1
            log.warning("Send queue full — gesture event dropped (%d total)", self._dropped_count)
            return False

    def send_raw(self, payload: dict) -> bool:
        """Send an arbitrary JSON payload (thread-safe)."""
        payload.setdefault("client_id", self._client_id)
        try:
            self._thread_queue.put_nowait(payload)
            return True
        except queue.Full:
            self._dropped_count += 1
            return False

    @property
    def connected(self) -> bool:
        return self._connected

    def stats(self) -> dict:
        return {
            "connected":        self._connected,
            "url":              self._url,
            "client_id":        self._client_id,
            "sent":             self._sent_count,
            "dropped":          self._dropped_count,
            "reconnect_count":  self._reconnect_count,
            "queue_size":       self._thread_queue.qsize(),
        }

    # ── Background event loop ──────────────────────────────────────────────────

    def _run_event_loop(self) -> None:
        """Entry point for the background thread."""
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._connect_loop())
        finally:
            self._loop.close()

    async def _connect_loop(self) -> None:
        """Reconnection loop with exponential back-off."""
        delay = self._base_delay
        while self._running:
            try:
                log.info("Connecting to %s …", self._url)
                async with websockets.connect(
                    self._url,
                    ping_interval=20,
                    ping_timeout=10,
                    close_timeout=5,
                ) as ws:
                    self._connected = True
                    self._reconnect_count += 1
                    delay = self._base_delay  # reset on success
                    log.info("Connected to %s", self._url)

                    await asyncio.gather(
                        self._sender(ws),
                        self._receiver(ws),
                    )

            except (ConnectionClosed, WebSocketException, OSError) as exc:
                self._connected = False
                log.warning("WebSocket error: %s — retrying in %.1f s", exc, delay)
            except asyncio.CancelledError:
                break
            except Exception as exc:
                self._connected = False
                log.exception("Unexpected WebSocket error: %s", exc)

            if not self._running:
                break
            await asyncio.sleep(delay)
            delay = min(delay * 2, self._max_delay)

        self._connected = False

    async def _sender(self, ws) -> None:
        """Drain the thread queue and send messages to the server."""
        while self._running:
            # Drain cross-thread queue into asyncio queue
            while not self._thread_queue.empty():
                try:
                    item = self._thread_queue.get_nowait()
                    await self._send_queue.put(item)
                except queue.Empty:
                    break

            try:
                payload = await asyncio.wait_for(self._send_queue.get(), timeout=0.05)
                msg = json.dumps(payload)
                await ws.send(msg)
                self._sent_count += 1
                log.debug("→ %s", msg[:120])
            except asyncio.TimeoutError:
                continue
            except Exception as exc:
                log.debug("Send failed: %s", exc)
                raise

    async def _receiver(self, ws) -> None:
        """Listen for messages from the server."""
        async for raw in ws:
            try:
                msg = json.loads(raw)
                log.debug("← %s", raw[:120])
                if self._on_message:
                    self._on_message(msg)
            except json.JSONDecodeError:
                log.debug("Non-JSON message from server: %s", raw[:60])

    # ── Context manager ────────────────────────────────────────────────────────

    def __enter__(self) -> "WebSocketClient":
        return self.start()

    def __exit__(self, *_) -> None:
        self.stop()
