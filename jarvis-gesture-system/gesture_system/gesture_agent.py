"""
gesture_system/gesture_agent.py
=================================
GestureAgent — the top-level orchestrator for the gesture pipeline.

                        ┌───────────────────────────────┐
                        │          GestureAgent          │
                        │                               │
                        │  frame in                     │
                        │     ↓                         │
                        │  HandTracker.process()        │
                        │     ↓                         │
                        │  GestureClassifier.classify() │
                        │     ↓                         │
                        │  GestureInterpreter.update()  │
                        │     ↓                         │
                        │  format JSON payload          │
                        │     ↓                         │
                        │  emit via callback / WS       │
                        └───────────────────────────────┘

Thread safety
─────────────
process_frame() is synchronous and NOT thread-safe by itself.
Call it from a single thread (the main loop in main.py).
The callback may be called from the same thread synchronously, or handed off
to an asyncio queue for async delivery.

Callbacks
─────────
Register one or more callbacks with on_gesture(cb):
    def my_cb(event: GestureEvent) -> None: ...

Callbacks are invoked synchronously in the process_frame() call.

For async callbacks, wrap with an asyncio.Queue and drain in a coroutine.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Callable

import numpy as np

from gesture_system.gesture_classifier import (
    GestureClassifier,
    ClassificationResult,
    UNKNOWN,
)
from gesture_system.gesture_interpreter import GestureEvent, GestureInterpreter
from gesture_system.hand_tracking import HandResult, HandTracker

log = logging.getLogger("jarvis.gesture.agent")

GestureCallback = Callable[[GestureEvent], None]


class GestureAgent:
    """
    Top-level pipeline: frame → landmarks → classification → event.

    Args:
        tracker:      HandTracker instance (must be started before use).
        classifier:   GestureClassifier (rule-based or ML).
        interpreter:  GestureInterpreter state machine.
        async_queue:  Optional asyncio.Queue to push events into.
    """

    def __init__(
        self,
        tracker:     HandTracker,
        classifier:  GestureClassifier,
        interpreter: GestureInterpreter,
        async_queue: asyncio.Queue | None = None,
    ) -> None:
        self._tracker     = tracker
        self._classifier  = classifier
        self._interpreter = interpreter
        self._queue       = async_queue
        self._callbacks:  list[GestureCallback] = []

        # Metrics
        self._frames_processed = 0
        self._events_emitted   = 0
        self._last_result: ClassificationResult | None = None
        self._last_event:  GestureEvent | None         = None
        self._frame_ts     = 0.0
        self._fps          = 0.0
        self._fps_count    = 0
        self._fps_ts       = time.perf_counter()

    # ── Callback registration ──────────────────────────────────────────────────

    def on_gesture(self, callback: GestureCallback) -> None:
        """Register a callback invoked when a gesture event is confirmed."""
        self._callbacks.append(callback)

    # ── Frame processing ───────────────────────────────────────────────────────

    def process_frame(self, frame: np.ndarray) -> tuple[np.ndarray, GestureEvent | None]:
        """
        Run the full gesture pipeline on a single BGR frame.

        Args:
            frame: OpenCV BGR numpy array from the camera.

        Returns:
            (annotated_frame, event_or_None)
            annotated_frame has landmarks, bounding boxes, and gesture labels drawn.
            event_or_None is a GestureEvent if a gesture was confirmed this frame.
        """
        t0 = time.perf_counter()

        # 1 — Hand detection + landmark extraction
        annotated, hands = self._tracker.process(frame)

        # 2 — Gesture classification
        if hands:
            result = self._classifier.classify(hands)
        else:
            result = ClassificationResult(UNKNOWN, 0.0, "rule")
        self._last_result = result

        # 3 — Temporal interpretation / stabilisation
        event = self._interpreter.update(result, hands)
        if event:
            self._last_event   = event
            self._events_emitted += 1
            self._fire_callbacks(event)
            if self._queue is not None:
                try:
                    self._queue.put_nowait(event)
                except asyncio.QueueFull:
                    log.warning("Event queue full — dropping gesture event")

        # 4 — Draw HUD on the annotated frame
        annotated = self._draw_hud(annotated, hands, result, event)

        # FPS tracking
        self._frames_processed += 1
        self._fps_count += 1
        now = time.perf_counter()
        if now - self._fps_ts >= 1.0:
            self._fps     = self._fps_count / (now - self._fps_ts)
            self._fps_count = 0
            self._fps_ts  = now

        return annotated, event

    # ── HUD rendering ─────────────────────────────────────────────────────────

    def _draw_hud(
        self,
        frame: np.ndarray,
        hands: list[HandResult],
        result: ClassificationResult,
        event:  GestureEvent | None,
    ) -> np.ndarray:
        """Draw gesture label, confidence, FPS, and state overlay."""
        import cv2

        h, w = frame.shape[:2]
        arc_blue  = (255, 229, 0)    # BGR arc reactor blue
        gold      = (0, 214, 255)
        green     = (118, 230, 0)
        red       = (71, 23, 255)
        grey      = (160, 160, 160)

        # ── Gesture label ──────────────────────────────────────────────────────
        label = result.gesture if result.gesture != UNKNOWN else ""
        conf  = result.confidence

        if label:
            color = green if event else arc_blue
            cv2.putText(frame, label,
                        (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 2, cv2.LINE_AA)
            cv2.putText(frame, f"{conf:.0%}  [{result.mode}]",
                        (20, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.45, grey, 1, cv2.LINE_AA)

        # ── Confirmed event banner ─────────────────────────────────────────────
        if event:
            banner = f"CMD: {event.command}"
            tw, _ = cv2.getTextSize(banner, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[:2]
            bx = w // 2 - tw // 2
            cv2.rectangle(frame, (bx - 10, 88), (bx + tw + 10, 118), (0, 0, 0), -1)
            cv2.putText(frame, banner,
                        (bx, 112), cv2.FONT_HERSHEY_SIMPLEX, 0.7, gold, 2, cv2.LINE_AA)

        # ── State indicator ────────────────────────────────────────────────────
        state_color = {"idle": grey, "candidate": arc_blue, "confirmed": green}.get(
            self._interpreter.state, grey
        )
        state_txt = f"STATE: {self._interpreter.state.upper()}"
        if self._interpreter.state == "candidate":
            state_txt += f"  [{self._interpreter.frame_count}/{5}]"
        cv2.putText(frame, state_txt,
                    (20, h - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.45, state_color, 1, cv2.LINE_AA)

        # ── FPS ────────────────────────────────────────────────────────────────
        cv2.putText(frame, f"FPS: {self._fps:.1f}",
                    (20, h - 38), cv2.FONT_HERSHEY_SIMPLEX, 0.45, grey, 1, cv2.LINE_AA)

        # ── Hand count ─────────────────────────────────────────────────────────
        hand_txt = f"HANDS: {len(hands)}"
        cv2.putText(frame, hand_txt,
                    (20, h - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, grey, 1, cv2.LINE_AA)

        # ── Velocity arrows (if last event was swipe) ──────────────────────────
        if self._last_event:
            vx, vy = self._last_event.velocity
            cx, cy = w // 2, h // 2
            ex = cx + int(vx * 30)
            ey = cy + int(vy * 30)
            if abs(vx) > 0.1 or abs(vy) > 0.1:
                cv2.arrowedLine(frame, (cx, cy), (ex, ey), gold, 2, tipLength=0.3)

        # ── Corner brackets ───────────────────────────────────────────────────
        margin, length, thick = 12, 20, 2
        pts = [
            [(margin, margin), (margin + length, margin), (margin, margin + length)],
            [(w - margin, margin), (w - margin - length, margin), (w - margin, margin + length)],
            [(margin, h - margin), (margin + length, h - margin), (margin, h - margin - length)],
            [(w - margin, h - margin), (w - margin - length, h - margin), (w - margin, h - margin - length)],
        ]
        for bracket in pts:
            origin = bracket[0]
            for pt in bracket[1:]:
                cv2.line(frame, origin, pt, arc_blue, thick)

        return frame

    # ── Callbacks ──────────────────────────────────────────────────────────────

    def _fire_callbacks(self, event: GestureEvent) -> None:
        for cb in self._callbacks:
            try:
                cb(event)
            except Exception as exc:
                log.exception("Gesture callback raised: %s", exc)

    # ── Diagnostics ────────────────────────────────────────────────────────────

    @property
    def fps(self) -> float:
        return round(self._fps, 1)

    @property
    def frames_processed(self) -> int:
        return self._frames_processed

    @property
    def events_emitted(self) -> int:
        return self._events_emitted

    @property
    def last_result(self) -> ClassificationResult | None:
        return self._last_result

    @property
    def last_event(self) -> GestureEvent | None:
        return self._last_event

    def stats(self) -> dict:
        return {
            "fps":              self.fps,
            "frames_processed": self._frames_processed,
            "events_emitted":   self._events_emitted,
            "state":            self._interpreter.state,
            "candidate":        self._interpreter.candidate,
            "last_gesture":     self._last_event.gesture if self._last_event else None,
            "ml_model_loaded":  self._classifier.model_loaded,
        }
