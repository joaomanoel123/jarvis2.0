"""
gesture_system/gesture_interpreter.py
========================================
GestureInterpreter — temporal state machine that converts raw per-frame
gesture classifications into stable, de-duplicated gesture events.

Problems solved
───────────────
1. Noise suppression — a gesture must be detected consistently for N frames
   before it is emitted as a real event.
2. Hold detection — brief pauses mid-gesture don't reset the state.
3. Duplicate suppression — the same gesture isn't re-emitted until the hand
   returns to a neutral state or a different gesture is detected.
4. Swipe/velocity inference — tracked wrist position over a short window is
   used to confirm directional swipes (SWIPE_LEFT/RIGHT/UP/DOWN).
5. Circle detection — angular trajectory around wrist centroid is analysed
   to confirm CIRCLE_CW / CIRCLE_CCW.

State machine
─────────────
  IDLE → CANDIDATE (first detection)
       → CANDIDATE (same gesture, count < threshold)
       → CONFIRMED (count >= threshold)  — emit event
       → IDLE (different gesture or timeout)

Output
──────
    GestureEvent(gesture, command, confidence, velocity, hand_count, timestamp)

Velocity
────────
  velocity = (vx, vy) in normalised-per-second units.
  Positive vx = rightward, positive vy = downward.
"""

from __future__ import annotations

import collections
import logging
import math
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Deque

import numpy as np

from gesture_system.gesture_classifier import (
    ClassificationResult,
    OPEN_HAND, FIST, POINT, PINCH, DOUBLE_PINCH, GRAB, RELEASE,
    SWIPE_LEFT, SWIPE_RIGHT, SWIPE_UP, SWIPE_DOWN,
    CIRCLE_CLOCKWISE, CIRCLE_COUNTERCLOCKWISE,
    ZOOM_IN, ZOOM_OUT, TWO_HAND_EXPAND, TWO_HAND_CONTRACT,
    UNKNOWN,
)
from gesture_system.hand_tracking import HandResult

log = logging.getLogger("jarvis.gesture.interpreter")

# ── Command mapping ────────────────────────────────────────────────────────────

GESTURE_COMMAND_MAP: dict[str, str] = {
    OPEN_HAND:               "open_menu",
    FIST:                    "close_menu",
    POINT:                   "select_element",
    PINCH:                   "select_object",
    DOUBLE_PINCH:            "confirm_action",
    GRAB:                    "grab_element",
    RELEASE:                 "release_element",
    SWIPE_LEFT:              "navigate_left",
    SWIPE_RIGHT:             "navigate_right",
    SWIPE_UP:                "scroll_up",
    SWIPE_DOWN:              "scroll_down",
    CIRCLE_CLOCKWISE:        "rotate_right",
    CIRCLE_COUNTERCLOCKWISE: "rotate_left",
    ZOOM_IN:                 "zoom_interface",
    ZOOM_OUT:                "shrink_interface",
    TWO_HAND_EXPAND:         "expand_view",
    TWO_HAND_CONTRACT:       "collapse_view",
    UNKNOWN:                 "no_op",
}

# Velocity thresholds for swipe confirmation (normalised units/second)
SWIPE_MIN_VX = 0.8   # minimum horizontal velocity to confirm a swipe
SWIPE_MIN_VY = 0.7   # minimum vertical velocity to confirm a swipe

# Minimum arc length (radians) to confirm a circle gesture
CIRCLE_MIN_ARC = 3.5   # ~200 degrees


# ── Output event ───────────────────────────────────────────────────────────────

@dataclass
class GestureEvent:
    """A confirmed, stable gesture event ready to send to the backend."""
    gesture:    str
    command:    str
    confidence: float
    velocity:   tuple[float, float]    = (0.0, 0.0)    # (vx, vy)
    hand_count: int                    = 1
    timestamp:  float                  = field(default_factory=time.time)
    metadata:   dict                   = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "type":       "gesture",
            "gesture":    self.gesture,
            "command":    self.command,
            "confidence": self.confidence,
            "velocity":   {"vx": round(self.velocity[0], 3),
                           "vy": round(self.velocity[1], 3)},
            "hand_count": self.hand_count,
            "timestamp":  self.timestamp,
        }


class _State(str, Enum):
    IDLE      = "idle"
    CANDIDATE = "candidate"
    CONFIRMED = "confirmed"


# ── Wrist history for velocity + circle detection ──────────────────────────────

class _WristHistory:
    """
    Rolling buffer of (x, y, timestamp) wrist positions.
    Used to compute velocity and detect circular motion.
    """

    def __init__(self, max_age: float = 0.8, max_size: int = 60) -> None:
        self._buf: Deque[tuple[float, float, float]] = collections.deque(maxlen=max_size)
        self._max_age = max_age

    def push(self, x: float, y: float) -> None:
        self._buf.append((x, y, time.monotonic()))

    def velocity(self) -> tuple[float, float]:
        """Estimate (vx, vy) in normalised-units/second over the recent window."""
        self._prune()
        pts = list(self._buf)
        if len(pts) < 5:
            return (0.0, 0.0)
        oldest, newest = pts[0], pts[-1]
        dt = newest[2] - oldest[2] + 1e-6
        return ((newest[0] - oldest[0]) / dt, (newest[1] - oldest[1]) / dt)

    def circle_arc(self) -> tuple[float, str]:
        """
        Compute signed total arc traversed in the recent window.
        Returns (total_radians, "clockwise" | "counterclockwise").
        """
        self._prune()
        pts = list(self._buf)
        if len(pts) < 10:
            return (0.0, "unknown")

        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        cx = sum(xs) / len(xs)
        cy = sum(ys) / len(ys)

        angles = [math.atan2(y - cy, x - cx) for x, y in zip(xs, ys)]
        total  = 0.0
        for i in range(1, len(angles)):
            da = angles[i] - angles[i-1]
            # Wrap to [-π, π]
            da = (da + math.pi) % (2 * math.pi) - math.pi
            total += da

        direction = "clockwise" if total < 0 else "counterclockwise"
        return (abs(total), direction)

    def clear(self) -> None:
        self._buf.clear()

    def _prune(self) -> None:
        cutoff = time.monotonic() - self._max_age
        while self._buf and self._buf[0][2] < cutoff:
            self._buf.popleft()


# ── GestureInterpreter ─────────────────────────────────────────────────────────

class GestureInterpreter:
    """
    Temporal state machine for gesture stabilisation.

    Args:
        confirm_frames:      Consecutive frames needed to confirm a gesture.
        hold_timeout:        Seconds of no detection before resetting state.
        cooldown:            Seconds before the same gesture can fire again.
        swipe_window:        Seconds to measure velocity for swipe gestures.
        enable_swipe_check:  Require velocity threshold for swipe confirmation.
        enable_circle_check: Require arc threshold for circle confirmation.
    """

    def __init__(
        self,
        confirm_frames:      int   = 5,
        hold_timeout:        float = 0.4,
        cooldown:            float = 0.8,
        swipe_window:        float = 0.6,
        enable_swipe_check:  bool  = True,
        enable_circle_check: bool  = True,
    ) -> None:
        self._confirm_n      = confirm_frames
        self._hold_timeout   = hold_timeout
        self._cooldown       = cooldown
        self._swipe_window   = swipe_window
        self._check_swipe    = enable_swipe_check
        self._check_circle   = enable_circle_check

        self._state:      _State = _State.IDLE
        self._candidate:  str    = UNKNOWN
        self._count:      int    = 0
        self._last_ts:    float  = 0.0
        self._last_event: str    = UNKNOWN
        self._last_emit:  float  = 0.0

        self._wrist = _WristHistory(max_age=swipe_window)

    def update(
        self,
        result: ClassificationResult,
        hands:  list[HandResult],
    ) -> GestureEvent | None:
        """
        Feed one frame's classification result into the state machine.

        Args:
            result: Output of GestureClassifier.classify().
            hands:  Raw HandResult list from this frame (for velocity tracking).

        Returns:
            A GestureEvent if a gesture was confirmed, else None.
        """
        now = time.monotonic()

        # Track primary wrist position
        if hands:
            w = hands[0].landmarks[0]
            self._wrist.push(w.x, w.y)

        gesture = result.gesture

        # ── State machine ──────────────────────────────────────────────────────

        if gesture == UNKNOWN or result.confidence < 0.45:
            # No detection — check hold timeout
            if self._state != _State.IDLE and (now - self._last_ts) > self._hold_timeout:
                self._reset()
            return None

        self._last_ts = now

        if self._state == _State.IDLE:
            self._state     = _State.CANDIDATE
            self._candidate = gesture
            self._count     = 1
            return None

        if gesture == self._candidate:
            self._count += 1
        else:
            # Different gesture resets the candidate
            self._candidate = gesture
            self._count     = 1
            self._state     = _State.CANDIDATE
            return None

        # Awaiting confirmation threshold
        if self._state == _State.CANDIDATE:
            if self._count >= self._confirm_n:
                self._state = _State.CONFIRMED
                event = self._build_event(result, hands, now)
                return event
            return None

        # Already confirmed — suppress repeats until cooldown or gesture change
        if self._state == _State.CONFIRMED:
            if (now - self._last_emit) > self._cooldown:
                # Allow re-emission after cooldown (e.g. held swipe)
                if gesture in (SWIPE_LEFT, SWIPE_RIGHT, SWIPE_UP, SWIPE_DOWN):
                    event = self._build_event(result, hands, now)
                    return event
            return None

        return None

    def _build_event(
        self,
        result: ClassificationResult,
        hands:  list[HandResult],
        now:    float,
    ) -> GestureEvent | None:
        """Validate the gesture with velocity/circle checks and build the event."""
        gesture = result.gesture
        vx, vy  = self._wrist.velocity()

        # ── Swipe validation ──────────────────────────────────────────────────
        if self._check_swipe and gesture in (SWIPE_LEFT, SWIPE_RIGHT, SWIPE_UP, SWIPE_DOWN):
            if gesture in (SWIPE_LEFT, SWIPE_RIGHT):
                if abs(vx) < SWIPE_MIN_VX:
                    log.debug("Swipe rejected — vx=%.2f < threshold", vx)
                    self._reset()
                    return None
                # Correct direction
                gesture = SWIPE_LEFT if vx < 0 else SWIPE_RIGHT
            else:
                if abs(vy) < SWIPE_MIN_VY:
                    log.debug("Swipe rejected — vy=%.2f < threshold", vy)
                    self._reset()
                    return None
                gesture = SWIPE_UP if vy < 0 else SWIPE_DOWN

        # ── Circle validation ─────────────────────────────────────────────────
        if self._check_circle and gesture in (CIRCLE_CLOCKWISE, CIRCLE_COUNTERCLOCKWISE):
            arc, direction = self._wrist.circle_arc()
            if arc < CIRCLE_MIN_ARC:
                log.debug("Circle rejected — arc=%.2f rad < %.2f threshold", arc, CIRCLE_MIN_ARC)
                self._reset()
                return None
            gesture = CIRCLE_CLOCKWISE if direction == "clockwise" else CIRCLE_COUNTERCLOCKWISE

        command = GESTURE_COMMAND_MAP.get(gesture, "no_op")

        # Cooldown gate for this specific gesture
        if gesture == self._last_event and (now - self._last_emit) < self._cooldown:
            return None

        self._last_event = gesture
        self._last_emit  = time.time()

        log.info("✓ GestureEvent: %s → %s  conf=%.2f  vx=%.2f vy=%.2f",
                 gesture, command, result.confidence, vx, vy)

        return GestureEvent(
            gesture=gesture,
            command=command,
            confidence=result.confidence,
            velocity=(round(vx, 3), round(vy, 3)),
            hand_count=result.hand_count,
        )

    def _reset(self) -> None:
        self._state     = _State.IDLE
        self._candidate = UNKNOWN
        self._count     = 0
        self._wrist.clear()

    @property
    def state(self) -> str:
        return self._state.value

    @property
    def candidate(self) -> str:
        return self._candidate

    @property
    def frame_count(self) -> int:
        return self._count
