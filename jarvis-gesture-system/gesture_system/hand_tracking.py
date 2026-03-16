"""
gesture_system/hand_tracking.py
================================
HandTracker — wraps the MediaPipe Tasks HandLandmarker (0.10+) and exposes
clean per-hand landmark data as HandResult objects.

MediaPipe 0.10+ migration note
───────────────────────────────
MediaPipe 0.10 dropped mp.solutions in favour of the Tasks API.
HandLandmarker requires a .task model file downloaded at first use.
The model is cached to ~/.cache/jarvis/hand_landmarker.task.

MediaPipe landmark indices (0–20)
──────────────────────────────────
  0  WRIST
  1  THUMB_CMC   2  THUMB_MCP   3  THUMB_IP    4  THUMB_TIP
  5  INDEX_MCP   6  INDEX_PIP   7  INDEX_DIP   8  INDEX_TIP
  9  MIDDLE_MCP  10 MIDDLE_PIP  11 MIDDLE_DIP  12 MIDDLE_TIP
 13  RING_MCP   14  RING_PIP   15  RING_DIP   16  RING_TIP
 17  PINKY_MCP  18  PINKY_PIP  19  PINKY_DIP  20  PINKY_TIP
"""

from __future__ import annotations

import logging
import os
import urllib.request
from pathlib import Path
from typing import NamedTuple

import cv2
import numpy as np

log = logging.getLogger("jarvis.gesture.hand_tracking")

# ── Model download ────────────────────────────────────────────────────────────
_MODEL_URL  = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
_CACHE_DIR  = Path(os.path.expanduser("~/.cache/jarvis"))
_MODEL_PATH = _CACHE_DIR / "hand_landmarker.task"

# Colour scheme
_ARC  = (255, 229, 0)   # BGR arc-reactor blue
_GOLD = (0, 214, 255)
_GREY = (160, 160, 160)


def _ensure_model() -> Path:
    """Download the MediaPipe hand landmarker model if not already cached."""
    if _MODEL_PATH.exists():
        return _MODEL_PATH
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    log.info("Downloading hand landmarker model → %s …", _MODEL_PATH)
    try:
        urllib.request.urlretrieve(_MODEL_URL, _MODEL_PATH)
        log.info("Model downloaded (%d KB)", _MODEL_PATH.stat().st_size // 1024)
    except Exception as exc:
        log.error("Model download failed: %s", exc)
        raise RuntimeError(
            f"Could not download MediaPipe hand landmarker model from {_MODEL_URL}.\n"
            f"Download it manually and place it at: {_MODEL_PATH}"
        ) from exc
    return _MODEL_PATH


# ── Data structures ───────────────────────────────────────────────────────────

class LM(NamedTuple):
    """Single hand landmark with (x, y, z) coordinates."""
    x: float
    y: float
    z: float


class HandResult:
    """
    All detected data for a single hand in one frame.

    Attributes:
        hand_index:   Positional index (0 = first detected, 1 = second).
        handedness:   "Left" or "Right" as classified by MediaPipe.
        landmarks:    21 normalised [0–1] landmarks.
        landmarks_px: 21 pixel-coordinate landmarks (int x, int y).
        bbox:         (x, y, w, h) bounding box in pixels.
        confidence:   Hand detection confidence [0–1].
    """

    # Landmark index constants
    WRIST       = 0
    THUMB_TIP   = 4
    INDEX_MCP   = 5
    INDEX_PIP   = 6
    INDEX_TIP   = 8
    MIDDLE_MCP  = 9
    MIDDLE_PIP  = 10
    MIDDLE_TIP  = 12
    RING_MCP    = 13
    RING_PIP    = 14
    RING_TIP    = 16
    PINKY_MCP   = 17
    PINKY_PIP   = 18
    PINKY_TIP   = 20
    THUMB_IP    = 3
    THUMB_MCP   = 2

    def __init__(
        self,
        hand_index:   int,
        handedness:   str,
        landmarks:    list[LM],
        landmarks_px: list[LM],
        bbox:         tuple[int, int, int, int],
        confidence:   float,
    ) -> None:
        self.hand_index   = hand_index
        self.handedness   = handedness
        self.landmarks    = landmarks
        self.landmarks_px = landmarks_px
        self.bbox         = bbox
        self.confidence   = confidence

    def __repr__(self) -> str:
        return f"<HandResult {self.handedness} conf={self.confidence:.2f}>"


# ── HandTracker ───────────────────────────────────────────────────────────────

class HandTracker:
    """
    Wraps MediaPipe Tasks HandLandmarker and returns HandResult objects.

    Args:
        max_hands:     Maximum number of hands to detect per frame (1 or 2).
        min_detect_cf: Minimum hand detection confidence.
        min_track_cf:  Minimum tracking confidence.
        draw:          Draw landmarks and bounding boxes on the frame.
        model_path:    Path to .task model file. Auto-downloads if None.
    """

    def __init__(
        self,
        max_hands:     int   = 2,
        min_detect_cf: float = 0.7,
        min_track_cf:  float = 0.6,
        draw:          bool  = True,
        model_path:    str | Path | None = None,
    ) -> None:
        self._max_hands     = max_hands
        self._min_detect_cf = min_detect_cf
        self._min_track_cf  = min_track_cf
        self._draw          = draw
        self._model_path    = Path(model_path) if model_path else None
        self._landmarker    = None

    def start(self) -> "HandTracker":
        """Initialise the HandLandmarker (downloads model if necessary)."""
        from mediapipe.tasks.python import vision, BaseOptions
        from mediapipe.tasks.python.vision import (
            HandLandmarker,
            HandLandmarkerOptions,
            RunningMode,
        )

        mp_path = self._model_path or _ensure_model()

        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(mp_path)),
            running_mode=RunningMode.IMAGE,
            num_hands=self._max_hands,
            min_hand_detection_confidence=self._min_detect_cf,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=self._min_track_cf,
        )
        self._landmarker = HandLandmarker.create_from_options(options)
        log.info("HandTracker started  max_hands=%d  model=%s", self._max_hands, mp_path.name)
        return self

    def stop(self) -> None:
        """Close the landmarker and release resources."""
        if self._landmarker:
            self._landmarker.close()
            self._landmarker = None
        log.info("HandTracker stopped")

    # ── Frame processing ──────────────────────────────────────────────────────

    def process(self, frame: np.ndarray) -> tuple[np.ndarray, list[HandResult]]:
        """
        Detect hands in a BGR frame and return annotated frame + results.

        Args:
            frame: OpenCV BGR frame (numpy ndarray H×W×3 uint8).

        Returns:
            (annotated_frame, hand_results)
        """
        assert self._landmarker is not None, "HandTracker not started"

        import mediapipe as mp

        h, w = frame.shape[:2]
        rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        mp_result = self._landmarker.detect(mp_image)
        annotated = frame.copy()

        if not mp_result.hand_landmarks:
            return annotated, []

        results: list[HandResult] = []

        for i, (lm_list, handedness_list) in enumerate(
            zip(mp_result.hand_landmarks, mp_result.handedness)
        ):
            # Normalised landmarks
            lms: list[LM] = []
            lms_px: list[LM] = []
            xs_px, ys_px = [], []

            for lm in lm_list:
                lms.append(LM(lm.x, lm.y, lm.z))
                px = int(lm.x * w)
                py = int(lm.y * h)
                lms_px.append(LM(px, py, lm.z))
                xs_px.append(px)
                ys_px.append(py)

            # Bounding box
            x0 = max(min(xs_px) - 12, 0)
            y0 = max(min(ys_px) - 12, 0)
            x1 = min(max(xs_px) + 12, w)
            y1 = min(max(ys_px) + 12, h)
            bbox = (x0, y0, x1 - x0, y1 - y0)

            # Handedness + confidence
            label = handedness_list[0].category_name   # "Left" or "Right"
            conf  = round(handedness_list[0].score, 3)

            hand = HandResult(
                hand_index=i,
                handedness=label,
                landmarks=lms,
                landmarks_px=lms_px,
                bbox=bbox,
                confidence=conf,
            )
            results.append(hand)

            if self._draw:
                self._draw_hand(annotated, hand, label, conf, w, h)

        return annotated, results

    # ── Drawing ───────────────────────────────────────────────────────────────

    def _draw_hand(
        self,
        frame: np.ndarray,
        hand:  "HandResult",
        label: str,
        conf:  float,
        w:     int,
        h:     int,
    ) -> None:
        """Draw landmarks, connections, and bounding box onto the frame."""
        from mediapipe.tasks.python.vision import HandLandmarksConnections

        lms_px = hand.landmarks_px

        # Draw connections
        for conn in HandLandmarksConnections.HAND_CONNECTIONS:
            a = lms_px[conn.start]
            b = lms_px[conn.end]
            cv2.line(frame, (int(a.x), int(a.y)), (int(b.x), int(b.y)), _GREY, 1, cv2.LINE_AA)

        # Draw landmark dots
        for j, lm in enumerate(lms_px):
            color = _GOLD if j in (4, 8, 12, 16, 20) else _ARC  # tips in gold
            cv2.circle(frame, (int(lm.x), int(lm.y)), 4, color, -1)

        # Bounding box
        x, y, bw, bh = hand.bbox
        cv2.rectangle(frame, (x, y), (x + bw, y + bh), _ARC, 1)

        # Label
        txt = f"{label}  {conf:.0%}"
        cv2.putText(frame, txt, (x, y - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, _ARC, 1, cv2.LINE_AA)

    # ── Context manager ───────────────────────────────────────────────────────

    def __enter__(self) -> "HandTracker":
        return self.start()

    def __exit__(self, *_) -> None:
        self.stop()

    def __repr__(self) -> str:
        state = "active" if self._landmarker else "stopped"
        return f"<HandTracker {state} max_hands={self._max_hands}>"
