"""
gesture_system/gesture_classifier.py
======================================
GestureClassifier — dual-mode gesture recognition.

Mode 1 — Rule-based (always available, no training required)
─────────────────────────────────────────────────────────────
Deterministic geometric rules derived directly from the 42-dimensional
feature vector:
  • Finger extension / curl thresholds
  • Pinch distance
  • Palm angle
  • Hand aspect ratio

Mode 2 — ML model (optional, loaded from disk)
───────────────────────────────────────────────
A scikit-learn RandomForestClassifier (or any sklearn-compatible model)
trained with GestureTrainer and saved as models/gesture_model.pkl.
When the model is loaded it takes priority over rule-based classification.

The rule engine serves as the fallback when:
  • No model file exists (first run / development)
  • Model confidence falls below ML_CONFIDENCE_THRESHOLD

Supported gestures (17)
───────────────────────
  OPEN_HAND, FIST, POINT, PINCH, DOUBLE_PINCH, GRAB, RELEASE,
  SWIPE_LEFT, SWIPE_RIGHT, SWIPE_UP, SWIPE_DOWN,
  CIRCLE_CLOCKWISE, CIRCLE_COUNTERCLOCKWISE,
  ZOOM_IN, ZOOM_OUT, TWO_HAND_EXPAND, TWO_HAND_CONTRACT

  UNKNOWN — returned when nothing matches

Output
──────
    ClassificationResult(gesture, confidence, mode, raw_scores)
"""

from __future__ import annotations

import logging
import math
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from gesture_system.hand_tracking import HandResult, LM
from gesture_system.feature_extraction import FeatureExtractor, FEATURE_DIM

log = logging.getLogger("jarvis.gesture.classifier")

# ── Gesture label constants ────────────────────────────────────────────────────

OPEN_HAND              = "OPEN_HAND"
FIST                   = "FIST"
POINT                  = "POINT"
PINCH                  = "PINCH"
DOUBLE_PINCH           = "DOUBLE_PINCH"
GRAB                   = "GRAB"
RELEASE                = "RELEASE"
SWIPE_LEFT             = "SWIPE_LEFT"
SWIPE_RIGHT            = "SWIPE_RIGHT"
SWIPE_UP               = "SWIPE_UP"
SWIPE_DOWN             = "SWIPE_DOWN"
CIRCLE_CLOCKWISE       = "CIRCLE_CLOCKWISE"
CIRCLE_COUNTERCLOCKWISE = "CIRCLE_COUNTERCLOCKWISE"
ZOOM_IN                = "ZOOM_IN"
ZOOM_OUT               = "ZOOM_OUT"
TWO_HAND_EXPAND        = "TWO_HAND_EXPAND"
TWO_HAND_CONTRACT      = "TWO_HAND_CONTRACT"
UNKNOWN                = "UNKNOWN"

ALL_GESTURES = [
    OPEN_HAND, FIST, POINT, PINCH, DOUBLE_PINCH, GRAB, RELEASE,
    SWIPE_LEFT, SWIPE_RIGHT, SWIPE_UP, SWIPE_DOWN,
    CIRCLE_CLOCKWISE, CIRCLE_COUNTERCLOCKWISE,
    ZOOM_IN, ZOOM_OUT, TWO_HAND_EXPAND, TWO_HAND_CONTRACT, UNKNOWN,
]

# Minimum confidence to accept ML classification (falls back to rules below this)
ML_CONFIDENCE_THRESHOLD = 0.55


# ── Result type ────────────────────────────────────────────────────────────────

@dataclass
class ClassificationResult:
    gesture:    str
    confidence: float
    mode:       str           # "rule" | "ml" | "two_hand"
    raw_scores: dict[str, float] = field(default_factory=dict)
    hand_count: int = 1


# ── Rule helpers ───────────────────────────────────────────────────────────────

def _finger_extended(lms: list[LM], tip: int, pip: int, mcp: int) -> bool:
    """True when the finger tip is significantly above the MCP joint (y axis)."""
    return (lms[mcp].y - lms[tip].y) > 0.04


def _finger_curled(lms: list[LM], tip: int, pip: int) -> bool:
    """True when the tip is below or very close to the PIP joint."""
    return lms[tip].y >= lms[pip].y - 0.02


def _pinch_distance(lms: list[LM]) -> float:
    t, i = lms[4], lms[8]
    return math.sqrt((t.x - i.x)**2 + (t.y - i.y)**2)


def _all_extended(lms: list[LM]) -> bool:
    fingers = [(8,7,5), (12,11,9), (16,15,13), (20,19,17)]
    return all(_finger_extended(lms, t, p, m) for t, p, m in fingers)


def _all_curled(lms: list[LM]) -> bool:
    return all(
        _finger_curled(lms, t, p)
        for t, p in [(8,7), (12,11), (16,15), (20,19)]
    )


def _count_extended(lms: list[LM]) -> int:
    fingers = [(8,7,5), (12,11,9), (16,15,13), (20,19,17)]
    return sum(_finger_extended(lms, t, p, m) for t, p, m in fingers)


# ── Rule-based classifier ──────────────────────────────────────────────────────

def _classify_single_rule(hand: HandResult) -> ClassificationResult:
    """
    Apply geometric rules to classify a single hand gesture.
    Returns ClassificationResult with mode="rule".
    """
    lms = hand.landmarks

    thumb_extended = (lms[2].y - lms[4].y) > 0.04
    pinch_dist     = _pinch_distance(lms)
    n_extended     = _count_extended(lms)

    # PINCH — thumb and index very close, others mostly curled
    if pinch_dist < 0.06 and n_extended <= 1:
        conf = max(0.0, 1.0 - pinch_dist / 0.06)
        return ClassificationResult(PINCH, round(conf, 2), "rule")

    # OPEN HAND — all 4 fingers extended
    if _all_extended(lms) and thumb_extended:
        return ClassificationResult(OPEN_HAND, 0.90, "rule")

    # FIST — all fingers curled
    if _all_curled(lms) and not thumb_extended:
        return ClassificationResult(FIST, 0.90, "rule")

    # GRAB — all curled, thumb may be extended (grasping pose)
    if _all_curled(lms) and thumb_extended:
        return ClassificationResult(GRAB, 0.80, "rule")

    # POINT — only index extended
    index_ext  = _finger_extended(lms, 8, 7, 5)
    middle_ext = _finger_extended(lms, 12, 11, 9)
    ring_ext   = _finger_extended(lms, 16, 15, 13)
    pinky_ext  = _finger_extended(lms, 20, 19, 17)

    if index_ext and not middle_ext and not ring_ext and not pinky_ext:
        return ClassificationResult(POINT, 0.88, "rule")

    # RELEASE — fingers spread open, thumb extended, relaxed pose
    if n_extended >= 3:
        return ClassificationResult(RELEASE, 0.75, "rule")

    # DOUBLE_PINCH — index AND middle close to thumb
    mid_pinch = math.sqrt((lms[4].x-lms[12].x)**2 + (lms[4].y-lms[12].y)**2)
    if pinch_dist < 0.10 and mid_pinch < 0.12:
        return ClassificationResult(DOUBLE_PINCH, 0.82, "rule")

    # Fallback
    return ClassificationResult(UNKNOWN, 0.40, "rule")


def _classify_two_hands_rule(
    hands: list[HandResult],
) -> ClassificationResult | None:
    """
    Classify two-hand gestures: TWO_HAND_EXPAND, TWO_HAND_CONTRACT, ZOOM_IN, ZOOM_OUT.
    Returns None if no two-hand gesture is detected.
    """
    if len(hands) < 2:
        return None

    h1, h2 = hands[0], hands[1]
    lms1, lms2 = h1.landmarks, h2.landmarks

    # Distance between the two wrists
    wrist_dist = math.sqrt(
        (lms1[0].x - lms2[0].x)**2 + (lms1[0].y - lms2[0].y)**2
    )

    # Distance between index fingertips
    tip_dist = math.sqrt(
        (lms1[8].x - lms2[8].x)**2 + (lms1[8].y - lms2[8].y)**2
    )

    # Both hands open with tips far apart → EXPAND
    if _all_extended(lms1) and _all_extended(lms2):
        if wrist_dist > 0.5:
            return ClassificationResult(TWO_HAND_EXPAND, 0.85, "two_hand", hand_count=2)
        elif wrist_dist < 0.25:
            return ClassificationResult(TWO_HAND_CONTRACT, 0.85, "two_hand", hand_count=2)

    # Both hands pinching → ZOOM
    p1 = _pinch_distance(lms1)
    p2 = _pinch_distance(lms2)
    if p1 < 0.10 and p2 < 0.10:
        if tip_dist > 0.40:
            return ClassificationResult(ZOOM_IN, 0.85, "two_hand", hand_count=2)
        else:
            return ClassificationResult(ZOOM_OUT, 0.85, "two_hand", hand_count=2)

    return None


# ── Main classifier ────────────────────────────────────────────────────────────

class GestureClassifier:
    """
    Two-mode gesture classifier.

    Falls back gracefully from ML → rule-based when no model is loaded
    or ML confidence is too low.

    Args:
        model_path: Path to a pickled sklearn model. None = rule-based only.
        confidence_threshold: Minimum ML confidence to accept ML result.
    """

    def __init__(
        self,
        model_path: str | Path | None = None,
        confidence_threshold: float = ML_CONFIDENCE_THRESHOLD,
    ) -> None:
        self._model_path = Path(model_path) if model_path else None
        self._threshold  = confidence_threshold
        self._model: Any | None = None
        self._classes: list[str] = []
        self._fe = FeatureExtractor()

    def load_model(self, path: str | Path | None = None) -> bool:
        """
        Load an ML model from disk.

        Returns:
            True if loaded successfully, False otherwise.
        """
        p = Path(path) if path else self._model_path
        if p is None or not p.exists():
            log.info("No ML model file found — using rule-based classification")
            return False

        try:
            with open(p, "rb") as f:
                payload = pickle.load(f)

            if isinstance(payload, dict):
                self._model   = payload["model"]
                self._classes = payload.get("classes", [])
            else:
                self._model = payload
                self._classes = list(getattr(payload, "classes_", []))

            log.info("ML model loaded from %s  classes=%s", p, self._classes)
            return True
        except Exception as exc:
            log.warning("Failed to load ML model: %s — falling back to rules", exc)
            self._model = None
            return False

    def classify(self, hands: list[HandResult]) -> ClassificationResult:
        """
        Classify the gesture from one or two detected hands.

        Args:
            hands: List of HandResult objects (1 or 2 elements).

        Returns:
            ClassificationResult with the best matching gesture.
        """
        if not hands:
            return ClassificationResult(UNKNOWN, 0.0, "rule")

        # Two-hand gestures take priority
        if len(hands) >= 2:
            two_hand = _classify_two_hands_rule(hands)
            if two_hand is not None:
                return two_hand

        # Primary hand = first detected
        primary = hands[0]

        # ML classification
        if self._model is not None:
            ml_result = self._classify_ml(primary)
            if ml_result.confidence >= self._threshold:
                return ml_result
            log.debug("ML confidence %.2f below threshold — using rules", ml_result.confidence)

        # Rule-based
        return _classify_single_rule(primary)

    def _classify_ml(self, hand: HandResult) -> ClassificationResult:
        """Run the ML model on a single hand."""
        vec = self._fe.extract(hand).reshape(1, -1)
        proba = self._model.predict_proba(vec)[0]
        best_idx  = int(np.argmax(proba))
        best_conf = float(proba[best_idx])
        label     = self._classes[best_idx] if self._classes else str(best_idx)
        scores    = {cls: float(p) for cls, p in zip(self._classes, proba)}
        return ClassificationResult(label, round(best_conf, 3), "ml", scores)

    @property
    def model_loaded(self) -> bool:
        return self._model is not None

    def __repr__(self) -> str:
        mode = f"ML+rules ({len(self._classes)} classes)" if self._model else "rules-only"
        return f"<GestureClassifier mode={mode}>"
