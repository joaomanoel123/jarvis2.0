"""
gesture_system/feature_extraction.py
======================================
FeatureExtractor — converts raw MediaPipe hand landmarks into a compact,
rotation-invariant feature vector suitable for ML classification.

Feature vector (42 dimensions)
────────────────────────────────
20 normalised inter-joint distances  (each pair relative to hand span)
 5 finger extension scores           (continuous, 0–1)
 5 finger curl scores                (continuous, 0–1)
 4 fingertip–wrist distances         (normalised)
 4 fingertip–palm angles             (radians / π)
 2 thumb–index pinch distances       (normalised + binary threshold)
 1 hand aspect ratio                 (bbox w/h)
 1 palm orientation angle            (radians / π, –1 to 1)
─────────────────────────────────────────
Total: 42 features per hand

Design rationale
────────────────
• All distances are normalised by the wrist-to-middle-MCP "hand span" so the
  vector is scale-invariant — works regardless of how close the hand is to cam.
• Angles are normalised to [–1, 1] via division by π.
• No raw (x, y, z) coordinates are included — the classifier does not need to
  know where on screen the hand is, only its shape.

Usage
─────
    fe = FeatureExtractor()
    vector = fe.extract(hand_result)   # np.ndarray shape (42,)
    matrix = fe.extract_batch([h1, h2])  # shape (2, 42)
"""

from __future__ import annotations

import math
import logging
from typing import Sequence

import numpy as np

from gesture_system.hand_tracking import HandResult, LM

log = logging.getLogger("jarvis.gesture.features")

# Finger definitions: (TIP_IDX, PIP_IDX, MCP_IDX, DIP_IDX)
_FINGER_DEFS = {
    "thumb":  (4,  3,  2,  1),
    "index":  (8,  7,  6,  5),
    "middle": (12, 11, 10,  9),
    "ring":   (16, 15, 14, 13),
    "pinky":  (20, 19, 18, 17),
}

# Key pairs for inter-joint distance features
_DISTANCE_PAIRS = [
    (4, 8),   # thumb–index
    (4, 12),  # thumb–middle
    (4, 16),  # thumb–ring
    (4, 20),  # thumb–pinky
    (8, 12),  # index–middle
    (8, 16),  # index–ring
    (8, 20),  # index–pinky
    (12, 16), # middle–ring
    (12, 20), # middle–pinky
    (16, 20), # ring–pinky
    (0,  8),  # wrist–index tip
    (0, 12),  # wrist–middle tip
    (0, 16),  # wrist–ring tip
    (0, 20),  # wrist–pinky tip
    (0,  4),  # wrist–thumb tip
    (5,  8),  # index mcp–tip
    (9, 12),  # middle mcp–tip
    (13, 16), # ring mcp–tip
    (17, 20), # pinky mcp–tip
    (5,  9),  # index mcp–middle mcp (palm spread)
]

FEATURE_DIM = 42


def _dist(a: LM, b: LM) -> float:
    return math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2 + (a.z - b.z) ** 2)


def _dist2d(a: LM, b: LM) -> float:
    return math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2)


def _angle2d(a: LM, b: LM) -> float:
    """Angle of vector a→b in radians."""
    return math.atan2(b.y - a.y, b.x - a.x)


class FeatureExtractor:
    """
    Converts a HandResult into a fixed-length, normalised feature vector.

    The vector is scale-invariant, orientation-partially-invariant, and
    ready to feed into scikit-learn or a PyTorch linear classifier.
    """

    def __init__(self, normalise: bool = True) -> None:
        self._normalise = normalise

    def extract(self, hand: HandResult) -> np.ndarray:
        """
        Extract features from a single HandResult.

        Returns:
            np.ndarray of shape (FEATURE_DIM,) — float32.
        """
        lms = hand.landmarks
        wrist = lms[0]
        mid_mcp = lms[9]

        # ── Normalisation factor: wrist → middle MCP distance ─────────────────
        hand_span = _dist(wrist, mid_mcp) + 1e-7

        features: list[float] = []

        # ── Block 1: Inter-joint distances (20) ───────────────────────────────
        for a_idx, b_idx in _DISTANCE_PAIRS:
            d = _dist(lms[a_idx], lms[b_idx]) / hand_span
            features.append(d)

        # ── Block 2: Finger extension scores (5) ─────────────────────────────
        # Extension = (tip_y - mcp_y) / hand_span; negative = extended (up)
        for finger, (tip, pip, mcp, _) in _FINGER_DEFS.items():
            ext = -(lms[tip].y - lms[mcp].y) / hand_span   # positive = extended
            features.append(float(np.clip(ext, -1.5, 1.5)))

        # ── Block 3: Finger curl scores (5) ──────────────────────────────────
        # Curl = angle at PIP joint (tip–pip–mcp); small angle = curled
        for finger, (tip, pip, mcp, dip) in _FINGER_DEFS.items():
            v1 = LM(lms[tip].x - lms[pip].x,  lms[tip].y - lms[pip].y,  0)
            v2 = LM(lms[mcp].x - lms[pip].x,  lms[mcp].y - lms[pip].y,  0)
            cos = (v1.x * v2.x + v1.y * v2.y) / (
                (_dist(LM(0,0,0), v1) + 1e-7) * (_dist(LM(0,0,0), v2) + 1e-7)
            )
            angle = math.acos(float(np.clip(cos, -1.0, 1.0)))
            features.append(angle / math.pi)

        # ── Block 4: Fingertip–wrist normalised distances (4) ─────────────────
        for tip_idx in (8, 12, 16, 20):
            features.append(_dist(wrist, lms[tip_idx]) / hand_span)

        # ── Block 5: Fingertip→wrist angles (4) ──────────────────────────────
        for tip_idx in (8, 12, 16, 20):
            ang = _angle2d(wrist, lms[tip_idx]) / math.pi
            features.append(ang)

        # ── Block 6: Thumb–index pinch (2) ───────────────────────────────────
        pinch_dist = _dist2d(lms[4], lms[8]) / hand_span
        features.append(pinch_dist)
        features.append(1.0 if pinch_dist < 0.15 else 0.0)  # binary pinch

        # ── Block 7: Hand aspect ratio (1) ────────────────────────────────────
        xs = [lm.x for lm in lms]
        ys = [lm.y for lm in lms]
        bw = max(xs) - min(xs) + 1e-7
        bh = max(ys) - min(ys) + 1e-7
        features.append(bw / bh)

        # ── Block 8: Palm orientation angle (1) ──────────────────────────────
        palm_angle = _angle2d(wrist, lms[9]) / math.pi
        features.append(palm_angle)

        assert len(features) == FEATURE_DIM, f"Feature length mismatch: {len(features)}"

        vec = np.array(features, dtype=np.float32)

        # Replace any NaN/Inf with 0
        vec = np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)

        return vec

    def extract_batch(self, hands: list[HandResult]) -> np.ndarray:
        """
        Extract features from a list of HandResults.

        Returns:
            np.ndarray of shape (N, FEATURE_DIM) — float32.
        """
        return np.stack([self.extract(h) for h in hands])

    @staticmethod
    def feature_names() -> list[str]:
        """Return human-readable names for all 42 features."""
        names = []
        for a, b in _DISTANCE_PAIRS:
            names.append(f"dist_{a}_{b}")
        for f in ("thumb", "index", "middle", "ring", "pinky"):
            names.append(f"ext_{f}")
        for f in ("thumb", "index", "middle", "ring", "pinky"):
            names.append(f"curl_{f}")
        for tip in (8, 12, 16, 20):
            names.append(f"tip{tip}_wrist_dist")
        for tip in (8, 12, 16, 20):
            names.append(f"tip{tip}_wrist_angle")
        names += ["pinch_dist", "pinch_binary", "hand_aspect_ratio", "palm_angle"]
        assert len(names) == FEATURE_DIM
        return names
