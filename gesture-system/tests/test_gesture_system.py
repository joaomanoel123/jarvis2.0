"""
tests/test_gesture_system.py
============================
Unit tests for the gesture recognition pipeline.

Run with:
    pytest tests/test_gesture_system.py -v

No camera or WebSocket required — all tests use synthetic hand landmarks.
"""

from __future__ import annotations

import math
import time
import unittest

import numpy as np

from gesture_system.hand_tracking import HandResult, LM
from gesture_system.feature_extraction import FeatureExtractor, FEATURE_DIM
from gesture_system.gesture_classifier import (
    GestureClassifier, ClassificationResult,
    OPEN_HAND, FIST, POINT, PINCH, GRAB, UNKNOWN,
)
from gesture_system.gesture_interpreter import (
    GestureInterpreter, GestureEvent, GESTURE_COMMAND_MAP,
)


# ── Landmark factories ─────────────────────────────────────────────────────────

def _lm(x: float, y: float, z: float = 0.0) -> LM:
    return LM(x, y, z)


def _make_hand(
    landmarks: list[tuple[float, float, float]],
    handedness: str = "Right",
    confidence: float = 0.95,
) -> HandResult:
    lms = [LM(x, y, z) for x, y, z in landmarks]
    lms_px = [LM(int(x * 1280), int(y * 720), z) for x, y, z in landmarks]
    xs = [int(x * 1280) for x, y, z in landmarks]
    ys = [int(y * 720)  for x, y, z in landmarks]
    bbox = (min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys))
    return HandResult(
        hand_index=0,
        handedness=handedness,
        landmarks=lms,
        landmarks_px=lms_px,
        bbox=bbox,
        confidence=confidence,
    )


def _open_hand() -> HandResult:
    """All 4 fingers fully extended upward."""
    lms = [
        (0.5, 0.9, 0),    # 0 WRIST
        (0.45, 0.82, 0),  # 1
        (0.42, 0.74, 0),  # 2
        (0.40, 0.67, 0),  # 3
        (0.37, 0.60, 0),  # 4 THUMB TIP
        (0.52, 0.72, 0),  # 5 INDEX MCP
        (0.53, 0.62, 0),  # 6 INDEX PIP
        (0.54, 0.52, 0),  # 7
        (0.54, 0.44, 0),  # 8 INDEX TIP
        (0.58, 0.71, 0),  # 9 MIDDLE MCP
        (0.59, 0.60, 0),  # 10
        (0.60, 0.50, 0),  # 11
        (0.60, 0.41, 0),  # 12 MIDDLE TIP
        (0.63, 0.72, 0),  # 13
        (0.64, 0.62, 0),  # 14
        (0.64, 0.53, 0),  # 15
        (0.64, 0.46, 0),  # 16 RING TIP
        (0.68, 0.74, 0),  # 17
        (0.69, 0.65, 0),  # 18
        (0.70, 0.57, 0),  # 19
        (0.70, 0.51, 0),  # 20 PINKY TIP
    ]
    return _make_hand(lms)


def _fist() -> HandResult:
    """All fingers curled down below the knuckles."""
    lms = [
        (0.5, 0.70, 0),   # 0 WRIST
        (0.45, 0.67, 0),  # 1
        (0.42, 0.62, 0),  # 2
        (0.42, 0.60, 0),  # 3
        (0.44, 0.62, 0),  # 4 THUMB TIP (tucked)
        (0.52, 0.60, 0),  # 5 INDEX MCP
        (0.53, 0.65, 0),  # 6 INDEX PIP (curled below MCP)
        (0.53, 0.68, 0),  # 7
        (0.52, 0.70, 0),  # 8 INDEX TIP (curled)
        (0.58, 0.59, 0),  # 9 MIDDLE MCP
        (0.59, 0.64, 0),  # 10
        (0.59, 0.67, 0),  # 11
        (0.59, 0.70, 0),  # 12 MIDDLE TIP (curled)
        (0.63, 0.60, 0),  # 13
        (0.64, 0.65, 0),  # 14
        (0.64, 0.68, 0),  # 15
        (0.64, 0.70, 0),  # 16 RING TIP (curled)
        (0.68, 0.62, 0),  # 17
        (0.69, 0.66, 0),  # 18
        (0.70, 0.69, 0),  # 19
        (0.70, 0.71, 0),  # 20 PINKY TIP (curled)
    ]
    return _make_hand(lms)


def _point() -> HandResult:
    """Index extended, others curled."""
    lms = _open_hand().landmarks
    # Curl middle, ring, pinky by pushing tips below PIP
    lms_raw = [(lm.x, lm.y, lm.z) for lm in lms]
    # Middle tip (12) below middle PIP (10)
    lms_raw[12] = (lms_raw[10][0], lms_raw[10][1] + 0.08, 0)
    lms_raw[16] = (lms_raw[14][0], lms_raw[14][1] + 0.08, 0)
    lms_raw[20] = (lms_raw[18][0], lms_raw[18][1] + 0.08, 0)
    return _make_hand(lms_raw)


def _pinch() -> HandResult:
    """Thumb tip very close to index tip."""
    lms = [(lm.x, lm.y, lm.z) for lm in _open_hand().landmarks]
    # Move thumb tip to very close to index tip
    lms[4] = (lms[8][0] + 0.01, lms[8][1] + 0.01, 0)
    # Curl other fingers
    lms[12] = (lms[10][0], lms[10][1] + 0.08, 0)
    lms[16] = (lms[14][0], lms[14][1] + 0.08, 0)
    lms[20] = (lms[18][0], lms[18][1] + 0.08, 0)
    return _make_hand(lms)


# ── Feature extraction tests ───────────────────────────────────────────────────

class TestFeatureExtractor(unittest.TestCase):

    def setUp(self):
        self.fe = FeatureExtractor()

    def test_vector_shape(self):
        vec = self.fe.extract(_open_hand())
        self.assertEqual(vec.shape, (FEATURE_DIM,))
        self.assertEqual(vec.dtype, np.float32)

    def test_no_nan_inf(self):
        for hand in [_open_hand(), _fist(), _point(), _pinch()]:
            vec = self.fe.extract(hand)
            self.assertFalse(np.any(np.isnan(vec)), "NaN in feature vector")
            self.assertFalse(np.any(np.isinf(vec)), "Inf in feature vector")

    def test_batch_shape(self):
        hands = [_open_hand(), _fist()]
        batch = self.fe.extract_batch(hands)
        self.assertEqual(batch.shape, (2, FEATURE_DIM))

    def test_feature_names_length(self):
        names = FeatureExtractor.feature_names()
        self.assertEqual(len(names), FEATURE_DIM)

    def test_pinch_distance_low_for_pinch(self):
        """Pinch distance feature (index 20) should be low for pinch gesture."""
        vec_pinch  = self.fe.extract(_pinch())
        vec_open   = self.fe.extract(_open_hand())
        # Feature index 20 is the first distance pair (thumb-index, pair 0)
        self.assertLess(vec_pinch[0], vec_open[0])

    def test_extension_positive_for_open_hand(self):
        """Extension scores should be positive for open hand."""
        vec = self.fe.extract(_open_hand())
        # Features 20–24 are finger extension scores
        for i in range(20, 25):
            self.assertGreater(vec[i], 0, f"Extension feature {i} not positive for open hand")

    def test_curl_different_for_fist_vs_open(self):
        """
        For a fist, fingertip y > PIP y (tip below pip in screen coords = curled).
        The extension score (features 20-24) should be higher for open hand than fist.
        The curl is checked via extension: open_hand extension > fist extension.
        """
        vec_fist = self.fe.extract(_fist())
        vec_open = self.fe.extract(_open_hand())
        # Extension scores (features 20-24): open hand has positive (up) extension
        # Fist has negative or near-zero extension (tips not above MCPs)
        open_ext = vec_open[20:25].sum()
        fist_ext = vec_fist[20:25].sum()
        self.assertGreater(open_ext, fist_ext,
            f"Open hand extension {open_ext:.2f} should exceed fist {fist_ext:.2f}")


# ── Gesture classifier tests ───────────────────────────────────────────────────

class TestGestureClassifier(unittest.TestCase):

    def setUp(self):
        self.clf = GestureClassifier(model_path=None)

    def test_open_hand_classified(self):
        result = self.clf.classify([_open_hand()])
        self.assertEqual(result.gesture, OPEN_HAND)
        self.assertGreater(result.confidence, 0.7)
        self.assertEqual(result.mode, "rule")

    def test_fist_classified(self):
        result = self.clf.classify([_fist()])
        self.assertEqual(result.gesture, FIST)
        self.assertGreater(result.confidence, 0.7)

    def test_point_classified(self):
        result = self.clf.classify([_point()])
        self.assertEqual(result.gesture, POINT)

    def test_pinch_classified(self):
        result = self.clf.classify([_pinch()])
        self.assertEqual(result.gesture, PINCH)
        self.assertGreater(result.confidence, 0.7)

    def test_empty_hands_returns_unknown(self):
        result = self.clf.classify([])
        self.assertEqual(result.gesture, UNKNOWN)

    def test_result_has_required_fields(self):
        result = self.clf.classify([_open_hand()])
        self.assertIsInstance(result.gesture,    str)
        self.assertIsInstance(result.confidence, float)
        self.assertIsInstance(result.mode,       str)


# ── Gesture interpreter tests ──────────────────────────────────────────────────

class TestGestureInterpreter(unittest.TestCase):

    def setUp(self):
        self.interp = GestureInterpreter(
            confirm_frames=3,
            cooldown=0.1,
            enable_swipe_check=False,   # disable for deterministic tests
            enable_circle_check=False,
        )

    def _run_n(self, gesture: str, n: int, confidence: float = 0.9) -> list:
        """Feed N consecutive frames of the same gesture."""
        events = []
        result = ClassificationResult(gesture, confidence, "rule")
        for _ in range(n):
            ev = self.interp.update(result, [_open_hand()])
            if ev:
                events.append(ev)
        return events

    def test_gesture_not_emitted_before_threshold(self):
        events = self._run_n(OPEN_HAND, 2)  # threshold is 3
        self.assertEqual(len(events), 0)

    def test_gesture_emitted_at_threshold(self):
        events = self._run_n(OPEN_HAND, 3)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].gesture, OPEN_HAND)

    def test_gesture_event_has_command(self):
        events = self._run_n(OPEN_HAND, 3)
        self.assertEqual(events[0].command, "open_menu")

    def test_different_gesture_resets_count(self):
        result_a = ClassificationResult(OPEN_HAND, 0.9, "rule")
        result_b = ClassificationResult(FIST,      0.9, "rule")
        hands = [_open_hand()]
        # 2 frames of OPEN_HAND
        self.interp.update(result_a, hands)
        self.interp.update(result_a, hands)
        # Switch to FIST — resets count
        self.interp.update(result_b, hands)
        # Only 1 frame of FIST so far — no event yet
        self.assertEqual(self.interp.frame_count, 1)

    def test_gesture_event_has_velocity(self):
        events = self._run_n(OPEN_HAND, 3)
        self.assertIsInstance(events[0].velocity, tuple)
        self.assertEqual(len(events[0].velocity), 2)

    def test_state_transitions(self):
        self.assertEqual(self.interp.state, "idle")
        result = ClassificationResult(OPEN_HAND, 0.9, "rule")
        hands  = [_open_hand()]
        self.interp.update(result, hands)
        self.assertEqual(self.interp.state, "candidate")
        self.interp.update(result, hands)
        self.interp.update(result, hands)
        self.assertEqual(self.interp.state, "confirmed")


# ── Command map tests ──────────────────────────────────────────────────────────

class TestGestureCommandMap(unittest.TestCase):

    def test_all_gestures_have_commands(self):
        from gesture_system.gesture_classifier import ALL_GESTURES
        for g in ALL_GESTURES:
            self.assertIn(g, GESTURE_COMMAND_MAP, f"Gesture {g} missing from command map")

    def test_swipe_left_command(self):
        self.assertEqual(GESTURE_COMMAND_MAP["SWIPE_LEFT"], "navigate_left")

    def test_swipe_right_command(self):
        self.assertEqual(GESTURE_COMMAND_MAP["SWIPE_RIGHT"], "navigate_right")

    def test_zoom_in_command(self):
        self.assertEqual(GESTURE_COMMAND_MAP["ZOOM_IN"], "zoom_interface")

    def test_event_to_dict(self):
        ev = GestureEvent(gesture="SWIPE_LEFT", command="navigate_left",
                          confidence=0.9, velocity=(-1.2, 0.0))
        d = ev.to_dict()
        self.assertEqual(d["type"],    "gesture")
        self.assertEqual(d["gesture"], "SWIPE_LEFT")
        self.assertEqual(d["command"], "navigate_left")
        self.assertIn("velocity", d)
        self.assertIn("timestamp", d)


if __name__ == "__main__":
    unittest.main(verbosity=2)
