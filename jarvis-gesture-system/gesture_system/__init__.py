"""gesture_system — JARVIS 2.0 real-time gesture recognition package."""

from .camera           import CameraManager
from .hand_tracking    import HandTracker, HandResult, LM
from .feature_extraction import FeatureExtractor, FEATURE_DIM
from .gesture_classifier import (
    GestureClassifier, ClassificationResult,
    OPEN_HAND, FIST, POINT, PINCH, DOUBLE_PINCH, GRAB, RELEASE,
    SWIPE_LEFT, SWIPE_RIGHT, SWIPE_UP, SWIPE_DOWN,
    CIRCLE_CLOCKWISE, CIRCLE_COUNTERCLOCKWISE,
    ZOOM_IN, ZOOM_OUT, TWO_HAND_EXPAND, TWO_HAND_CONTRACT,
    UNKNOWN, ALL_GESTURES,
)
from .gesture_interpreter import GestureInterpreter, GestureEvent, GESTURE_COMMAND_MAP
from .gesture_agent      import GestureAgent
from .websocket_client   import WebSocketClient

__all__ = [
    "CameraManager",
    "HandTracker", "HandResult", "LM",
    "FeatureExtractor", "FEATURE_DIM",
    "GestureClassifier", "ClassificationResult",
    "OPEN_HAND", "FIST", "POINT", "PINCH", "DOUBLE_PINCH", "GRAB", "RELEASE",
    "SWIPE_LEFT", "SWIPE_RIGHT", "SWIPE_UP", "SWIPE_DOWN",
    "CIRCLE_CLOCKWISE", "CIRCLE_COUNTERCLOCKWISE",
    "ZOOM_IN", "ZOOM_OUT", "TWO_HAND_EXPAND", "TWO_HAND_CONTRACT",
    "UNKNOWN", "ALL_GESTURES",
    "GestureInterpreter", "GestureEvent", "GESTURE_COMMAND_MAP",
    "GestureAgent",
    "WebSocketClient",
]
