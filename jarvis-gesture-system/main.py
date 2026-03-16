"""
main.py
=======
JARVIS 2.0 Gesture Recognition System — production entry point.

Usage
─────
    # Default (camera 0, connect to ws://localhost:8000/ws/gestures)
    python main.py

    # Custom camera and backend
    python main.py --camera 1 --backend ws://192.168.1.10:8000/ws/gestures

    # Rule-based only (no ML model)
    python main.py --no-model

    # Load specific ML model
    python main.py --model models/my_model.pkl

    # Headless (no display window)
    python main.py --headless

    # Collect training data
    python -m gesture_system.trainer --mode collect --gesture open_hand

    # Train model
    python -m gesture_system.trainer --mode train

Keyboard shortcuts (when window is open)
─────────────────────────────────────────
    Q       — quit
    R       — reset interpreter state
    S       — print stats to terminal
    M       — toggle ML / rule-based mode
    H       — toggle overlay HUD

Architecture (single-threaded main loop)
─────────────────────────────────────────
  Main thread:  camera read → gesture pipeline → HUD render → cv2.imshow
  BG thread:    WebSocket client reconnection + send loop
  BG thread:    asyncio event loop for WS (inside WS thread)
"""

from __future__ import annotations

import argparse
import logging
import signal
import sys
import time
from pathlib import Path

import cv2

from gesture_system.camera            import CameraManager
from gesture_system.hand_tracking     import HandTracker
from gesture_system.gesture_classifier import GestureClassifier
from gesture_system.gesture_interpreter import GestureInterpreter
from gesture_system.gesture_agent     import GestureAgent
from gesture_system.websocket_client  import WebSocketClient

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("jarvis.gesture.main")

# ── Constants ──────────────────────────────────────────────────────────────────
TARGET_FPS    = 30
DEFAULT_WS    = "ws://localhost:8000/ws/gestures"
DEFAULT_MODEL = Path("models/gesture_model.pkl")
WINDOW_TITLE  = "JARVIS 2.0 — Gesture Recognition"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="JARVIS 2.0 Gesture Recognition System")
    p.add_argument("--camera",    type=int,   default=0,          help="Camera device index")
    p.add_argument("--width",     type=int,   default=1280,        help="Camera width")
    p.add_argument("--height",    type=int,   default=720,         help="Camera height")
    p.add_argument("--fps",       type=int,   default=TARGET_FPS,  help="Target FPS")
    p.add_argument("--backend",   type=str,   default=DEFAULT_WS,  help="WebSocket URL")
    p.add_argument("--model",     type=str,   default=None,        help="Path to ML model pickle")
    p.add_argument("--no-model",  action="store_true",             help="Disable ML model")
    p.add_argument("--no-ws",     action="store_true",             help="Disable WebSocket")
    p.add_argument("--headless",  action="store_true",             help="No display window")
    p.add_argument("--confirm-frames", type=int, default=5,        help="Frames to confirm gesture")
    p.add_argument("--cooldown",  type=float, default=0.8,         help="Gesture cooldown seconds")
    p.add_argument("--max-hands", type=int,   default=2,           help="Max hands to track")
    p.add_argument("--confidence",type=float, default=0.7,         help="Min hand detection confidence")
    p.add_argument("--verbose",   action="store_true",             help="Debug logging")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    log.info("═══ JARVIS 2.0 Gesture Recognition System ═══")
    log.info("Camera: %d  Resolution: %dx%d  FPS: %d",
             args.camera, args.width, args.height, args.fps)

    # ── Component setup ────────────────────────────────────────────────────────
    cam = CameraManager(
        device=args.camera,
        width=args.width,
        height=args.height,
        fps=args.fps,
        flip=True,
    )

    tracker = HandTracker(
        max_hands=args.max_hands,
        min_detect_cf=args.confidence,
        min_track_cf=0.6,
        draw=not args.headless,
    )

    model_path = None
    if not args.no_model:
        if args.model:
            model_path = Path(args.model)
        elif DEFAULT_MODEL.exists():
            model_path = DEFAULT_MODEL

    classifier = GestureClassifier(model_path=model_path)
    if model_path:
        loaded = classifier.load_model()
        if loaded:
            log.info("ML model loaded: %s", model_path)
        else:
            log.info("Rule-based mode (no ML model)")
    else:
        log.info("Rule-based mode")

    interpreter = GestureInterpreter(
        confirm_frames=args.confirm_frames,
        cooldown=args.cooldown,
        enable_swipe_check=True,
        enable_circle_check=True,
    )

    ws_client: WebSocketClient | None = None
    if not args.no_ws:
        try:
            ws_client = WebSocketClient(url=args.backend)
            ws_client.start()
            log.info("WebSocket client started → %s", args.backend)
        except ImportError:
            log.warning("websockets not installed — running without WebSocket")
            ws_client = None

    agent = GestureAgent(
        tracker=tracker,
        classifier=classifier,
        interpreter=interpreter,
    )

    # Wire gesture events to WebSocket sender
    if ws_client:
        def _send_event(event):
            ws_client.send_gesture(event)
            log.info("WS → %s | %s", event.gesture, event.command)
        agent.on_gesture(_send_event)

    # Also log every event
    agent.on_gesture(lambda e: log.info(
        "✓ GESTURE: %-22s CMD: %-20s CONF: %.0f%%",
        e.gesture, e.command, e.confidence * 100
    ))

    # ── Startup ────────────────────────────────────────────────────────────────
    cam.start()
    tracker.start()

    if not args.headless:
        cv2.namedWindow(WINDOW_TITLE, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW_TITLE, args.width, args.height)

    log.info("Pipeline running — press Q to quit")
    log.info("Shortcuts: R=reset  S=stats  M=toggle ML  H=toggle HUD")

    # ── Graceful shutdown signal ───────────────────────────────────────────────
    running = True

    def _handle_signal(sig, frame):
        nonlocal running
        log.info("Shutdown signal received")
        running = False

    signal.signal(signal.SIGINT,  _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    # ── Main loop ──────────────────────────────────────────────────────────────
    frame_deadline = time.perf_counter()
    show_hud = True

    while running:
        # Pace to target FPS
        now = time.perf_counter()
        if now < frame_deadline:
            wait_ms = max(1, int((frame_deadline - now) * 1000))
            if not args.headless:
                key = cv2.waitKey(wait_ms) & 0xFF
                if key == ord('q'):
                    running = False; continue
                elif key == ord('r'):
                    interpreter._reset()
                    log.info("Interpreter state reset")
                elif key == ord('s'):
                    import pprint
                    pprint.pprint(agent.stats())
                    if ws_client:
                        pprint.pprint(ws_client.stats())
                elif key == ord('m'):
                    log.info("ML model loaded: %s", classifier.model_loaded)
                elif key == ord('h'):
                    show_hud = not show_hud
                    tracker._draw = show_hud
            else:
                time.sleep((frame_deadline - now))
        frame_deadline = time.perf_counter() + 1.0 / args.fps

        # Read frame
        frame = cam.read()
        if frame is None:
            continue

        if not cam.is_running():
            log.error("Camera stopped unexpectedly — exiting")
            break

        # Run gesture pipeline
        annotated, event = agent.process_frame(frame)

        # Display
        if not args.headless:
            cv2.imshow(WINDOW_TITLE, annotated)
            # poll for events without extra wait (already waited above)
            cv2.waitKey(1)

    # ── Cleanup ────────────────────────────────────────────────────────────────
    log.info("Shutting down …")
    log.info("Stats: %s", agent.stats())

    cam.stop()
    tracker.stop()
    if ws_client:
        ws_client.stop()
    if not args.headless:
        cv2.destroyAllWindows()

    log.info("Goodbye.")


if __name__ == "__main__":
    main()
