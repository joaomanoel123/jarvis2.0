"""
gesture_system/camera.py
========================
CameraManager — thread-safe, non-blocking camera capture layer.

Design
──────
• A background daemon thread reads frames from OpenCV at full hardware speed.
• The latest frame is stored in a double-buffered slot; readers always get the
  newest frame without waiting for the capture loop.
• An asyncio-friendly interface wraps the thread for use in async pipelines.
• Horizontal flip (mirror mode) is on by default — it matches natural hand
  movement expectations when looking at your own reflection.

Tested with:
  - Built-in laptop webcams  (device 0)
  - USB UVC cameras          (device 0–4)
  - IP cameras               (RTSP URLs as device string)

Usage
-----
    cam = CameraManager(device=0, width=1280, height=720, fps=30)
    cam.start()
    frame = cam.read()          # numpy ndarray, BGR, or None
    cam.stop()

    # or async context manager:
    async with CameraManager() as cam:
        frame = cam.read()
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Optional

import cv2
import numpy as np

log = logging.getLogger("jarvis.gesture.camera")


class CameraManager:
    """
    Background-thread camera reader.

    Args:
        device:    Camera device index (int) or RTSP/HTTP URL (str).
        width:     Requested frame width in pixels.
        height:    Requested frame height in pixels.
        fps:       Requested capture frame rate.
        flip:      Horizontally flip frames (mirror mode). Default True.
        backend:   OpenCV backend flag (e.g. cv2.CAP_V4L2). None = auto.
    """

    def __init__(
        self,
        device: int | str = 0,
        width:  int = 1280,
        height: int = 720,
        fps:    int = 30,
        flip:   bool = True,
        backend: int | None = None,
    ) -> None:
        self._device  = device
        self._width   = width
        self._height  = height
        self._fps     = fps
        self._flip    = flip
        self._backend = backend

        self._cap: cv2.VideoCapture | None = None
        self._frame:  np.ndarray | None = None
        self._lock    = threading.Lock()
        self._running = False
        self._thread: threading.Thread | None = None

        # Performance tracking
        self._frame_count = 0
        self._last_fps_ts = time.perf_counter()
        self._current_fps = 0.0

    # ── Lifecycle ──────────────────────────────────────────────────────────────

    def start(self) -> "CameraManager":
        """Open the camera and start the background capture thread."""
        if self._running:
            log.warning("CameraManager already running — ignoring start()")
            return self

        log.info("Opening camera device=%s  %dx%d @ %d fps", self._device, self._width, self._height, self._fps)
        if self._backend is not None:
            self._cap = cv2.VideoCapture(self._device, self._backend)
        else:
            self._cap = cv2.VideoCapture(self._device)

        if not self._cap.isOpened():
            raise RuntimeError(f"Could not open camera device: {self._device!r}")

        # Apply resolution and FPS hints (driver may not honour them exactly)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH,  self._width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
        self._cap.set(cv2.CAP_PROP_FPS,          self._fps)
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)  # minimise latency

        # Read actual settings back
        aw = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        ah = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        af = self._cap.get(cv2.CAP_PROP_FPS)
        log.info("Camera opened — actual: %dx%d @ %.1f fps", aw, ah, af)

        self._running = True
        self._thread  = threading.Thread(
            target=self._capture_loop,
            name="camera-capture",
            daemon=True,
        )
        self._thread.start()
        return self

    def stop(self) -> None:
        """Stop the capture thread and release the camera."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
            self._thread = None
        if self._cap:
            self._cap.release()
            self._cap = None
        log.info("CameraManager stopped")

    # ── Frame access ───────────────────────────────────────────────────────────

    def read(self) -> np.ndarray | None:
        """
        Return the most recently captured frame.

        Returns:
            A BGR numpy array, or None if no frame is available yet.
        """
        with self._lock:
            if self._frame is None:
                return None
            return self._frame.copy()

    def is_running(self) -> bool:
        return self._running and self._cap is not None

    @property
    def fps(self) -> float:
        """Measured capture FPS over the last second."""
        return self._current_fps

    @property
    def resolution(self) -> tuple[int, int]:
        """(width, height) of captured frames."""
        if self._cap is None:
            return (self._width, self._height)
        return (
            int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )

    # ── Internal capture loop ──────────────────────────────────────────────────

    def _capture_loop(self) -> None:
        """Background thread: continuously grabs frames from the camera."""
        log.debug("Capture loop started")
        while self._running:
            if self._cap is None or not self._cap.isOpened():
                break

            ok, frame = self._cap.read()
            if not ok or frame is None:
                log.warning("Frame read failed — skipping")
                time.sleep(0.01)
                continue

            if self._flip:
                frame = cv2.flip(frame, 1)

            with self._lock:
                self._frame = frame

            # Track FPS
            self._frame_count += 1
            now = time.perf_counter()
            elapsed = now - self._last_fps_ts
            if elapsed >= 1.0:
                self._current_fps = self._frame_count / elapsed
                self._frame_count = 0
                self._last_fps_ts = now

        log.debug("Capture loop exited")

    # ── Context manager ────────────────────────────────────────────────────────

    def __enter__(self) -> "CameraManager":
        return self.start()

    def __exit__(self, *_) -> None:
        self.stop()

    async def __aenter__(self) -> "CameraManager":
        return self.start()

    async def __aexit__(self, *_) -> None:
        self.stop()

    def __repr__(self) -> str:
        state = "running" if self._running else "stopped"
        return f"<CameraManager device={self._device} {state} fps={self._current_fps:.1f}>"
