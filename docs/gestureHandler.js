/**
 * frontend/gestureHandler.js
 * ─────────────────────────────────────────────────────────────
 * Gesture input layer.
 *
 * Mode A — Simulated (default, no hardware required):
 *   Programmatic dispatch via sendGesture(id).
 *   Used by gesture buttons in the UI.
 *
 * Mode B — WebSocket bridge (gesture-system running locally):
 *   Connects to ws://localhost:8001/gestures and forwards
 *   real MediaPipe events from the Python gesture-system.
 *
 * All gestures go through ApiClient.sendGesture() to the backend.
 * ─────────────────────────────────────────────────────────────
 */

const GestureHandler = (() => {

  let _onGesture    = null;   // (gestureId, response) => void
  let _wsGesture    = null;   // WebSocket to local gesture-system
  let _wsActive     = false;

  const SUPPORTED_GESTURES = [
    'swipe_right', 'swipe_left', 'swipe_up', 'swipe_down',
    'open_hand', 'fist', 'pinch', 'grab', 'release',
    'zoom_in', 'zoom_out', 'thumbs_up', 'thumbs_down',
    'wave', 'point', 'circle_clockwise', 'circle_counterclockwise',
  ];

  // ── Callback registration ──────────────────────────────────

  /**
   * Register a callback invoked after every gesture is processed.
   * @param {function} fn - (gestureId: string, backendResponse: object) => void
   */
  function onGesture(fn) { _onGesture = fn; }

  // ── Send gesture to backend ────────────────────────────────

  /**
   * Send a gesture to the JARVIS backend and invoke the callback.
   * @param {string} gestureId  - e.g. "swipe_right"
   * @param {number} confidence - Classifier confidence [0–1]
   */
  async function sendGesture(gestureId, confidence = 0.95) {
    if (!SUPPORTED_GESTURES.includes(gestureId)) {
      console.warn(`[GestureHandler] Unknown gesture: ${gestureId}`);
    }

    console.log(`[GestureHandler] Sending: ${gestureId}`);

    try {
      const response = await ApiClient.sendGesture(gestureId, confidence);
      if (_onGesture) _onGesture(gestureId, response);
      return response;
    } catch (err) {
      console.error('[GestureHandler] Error:', err.message);
      if (_onGesture) _onGesture(gestureId, null);
      return null;
    }
  }

  // ── WebSocket bridge (gesture-system → frontend → backend) ─

  /**
   * Connect to the local gesture-system WebSocket bridge.
   * The gesture-system runs on ws://localhost:8001/gestures
   * and forwards real MediaPipe gesture events.
   *
   * @param {string} wsUrl - Gesture-system WS URL (default: ws://localhost:8001/gestures)
   */
  function connectGestureSystem(wsUrl = 'ws://localhost:8001/gestures') {
    if (_wsGesture) _wsGesture.close();

    _wsGesture = new WebSocket(wsUrl);

    _wsGesture.onopen = () => {
      _wsActive = true;
      console.log('[GestureHandler] Gesture-system WS connected');
    };

    _wsGesture.onmessage = (event) => {
      try {
        const msg = JSON.parse(event.data);
        // Gesture-system sends: { type: "gesture", gesture: "swipe_right", confidence: 0.93 }
        if (msg.type === 'gesture' && msg.gesture) {
          sendGesture(msg.gesture, msg.confidence || 0.95);
        }
      } catch (_) {}
    };

    _wsGesture.onclose = () => {
      _wsActive = false;
      console.log('[GestureHandler] Gesture-system WS disconnected');
    };

    _wsGesture.onerror = () => {
      console.warn('[GestureHandler] Could not connect to gesture-system at', wsUrl);
    };
  }

  // ── Keyboard shortcut simulation (development mode) ────────

  /**
   * Register keyboard shortcuts for testing gestures without hardware.
   * Active only when called explicitly.
   *
   * Shortcuts:
   *   ArrowRight → swipe_right    ArrowLeft → swipe_left
   *   ArrowUp    → swipe_up       ArrowDown → swipe_down
   *   O          → open_hand      F         → fist
   *   P          → pinch          G         → grab
   *   Z          → zoom_in        X         → zoom_out
   */
  function enableKeyboardSimulation() {
    const keyMap = {
      ArrowRight: 'swipe_right', ArrowLeft: 'swipe_left',
      ArrowUp:    'swipe_up',    ArrowDown: 'swipe_down',
      o: 'open_hand', f: 'fist', p: 'pinch', g: 'grab',
      z: 'zoom_in',   x: 'zoom_out',
    };

    document.addEventListener('keydown', (e) => {
      // Don't intercept when focus is on an input
      if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;

      const gestureId = keyMap[e.key];
      if (gestureId) {
        e.preventDefault();
        sendGesture(gestureId, 0.90);
      }
    });

    console.log('[GestureHandler] Keyboard simulation enabled');
  }

  // ── Info ───────────────────────────────────────────────────

  function isGestureSystemConnected() { return _wsActive; }
  function supportedGestures()        { return [...SUPPORTED_GESTURES]; }

  // ── Public API ─────────────────────────────────────────────
  return {
    onGesture,
    sendGesture,
    connectGestureSystem,
    enableKeyboardSimulation,
    isGestureSystemConnected,
    supportedGestures,
  };

})();
