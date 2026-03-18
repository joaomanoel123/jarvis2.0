/**
 * frontend/apiClient.js
 * ─────────────────────────────────────────────────────────────
 * All backend communication goes through this module.
 * Maintains session_id automatically across all requests.
 * ─────────────────────────────────────────────────────────────
 */

const ApiClient = (() => {

  let _base      = localStorage.getItem('j_api') || 'http://localhost:8000';
  let _sessionId = localStorage.getItem('j_sid') || null;
  let _ws        = null;
  let _wsReconnect = 0;
  let _onWsMessage = null;

  // ── Session ────────────────────────────────────────────────

  function setBase(url) {
    _base = url.replace(/\/$/, '');
    localStorage.setItem('j_api', _base);
  }

  function getBase() { return _base; }

  function getSessionId() { return _sessionId; }

  function _saveSession(sid) {
    if (sid && sid !== _sessionId) {
      _sessionId = sid;
      localStorage.setItem('j_sid', _sessionId);
    }
  }

  // ── Core fetch ─────────────────────────────────────────────

  async function _post(path, body = {}) {
    const payload = { ...body };
    if (_sessionId) payload.session_id = _sessionId;

    const res = await fetch(_base + path, {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify(payload),
    });

    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: res.statusText }));
      throw new Error(err.detail || `HTTP ${res.status}`);
    }

    const data = await res.json();
    if (data.session_id) _saveSession(data.session_id);
    return data;
  }

  async function _get(path) {
    const res = await fetch(_base + path);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    return res.json();
  }

  // ── Endpoints ──────────────────────────────────────────────

  /** Send a text message to JARVIS. */
  async function sendMessage(message) {
    return _post('/chat', { message });
  }

  /** Send a voice-transcribed command. */
  async function sendVoiceCommand(text, confidence = 1.0) {
    return _post('/voice-command', { text, confidence, source: 'voice' });
  }

  /** Send a gesture event. */
  async function sendGesture(gestureId, confidence = 1.0) {
    return _post('/gesture-command', { gesture_id: gestureId, confidence });
  }

  /** Full system status. */
  async function fetchStatus() {
    return _get('/status');
  }

  /** Lightweight health check. */
  async function fetchHealth() {
    return _get('/health');
  }

  // ── WebSocket ──────────────────────────────────────────────

  function connectWebSocket(onMessage) {
    _onWsMessage = onMessage;
    const url = _base.replace('https://', 'wss://').replace('http://', 'ws://') + '/ws/ui';

    try {
      _ws = new WebSocket(url);

      _ws.onopen = () => {
        _wsReconnect = 0;
        if (_onWsMessage) _onWsMessage({ type: 'ws_connected' });
      };

      _ws.onmessage = (e) => {
        try {
          const msg = JSON.parse(e.data);
          if (msg.type === 'ping') { _ws.send(JSON.stringify({ type: 'pong' })); return; }
          if (_onWsMessage) _onWsMessage(msg);
        } catch (_) {}
      };

      _ws.onclose = () => {
        if (_onWsMessage) _onWsMessage({ type: 'ws_disconnected' });
        const delay = Math.min(1000 * Math.pow(2, _wsReconnect++), 30000);
        setTimeout(() => connectWebSocket(_onWsMessage), delay);
      };

      _ws.onerror = () => {};
    } catch (_) {}
  }

  function isWsConnected() {
    return _ws && _ws.readyState === WebSocket.OPEN;
  }

  // ── Public API ─────────────────────────────────────────────
  return { setBase, getBase, getSessionId, sendMessage, sendVoiceCommand, sendGesture, fetchStatus, fetchHealth, connectWebSocket, isWsConnected };

})();
