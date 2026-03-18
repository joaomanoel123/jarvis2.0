/**
 * frontend/main.js
 * ─────────────────────────────────────────────────────────────
 * Main controller for JARVIS 2.0 frontend.
 * Wires ApiClient, CommandExecutor, VoiceHandler, GestureHandler.
 * Manages UI state machine: standby → listening → processing → responding
 * ─────────────────────────────────────────────────────────────
 */

// ── State ──────────────────────────────────────────────────────────────

const JARVIS = {
  mode:      'standby',   // standby | listening | processing | responding
  msgCount:  0,
  sessionId: null,
};

// ── UI element cache ───────────────────────────────────────────────────

const UI = {
  input:      () => document.getElementById('j-input'),
  sendBtn:    () => document.getElementById('j-send'),
  voiceBtn:   () => document.getElementById('j-voice'),
  log:        () => document.getElementById('j-log'),
  status:     () => document.getElementById('j-status'),
  statusDot:  () => document.getElementById('j-dot'),
  intentBadge:() => document.getElementById('j-intent'),
  latency:    () => document.getElementById('j-latency'),
  modelLabel: () => document.getElementById('j-model'),
};

// ── Mode machine ────────────────────────────────────────────────────────

const MODE_LABELS = {
  standby:    'STANDBY',
  listening:  'LISTENING',
  processing: 'PROCESSING',
  responding: 'RESPONDING',
};

function setMode(mode) {
  JARVIS.mode = mode;
  const dot   = UI.statusDot();
  const label = UI.status();
  if (dot)   { dot.className   = `j-dot ${mode}`; }
  if (label) { label.textContent = MODE_LABELS[mode] || mode.toUpperCase(); }
}

// ── Chat log ────────────────────────────────────────────────────────────

function appendUser(text, source = 'text') {
  const log = UI.log();
  if (!log) return;
  JARVIS.msgCount++;
  const ts = _ts();
  const sourceBadge = source !== 'text' ? `<span class="j-badge">${source.toUpperCase()}</span>` : '';
  const div = document.createElement('div');
  div.className = 'j-msg j-user';
  div.innerHTML = `
    <div class="j-av">YOU</div>
    <div class="j-body">
      <div class="j-meta"><span>${ts}</span>${sourceBadge}</div>
      <div class="j-text">${_esc(text)}</div>
    </div>`;
  log.appendChild(div);
  log.scrollTop = log.scrollHeight;
}

function appendBot(text, meta = {}) {
  const log = UI.log();
  if (!log) return;
  JARVIS.msgCount++;
  const ts      = _ts();
  const intent  = meta.intent  ? `<span class="j-badge">${meta.intent.toUpperCase()}</span>` : '';
  const lat     = meta.latency_ms ? `<span class="j-badge">${meta.latency_ms | 0}ms</span>` : '';
  const action  = meta.action && meta.action.command
    ? `<div class="j-action"><span class="j-ak">action:</span> ${_esc(meta.action.command)} — ${_esc(meta.action.type || '')}</div>`
    : '';

  const div = document.createElement('div');
  div.className = 'j-msg j-bot';
  div.innerHTML = `
    <div class="j-av">JAR</div>
    <div class="j-body">
      <div class="j-meta"><span>${ts}</span>${intent}${lat}</div>
      <div class="j-text">${_esc(text)}</div>
      ${action}
    </div>`;
  log.appendChild(div);
  log.scrollTop = log.scrollHeight;

  if (UI.intentBadge()) UI.intentBadge().textContent = (meta.intent || '').toUpperCase();
  if (UI.latency())     UI.latency().textContent      = meta.latency_ms ? `${meta.latency_ms | 0} ms` : '';
}

function appendThinking() {
  const log = UI.log();
  if (!log) return;
  const id  = `think-${Date.now()}`;
  const div = document.createElement('div');
  div.id        = id;
  div.className = 'j-msg j-bot';
  div.innerHTML = `<div class="j-av">JAR</div><div class="j-body"><div class="j-text"><span class="j-dots"><span></span><span></span><span></span></span></div></div>`;
  log.appendChild(div);
  log.scrollTop = log.scrollHeight;
  return id;
}

function removeThinking(id) {
  document.getElementById(id)?.remove();
}

// ── Response handler ────────────────────────────────────────────────────

function handleResponse(response) {
  if (!response) return;

  // Update session
  if (response.session_id) JARVIS.sessionId = response.session_id;

  // Render message
  appendBot(response.response || '', response);

  // Execute action if present
  if (response.action && response.action.command) {
    const result = CommandExecutor.executeCommand(response.action);
    console.log('[Main] Action result:', result);

    // Show toast for successful browser commands
    if (response.action.type === 'browser' && result.success) {
      CommandExecutor.toast(response.response, 'info', 2500);
    }
  }
}

// ── Send text ────────────────────────────────────────────────────────────

async function sendMessage() {
  const input = UI.input();
  const text  = input?.value?.trim();
  if (!text || JARVIS.mode === 'processing') return;
  input.value = '';

  appendUser(text, 'text');
  setMode('processing');
  const thinkId = appendThinking();

  try {
    const response = await ApiClient.sendMessage(text);
    removeThinking(thinkId);
    setMode('responding');
    handleResponse(response);
    setTimeout(() => setMode('standby'), 1000);
  } catch (err) {
    removeThinking(thinkId);
    appendBot(`Error: ${err.message}`, { intent: 'error' });
    setMode('standby');
  }
}

// ── Voice ────────────────────────────────────────────────────────────────

function toggleVoice() {
  if (!VoiceHandler.isSupported()) {
    appendBot('Voice recognition is not supported in this browser.', { intent: 'error' });
    return;
  }

  const btn = UI.voiceBtn();

  if (VoiceHandler.isActive()) {
    VoiceHandler.stop();
    if (btn) { btn.classList.remove('active'); btn.textContent = '🎤'; }
    setMode('standby');
    return;
  }

  VoiceHandler.start();
  if (btn) { btn.classList.add('active'); btn.textContent = '⏹'; }
  setMode('listening');
}

// ── Gesture ──────────────────────────────────────────────────────────────

async function sendGesture(gestureId) {
  if (JARVIS.mode === 'processing') return;
  setMode('processing');
  const thinkId = appendThinking();

  const response = await GestureHandler.sendGesture(gestureId);
  removeThinking(thinkId);

  if (response) {
    setMode('responding');
    handleResponse(response);
    setTimeout(() => setMode('standby'), 1000);
  } else {
    setMode('standby');
  }
}

// ── Status poll ──────────────────────────────────────────────────────────

async function fetchStatus() {
  try {
    const s = await ApiClient.fetchStatus();
    if (UI.modelLabel() && s.model) {
      UI.modelLabel().textContent = (s.model.model_id || '—').split('/').pop();
    }
  } catch (_) {}
}

// ── Boot ─────────────────────────────────────────────────────────────────

function boot() {
  // ── Voice callbacks ────────────────────────────────────────
  VoiceHandler.onResult((text, confidence) => {
    if (UI.input()) UI.input().value = text;
    appendUser(text, 'voice');
    if (UI.voiceBtn()) { UI.voiceBtn().classList.remove('active'); UI.voiceBtn().textContent = '🎤'; }
    setMode('processing');
    const thinkId = appendThinking();

    ApiClient.sendVoiceCommand(text, confidence)
      .then(response => {
        removeThinking(thinkId);
        setMode('responding');
        handleResponse(response);
        setTimeout(() => setMode('standby'), 1000);
      })
      .catch(err => {
        removeThinking(thinkId);
        appendBot(`Error: ${err.message}`, { intent: 'error' });
        setMode('standby');
      });
  });

  VoiceHandler.onStateChange((state) => {
    const btn = UI.voiceBtn();
    if (state === 'listening') { setMode('listening'); if (btn) btn.classList.add('active'); }
    if (state === 'stopped')   { if (JARVIS.mode === 'listening') setMode('standby'); if (btn) { btn.classList.remove('active'); btn.textContent = '🎤'; } }
    if (state === 'permission_denied') { appendBot('Microphone permission denied. Please allow access in your browser.', { intent: 'error' }); setMode('standby'); }
  });

  // ── Gesture callback ───────────────────────────────────────
  GestureHandler.onGesture((gestureId, response) => {
    console.log('[Main] Gesture response:', gestureId, response?.intent);
  });

  // ── WS updates ─────────────────────────────────────────────
  ApiClient.connectWebSocket((msg) => {
    if (msg.type === 'jarvis_update') handleResponse(msg);
  });

  // ── Input enter key ────────────────────────────────────────
  UI.input()?.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendMessage(); }
  });

  // ── Keyboard gesture shortcuts ─────────────────────────────
  GestureHandler.enableKeyboardSimulation();

  // ── UI events for jarvis:ui ────────────────────────────────
  document.addEventListener('jarvis:ui', (e) => {
    console.log('[UI Event]', e.detail);
    // Hologram or panel transitions handled by index.html canvas layer
  });

  // ── Status poll ────────────────────────────────────────────
  fetchStatus();
  setInterval(fetchStatus, 15000);

  // ── Boot message ───────────────────────────────────────────
  setMode('standby');
  appendBot(
    'JARVIS 2.0 online. All systems operational. Send a message, speak a command, or use a gesture.',
    { intent: 'system_control', action: { command: 'boot', type: 'system' } }
  );

  console.log('[Main] JARVIS 2.0 frontend loaded. API:', ApiClient.getBase());
}

// ── Helpers ───────────────────────────────────────────────────────────────

function _ts() {
  return new Date().toLocaleTimeString('en', { hour12: false, hour: '2-digit', minute: '2-digit', second: '2-digit' });
}

function _esc(s) {
  return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/\n/g,'<br>');
}

// ── Expose globals for HTML event handlers ────────────────────────────────

window.sendMessage  = sendMessage;
window.toggleVoice  = toggleVoice;
window.sendGesture  = sendGesture;
window.JARVIS       = JARVIS;

// ── Auto-boot ─────────────────────────────────────────────────────────────

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', boot);
} else {
  boot();
}
