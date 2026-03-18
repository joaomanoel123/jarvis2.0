/**
 * frontend/voiceHandler.js
 * ─────────────────────────────────────────────────────────────
 * Web Speech API wrapper.
 * Captures microphone audio, converts to text, fires callbacks.
 * ─────────────────────────────────────────────────────────────
 */

const VoiceHandler = (() => {

  let _recognition  = null;
  let _active       = false;
  let _onResult     = null;   // (text, confidence) => void
  let _onStateChange = null;  // (state) => void  — 'listening' | 'stopped' | 'error'

  const SR = window.SpeechRecognition || window.webkitSpeechRecognition;

  /** True when the browser supports Web Speech API. */
  function isSupported() { return !!SR; }

  /**
   * Register callbacks before calling start().
   * @param {function} onResult      - Called with (text: string, confidence: number)
   * @param {function} onStateChange - Called with (state: string)
   */
  function onResult(fn)       { _onResult      = fn; }
  function onStateChange(fn)  { _onStateChange = fn; }

  function _emit(state) { if (_onStateChange) _onStateChange(state); }

  /**
   * Start listening.
   * Calls onResult once speech is detected and recognised.
   */
  function start() {
    if (!isSupported()) { console.warn('[VoiceHandler] Not supported'); _emit('error'); return; }
    if (_active) stop();

    _recognition = new SR();
    _recognition.lang            = 'en-US';
    _recognition.interimResults  = false;
    _recognition.maxAlternatives = 1;
    _recognition.continuous      = false;

    _recognition.onstart = () => { _active = true; _emit('listening'); };

    _recognition.onresult = (e) => {
      const transcript = Array.from(e.results)
        .map(r => r[0].transcript)
        .join(' ')
        .trim();
      const confidence = e.results[0]?.[0]?.confidence ?? 1.0;
      console.log(`[VoiceHandler] Heard: "${transcript}" (${(confidence * 100).toFixed(0)}%)`);
      if (_onResult) _onResult(transcript, confidence);
    };

    _recognition.onerror = (e) => {
      console.warn('[VoiceHandler] Error:', e.error);
      _active = false;
      _emit(e.error === 'not-allowed' ? 'permission_denied' : 'error');
    };

    _recognition.onend = () => {
      _active = false;
      _emit('stopped');
    };

    _recognition.start();
  }

  /** Stop the current recognition session. */
  function stop() {
    if (_recognition) { _recognition.abort(); _recognition = null; }
    _active = false;
    _emit('stopped');
  }

  /** Toggle start / stop. Returns new active state. */
  function toggle() {
    if (_active) { stop(); return false; }
    start(); return true;
  }

  /** True while actively listening. */
  function isActive() { return _active; }

  // ── Public API ─────────────────────────────────────────────
  return { isSupported, onResult, onStateChange, start, stop, toggle, isActive };

})();
