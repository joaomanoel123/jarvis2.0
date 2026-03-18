/**
 * frontend/commandExecutor.js
 * ─────────────────────────────────────────────────────────────
 * Receives a structured action from the backend and executes it.
 * Dispatches CustomEvents on document for any UI layer to listen.
 * ─────────────────────────────────────────────────────────────
 */

const CommandExecutor = (() => {

  const _log = [];

  // ── Main dispatch ──────────────────────────────────────────

  function executeCommand(action) {
    if (!action || !action.command) return { success: false, result: 'No action' };

    const map = {
      browser:           _browser,
      ui:                _ui,
      media:             _media,
      system:            _system,
      notification:      _notification,
      gesture_execution: _gestureExec,
    };

    const handler = map[action.type] || _unknown;
    const result  = handler(action.command, action.parameters || {});

    _log.unshift({ ts: Date.now(), ...action, result: result.result });
    if (_log.length > 50) _log.pop();
    console.log(`[Executor] ${action.type}::${action.command}`, '→', result.result);
    return result;
  }

  // ── Browser ────────────────────────────────────────────────

  function _browser(cmd, p) {
    if (cmd === 'open_url') {
      const url = _proto(p.url || '');
      if (!url) return { success: false, result: 'No URL' };
      window.open(url, '_blank', 'noopener,noreferrer');
      return { success: true, result: `Opened: ${url}` };
    }
    if (cmd === 'search') {
      const q = encodeURIComponent(p.query || '');
      const engines = { google: `https://www.google.com/search?q=${q}`, youtube: `https://www.youtube.com/results?search_query=${q}` };
      window.open(engines[p.engine] || engines.google, '_blank', 'noopener,noreferrer');
      return { success: true, result: `Search: ${decodeURIComponent(q)}` };
    }
    if (cmd === 'navigate_back')    { window.history.back();    return { success: true, result: 'Back' }; }
    if (cmd === 'navigate_forward') { window.history.forward(); return { success: true, result: 'Forward' }; }
    return _unknown(cmd, p);
  }

  // ── UI ─────────────────────────────────────────────────────

  function _ui(cmd, p) {
    const uiCmds = ['next_screen','previous_screen','open_menu','close_menu','zoom_in','zoom_out','scroll_top','scroll_bottom','reset','rotate_right','rotate_left','select_object','grab_element','release_element'];
    if (uiCmds.includes(cmd)) {
      document.dispatchEvent(new CustomEvent('jarvis:ui', { detail: { command: cmd, ...p } }));
      if (cmd === 'scroll_top')    window.scrollTo({ top: 0,                        behavior: 'smooth' });
      if (cmd === 'scroll_bottom') window.scrollTo({ top: document.body.scrollHeight, behavior: 'smooth' });
      return { success: true, result: cmd };
    }
    document.dispatchEvent(new CustomEvent('jarvis:ui', { detail: { command: cmd, ...p } }));
    return { success: true, result: `UI: ${cmd}` };
  }

  // ── Media ──────────────────────────────────────────────────

  function _media(cmd, p) {
    const el = document.querySelector('video, audio');
    if (cmd === 'play')   { el?.play?.(); }
    if (cmd === 'pause')  { el?.pause?.(); }
    if (cmd === 'stop')   { if (el) { el.pause(); el.currentTime = 0; } }
    if (cmd === 'set_volume' && el) { el.volume = Math.max(0, Math.min(1, (p.level || 50) / 100)); }
    document.dispatchEvent(new CustomEvent('jarvis:media', { detail: { command: cmd, ...p } }));
    return { success: true, result: `Media: ${cmd}` };
  }

  // ── System ─────────────────────────────────────────────────

  function _system(cmd, p) {
    if (cmd === 'fullscreen') {
      !document.fullscreenElement ? document.documentElement.requestFullscreen().catch(()=>{}) : document.exitFullscreen().catch(()=>{});
      return { success: true, result: 'Fullscreen toggled' };
    }
    if (cmd === 'copy' && p.text) { navigator.clipboard?.writeText(p.text).catch(()=>{}); return { success: true, result: 'Copied' }; }
    document.dispatchEvent(new CustomEvent('jarvis:system', { detail: { command: cmd, ...p } }));
    return { success: true, result: `System: ${cmd}` };
  }

  // ── Notification ───────────────────────────────────────────

  function _notification(cmd, p) {
    const msg = p.message || p.text || '';
    if (cmd === 'show_toast') { _toast(msg, p.level || 'info', p.duration || 3000); }
    else document.dispatchEvent(new CustomEvent('jarvis:notification', { detail: { command: cmd, message: msg, level: p.level } }));
    return { success: true, result: `Notify: ${msg}` };
  }

  // ── Gesture exec ───────────────────────────────────────────

  function _gestureExec(cmd, p) {
    const routes = {
      next_screen:      () => _ui('next_screen', {}),
      previous_screen:  () => _ui('previous_screen', {}),
      open_menu:        () => _ui('open_menu', {}),
      close_menu:       () => _ui('close_menu', {}),
      zoom_in:          () => _ui('zoom_in', {}),
      zoom_out:         () => _ui('zoom_out', {}),
      select_object:    () => _ui('select_object', {}),
      grab_element:     () => _ui('grab_element', {}),
      release_element:  () => _ui('release_element', {}),
      scroll_top:       () => _ui('scroll_top', {}),
      scroll_bottom:    () => _ui('scroll_bottom', {}),
      rotate_right:     () => _ui('rotate_right', {}),
      rotate_left:      () => _ui('rotate_left', {}),
      zoom_interface:   () => _ui('zoom_in', {}),
      shrink_interface: () => _ui('zoom_out', {}),
      navigate_right:   () => _ui('next_screen', {}),
      navigate_left:    () => _ui('previous_screen', {}),
    };
    const route = routes[cmd];
    if (route) return route();
    document.dispatchEvent(new CustomEvent('jarvis:gesture', { detail: { command: cmd, ...p } }));
    return { success: true, result: `Gesture: ${cmd}` };
  }

  // ── Unknown ────────────────────────────────────────────────

  function _unknown(cmd, p) {
    console.warn('[Executor] Unknown:', cmd, p);
    document.dispatchEvent(new CustomEvent('jarvis:unknown', { detail: { command: cmd, params: p } }));
    return { success: false, result: `Unknown: ${cmd}` };
  }

  // ── Toast ──────────────────────────────────────────────────

  function _toast(msg, level = 'info', ms = 3000) {
    const colors = { info: '#00e5ff', success: '#00e676', warning: '#ffd600', error: '#ff1744' };
    const t = document.createElement('div');
    Object.assign(t.style, {
      position: 'fixed', bottom: '90px', left: '50%', transform: 'translateX(-50%)',
      background: 'rgba(0,4,12,0.95)', border: `1px solid ${colors[level]||colors.info}`,
      color: colors[level]||colors.info, fontFamily: '"Share Tech Mono",monospace',
      fontSize: '11px', letterSpacing: '1px', padding: '8px 20px', zIndex: '9999',
      pointerEvents: 'none', opacity: '0', transition: 'opacity 0.2s ease',
    });
    t.textContent = msg;
    document.body.appendChild(t);
    requestAnimationFrame(() => { t.style.opacity = '1'; });
    setTimeout(() => { t.style.opacity = '0'; setTimeout(() => t.remove(), 250); }, ms);
  }

  // ── URL helper ─────────────────────────────────────────────

  function _proto(url) {
    if (!url) return '';
    if (/^https?:\/\//i.test(url)) return url;
    if (url.includes('.')) return `https://${url}`;
    return `https://www.google.com/search?q=${encodeURIComponent(url)}`;
  }

  // ── Public API ─────────────────────────────────────────────
  return { executeCommand, toast: _toast, getLog: () => [..._log] };

})();
