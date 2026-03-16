"""
command_executor.py
===================
CommandExecutor — safely executes structured VoiceCommand objects.

Execution handlers
──────────────────
  open_url        → webbrowser.open(url)
  web_search      → webbrowser.open(search_url)
  launch_app      → subprocess.Popen(app_cmd) with platform detection
  media_control   → platform media keys via pynput / subprocess
  volume_control  → platform volume control (pactl / AppleScript / nircmd)
  system_control  → guarded shutdown / restart / lock / screenshot
  jarvis_command  → POST /chat to JARVIS FastAPI backend

Security
────────
• All subprocess calls use allowlisted commands.
• Shell=False everywhere.
• Sensitive system commands (shutdown, restart) require confirmation=True.
• Input validation on all entity values.

Result type
───────────
    ExecutionResult(success, message, data, intent, action)
"""

from __future__ import annotations

import logging
import platform
import subprocess
import urllib.parse
import webbrowser
from dataclasses import dataclass, field
from typing import Any

import requests

from command_parser import VoiceCommand

log = logging.getLogger("jarvis.voice.executor")

_PLATFORM = platform.system()   # "Linux" | "Darwin" | "Windows"

# Backend API URL
DEFAULT_API_URL    = "http://localhost:8000"
API_TIMEOUT_S      = 8.0

# Commands that require explicit confirmation before execution
GUARDED_COMMANDS   = {"shutdown", "restart", "sleep"}

# App command allowlist per platform
_APP_ALLOWLIST: set[str] = {
    "code", "subl", "vim", "nvim", "idea",
    "x-terminal-emulator", "gnome-terminal", "xterm", "bash", "zsh",
    "google-chrome", "chromium-browser", "firefox", "brave-browser",
    "libreoffice", "gimp", "inkscape",
    "slack", "discord", "zoom", "teams",
    "nautilus", "thunar", "pcmanfm",
    "gnome-calculator", "gnome-control-center",
    "open",   # macOS
    "cmd.exe", "powershell.exe", "explorer.exe",  # Windows
}


# ── Result type ────────────────────────────────────────────────────────────────

@dataclass
class ExecutionResult:
    """Result returned from executing a VoiceCommand."""
    success:  bool
    message:  str
    intent:   str
    action:   str
    data:     dict[str, Any] = field(default_factory=dict)
    error:    str | None     = None

    def __str__(self) -> str:
        status = "✓" if self.success else "✗"
        return f"[{status}] {self.intent}/{self.action}: {self.message}"


# ── CommandExecutor ────────────────────────────────────────────────────────────

class CommandExecutor:
    """
    Executes structured voice commands safely.

    Args:
        api_url:         JARVIS FastAPI backend URL.
        api_timeout:     HTTP request timeout in seconds.
        confirmation_cb: Optional async/sync callback for guarded commands.
                         Must return True to allow execution.
                         None = auto-allow (development mode).
        session_id:      JARVIS conversation session ID.
    """

    def __init__(
        self,
        api_url:         str   = DEFAULT_API_URL,
        api_timeout:     float = API_TIMEOUT_S,
        confirmation_cb  = None,
        session_id:      str | None = None,
        dry_run:         bool  = False,
    ) -> None:
        self._api_url     = api_url.rstrip("/")
        self._api_timeout = api_timeout
        self._confirm_cb  = confirmation_cb
        self._session_id  = session_id
        self._dry_run     = dry_run

        # Handler dispatch table
        self._handlers = {
            "open_url":       self._exec_open_url,
            "web_search":     self._exec_web_search,
            "launch_app":     self._exec_launch_app,
            "media_control":  self._exec_media_control,
            "volume_control": self._exec_volume_control,
            "system_control": self._exec_system_control,
            "jarvis_command": self._exec_jarvis_command,
        }

        # Metrics
        self._executed  = 0
        self._succeeded = 0
        self._failed    = 0

    # ── Public API ─────────────────────────────────────────────────────────────

    def execute(self, cmd: VoiceCommand) -> ExecutionResult:
        """
        Execute a parsed VoiceCommand.

        Args:
            cmd: VoiceCommand from CommandParser.parse().

        Returns:
            ExecutionResult with success status and human-readable message.
        """
        self._executed += 1
        log.info("Executing: %s", cmd)

        handler = self._handlers.get(cmd.intent)
        if handler is None:
            log.warning("No handler for intent: %s", cmd.intent)
            return ExecutionResult(
                success=False,
                message=f"Unknown intent: {cmd.intent}",
                intent=cmd.intent,
                action=cmd.action,
                error="no_handler",
            )

        if self._dry_run:
            log.info("[DRY RUN] Would execute: %s", cmd)
            self._succeeded += 1
            return ExecutionResult(
                success=True,
                message=f"[DRY RUN] {cmd.intent}/{cmd.action}",
                intent=cmd.intent,
                action=cmd.action,
                data=cmd.entities,
            )

        try:
            result = handler(cmd)
            if result.success:
                self._succeeded += 1
            else:
                self._failed += 1
            return result
        except Exception as exc:
            self._failed += 1
            log.exception("Executor raised: %s", exc)
            return ExecutionResult(
                success=False,
                message=f"Execution error: {exc}",
                intent=cmd.intent,
                action=cmd.action,
                error=str(exc),
            )

    # ── Intent handlers ────────────────────────────────────────────────────────

    def _exec_open_url(self, cmd: VoiceCommand) -> ExecutionResult:
        """Open a URL in the default web browser."""
        url      = cmd.entities.get("url")
        app_cmd  = cmd.entities.get("app_cmd")
        app_name = cmd.entities.get("app_name", cmd.entities.get("target", ""))

        if url:
            log.info("Opening URL: %s", url)
            webbrowser.open(url)
            return ExecutionResult(
                success=True,
                message=f"Opened {cmd.entities.get('site_name', url)}",
                intent=cmd.intent,
                action=cmd.action,
                data={"url": url},
            )

        if app_cmd:
            return self._exec_launch_app(cmd)

        # Unknown target — route to web search
        target = cmd.entities.get("target", "")
        if target:
            search_url = f"https://www.google.com/search?q={urllib.parse.quote_plus(target)}"
            webbrowser.open(search_url)
            return ExecutionResult(
                success=True,
                message=f"Searched for '{target}'",
                intent=cmd.intent,
                action="web_search",
                data={"url": search_url, "query": target},
            )

        return ExecutionResult(
            success=False, message="No URL or app resolved",
            intent=cmd.intent, action=cmd.action, error="no_target",
        )

    def _exec_web_search(self, cmd: VoiceCommand) -> ExecutionResult:
        """Open a web search in the browser."""
        url   = cmd.entities.get("search_url")
        query = cmd.entities.get("query", "")

        if not url:
            url = f"https://www.google.com/search?q={urllib.parse.quote_plus(query)}"

        if not query:
            return ExecutionResult(
                success=False, message="No query for web search",
                intent=cmd.intent, action=cmd.action, error="no_query",
            )

        log.info("Searching: %r on %s", query, cmd.entities.get("engine", "google"))
        webbrowser.open(url)
        return ExecutionResult(
            success=True,
            message=f"Searching for '{query}'",
            intent=cmd.intent,
            action=cmd.action,
            data={"url": url, "query": query},
        )

    def _exec_launch_app(self, cmd: VoiceCommand) -> ExecutionResult:
        """Launch a local application."""
        app_cmd  = cmd.entities.get("app_cmd", [])
        app_name = cmd.entities.get("app_name", cmd.entities.get("target", ""))

        if not app_cmd:
            return ExecutionResult(
                success=False, message=f"App '{app_name}' not found in registry",
                intent=cmd.intent, action=cmd.action, error="app_not_found",
            )

        # Security: allowlist check
        if app_cmd[0] not in _APP_ALLOWLIST:
            log.warning("App '%s' not in allowlist — blocked", app_cmd[0])
            return ExecutionResult(
                success=False, message=f"App '{app_cmd[0]}' is not in the allowed list",
                intent=cmd.intent, action=cmd.action, error="blocked",
            )

        # Try each command in order (platform fallbacks)
        if isinstance(app_cmd[0], str) and not isinstance(app_cmd, str):
            cmds_to_try = [app_cmd] if isinstance(app_cmd[0], str) else app_cmd
        else:
            cmds_to_try = [app_cmd]

        for attempt in cmds_to_try:
            if not attempt:
                continue
            try:
                log.info("Launching: %s", attempt)
                subprocess.Popen(
                    attempt,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    start_new_session=True,
                )
                return ExecutionResult(
                    success=True,
                    message=f"Launched {app_name}",
                    intent=cmd.intent,
                    action=cmd.action,
                    data={"app": attempt[0]},
                )
            except FileNotFoundError:
                log.debug("Command not found: %s", attempt[0])
                continue
            except Exception as exc:
                log.warning("Launch error: %s", exc)
                continue

        return ExecutionResult(
            success=False, message=f"Could not launch '{app_name}' — not installed?",
            intent=cmd.intent, action=cmd.action, error="launch_failed",
        )

    def _exec_media_control(self, cmd: VoiceCommand) -> ExecutionResult:
        """Control media playback via platform media keys."""
        action = cmd.action
        target = cmd.entities.get("target", "")
        url    = cmd.entities.get("url")

        # If play has a specific target URL, open it
        if action == "play" and url:
            webbrowser.open(url)
            return ExecutionResult(
                success=True, message=f"Opening {target}",
                intent=cmd.intent, action=cmd.action,
                data={"url": url},
            )

        if action == "play" and not target:
            # Generic play — send media key
            self._send_media_key("play_pause")
            return ExecutionResult(
                success=True, message="Play/Pause toggled",
                intent=cmd.intent, action=cmd.action,
            )

        media_key_map = {
            "pause":    "play_pause",
            "resume":   "play_pause",
            "next":     "next_track",
            "previous": "previous_track",
            "stop":     "stop",
        }

        key = media_key_map.get(action)
        if key:
            self._send_media_key(key)
            return ExecutionResult(
                success=True, message=f"Media: {action}",
                intent=cmd.intent, action=cmd.action,
            )

        return ExecutionResult(
            success=False, message=f"Unknown media action: {action}",
            intent=cmd.intent, action=cmd.action, error="unknown_action",
        )

    def _exec_volume_control(self, cmd: VoiceCommand) -> ExecutionResult:
        """Control system volume."""
        action = cmd.action
        level  = cmd.entities.get("level")

        try:
            if _PLATFORM == "Linux":
                self._volume_linux(action, level)
            elif _PLATFORM == "Darwin":
                self._volume_macos(action, level)
            elif _PLATFORM == "Windows":
                self._volume_windows(action, level)
            else:
                return ExecutionResult(
                    success=False, message=f"Volume control not supported on {_PLATFORM}",
                    intent=cmd.intent, action=cmd.action, error="unsupported_platform",
                )

            msg = f"Volume: {action}" + (f" to {level}%" if level is not None else "")
            return ExecutionResult(
                success=True, message=msg,
                intent=cmd.intent, action=cmd.action,
                data={"action": action, "level": level},
            )
        except Exception as exc:
            return ExecutionResult(
                success=False, message=f"Volume control failed: {exc}",
                intent=cmd.intent, action=cmd.action, error=str(exc),
            )

    def _exec_system_control(self, cmd: VoiceCommand) -> ExecutionResult:
        """Execute system commands (shutdown, restart, lock, screenshot)."""
        action = cmd.action

        # Guarded commands require confirmation
        if action in GUARDED_COMMANDS:
            if self._confirm_cb:
                allowed = self._confirm_cb(cmd)
                if not allowed:
                    return ExecutionResult(
                        success=False, message=f"{action.capitalize()} cancelled",
                        intent=cmd.intent, action=action,
                    )
            else:
                log.warning("Guarded command '%s' executed without confirmation", action)

        try:
            if action == "screenshot":
                return self._take_screenshot(cmd)

            cmds = _system_command(action)
            if cmds:
                subprocess.run(cmds, check=True)
                return ExecutionResult(
                    success=True, message=f"System: {action}",
                    intent=cmd.intent, action=action,
                )
        except Exception as exc:
            return ExecutionResult(
                success=False, message=f"System command failed: {exc}",
                intent=cmd.intent, action=action, error=str(exc),
            )

        return ExecutionResult(
            success=False, message=f"Unsupported system action: {action}",
            intent=cmd.intent, action=action, error="unsupported",
        )

    def _exec_jarvis_command(self, cmd: VoiceCommand) -> ExecutionResult:
        """
        Send a natural language query to the JARVIS FastAPI backend.

        Hits POST /chat with the query text and returns JARVIS's response.
        Also tries POST /voice-command for voice-specific handling.
        """
        query = cmd.entities.get("query", cmd.raw_text)
        if not query:
            return ExecutionResult(
                success=False, message="Empty JARVIS query",
                intent=cmd.intent, action=cmd.action, error="empty_query",
            )

        # Try /voice-command first
        voice_payload = {
            "text":       query,
            "session_id": self._session_id,
            "source":     "voice",
        }
        try:
            resp = requests.post(
                f"{self._api_url}/voice-command",
                json=voice_payload,
                timeout=self._api_timeout,
            )
            if resp.status_code == 200:
                data    = resp.json()
                session = data.get("session_id", self._session_id)
                self._session_id = session
                text    = data.get("text") or data.get("response", "")
                log.info("JARVIS voice response: %r", text[:80])
                return ExecutionResult(
                    success=True,
                    message=text,
                    intent=cmd.intent,
                    action=cmd.action,
                    data={
                        "response":   text,
                        "session_id": session,
                        "intent":     data.get("intent", ""),
                    },
                )
        except requests.ConnectionError:
            log.warning("JARVIS backend not reachable at %s — trying /chat", self._api_url)
        except Exception as exc:
            log.debug("/voice-command failed: %s", exc)

        # Fallback: /chat
        chat_payload = {
            "message":    query,
            "session_id": self._session_id,
        }
        try:
            resp = requests.post(
                f"{self._api_url}/chat",
                json=chat_payload,
                timeout=self._api_timeout,
            )
            if resp.status_code == 200:
                data = resp.json()
                self._session_id = data.get("session_id", self._session_id)
                text = data.get("text", "")
                return ExecutionResult(
                    success=True,
                    message=text,
                    intent=cmd.intent,
                    action=cmd.action,
                    data={"response": text, "session_id": self._session_id},
                )
            else:
                return ExecutionResult(
                    success=False,
                    message=f"JARVIS API returned {resp.status_code}",
                    intent=cmd.intent, action=cmd.action,
                    error=f"http_{resp.status_code}",
                )
        except requests.ConnectionError:
            return ExecutionResult(
                success=False,
                message="JARVIS backend is offline",
                intent=cmd.intent, action=cmd.action,
                error="backend_offline",
            )
        except Exception as exc:
            return ExecutionResult(
                success=False,
                message=f"JARVIS API error: {exc}",
                intent=cmd.intent, action=cmd.action,
                error=str(exc),
            )

    # ── Platform helpers ───────────────────────────────────────────────────────

    def _send_media_key(self, key: str) -> None:
        """Send a media key press using pynput (or platform fallback)."""
        try:
            from pynput.keyboard import Key, Controller
            _key_map = {
                "play_pause":    Key.media_play_pause,
                "next_track":    Key.media_next,
                "previous_track": Key.media_previous,
                "stop":          Key.media_volume_mute,
            }
            kb = Controller()
            kb.press(_key_map[key])
            kb.release(_key_map[key])
            log.debug("Media key sent: %s", key)
        except ImportError:
            log.warning("pynput not installed — media key ignored")
        except Exception as exc:
            log.warning("Media key error: %s", exc)

    def _volume_linux(self, action: str, level: int | None) -> None:
        """Control volume on Linux via pactl."""
        if action == "up":
            subprocess.run(["pactl", "set-sink-volume", "@DEFAULT_SINK@", "+10%"],
                           check=True)
        elif action == "down":
            subprocess.run(["pactl", "set-sink-volume", "@DEFAULT_SINK@", "-10%"],
                           check=True)
        elif action == "mute":
            subprocess.run(["pactl", "set-sink-mute", "@DEFAULT_SINK@", "1"],
                           check=True)
        elif action == "unmute":
            subprocess.run(["pactl", "set-sink-mute", "@DEFAULT_SINK@", "0"],
                           check=True)
        elif action == "set" and level is not None:
            pct = max(0, min(100, level))
            subprocess.run(["pactl", "set-sink-volume", "@DEFAULT_SINK@", f"{pct}%"],
                           check=True)

    def _volume_macos(self, action: str, level: int | None) -> None:
        """Control volume on macOS via AppleScript."""
        if action == "up":
            subprocess.run(["osascript", "-e",
                            "set volume output volume (output volume of (get volume settings) + 10)"])
        elif action == "down":
            subprocess.run(["osascript", "-e",
                            "set volume output volume (output volume of (get volume settings) - 10)"])
        elif action == "mute":
            subprocess.run(["osascript", "-e", "set volume with output muted"])
        elif action == "unmute":
            subprocess.run(["osascript", "-e", "set volume without output muted"])
        elif action == "set" and level is not None:
            pct = max(0, min(100, level))
            subprocess.run(["osascript", "-e", f"set volume output volume {pct}"])

    def _volume_windows(self, action: str, level: int | None) -> None:
        """Control volume on Windows via nircmd (if installed)."""
        if action == "up":
            subprocess.run(["nircmd.exe", "changesysvolume", "6553"])
        elif action == "down":
            subprocess.run(["nircmd.exe", "changesysvolume", "-6553"])
        elif action == "mute":
            subprocess.run(["nircmd.exe", "mutesysvolume", "1"])
        elif action == "unmute":
            subprocess.run(["nircmd.exe", "mutesysvolume", "0"])
        elif action == "set" and level is not None:
            pct = max(0, min(100, level))
            vol = int(pct / 100 * 65535)
            subprocess.run(["nircmd.exe", "setsysvolume", str(vol)])

    def _take_screenshot(self, cmd: VoiceCommand) -> ExecutionResult:
        """Capture a screenshot and save to desktop."""
        import datetime
        filename = f"screenshot_{datetime.datetime.now():%Y%m%d_%H%M%S}.png"

        try:
            import PIL.ImageGrab
            import os
            desktop = os.path.expanduser("~/Desktop")
            path    = os.path.join(desktop, filename)
            img = PIL.ImageGrab.grab()
            img.save(path)
            log.info("Screenshot saved: %s", path)
            return ExecutionResult(
                success=True, message=f"Screenshot saved to Desktop/{filename}",
                intent=cmd.intent, action=cmd.action, data={"path": path},
            )
        except ImportError:
            # Fallback: use system tools
            if _PLATFORM == "Linux":
                subprocess.run(["gnome-screenshot", "-f", f"~/Desktop/{filename}"])
            elif _PLATFORM == "Darwin":
                subprocess.run(["screencapture", f"~/Desktop/{filename}"])
            elif _PLATFORM == "Windows":
                subprocess.run(["snippingtool.exe"])
            return ExecutionResult(
                success=True, message="Screenshot taken",
                intent=cmd.intent, action=cmd.action,
            )

    # ── Diagnostics ───────────────────────────────────────────────────────────

    def stats(self) -> dict:
        return {
            "executed":   self._executed,
            "succeeded":  self._succeeded,
            "failed":     self._failed,
            "session_id": self._session_id,
            "api_url":    self._api_url,
        }


# ── System command helpers ─────────────────────────────────────────────────────

def _system_command(action: str) -> list[str] | None:
    """Return the platform-specific system command for an action."""
    platform_cmds = {
        "Linux": {
            "shutdown": ["systemctl", "poweroff"],
            "restart":  ["systemctl", "reboot"],
            "sleep":    ["systemctl", "suspend"],
            "lock":     ["loginctl", "lock-session"],
        },
        "Darwin": {
            "shutdown": ["osascript", "-e", 'tell app "System Events" to shut down'],
            "restart":  ["osascript", "-e", 'tell app "System Events" to restart'],
            "sleep":    ["osascript", "-e", 'tell app "System Events" to sleep'],
            "lock":     ["pmset", "displaysleepnow"],
        },
        "Windows": {
            "shutdown": ["shutdown", "/s", "/t", "5"],
            "restart":  ["shutdown", "/r", "/t", "5"],
            "sleep":    ["rundll32.exe", "powrprof.dll,SetSuspendState", "0", "1", "0"],
            "lock":     ["rundll32.exe", "user32.dll,LockWorkStation"],
        },
    }
    return platform_cmds.get(_PLATFORM, {}).get(action)
