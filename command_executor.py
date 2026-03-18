"""
backend/command_executor.py
Maps parsed commands to structured action payloads the frontend will execute.
The backend NEVER opens URLs directly — it only tells the frontend what to do.
"""

from dataclasses import dataclass, field


@dataclass
class ActionPayload:
    """Structured action sent to the frontend."""
    type:       str
    command:    str
    parameters: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "type":       self.type,
            "command":    self.command,
            "parameters": self.parameters,
        }


@dataclass
class CommandResult:
    """Full result returned by the executor."""
    success:  bool
    response: str
    action:   ActionPayload | None = None

    def action_dict(self) -> dict:
        return self.action.to_dict() if self.action else {}


# ── Gesture command map ────────────────────────────────────────────────
_GESTURE_MAP: dict[str, dict] = {
    "swipe_right":           {"type": "ui",    "command": "next_screen"},
    "swipe_left":            {"type": "ui",    "command": "previous_screen"},
    "swipe_up":              {"type": "ui",    "command": "scroll_top"},
    "swipe_down":            {"type": "ui",    "command": "scroll_bottom"},
    "open_hand":             {"type": "ui",    "command": "open_menu"},
    "fist":                  {"type": "ui",    "command": "close_menu"},
    "pinch":                 {"type": "ui",    "command": "select_object"},
    "grab":                  {"type": "ui",    "command": "grab_element"},
    "release":               {"type": "ui",    "command": "release_element"},
    "zoom_in":               {"type": "ui",    "command": "zoom_in"},
    "zoom_out":              {"type": "ui",    "command": "zoom_out"},
    "thumbs_up":             {"type": "notification", "command": "show_toast",
                              "parameters": {"message": "Confirmed", "level": "success"}},
    "thumbs_down":           {"type": "notification", "command": "show_toast",
                              "parameters": {"message": "Rejected", "level": "error"}},
    "wave":                  {"type": "system", "command": "wake"},
    "circle_clockwise":      {"type": "ui",    "command": "rotate_right"},
    "circle_counterclockwise": {"type": "ui", "command": "rotate_left"},
}

_GESTURE_LABELS: dict[str, str] = {
    "swipe_right":           "Navigating right",
    "swipe_left":            "Navigating left",
    "swipe_up":              "Scrolling up",
    "swipe_down":            "Scrolling down",
    "open_hand":             "Menu activated",
    "fist":                  "Menu closed",
    "pinch":                 "Object selected",
    "grab":                  "Element grabbed",
    "release":               "Element released",
    "zoom_in":               "Zooming in",
    "zoom_out":              "Zooming out",
    "thumbs_up":             "Confirmed",
    "thumbs_down":           "Rejected",
    "wave":                  "JARVIS activated",
    "circle_clockwise":      "Rotating right",
    "circle_counterclockwise": "Rotating left",
}


class CommandExecutor:
    """
    Converts intent + parameters into a structured ActionPayload.

    Supported commands:
        open_url, search, set_volume, gesture_execution, system_action
    """

    def execute(self, command: str, parameters: dict) -> CommandResult:
        """
        Dispatch to the correct handler.

        Args:
            command:    Command name from intent_detector
            parameters: Extracted parameters

        Returns:
            CommandResult
        """
        handlers = {
            "open_url":          self._open_url,
            "search":            self._search,
            "set_volume":        self._set_volume,
            "gesture_execution": self._gesture,
            "system_action":     self._system,
        }

        handler = handlers.get(command, self._unknown)
        return handler(parameters)

    # ── Handlers ───────────────────────────────────────────────────────

    def _open_url(self, params: dict) -> CommandResult:
        url  = params.get("url", "")
        site = params.get("site", url)
        if not url:
            return CommandResult(success=False, response="No URL specified.")
        return CommandResult(
            success=True,
            response=f"Opening {site}.",
            action=ActionPayload(
                type="browser",
                command="open_url",
                parameters={"url": url},
            ),
        )

    def _search(self, params: dict) -> CommandResult:
        query  = params.get("query", "")
        engine = params.get("engine", "google")
        if not query:
            return CommandResult(success=False, response="No search query provided.")

        engines = {
            "google":  f"https://www.google.com/search?q={query.replace(' ', '+')}",
            "youtube": f"https://www.youtube.com/results?search_query={query.replace(' ', '+')}",
        }
        url = engines.get(engine, engines["google"])

        return CommandResult(
            success=True,
            response=f"Searching for '{query}' on {engine}.",
            action=ActionPayload(
                type="browser",
                command="open_url",
                parameters={"url": url},
            ),
        )

    def _set_volume(self, params: dict) -> CommandResult:
        level = params.get("level", 50)
        return CommandResult(
            success=True,
            response=f"Volume set to {level}%.",
            action=ActionPayload(
                type="media",
                command="set_volume",
                parameters={"level": level},
            ),
        )

    def _gesture(self, params: dict) -> CommandResult:
        gesture = params.get("gesture", "")
        mapping = _GESTURE_MAP.get(gesture)
        label   = _GESTURE_LABELS.get(gesture, f"Gesture: {gesture}")

        if not mapping:
            return CommandResult(
                success=False,
                response=f"Gesture '{gesture}' not mapped.",
            )

        action_params = dict(mapping.get("parameters", {}))
        return CommandResult(
            success=True,
            response=label,
            action=ActionPayload(
                type="gesture_execution",
                command=mapping["command"],
                parameters=action_params,
            ),
        )

    def _system(self, params: dict) -> CommandResult:
        instruction = params.get("instruction", "")
        return CommandResult(
            success=True,
            response=f"System: {instruction}",
            action=ActionPayload(
                type="system",
                command="system_action",
                parameters={"instruction": instruction},
            ),
        )

    def _unknown(self, params: dict) -> CommandResult:
        return CommandResult(
            success=False,
            response="I didn't understand that command.",
        )


# Module singleton
command_executor = CommandExecutor()
