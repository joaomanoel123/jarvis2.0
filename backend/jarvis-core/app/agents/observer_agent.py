"""
app/agents/observer_agent.py
Monitors system performance, records metrics, and detects error patterns.
"""

from __future__ import annotations

import time
from collections import defaultdict, deque

from app.memory.database import Database
from app.utils.logger    import get_logger

log = get_logger("jarvis.agent.observer")


class ObserverAgent:
    """
    Passively monitors every request:
      - latency per intent
      - error rate per session
      - command success rate
      - anomaly detection (latency spikes)

    Writes observations to the database for historical analysis.
    """

    def __init__(self, db: Database) -> None:
        self._db          = db
        self._latencies   = defaultdict(lambda: deque(maxlen=100))
        self._errors      = defaultdict(int)
        self._total       = 0
        self._error_count = 0

    async def record(self, intent: str, latency_ms: float, success: bool) -> None:
        """Record one request observation."""
        self._total += 1
        self._latencies[intent].append(latency_ms)

        if not success:
            self._error_count += 1
            self._errors[intent] += 1

        # Persist to DB (fire-and-forget — don't block the response)
        try:
            await self._db.record_observation(
                metric=f"latency.{intent}",
                value=latency_ms,
                meta={"success": success},
            )
        except Exception:
            pass

        # Anomaly: latency spike (> 3× rolling average)
        window = list(self._latencies[intent])
        if len(window) >= 5:
            avg = sum(window[:-1]) / (len(window) - 1)
            if latency_ms > avg * 3 and latency_ms > 2000:
                log.warning("Latency anomaly: intent=%s  %.0f ms (avg %.0f ms)",
                            intent, latency_ms, avg)

    def metrics(self) -> dict:
        """Return in-process metrics summary."""
        result = {"total": self._total, "errors": self._error_count, "by_intent": {}}
        for intent, lats in self._latencies.items():
            window = list(lats)
            result["by_intent"][intent] = {
                "count":   len(window),
                "avg_ms":  round(sum(window) / len(window), 1) if window else 0,
                "max_ms":  round(max(window), 1) if window else 0,
                "errors":  self._errors.get(intent, 0),
            }
        return result
