# -*- coding: utf-8 -*-
"""
CoffeeTech — the service's minimal persistent state.

Keeps on disk the little the model has to remember across runs and restarts. Today that is when
each lab reminder was last sent to each device, so it can go out on a cadence instead of riding
along in every payload.

Standard library only. The file is plain JSON and loading is tolerant: missing or corrupt, it
starts from scratch, and the only cost is sending one reminder twice.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Dict, Optional

logger = logging.getLogger(__name__)

# Path to the state file (not versioned; see .gitignore).
DEFAULT_STATE_FILE = os.getenv("COFFEETECH_STATE_FILE", "coffeetech_state.json")


class ReminderState:
    """Last send date of each reminder, keyed by (reminder, device).

    Indexed by `reminder_id` because reminders have very different cadences (soil analysis ~24
    months, liming ~12): one counter per device would not do."""

    _LEGACY_KEY = "lab_reminder_last_sent"          # previous format: one counter per device
    _LEGACY_REMINDER = "LAB_SOIL_ANALYSIS"          # the reminder that counter belonged to

    def __init__(self, path: Optional[str] = None):
        self.path = path or DEFAULT_STATE_FILE
        # {reminder_id: {device_hub_id: iso8601}}
        self._last_sent: Dict[str, Dict[str, str]] = {}
        self._load()

    def _load(self) -> None:
        try:
            with open(self.path, encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, dict):
                return
            raw = data.get("reminders", {})
            if isinstance(raw, dict):
                self._last_sent = {
                    str(rid): {str(d): str(t) for d, t in devs.items()}
                    for rid, devs in raw.items() if isinstance(devs, dict)
                }
            # Migration from the previous format (a single lab reminder).
            legacy = data.get(self._LEGACY_KEY)
            if isinstance(legacy, dict) and self._LEGACY_REMINDER not in self._last_sent:
                self._last_sent[self._LEGACY_REMINDER] = {str(k): str(v) for k, v in legacy.items()}
        except FileNotFoundError:
            pass
        except Exception as exc:
            logger.warning(f"Estado ilegible en {self.path} ({exc}); se parte de cero.")

    def _save(self) -> None:
        tmp = f"{self.path}.tmp"
        try:
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump({"reminders": self._last_sent}, f, ensure_ascii=False, indent=2)
            os.replace(tmp, self.path)  # atomic: never leaves a half-written JSON
        except Exception as exc:
            logger.error(f"No se pudo persistir el estado en {self.path}: {exc}")

    def last_sent(self, device_hub_id: str, reminder_id: str) -> Optional[datetime]:
        raw = self._last_sent.get(str(reminder_id), {}).get(str(device_hub_id))
        if not raw:
            return None
        try:
            dt = datetime.fromisoformat(raw)
        except ValueError:
            return None
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)

    def is_due(self, device_hub_id: str, reminder_id: str, *, every_days: float,
               now: Optional[datetime] = None) -> bool:
        """True when that reminder has never been sent to this device, or its cadence has elapsed."""
        last = self.last_sent(device_hub_id, reminder_id)
        if last is None:
            return True
        now = now or datetime.now(timezone.utc)
        return (now - last) >= timedelta(days=every_days)

    def mark_sent(self, device_hub_id: str, reminder_id: str,
                  when: Optional[datetime] = None) -> None:
        when = when or datetime.now(timezone.utc)
        self._last_sent.setdefault(str(reminder_id), {})[str(device_hub_id)] = when.isoformat()
        self._save()
