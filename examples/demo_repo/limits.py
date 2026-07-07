from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass
class SlidingWindowLimiter:
    """Track recent events and allow only a fixed number per window."""

    max_events: int
    window_seconds: int
    events: dict[str, list[float]] = field(default_factory=dict)

    def allow(self, key: str) -> bool:
        now = time.time()
        floor = now - self.window_seconds
        recent = [ts for ts in self.events.get(key, []) if ts >= floor]
        if len(recent) >= self.max_events:
            self.events[key] = recent
            return False
        recent.append(now)
        self.events[key] = recent
        return True


def client_key(user_id: str, ip_address: str) -> str:
    """Build the rate-limit key used around login."""
    return f"{user_id}:{ip_address}"
