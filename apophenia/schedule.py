"""Часы работы галереи и ритм циклов."""

from __future__ import annotations

from datetime import datetime, time, timedelta
from typing import Any, Dict, Optional


def _parse(t: str) -> time:
    h, m = t.split(":")
    return time(int(h), int(m))


class Schedule:
    def __init__(self, cfg: Dict[str, Any]):
        s = cfg.get("schedule", {})
        self.always = bool(s.get("always", True))   # true — часов работы нет, циклы идут, пока включён компьютер
        self.open = _parse(s.get("open", "11:00"))
        self.close = _parse(s.get("close", "20:00"))
        self.days = set(int(d) for d in s.get("days", [1, 2, 3, 4, 5, 6, 7]))
        self.interval = timedelta(minutes=max(0.0, float(s.get("interval_minutes", 60))))

    def is_open(self, now: Optional[datetime] = None) -> bool:
        if self.always:
            return True
        now = now or datetime.now()
        if now.isoweekday() not in self.days:
            return False
        return self.open <= now.time() < self.close

    def next_open(self, now: Optional[datetime] = None) -> datetime:
        now = now or datetime.now()
        if self.always:
            return now
        for d in range(0, 8):
            day = (now + timedelta(days=d)).date()
            if day.isoweekday() not in self.days:
                continue
            start = datetime.combine(day, self.open)
            if start > now:
                return start
        return now + timedelta(days=1)

    def seconds_until_open(self, now: Optional[datetime] = None) -> float:
        now = now or datetime.now()
        if self.is_open(now):
            return 0.0
        return max(0.0, (self.next_open(now) - now).total_seconds())
