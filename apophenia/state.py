"""Состояние службы на диске: сквозной номер цикла и данные для экрана."""

from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

from .config import path_dir


def _atomic_write(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=".tmp_", suffix=".json")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=1)
    os.replace(tmp, path)


class State:
    def __init__(self, cfg: Dict[str, Any]):
        self.dir = path_dir(cfg, "state")
        self.path = self.dir / "state.json"
        self.display_path = self.dir / "display.json"
        self.data: Dict[str, Any] = {"last_cycle": 0, "started_at": None}
        if self.path.exists():
            try:
                self.data.update(json.loads(self.path.read_text(encoding="utf-8")))
            except json.JSONDecodeError:
                pass

    def next_cycle_number(self) -> int:
        self.data["last_cycle"] = int(self.data.get("last_cycle", 0)) + 1
        self.save()
        return self.data["last_cycle"]

    def rollback_cycle_number(self) -> None:
        """Цикл отброшен модерацией до печати: номер не расходуется."""
        self.data["last_cycle"] = max(0, int(self.data.get("last_cycle", 0)) - 1)
        self.save()

    def save(self) -> None:
        _atomic_write(self.path, self.data)

    def set_display(self, **fields: Any) -> None:
        cur: Dict[str, Any] = {}
        if self.display_path.exists():
            try:
                cur = json.loads(self.display_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                cur = {}
        cur.update(fields)
        cur["updated_at"] = datetime.now().isoformat(timespec="seconds")
        _atomic_write(self.display_path, cur)
