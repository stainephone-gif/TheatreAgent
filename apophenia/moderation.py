"""Модерация текста до печати и до экрана: стоп-список и проверка моделью."""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional

from .config import resolve
from .llm import BaseLLM, LLMRefusal, extract_json
from .prompts import Prompt

log = logging.getLogger("apophenia.moderation")


class Stoplist:
    def __init__(self, entries: List[str]):
        self.patterns: List[re.Pattern] = []
        for e in entries:
            e = e.strip()
            if not e or e.startswith("#"):
                continue
            if e.startswith("re:"):
                self.patterns.append(re.compile(e[3:], re.I | re.U))
            else:
                # слово целиком с русским окончанием до 4 букв: «путин» → «путина», «крым» → «крымский»
                self.patterns.append(re.compile(r"\b" + re.escape(e.lower()) + r"[а-яё]{0,4}\b", re.I | re.U))

    @classmethod
    def load(cls, cfg: Dict[str, Any]) -> "Stoplist":
        p = cfg.get("moderation", {}).get("stoplist")
        path = resolve(cfg, p) if p else None
        if not path or not path.exists():
            return cls([])
        return cls(path.read_text(encoding="utf-8").splitlines())

    def hits(self, text: str) -> List[str]:
        t = (text or "").replace("ё", "е").replace("Ё", "Е")
        return [m.group(0) for p in self.patterns for m in [p.search(t)] if m]


class Moderator:
    def __init__(self, cfg: Dict[str, Any], llm: BaseLLM, prompt: Optional[Prompt]):
        m = cfg.get("moderation", {})
        self.enabled = bool(m.get("enabled", True))
        self.llm_check = bool(m.get("llm_check", True)) and prompt is not None
        self.stoplist = Stoplist.load(cfg)
        self.llm = llm
        self.prompt = prompt
        self.temperature = float(cfg["llm"].get("temperatures", {}).get("moderate", 0.0))

    def check(self, text: str) -> Dict[str, Any]:
        """{ok, stoplist_hits, llm: {flagged, topics, reason} | None}."""
        if not self.enabled:
            return {"ok": True, "stoplist_hits": [], "llm": None}
        hits = self.stoplist.hits(text)
        result: Dict[str, Any] = {"ok": not hits, "stoplist_hits": hits, "llm": None}
        if hits:
            log.warning("Стоп-список: %s", hits)
            return result
        if self.llm_check:
            try:
                resp = self.llm.complete("moderate", self.prompt.system, self.prompt.user(text=text), self.temperature)
                data = extract_json(resp.content) or {}
            except LLMRefusal as e:
                # цензура провайдера сработала на самом тексте — считаем помеченным
                data = {"flagged": True, "topics": ["provider_blacklist"], "reason": str(e)}
            flagged = bool(data.get("flagged", False))
            result["llm"] = {"flagged": flagged, "topics": data.get("topics", []), "reason": data.get("reason", "")}
            result["ok"] = not flagged
            if flagged:
                log.warning("Модерация моделью: %s %s", data.get("topics"), data.get("reason"))
        return result
