"""Детекторы исходов цикла: STABILIZED, OSCILLATION, UNFINISHED (потолок или вырождение).

Чистые функции над списком итераций [{primary, confidence, text}], без API — покрываются тестами.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Any, Dict, List, Optional


def char_trigrams(text: str) -> Counter:
    s = " ".join((text or "").lower().split())
    return Counter(s[i:i + 3] for i in range(max(0, len(s) - 2)))


def cosine_trigrams(a: str, b: str) -> float:
    ca, cb = char_trigrams(a), char_trigrams(b)
    if not ca or not cb:
        return 0.0
    dot = sum(v * cb.get(k, 0) for k, v in ca.items())
    na = math.sqrt(sum(v * v for v in ca.values()))
    nb = math.sqrt(sum(v * v for v in cb.values()))
    return dot / (na * nb) if na and nb else 0.0


def is_stabilized(iters: List[Dict[str, Any]], streak: int = 3, threshold: float = 0.81) -> Optional[str]:
    """Последние `streak` итераций дают один класс (не UND) с уверенностью выше порога. Возвращает класс."""
    if len(iters) < streak:
        return None
    tail = iters[-streak:]
    cls = tail[0]["primary"]
    if cls == "UND":
        return None
    if all(it["primary"] == cls and float(it["confidence"]) > threshold for it in tail):
        return cls
    return None


def find_oscillation(iters: List[Dict[str, Any]], window: int = 6, periods=(2, 3)) -> Optional[Dict[str, Any]]:
    """Петля с периодом p в окне последних `window` классов: c[i] == c[i-p] для всех i, и последовательность не постоянна."""
    if len(iters) < window:
        return None
    seq = [it["primary"] for it in iters[-window:]]
    if len(set(seq)) < 2:
        return None
    for p in periods:
        if all(seq[i] == seq[i - p] for i in range(p, window)):
            return {"period": p, "pattern": seq[-p:]}
    return None


def is_degenerate(prev_text: str, text: str, threshold: float = 0.97) -> Optional[float]:
    c = cosine_trigrams(prev_text, text)
    return c if c > threshold else None


def detect(iters: List[Dict[str, Any]], ccfg: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Итог после очередной итерации или None, если цикл продолжается."""
    cls = is_stabilized(iters, int(ccfg.get("stabilized_streak", 3)), float(ccfg.get("stabilized_confidence", 0.81)))
    if cls:
        return {"outcome": "STABILIZED", "class": cls, "detail": f"{ccfg.get('stabilized_streak', 3)} итерации подряд: {cls}"}
    osc = find_oscillation(iters, int(ccfg.get("oscillation_window", 6)), tuple(ccfg.get("oscillation_periods", [2, 3])))
    if osc:
        return {"outcome": "OSCILLATION", "period": osc["period"], "pattern": osc["pattern"],
                "detail": "петля с периодом %d: %s" % (osc["period"], " → ".join(osc["pattern"]))}
    if len(iters) >= 2:
        c = is_degenerate(iters[-2]["text"], iters[-1]["text"], float(ccfg.get("degeneration_cosine", 0.97)))
        if c is not None:
            return {"outcome": "UNFINISHED", "reason": "degenerate", "cosine": round(c, 4),
                    "detail": f"текст выродился: близость соседних версий {c:.3f}"}
    if len(iters) >= int(ccfg.get("max_iterations", 25)):
        return {"outcome": "UNFINISHED", "reason": "ceiling", "detail": f"достигнут потолок в {len(iters)} итераций"}
    return None
