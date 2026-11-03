"""Верификация цитат: проверка, что span действительно есть в тексте, и отбор независимых фрагментов.

Вынесено из enhanced_classifier.py (v2) без изменения логики сопоставления:
  - exact: нормализованное (нижний регистр, без пунктуации) вхождение подстроки;
  - token_overlap: доля токенов span (длиннее 2 символов), найденных в тексте,
    с префиксным матчингом для русских словоформ.
Добавлено: позиция точного вхождения в токенах и отбор независимых (неперекрывающихся) фрагментов.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

_PUNCT_RE = re.compile(r"[^\w\s]", re.UNICODE)
_WS_RE = re.compile(r"\s+")


def normalize(s: str) -> str:
    s = (s or "").lower().replace("ё", "е")
    s = _PUNCT_RE.sub(" ", s)
    return _WS_RE.sub(" ", s).strip()


def tokens(s: str) -> List[str]:
    return normalize(s).split()


def _token_matches(t: str, source_tokens: set) -> bool:
    if t in source_tokens:
        return True
    if len(t) > 5:
        prefix = t[: max(5, len(t) - 3)]
        return any(st.startswith(prefix) for st in source_tokens)
    return False


def locate_exact(span: str, text: str) -> Optional[Tuple[int, int]]:
    """Позиция span в тексте как интервал токенов [start, end) нормализованного текста, если есть точное вхождение."""
    sp, tx = tokens(span), tokens(text)
    n = len(sp)
    if n == 0 or n > len(tx):
        return None
    for i in range(len(tx) - n + 1):
        if tx[i:i + n] == sp:
            return (i, i + n)
    return None


def verify_span(span: str, text: str, threshold: float = 0.7, require_exact: bool = False) -> Dict[str, Any]:
    """Проверка цитаты. Возвращает verified, method, token_overlap, loc (интервал токенов или None)."""
    if not span or not span.strip():
        return {"verified": False, "method": "empty", "token_overlap": 0.0, "loc": None}
    loc = locate_exact(span, text)
    if loc is not None:
        return {"verified": True, "method": "exact", "token_overlap": 1.0, "loc": list(loc)}
    if require_exact:
        return {"verified": False, "method": "exact_required", "token_overlap": 0.0, "loc": None}
    sp = [t for t in tokens(span) if len(t) > 2]
    if not sp:
        return {"verified": False, "method": "too_short", "token_overlap": 0.0, "loc": None}
    src = set(tokens(text))
    overlap = sum(1 for t in sp if _token_matches(t, src)) / len(sp)
    return {
        "verified": overlap >= threshold,
        "method": "token_overlap_with_prefix",
        "token_overlap": round(overlap, 3),
        "loc": None,
    }


def _jaccard(a: List[str], b: List[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def is_independent(span: Dict[str, Any], kept: List[Dict[str, Any]], jaccard_max: float = 0.5) -> bool:
    """Фрагмент независим от уже отобранных, если не перекрывается с ними по позиции
    (точные вхождения) и не совпадает по составу токенов (неточные)."""
    loc = span.get("loc")
    st = tokens(span["span"])
    for k in kept:
        kl = k.get("loc")
        if loc and kl and loc[0] < kl[1] and kl[0] < loc[1]:
            return False
        kt = tokens(k["span"])
        if _jaccard(st, kt) >= jaccard_max:
            return False
        if st and kt and (" ".join(st) in " ".join(kt) or " ".join(kt) in " ".join(st)):
            return False
    return True


def select_independent(spans: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Делит верифицированные фрагменты на независимые и дубликаты (перекрытия). Порядок: как в ответе модели."""
    kept: List[Dict[str, Any]] = []
    dropped: List[Dict[str, Any]] = []
    for s in spans:
        if is_independent(s, kept):
            kept.append(s)
        else:
            dropped.append(s)
    return kept, dropped
