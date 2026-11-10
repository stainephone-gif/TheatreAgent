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


def shares_run(span: str, text: str, n: int = 4, min_content: int = 0) -> bool:
    """Есть ли у цитаты и текста общая цепочка из n слов подряд (после нормализации).
    min_content — сколько слов длиннее трёх букв должно быть в цепочке: отсекает совпадения из одних служебных слов."""
    sp, tx = tokens(span), tokens(text)
    if n <= 0 or len(sp) < n or len(tx) < n:
        return False
    grams = {tuple(tx[i:i + n]) for i in range(len(tx) - n + 1)}
    for i in range(len(sp) - n + 1):
        g = tuple(sp[i:i + n])
        if g in grams and sum(len(w) > 3 for w in g) >= min_content:
            return True
    return False


def content_overlap(span: str, text: str) -> Tuple[float, int]:
    """(доля, число): сколько знаменательных слов цитаты (длиннее трёх букв) есть в тексте, с учётом словоформ."""
    words = [w for w in tokens(span) if len(w) > 3]
    if not words:
        return 0.0, 0
    src = set(tokens(text))
    return sum(_token_matches(w, src) for w in words) / len(words), len(words)


_NOT_QUOTE_RE = re.compile(r"[\[\]{}<>]|пересказ|не цитата|по памяти|примерно так|что-то вроде|дословно не помн", re.I)


def ghost_defect(span: str, seen: str, gcfg: Dict[str, Any]) -> Optional[str]:
    """Почему непроверенная «цитата» не годится в призраки: not_a_quote — пометка или пересказ вместо фразы;
    codebook_echo — формулировка промпта читателя, выданная за цитату. None — годится.

    ghosts.codebook_echo_words: N — общая цепочка из N слов подряд, из них не меньше двух знаменательных;
    ghosts.codebook_echo_overlap: доля знаменательных слов цитаты, найденных в промпте (при трёх и более словах)."""
    if _NOT_QUOTE_RE.search(span or ""):
        return "not_a_quote"
    n = int(gcfg.get("codebook_echo_words", 0) or 0)
    if n and shares_run(span, seen, n, min_content=min(2, n)):
        return "codebook_echo"
    thr = float(gcfg.get("codebook_echo_overlap", 0) or 0)
    if thr:
        share, count = content_overlap(span, seen)
        if count >= 3 and share >= thr:
            return "codebook_echo"
    return None
