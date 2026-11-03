"""Чтение текста классификатором: валидация свидетельств и агрегация.

Логика скопирована из AI-art pipeline/aggregate.py (validate_theory_response, aggregate_text)
и дополнена числовой уверенностью: доля фрагментов ведущего класса среди всех верифицированных.
"""

from __future__ import annotations

import hashlib
import random
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

from . import CLASSES
from .corpus import count_words
from .verify_quotes import select_independent, verify_span

LEVELS = {"explicit", "scientific_metaphor", "meta_metaphor"}


def validate_theory_response(resp: Optional[Dict[str, Any]], text: str, vcfg: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Все фрагменты ответа с полем kept и drop_reason (unverified — «призрачные цитаты»)."""
    thr = float(vcfg.get("token_overlap_threshold", 0.7))
    require_exact = bool(vcfg.get("require_exact", False))
    min_words = int(vcfg.get("min_span_words", 2))
    require_excl = bool(vcfg.get("require_exclusion_checked", True))
    records: List[Dict[str, Any]] = []
    items = (resp or {}).get("evidence", []) if isinstance(resp, dict) else []
    if not isinstance(items, list):
        items = []
    for item in items:
        if not isinstance(item, dict):
            records.append({"kept": False, "drop_reason": "malformed", "raw": item, "class": None, "span": ""})
            continue
        cls = str(item.get("class", "")).upper().strip()
        span = str(item.get("span", "") or "").strip()
        mapping = item.get("mapping") if isinstance(item.get("mapping"), dict) else None
        rec = {
            "class": cls, "span": span, "mapping": mapping,
            "level": item.get("level") if item.get("level") in LEVELS else None,
            "exclusion_checked": bool(item.get("exclusion_checked", False)),
            "reasoning": item.get("reasoning"),
            "kept": False, "drop_reason": None, "verification": None,
        }
        if cls not in CLASSES:
            rec["drop_reason"] = "invalid_class"
        elif len(span.split()) < min_words:
            rec["drop_reason"] = "span_too_short"
        elif not mapping or not str(mapping.get("source", "")).strip() or not str(mapping.get("target", "")).strip():
            rec["drop_reason"] = "no_mapping"
        elif require_excl and not rec["exclusion_checked"]:
            rec["drop_reason"] = "exclusion_not_checked"
        else:
            v = verify_span(span, text, threshold=thr, require_exact=require_exact)
            rec["verification"] = v
            rec["loc"] = v.get("loc")
            if not v["verified"]:
                rec["drop_reason"] = "unverified"
        records.append(rec)
    for cls in CLASSES:
        cand = [r for r in records if r.get("class") == cls and r["drop_reason"] is None]
        kept, dup = select_independent(cand)
        for r in kept:
            r["kept"] = True
        for r in dup:
            r["drop_reason"] = "duplicate"
    stats = {
        "n_total": len(records),
        "n_kept": sum(1 for r in records if r["kept"]),
        "dropped": dict(Counter(r["drop_reason"] for r in records if not r["kept"])),
    }
    return records, stats


def aggregate(records: List[Dict[str, Any]], text: str) -> Dict[str, Any]:
    """primary, n_spans по классам, confidence = доля фрагментов primary среди всех kept."""
    kept = [r for r in records if r["kept"]]
    n_spans = {c: sum(1 for r in kept if r["class"] == c) for c in CLASSES}
    total = sum(n_spans.values())
    n_words = count_words(text)
    if total == 0:
        return {"primary": "UND", "confidence": 0.0, "n_spans": n_spans, "n_kept": 0, "n_words": n_words, "primary_tie": False}
    best = max(n_spans.values())
    tied = [c for c in CLASSES if n_spans[c] == best]
    primary = tied[0]
    if len(tied) > 1:
        # при равенстве — по покрытию текста фрагментами (как в AI-art); при полном равенстве —
        # случайно с зерном от текста, а не по порядку списка (иначе COMP всегда впереди)
        def coverage(c: str) -> int:
            return sum(len(r["span"].split()) for r in kept if r["class"] == c)
        best_cov = max(coverage(c) for c in tied)
        tied = [c for c in tied if coverage(c) == best_cov]
        rng = random.Random(hashlib.sha256(text.encode("utf-8")).hexdigest())
        primary = tied[0] if len(tied) == 1 else rng.choice(tied)
    return {
        "primary": primary,
        "confidence": round(n_spans[primary] / total, 3),
        "n_spans": n_spans,
        "n_kept": total,
        "n_words": n_words,
        "primary_tie": len(tied) > 1,
    }
