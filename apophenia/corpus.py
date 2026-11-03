"""Корпус кураторских текстов (копия combined_ai_preprocessed.xlsx из AI-art)."""

from __future__ import annotations

import json
import random
import re
from typing import Any, Dict, List, Optional

import pandas as pd

from .config import resolve

_WORD_RE = re.compile(r"[\w'’-]+", re.UNICODE)


def clean_text(s: Any) -> str:
    if s is None or (isinstance(s, float) and pd.isna(s)):
        return ""
    s = str(s).replace("_x000D_", " ").replace("\r", " ")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\s*\n\s*", "\n", s)
    return s.strip()


def count_words(text: str) -> int:
    return len(_WORD_RE.findall(text or ""))


def load_corpus(cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    """[{id, source, title, author, text, n_words}], id — позиция строки в xlsx, как в AI-art."""
    c = cfg["corpus"]
    df = pd.read_excel(resolve(cfg, c["path"]), sheet_name=c.get("sheet", 0))
    text_col = c.get("text_column", "descr_clean")
    fb_col = c.get("fallback_text_column", "descr")
    excluded = set()
    ex_path = resolve(cfg, c.get("exclude_file", "")) if c.get("exclude_file") else None
    if ex_path and ex_path.exists():
        excluded = set(json.loads(ex_path.read_text(encoding="utf-8")).get("excluded_ids", []))
    out = []
    for i, row in df.iterrows():
        t = clean_text(row.get(text_col)) if text_col in df.columns else ""
        if not t and fb_col in df.columns:
            t = clean_text(row.get(fb_col))
        n = count_words(t)
        if n < int(c.get("min_words", 0)) or int(i) in excluded:
            continue
        out.append({
            "id": int(i),
            "source": str(row.get(c.get("source_column", "__source"), "")),
            "title": clean_text(row.get(c.get("title_column", "title"))) or f"Текст {i}",
            "author": clean_text(row.get(c.get("author_column", "descr_author"))),
            "text": t,
            "n_words": n,
        })
    return out


def pick_texts(corpus: List[Dict[str, Any]], k: int, seed: Optional[int] = None) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    return rng.sample(corpus, min(k, len(corpus)))
