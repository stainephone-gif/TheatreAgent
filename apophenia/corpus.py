"""Корпус: таблица xlsx (кураторские тексты AI-art) или папка txt-файлов, нарезанных на фрагменты (проза)."""

from __future__ import annotations

import json
import random
import re
from pathlib import Path
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


def split_chunks(text: str, target_words: int, max_words: int) -> List[str]:
    """Режет прозу по абзацам на фрагменты около target_words слов; абзац длиннее max_words режется по предложениям."""
    paras = [p.strip() for p in re.split(r"\n\s*\n|\n", text) if p.strip()]
    units: List[str] = []
    for p in paras:
        if count_words(p) <= max_words:
            units.append(p)
        else:
            units.extend(s.strip() for s in re.split(r"(?<=[.!?…»])\s+", p) if s.strip())
    chunks, cur, n = [], [], 0
    for u in units:
        w = count_words(u)
        if cur and n + w > max_words:
            chunks.append("\n".join(cur))
            cur, n = [], 0
        cur.append(u)
        n += w
        if n >= target_words:
            chunks.append("\n".join(cur))
            cur, n = [], 0
    if cur:
        if chunks and n < target_words // 3:
            chunks[-1] += "\n" + "\n".join(cur)   # короткий хвост приклеивается к предыдущему
        else:
            chunks.append("\n".join(cur))
    return chunks


def load_txt_corpus(cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Папка с *.txt (одна книга или рассказ на файл, UTF-8). Каждый файл режется на фрагменты;
    title = имя файла без расширения + номер фрагмента, author = corpus.author. id сквозной по порядку."""
    c = cfg["corpus"]
    d = resolve(cfg, c["dir"])
    if not d.is_dir():
        raise FileNotFoundError(f"Папка корпуса не найдена: {d} (положите туда txt-файлы, см. README)")
    target = int(c.get("chunk_words", 300))
    max_w = int(c.get("chunk_max_words", target * 2))
    min_w = int(c.get("min_words", 0))
    author = str(c.get("author", ""))
    excluded = set()
    ex_path = resolve(cfg, c.get("exclude_file", "")) if c.get("exclude_file") else None
    if ex_path and ex_path.exists():
        excluded = set(json.loads(ex_path.read_text(encoding="utf-8")).get("excluded_ids", []))
    out, i = [], 0
    for f in sorted(d.glob("*.txt")):
        raw = clean_text(f.read_text(encoding="utf-8-sig"))
        title = f.stem.replace("_", " ")
        for k, chunk in enumerate(split_chunks(raw, target, max_w), start=1):
            n = count_words(chunk)
            if n >= min_w and i not in excluded:
                out.append({"id": i, "source": f.name, "title": f"{title}, фрагмент {k}", "author": author,
                            "text": chunk, "n_words": n})
            i += 1
    return out


def load_corpus(cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    """[{id, source, title, author, text, n_words}], id — позиция строки в xlsx, как в AI-art."""
    c = cfg["corpus"]
    if c.get("format", "xlsx") == "txt":
        return load_txt_corpus(cfg)
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
