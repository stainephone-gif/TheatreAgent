#!/usr/bin/env python3
"""Готовит книгу «Компромисс» для корпуса: режет txt на вступление и двенадцать новелл.

    python tools/prepare_kompromiss.py путь/к/Компромисс.txt [--out data/kompromiss]

Кодировка определяется сама (UTF-8 или Windows-1251). Выбрасываются титульные страницы и копирайт (всё до строки
«* * *» перед первой новеллой), послесловие и примечания (с строки «Послесловие»). Результат:
00 Вступление.txt, 01 Компромисс первый.txt … 12 Компромисс двенадцатый.txt — в UTF-8, по абзацу в строке.
Папка вывода в git не попадает: текст охраняется авторским правом.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import List, Tuple

ORDINALS = ["первый", "второй", "третий", "четвертый", "пятый", "шестой", "седьмой", "восьмой",
            "девятый", "десятый", "одиннадцатый", "двенадцатый"]
HEAD_RE = re.compile(r"^Компромисс\s+(" + "|".join(ORDINALS) + r")$", re.I)
END_RE = re.compile(r"^(Послесловие|ПОСЛЕСЛОВИЕ|notes|Примечания)$")


def read_text(path: Path) -> str:
    raw = path.read_bytes()
    for enc in ("utf-8-sig", "cp1251"):
        try:
            text = raw.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    else:
        raise SystemExit(f"Не удалось определить кодировку {path}")
    return text.replace("\r\n", "\n").replace("\r", "\n")


def split_book(text: str) -> List[Tuple[str, str]]:
    """[(имя файла без .txt, текст)]: вступление и новеллы по порядку."""
    lines = [ln.rstrip() for ln in text.split("\n")]
    heads = [i for i, ln in enumerate(lines) if HEAD_RE.match(ln.strip().replace("ё", "е"))]
    if len(heads) != 12:
        found = ", ".join(lines[i].strip() for i in heads) or "ни одного"
        raise SystemExit(f"Ожидалось 12 заголовков «Компромисс первый…двенадцатый», найдено {len(heads)}: {found}")
    end = next((i for i in range(heads[-1], len(lines)) if END_RE.match(lines[i].strip())), len(lines))
    start = next((i + 1 for i in range(heads[0] - 1, -1, -1) if lines[i].strip() == "* * *"), heads[0])
    bounds = [start] + heads + [end]

    def para(a: int, b: int) -> str:
        return "\n".join(ln.strip() for ln in lines[a:b] if ln.strip())

    out = [("00 Вступление", para(bounds[0], bounds[1]))]
    for n, h in enumerate(heads, start=1):
        title = "Компромисс " + ORDINALS[n - 1]
        out.append((f"{n:02d} {title}", para(h + 1, bounds[n + 1])))   # заголовок новеллы в текст не входит
    return [(name, body) for name, body in out if body]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source", type=Path)
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parent.parent / "data" / "kompromiss")
    a = ap.parse_args()
    parts = split_book(read_text(a.source))
    a.out.mkdir(parents=True, exist_ok=True)
    for old in a.out.glob("*.txt"):
        old.unlink()
    for name, body in parts:
        (a.out / f"{name}.txt").write_text(body + "\n", encoding="utf-8")
        print(f"{name}: {len(body.split())} слов")
    print(f"Готово: {a.out}")


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
