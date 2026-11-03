#!/usr/bin/env python3
"""Предварительная проверка корпуса: прогоняет все кураторские тексты через модерацию
(стоп-список + модель) и записывает список исключённых id в data/corpus_excluded.json.
Запускать один раз перед выставкой:  python tools/screen_corpus.py [--config config.yaml] [--mock]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from apophenia.config import load_config, resolve  # noqa: E402
from apophenia.corpus import load_corpus  # noqa: E402
from apophenia.llm import make_llm  # noqa: E402
from apophenia.moderation import Moderator  # noqa: E402
from apophenia.prompts import load_prompt  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--mock", action="store_true")
    a = ap.parse_args()
    cfg = load_config(a.config)
    if a.mock:
        cfg["llm"]["provider"] = "mock"
    cfg["corpus"]["exclude_file"] = ""  # проверяем весь корпус
    llm = make_llm(cfg)
    mod = Moderator(cfg, llm, load_prompt(cfg, "moderate"))
    corpus = load_corpus(cfg)
    excluded, details = [], []
    for i, t in enumerate(corpus, 1):
        r = mod.check(t["text"][:6000])
        flag = not r["ok"]
        print(f"[{i}/{len(corpus)}] {t['id']:4d} {'ИСКЛЮЧИТЬ' if flag else 'ок       '} «{t['title'][:50]}»"
              + (f"  {r['stoplist_hits'] or (r['llm'] or {}).get('topics')}" if flag else ""))
        if flag:
            excluded.append(t["id"])
            details.append({"id": t["id"], "title": t["title"], "stoplist_hits": r["stoplist_hits"], "llm": r["llm"]})
    out = resolve(cfg, "data/corpus_excluded.json")
    out.write_text(json.dumps({"excluded_ids": excluded, "details": details, "n_checked": len(corpus)},
                              ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nИсключено {len(excluded)} из {len(corpus)}. Записано: {out}")


if __name__ == "__main__":
    main()
