"""Призрачные цитаты: модель при высокой температуре читает кураторский текст тем же промптом,
что и в исследовании; фрагменты, не прошедшие верификацию (drop_reason == unverified), и есть материал цикла."""

from __future__ import annotations

import logging
from itertools import zip_longest
from typing import Any, Dict, List

from .corpus import pick_texts
from .llm import BaseLLM, LLMRefusal, parse_evidence
from .prompts import Prompt
from .reading import validate_theory_response
from .verify_quotes import ghost_defect

log = logging.getLogger("apophenia.ghosts")


def collect_ghosts(cfg: Dict[str, Any], llm: BaseLLM, theory: Prompt, corpus: List[Dict[str, Any]], seed: int) -> Dict[str, Any]:
    g = cfg.get("ghosts", {})
    min_q, max_q = int(g.get("min_quotes", 3)), int(g.get("max_quotes", 8))
    # ghosts.min_sources: из скольких разных новелл (файлов) набирать призраков, пока хватает текстов на цикл;
    # ghosts.max_per_source: не больше стольких цитат из одной новеллы (0 — без ограничения)
    min_src, per_src = int(g.get("min_sources", 1)), int(g.get("max_per_source", 0) or 0)
    # верификация для этой роли: по умолчанию как в исследовании, но ghosts.verification переопределяет
    vcfg = {**cfg.get("verification", {}), **(g.get("verification") or {})}
    temp = float(cfg["llm"].get("temperatures", {}).get("ghosts", 1.0))
    texts = pick_texts(corpus, int(g.get("max_texts_per_cycle", 6)), seed=seed,
                       by=str(cfg.get("corpus", {}).get("sample_by", "")))
    # с чем сверять цитату: text — с прочитанным фрагментом; corpus — со всем корпусом («в книге не найдены»)
    whole = "\n".join(x["text"] for x in corpus) if str(g.get("verify_against", "text")) == "corpus" else None
    quotes: List[Dict[str, Any]] = []
    readings: List[Dict[str, Any]] = []
    for t in texts:
        try:
            resp = llm.complete("ghosts", theory.system, theory.user(text=t["text"], title=str(t["title"])), temp)
        except LLMRefusal as e:
            log.warning("Текст %s: отказ модели (%s), пропускаем", t["id"], e)
            continue
        data, how = parse_evidence(resp.content)
        if how == "salvaged":
            log.warning("Текст %s: JSON не разобран, фрагменты извлечены из текста ответа", t["id"])
        records, stats = validate_theory_response(data, whole or t["text"], vcfg)
        ghosts, defects = [], {"codebook_echo": [], "not_a_quote": []}
        for r in records:
            if r.get("drop_reason") != "unverified":
                continue
            why = ghost_defect(r["span"], theory.system, g)      # сверяем с тем, что читатель видел: его промпт
            if why:
                defects[why].append(r["span"])
            else:
                ghosts.append(r)
        reading = {"text_id": t["id"], "title": t["title"], "source": t["source"],
                   "n_total": stats["n_total"], "n_kept": stats["n_kept"], "n_unverified": len(ghosts),
                   "codebook_echo": defects["codebook_echo"], "not_a_quote": defects["not_a_quote"],
                   "dropped": stats["dropped"], "finish_reason": resp.finish_reason}
        if stats["n_total"] == 0:
            # ничего не нашлось: сохраняем начало сырого ответа, чтобы понять, JSON это или отказ
            reading["raw_preview"] = (resp.content or "")[:2000]
            reading["parsed"] = how
            log.warning("Текст %s: 0 фрагментов; разбор: %s; ответ: %s", t["id"], how,
                        (resp.content or "")[:200].replace("\n", " "))
        readings.append(reading)
        have = sum(1 for q in quotes if q["source"] == t["source"])
        for r in (ghosts[:max(0, per_src - have)] if per_src else ghosts):
            quotes.append({
                "span": r["span"], "class": r["class"], "mapping": r.get("mapping"), "level": r.get("level"),
                "reasoning": r.get("reasoning"),
                "token_overlap": (r.get("verification") or {}).get("token_overlap"),
                "text_id": t["id"], "title": t["title"], "author": t.get("author", ""), "source": t["source"],
            })
        log.info("Текст %s «%s»: %d фрагментов, %d призрачных", t["id"], t["title"][:40], stats["n_total"], len(ghosts))
        if len(quotes) >= min_q and len({q["source"] for q in quotes}) >= min_src:
            break
    # без дублей по нормализованному тексту цитаты
    seen, uniq = set(), []
    for q in quotes:
        key = " ".join(q["span"].lower().split())
        if key not in seen:
            seen.add(key)
            uniq.append(q)
    if len({q["source"] for q in uniq}) > 1:           # вперемешку по новеллам, чтобы обрезка до max_quotes не срезала одну из них
        by_src: Dict[str, List[Dict[str, Any]]] = {}
        for q in uniq:
            by_src.setdefault(q["source"], []).append(q)
        uniq = [q for row in zip_longest(*by_src.values()) for q in row if q is not None]
    elif min_src > 1 and uniq:
        log.info("Призраки только из одной новеллы: в остальных прочитанных текстах их не нашлось")
    return {"quotes": uniq[:max_q], "readings": readings}


def format_quotes(quotes: List[Dict[str, Any]], group: bool = False) -> str:
    """Список цитат для промптов письма. group (ghosts.group_by_source): цитаты из разных источников идут
    отдельными группами без названий — пишущая модель видит, что материал из двух мест, и должна свести их."""
    sources = list(dict.fromkeys(q.get("source", "") for q in quotes))
    if not group or len(sources) < 2:
        return "\n".join(f"- «{q['span']}»" for q in quotes)
    names = ["Первая глава", "Вторая глава", "Третья глава", "Четвёртая глава"]
    blocks = []
    for i, src in enumerate(sources):
        head = names[i] if i < len(names) else f"Глава {i + 1}"
        blocks.append(head + ":\n" + "\n".join(f"- «{q['span']}»" for q in quotes if q.get("source", "") == src))
    return "\n".join(blocks)
