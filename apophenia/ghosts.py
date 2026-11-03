"""Призрачные цитаты: модель при высокой температуре читает кураторский текст тем же промптом,
что и в исследовании; фрагменты, не прошедшие верификацию (drop_reason == unverified), и есть материал цикла."""

from __future__ import annotations

import logging
from typing import Any, Dict, List

from .corpus import pick_texts
from .llm import BaseLLM, LLMRefusal, parse_evidence
from .prompts import Prompt
from .reading import validate_theory_response

log = logging.getLogger("apophenia.ghosts")


def collect_ghosts(cfg: Dict[str, Any], llm: BaseLLM, theory: Prompt, corpus: List[Dict[str, Any]], seed: int) -> Dict[str, Any]:
    g = cfg.get("ghosts", {})
    min_q, max_q = int(g.get("min_quotes", 3)), int(g.get("max_quotes", 8))
    # верификация для этой роли: по умолчанию как в исследовании, но ghosts.verification переопределяет
    vcfg = {**cfg.get("verification", {}), **(g.get("verification") or {})}
    temp = float(cfg["llm"].get("temperatures", {}).get("ghosts", 1.0))
    texts = pick_texts(corpus, int(g.get("max_texts_per_cycle", 6)), seed=seed)
    quotes: List[Dict[str, Any]] = []
    readings: List[Dict[str, Any]] = []
    for t in texts:
        try:
            resp = llm.complete("ghosts", theory.system, theory.user(text=t["text"]), temp)
        except LLMRefusal as e:
            log.warning("Текст %s: отказ модели (%s), пропускаем", t["id"], e)
            continue
        data, how = parse_evidence(resp.content)
        if how == "salvaged":
            log.warning("Текст %s: JSON не разобран, фрагменты извлечены из текста ответа", t["id"])
        records, stats = validate_theory_response(data, t["text"], vcfg)
        ghosts = [r for r in records if r.get("drop_reason") == "unverified"]
        reading = {"text_id": t["id"], "title": t["title"], "source": t["source"],
                   "n_total": stats["n_total"], "n_kept": stats["n_kept"], "n_unverified": len(ghosts),
                   "dropped": stats["dropped"], "finish_reason": resp.finish_reason}
        if stats["n_total"] == 0:
            # ничего не нашлось: сохраняем начало сырого ответа, чтобы понять, JSON это или отказ
            reading["raw_preview"] = (resp.content or "")[:2000]
            reading["parsed"] = how
            log.warning("Текст %s: 0 фрагментов; разбор: %s; ответ: %s", t["id"], how,
                        (resp.content or "")[:200].replace("\n", " "))
        readings.append(reading)
        for r in ghosts:
            quotes.append({
                "span": r["span"], "class": r["class"], "mapping": r.get("mapping"), "level": r.get("level"),
                "reasoning": r.get("reasoning"),
                "token_overlap": (r.get("verification") or {}).get("token_overlap"),
                "text_id": t["id"], "title": t["title"], "author": t.get("author", ""), "source": t["source"],
            })
        log.info("Текст %s «%s»: %d фрагментов, %d призрачных", t["id"], t["title"][:40], stats["n_total"], len(ghosts))
        if len(quotes) >= min_q:
            break
    # без дублей по нормализованному тексту цитаты
    seen, uniq = set(), []
    for q in quotes:
        key = " ".join(q["span"].lower().split())
        if key not in seen:
            seen.add(key)
            uniq.append(q)
    return {"quotes": uniq[:max_q], "readings": readings}


def format_quotes(quotes: List[Dict[str, Any]]) -> str:
    return "\n".join(f"- «{q['span']}»" for q in quotes)
