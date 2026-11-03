"""Один цикл инсталляции: призрачные цитаты → самоэкспликация → (чтение → переписывание)* → исход."""

from __future__ import annotations

import logging
import subprocess
import time
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

from . import CLASS_NAMES_RU, __version__
from .ghosts import collect_ghosts, format_quotes
from .llm import BaseLLM, LLMRefusal, parse_evidence
from .moderation import Moderator
from .outcomes import detect
from .prompts import Prompt, class_definition, class_short_definitions
from .reading import aggregate, validate_theory_response

log = logging.getLogger("apophenia.cycle")

Progress = Callable[..., None]


class CycleRejected(Exception):
    """Цикл отброшен до печати (модерация или нет материала). Атрибут info — подробности для журнала."""

    def __init__(self, reason: str, info: Optional[Dict[str, Any]] = None):
        super().__init__(reason)
        self.reason = reason
        self.info = info or {}


def code_version(root: Optional[str]) -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=root, capture_output=True,
                             text=True, timeout=5, check=True).stdout.strip()
        return f"{out} (apophenia {__version__})"
    except Exception:  # noqa: BLE001
        return f"apophenia {__version__}"


class Engine:
    def __init__(self, cfg: Dict[str, Any], llm: BaseLLM, prompts: Dict[str, Prompt],
                 corpus: List[Dict[str, Any]], moderator: Moderator):
        self.cfg = cfg
        self.llm = llm
        self.prompts = prompts
        self.corpus = corpus
        self.moderator = moderator
        self.temps = cfg["llm"].get("temperatures", {})
        self.ccfg = cfg.get("cycle", {})
        self.max_regen = int(cfg.get("moderation", {}).get("max_regenerations", 2))
        lo, hi = self.ccfg.get("explication_words", [150, 260])
        self.words = {"min_words": str(lo), "max_words": str(hi)}

    # -- helpers --------------------------------------------------------------
    def _moderated(self, role: str, system: str, user: str, temperature: float, stage: str) -> Dict[str, Any]:
        """Вызов модели с модерацией результата; при пометке — повтор, затем CycleRejected."""
        attempts = []
        for attempt in range(self.max_regen + 1):
            try:
                resp = self.llm.complete(role, system, user, temperature)
                text = resp.content.strip().strip('"').strip()
            except LLMRefusal as e:
                attempts.append({"attempt": attempt, "refusal": str(e)})
                log.warning("%s: отказ модели, попытка %d", stage, attempt + 1)
                continue
            check = self.moderator.check(text)
            attempts.append({"attempt": attempt, "moderation": check, "n_words": len(text.split())})
            if check["ok"] and text:
                return {"text": text, "attempts": attempts}
            log.warning("%s: текст помечен модерацией, попытка %d", stage, attempt + 1)
        raise CycleRejected(f"moderation:{stage}", {"attempts": attempts})

    def read(self, text: str) -> Dict[str, Any]:
        """Чтение текста классификатором: тот же промпт и та же верификация, что в исследовании."""
        theory = self.prompts["theory"]
        try:
            resp = self.llm.complete("classify", theory.system, theory.user(text=text), float(self.temps.get("classify", 0.0)))
            data, how = parse_evidence(resp.content)
            if how != "json":
                log.warning("Чтение: разбор ответа классификатора: %s", how)
        except LLMRefusal:
            data = {"evidence": [], "notes": "refusal"}
        records, stats = validate_theory_response(data, text, self.cfg.get("verification", {}))
        agg = aggregate(records, text)
        evidence = [{"class": r["class"], "span": r["span"], "mapping": r.get("mapping"), "level": r.get("level")}
                    for r in records if r["kept"]]
        ghosts_in_reading = [r["span"] for r in records if r.get("drop_reason") == "unverified"]
        return {**agg, "evidence": evidence, "unverified": ghosts_in_reading, "filter": stats}

    # -- cycle ----------------------------------------------------------------
    def run(self, cycle_no: int, progress: Optional[Progress] = None) -> Dict[str, Any]:
        progress = progress or (lambda **kw: None)
        started = datetime.now()
        seed = int(time.time()) ^ (cycle_no * 7919)
        record: Dict[str, Any] = {
            "cycle": cycle_no, "started_at": started.isoformat(timespec="seconds"), "seed": seed,
            "model": getattr(self.llm, "model", type(self.llm).__name__),
            "prompts": {k: {"file": p.name, "hash": p.hash, "codebook_hash": p.codebook_hash} for k, p in self.prompts.items()},
            "code_version": code_version(self.cfg.get("_root")),
            "config": {"cycle": self.ccfg, "verification": self.cfg.get("verification", {}), "ghosts": self.cfg.get("ghosts", {})},
        }

        # 1. Призрачные цитаты
        progress(phase="ghosts", cycle=cycle_no, iterations=[], quotes=[])
        ghosts = collect_ghosts(self.cfg, self.llm, self.prompts.get("ghosts", self.prompts["theory"]), self.corpus, seed)
        quotes = ghosts["quotes"]
        record["ghosts"] = ghosts
        min_q = int(self.cfg.get("ghosts", {}).get("min_quotes", 3))
        if len(quotes) < min_q:
            raise CycleRejected("no_ghosts", {"found": len(quotes), "readings": ghosts["readings"]})
        qcheck = self.moderator.check("\n".join(q["span"] for q in quotes))
        record["ghosts"]["moderation"] = qcheck
        if not qcheck["ok"]:
            raise CycleRejected("moderation:ghosts", {"moderation": qcheck, "quotes": [q["span"] for q in quotes]})
        quotes_block = format_quotes(quotes)
        progress(phase="writing", cycle=cycle_no, quotes=quotes, iterations=[])

        # 2. Самоэкспликация
        ex = self.prompts["explicate"]
        gen = self._moderated("explicate", ex.system.replace("{min_words}", self.words["min_words"]).replace("{max_words}", self.words["max_words"]),
                              ex.user(quotes=quotes_block), float(self.temps.get("explicate", 0.8)), "explicate")
        text = gen["text"]
        record["explication_attempts"] = gen["attempts"]

        # 3. Чтение и переписывание
        iterations: List[Dict[str, Any]] = []
        outcome: Optional[Dict[str, Any]] = None
        rw = self.prompts["rewrite"]
        rw_system = (rw.system.replace("{min_words}", self.words["min_words"]).replace("{max_words}", self.words["max_words"])
                     .replace("{theories}", class_short_definitions(self.cfg)))
        while True:
            n = len(iterations) + 1
            progress(phase="reading", cycle=cycle_no, quotes=quotes, iterations=iterations, text=text, iteration=n)
            reading = self.read(text)
            it = {"n": n, "text": text, **reading, "at": datetime.now().isoformat(timespec="seconds")}
            iterations.append(it)
            log.info("Цикл %d, итерация %d: %s (%.2f, %d фрагментов)", cycle_no, n, it["primary"], it["confidence"], it["n_kept"])
            progress(phase="read", cycle=cycle_no, quotes=quotes, iterations=iterations, text=text, iteration=n)
            outcome = detect(iterations, self.ccfg)
            if outcome:
                break
            progress(phase="rewriting", cycle=cycle_no, quotes=quotes, iterations=iterations, text=text, iteration=n)
            ev_lines = "\n".join(
                f"- «{e['span']}» — {e['class']}: {e['mapping'].get('source', '')} → {e['mapping'].get('target', '')}"
                for e in it["evidence"] if e.get("mapping")) or "- свидетельств не найдено"
            user = rw.user(
                quotes=quotes_block, iteration=str(n), text=text, primary=it["primary"],
                primary_name=CLASS_NAMES_RU.get(it["primary"], ""), confidence=f"{it['confidence']:.2f}",
                reading=ev_lines,
                definition=class_definition(self.cfg, it["primary"]) if it["primary"] != "UND"
                else "UND — свидетельств ни одной теории не найдено: в тексте нет переноса между областью ума и областью машины.",
            )
            gen = self._moderated("rewrite", rw_system, user, float(self.temps.get("explicate", 0.8)), f"rewrite:{n}")
            it["rewrite_attempts"] = gen["attempts"]
            text = gen["text"]

        finished = datetime.now()
        record.update({
            "iterations": iterations,
            "outcome": outcome,
            "final_text": iterations[-1]["text"],
            "trajectory": [it["primary"] for it in iterations],
            "confidences": [it["confidence"] for it in iterations],
            "finished_at": finished.isoformat(timespec="seconds"),
            "duration_s": round((finished - started).total_seconds(), 1),
        })
        return record
