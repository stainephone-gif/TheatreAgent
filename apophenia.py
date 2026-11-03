#!/usr/bin/env python3
"""Метасознание — служба инсталляции.

  python apophenia.py run                 # основной режим: циклы по расписанию, экран, очередь печати
  python apophenia.py once [--no-print]   # один цикл сейчас (проверка), вне расписания
  python apophenia.py reprint 12          # допечатать лист цикла 12 из архива
  python apophenia.py render 12           # пересобрать PDF цикла 12 из JSON
  python apophenia.py test-print          # напечатать пробный лист
  python apophenia.py models              # список моделей GigaChat, доступных по ключу
  python apophenia.py status              # состояние: последний цикл, очередь, расписание
  python apophenia.py probe [id]          # сырой ответ модели на один текст корпуса (отладка призрачных цитат)
  python apophenia.py stats [папка]       # распределение классов по архиву (по умолчанию archive/; для теста: stats archive_test)
  python apophenia.py run --config config.test.yaml   # тестовый режим (test.bat): циклы подряд, свой архив и принтер
Флаги: --config путь/к/config.yaml, --mock (заглушка вместо API и принтера), --scenario drift.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import logging.handlers
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from apophenia import CLASS_NAMES_RU, OUTCOMES, __version__
from apophenia.config import load_config, path_dir
from apophenia.corpus import load_corpus
from apophenia.cycle import CycleRejected, Engine
from apophenia.display import DisplayServer
from apophenia.llm import LLMError, MockLLM, make_llm
from apophenia.moderation import Moderator
from apophenia.printer import PrintQueue
from apophenia.prompts import load_all
from apophenia.schedule import Schedule
from apophenia.sheet import render_protocol, render_sheet
from apophenia.state import State
from apophenia.telegram import Notifier

log = logging.getLogger("apophenia")


def setup_logging(cfg: Dict[str, Any]) -> None:
    logs = path_dir(cfg, "logs")
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.handlers.clear()
    if sys.stdout is not None:  # под pythonw.exe консоли нет, пишем только в файл
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(fmt)
        root.addHandler(sh)
    fh = logging.handlers.RotatingFileHandler(logs / "apophenia.log", maxBytes=5_000_000, backupCount=5, encoding="utf-8")
    fh.setFormatter(fmt)
    root.addHandler(fh)


class Service:
    def __init__(self, cfg: Dict[str, Any], mock: bool = False, scenario: str = "drift"):
        self.cfg = cfg
        if mock:
            cfg["llm"]["provider"] = "mock"
            cfg["llm"]["mock_scenario"] = scenario
            cfg["printer"]["enabled"] = False
        self.llm = make_llm(cfg)
        self.prompts = load_all(cfg)
        self.corpus = load_corpus(cfg)
        self.moderator = Moderator(cfg, self.llm, self.prompts["moderate"])
        self.engine = Engine(cfg, self.llm, self.prompts, self.corpus, self.moderator)
        self.state = State(cfg)
        self.schedule = Schedule(cfg)
        self.notifier = Notifier(cfg)
        self.queue = PrintQueue(cfg, on_error=lambda m: self.notifier.send(m, "printer_error"))
        self.display = DisplayServer(cfg)
        self.archive = path_dir(cfg, "archive")
        self.rejected = path_dir(cfg, "rejected")
        local = Path(cfg["_root"]) / "config.local.yaml"
        cfg_name = Path(cfg["_config_path"]).name
        if cfg_name == "config.local.yaml":
            cfg_name = "config.yaml"
        log.info("Конфиг: %s%s; корпус: %d текстов; модель: %s; принтер: %s (%s)",
                 cfg_name, " + config.local.yaml" if local.exists() else "",
                 len(self.corpus), getattr(self.llm, "model", "mock"),
                 cfg["printer"].get("name") if cfg["printer"].get("enabled") else "выключен",
                 "gdi, средствами Windows" if self.queue.backend == "gdi" else "SumatraPDF/команда")
        from apophenia.printer import GDI_IMPORT_ERROR
        if os.name == "nt" and self.queue.backend != "gdi" and GDI_IMPORT_ERROR:
            log.warning("Печать средствами Windows недоступна: %s. Выполните: "
                        ".venv\\Scripts\\python.exe -m pip install -r requirements.txt", GDI_IMPORT_ERROR)

    # -- один цикл ------------------------------------------------------------
    def progress(self, **fields: Any) -> None:
        iters = fields.get("iterations")
        if iters is not None:
            debug = bool(self.cfg.get("display", {}).get("debug", False))
            fields["iterations"] = [
                {"n": it["n"], "primary": it["primary"], "confidence": it["confidence"],
                 **({"evidence": it.get("evidence", []), "n_kept": it.get("n_kept", 0)} if debug else {})}
                for it in iters]
        fields.setdefault("test_mode", bool(self.cfg.get("display", {}).get("debug", False)))
        self.state.set_display(**fields)

    def run_cycle(self, do_print: bool = True) -> Optional[Dict[str, Any]]:
        cycle_no = self.state.next_cycle_number()
        log.info("=== Цикл %05d ===", cycle_no)
        try:
            rec = self.engine.run(cycle_no, progress=self.progress)
        except LLMError:
            self.state.rollback_cycle_number()  # сеть/API: номер не расходуется, цикл начнётся заново
            raise
        except CycleRejected as e:
            self.state.rollback_cycle_number()
            path = self.rejected / f"rejected_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{e.reason.replace(':', '_')}.json"
            path.write_text(json.dumps({"cycle": cycle_no, "reason": e.reason, "info": e.info}, ensure_ascii=False, indent=1), encoding="utf-8")
            log.warning("Цикл отброшен: %s (%s)", e.reason, path.name)
            self.notifier.send(f"Цикл {cycle_no:05d} отброшен: {e.reason}. Подробности: {path.name}", "rejected")
            self.progress(phase="idle", cycle=cycle_no - 1)
            return None
        jpath = self.archive / f"cycle_{cycle_no:05d}.json"
        jpath.write_text(json.dumps(rec, ensure_ascii=False, indent=1), encoding="utf-8")
        pdf = render_sheet(rec, self.cfg, self.archive / f"cycle_{cycle_no:05d}.pdf")
        protocol = render_protocol(rec, self.cfg, self.archive / f"cycle_{cycle_no:05d}_protocol.pdf")
        o = rec["outcome"]
        summary = (f"Цикл {cycle_no:05d}: {OUTCOMES.get(o['outcome'], o['outcome'])}"
                   f"{' — ' + o['class'] + ' (' + CLASS_NAMES_RU.get(o['class'], '') + ')' if o.get('class') else ''}, "
                   f"{len(rec['iterations'])} итераций, {rec['duration_s']} с. Траектория: {' → '.join(rec['trajectory'])}")
        log.info(summary)
        self.progress(phase="printing", cycle=cycle_no, quotes=rec["ghosts"]["quotes"], iterations=rec["iterations"],
                      text=rec["final_text"], last_outcome=o)
        m = self.cfg.get("moderation", {})
        if do_print and m.get("telegram_review") and self.notifier.enabled:
            self.notifier.send(f"{summary}\n\nТекст листа:\n{rec['final_text']}\n\nОтправьте «стоп» в течение "
                               f"{m.get('review_minutes', 20)} мин, чтобы не печатать.", "cycle_done")
            if self.notifier.wait_for_veto(cycle_no, float(m.get("review_minutes", 20))):
                rec["vetoed"] = True
                jpath.write_text(json.dumps(rec, ensure_ascii=False, indent=1), encoding="utf-8")
                log.warning("Цикл %05d: печать отменена художником", cycle_no)
                do_print = False
        else:
            self.notifier.send(summary, "cycle_done")
        if do_print:
            self.queue.enqueue(pdf)
            if self.cfg.get("printer", {}).get("protocol", False):
                self.queue.enqueue(protocol)
        self.progress(phase="idle", cycle=cycle_no, last_outcome=o)
        return rec

    # -- основной цикл службы -------------------------------------------------
    def run_forever(self) -> None:
        self.display.start()
        self.queue.start()
        self.state.data["started_at"] = datetime.now().isoformat(timespec="seconds")
        self.state.save()
        self.progress(phase="idle", cycle=self.state.data.get("last_cycle", 0))
        sch = self.cfg.get("schedule", {})
        interval = self.schedule.interval.total_seconds()
        pause = float(sch.get("pause_seconds", 60))
        max_cycles = int(sch.get("max_cycles", 0) or 0)
        done_cycles = 0
        next_at = time.time() if sch.get("start_immediately", True) else time.time() + interval
        while True:
            try:
                if not self.schedule.is_open():
                    wait = self.schedule.seconds_until_open()
                    log.info("Галерея закрыта, следующий цикл через %.0f мин", wait / 60)
                    self.progress(phase="offhours", next_at=self.schedule.next_open().isoformat(timespec="minutes"))
                    time.sleep(min(wait, 300))
                    next_at = time.time()
                    continue
                if max_cycles and done_cycles >= max_cycles:
                    self.progress(phase="idle", message=f"выполнено {done_cycles} циклов, служба ждёт")
                    time.sleep(30)
                    continue
                now = time.time()
                if now < next_at:
                    self.progress(phase="idle", next_at=datetime.fromtimestamp(next_at).isoformat(timespec="minutes"))
                    time.sleep(min(30, next_at - now))
                    continue
                started = time.time()
                if self.run_cycle() is not None:
                    done_cycles += 1
                    if max_cycles and done_cycles >= max_cycles:
                        log.info("Выполнено %d циклов (schedule.max_cycles), новые не запускаются", done_cycles)
                next_at = max(started + interval, time.time() + pause)
            except LLMError as e:
                log.error("API недоступен: %s. Пауза 60 с", e)
                self.progress(phase="nointernet", message="нет соединения")
                self.notifier.send(f"API/сеть: {e}", "no_internet")
                time.sleep(60)
                next_at = time.time()
            except KeyboardInterrupt:
                log.info("Остановка по Ctrl+C")
                break
            except Exception as e:  # noqa: BLE001 — служба не должна падать
                log.exception("Необработанная ошибка: %s", e)
                self.progress(phase="error", message="ошибка, перезапуск цикла")
                self.notifier.send(f"Ошибка: {e}", "api_error")
                time.sleep(60)
                next_at = time.time()
        self.queue.stop()
        self.display.stop()


# --------------------------------------------------------------------------- #
# Команды
# --------------------------------------------------------------------------- #

def cmd_reprint(svc: Service, n: int, render_only: bool = False) -> None:
    jpath = svc.archive / f"cycle_{n:05d}.json"
    if not jpath.exists():
        sys.exit(f"Нет файла {jpath}")
    rec = json.loads(jpath.read_text(encoding="utf-8"))
    pdf = render_sheet(rec, svc.cfg, svc.archive / f"cycle_{n:05d}.pdf")
    print(f"PDF: {pdf}")
    if not render_only:
        ok = svc.queue.print_now(pdf)
        print("Отправлено на печать" if ok else f"Ошибка печати: {svc.queue.last_error}; файл остался в очереди и уйдёт при следующем запуске службы")


def cmd_test_print(svc: Service) -> None:
    rec = {
        "cycle": 0, "finished_at": datetime.now().isoformat(timespec="seconds"), "model": "тест", "code_version": __version__,
        "prompts": {"theory": {"hash": "—", "codebook_hash": "—"}}, "duration_s": 0,
        "ghosts": {"quotes": [{"span": "пробный лист: принтер, бумага, тонер", "title": "тест", "class": "COMP",
                               "mapping": {"source": "машина", "target": "ум"}}]},
        "trajectory": ["COMP", "PRED", "ENACT", "ENACT", "ENACT"], "confidences": [0.5, 0.8, 0.9, 1.0, 1.0],
        "iterations": [{}] * 5, "outcome": {"outcome": "STABILIZED", "class": "ENACT"},
        "final_text": "Это пробный лист. Если вы его читаете, принтер подключён, шрифты с кириллицей найдены, "
                      "очередь печати работает. Настоящие листы будут появляться по одному в час в часы работы галереи.",
    }
    pdf = render_sheet(rec, svc.cfg, path_dir(svc.cfg, "queue") / "test_sheet.pdf")
    ok = svc.queue.print_file(pdf)
    print("Отправлено на печать" if ok else f"Ошибка печати: {svc.queue.last_error}")


def cmd_probe(svc: Service, text_id: Optional[str]) -> None:
    """Сырой ответ модели на один текст корпуса промптом классификатора при температуре призрачных цитат."""
    from apophenia.llm import parse_evidence
    from apophenia.reading import validate_theory_response
    corpus = svc.corpus
    t = next((x for x in corpus if text_id and str(x["id"]) == str(text_id)), None) or corpus[0]
    theory = svc.prompts.get("ghosts", svc.prompts["theory"])
    temp = float(svc.cfg["llm"].get("temperatures", {}).get("ghosts", 1.0))
    model = getattr(svc.llm, "models", {}).get("ghosts", getattr(svc.llm, "model", "mock"))
    print(f"Текст {t['id']} «{t['title']}», {t['n_words']} слов, температура {temp}, модель {model}")
    print("-" * 60)
    print(t["text"][:1500])
    print("-" * 60)
    resp = svc.llm.complete("ghosts", theory.system, theory.user(text=t["text"]), temp)
    print(f"finish_reason: {resp.finish_reason}; usage: {resp.raw.get('usage')}")
    print("СЫРОЙ ОТВЕТ:")
    print(resp.content)
    print("-" * 60)
    data, how = parse_evidence(resp.content)
    print("Разбор ответа:", {"json": "JSON разобран", "salvaged": "JSON не разобран, фрагменты извлечены из текста", "none": "не удалось"}[how])
    vcfg = {**svc.cfg.get("verification", {}), **(svc.cfg.get("ghosts", {}).get("verification") or {})}
    records, stats = validate_theory_response(data, t["text"], vcfg)
    print(f"фрагментов {stats['n_total']}, в тексте есть {stats['n_kept']}, призрачных {stats['dropped'].get('unverified', 0)}, "
          f"прочее {dict((k, v) for k, v in stats['dropped'].items() if k != 'unverified')}")
    for r in records:
        v = r.get("verification") or {}
        tag = "ЕСТЬ   " if r["kept"] else ("ПРИЗРАК" if r["drop_reason"] == "unverified" else r["drop_reason"])
        print(f"  [{r.get('class')}] {tag:<22} совпадение {v.get('token_overlap', '-')}  «{r.get('span', '')[:90]}»")


def cmd_stats(cfg: Dict[str, Any], archive_arg: Optional[str]) -> None:
    """Распределение классов по архиву: первые чтения, все итерации, итоги по исходам, классы призрачных цитат."""
    from collections import Counter
    from apophenia import ALL_CLASSES
    d = Path(archive_arg) if archive_arg else path_dir(cfg, "archive")
    files = sorted(d.glob("cycle_*.json"))
    if not files:
        print(f"В {d} нет циклов")
        return
    first, every, final, ghosts, outcomes, lengths = Counter(), Counter(), Counter(), Counter(), Counter(), []
    for f in files:
        rec = json.loads(f.read_text(encoding="utf-8"))
        tr = rec.get("trajectory", [])
        if tr:
            first[tr[0]] += 1
            every.update(tr)
            final[tr[-1]] += 1
        o = rec.get("outcome", {}) or {}
        outcomes[o.get("outcome", "?") + (" " + o.get("class", "") if o.get("class") else "")] += 1
        ghosts.update(q.get("class", "?") for q in rec.get("ghosts", {}).get("quotes", []))
        lengths.append(len(tr))

    def row(counter: Counter) -> str:
        total = sum(counter.values()) or 1
        return "  ".join(f"{c}:{counter.get(c, 0)} ({100 * counter.get(c, 0) / total:.0f}%)" for c in ALL_CLASSES if counter.get(c, 0))

    print(f"Циклов: {len(files)}; итераций в среднем {sum(lengths) / len(lengths):.1f}")
    print("Первое чтение:     ", row(first))
    print("Все итерации:      ", row(every))
    print("Последнее чтение:  ", row(final))
    print("Призрачные цитаты: ", row(ghosts))
    print("Исходы:            ", "  ".join(f"{k}:{v}" for k, v in outcomes.most_common()))


def cmd_status(svc: Service) -> None:
    st = svc.state.data
    print(f"Версия {__version__}; последний цикл: {st.get('last_cycle', 0)}; запуск службы: {st.get('started_at')}")
    print(f"Галерея {'открыта' if svc.schedule.is_open() else 'закрыта'}; интервал {svc.schedule.interval}; "
          f"следующее открытие {svc.schedule.next_open().isoformat(timespec='minutes')}")
    print(f"В очереди печати: {len(svc.queue.pending())}; архив: {len(list(svc.archive.glob('cycle_*.json')))} циклов; "
          f"отброшено: {len(list(svc.rejected.glob('*.json')))}")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Метасознание — служба инсталляции")
    ap.add_argument("command", nargs="?", default="run",
                    choices=["run", "once", "reprint", "render", "test-print", "models", "status", "probe", "stats"])
    ap.add_argument("arg", nargs="?")
    ap.add_argument("--config", default=None)
    ap.add_argument("--mock", action="store_true", help="заглушка вместо API и принтера")
    ap.add_argument("--scenario", default="drift", help="сценарий заглушки: stabilize|drift|oscillate|oscillate3|wander|degenerate|und")
    ap.add_argument("--no-print", action="store_true")
    a = ap.parse_args(argv)
    cfg = load_config(a.config)
    setup_logging(cfg)
    if a.command == "stats":
        cmd_stats(cfg, a.arg)
        return
    if a.command == "models":
        llm = make_llm(cfg)
        print("\n".join(llm.list_models()) if hasattr(llm, "list_models") else "mock")
        return
    svc = Service(cfg, mock=a.mock, scenario=a.scenario)
    if a.command == "run":
        svc.run_forever()
    elif a.command == "once":
        svc.queue.start()
        rec = svc.run_cycle(do_print=not a.no_print)
        if rec:
            print(f"Готово: archive/cycle_{rec['cycle']:05d}.pdf")
        time.sleep(3)
    elif a.command in ("reprint", "render"):
        if not a.arg:
            sys.exit("Укажите номер цикла")
        cmd_reprint(svc, int(a.arg), render_only=(a.command == "render"))
    elif a.command == "test-print":
        cmd_test_print(svc)
    elif a.command == "status":
        cmd_status(svc)
    elif a.command == "probe":
        cmd_probe(svc, a.arg)


if __name__ == "__main__":
    main()
