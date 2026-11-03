import json
from datetime import datetime

from apophenia.schedule import Schedule


def test_schedule_open_hours(cfg):
    cfg["schedule"].update({"always": False, "open": "11:00", "close": "20:00", "days": [1, 2, 3, 4, 5]})
    s = Schedule(cfg)
    assert s.is_open(datetime(2026, 9, 21, 12, 0))       # понедельник
    assert not s.is_open(datetime(2026, 9, 21, 20, 0))   # закрытие не включительно
    assert not s.is_open(datetime(2026, 9, 26, 12, 0))   # суббота
    assert s.next_open(datetime(2026, 9, 26, 12, 0)) == datetime(2026, 9, 28, 11, 0)
    assert s.next_open(datetime(2026, 9, 21, 12, 0)) == datetime(2026, 9, 22, 11, 0)


def test_service_once_and_reprint(cfg, tmp_path, main_mod):
    svc = main_mod.Service(cfg, mock=True, scenario="drift")
    rec = svc.run_cycle(do_print=True)
    assert rec["cycle"] == 1
    archive = tmp_path / "archive"
    assert (archive / "cycle_00001.json").exists() and (archive / "cycle_00001.pdf").exists()
    assert (tmp_path / "queue" / "cycle_00001.pdf").exists()  # в очереди, рабочий поток не запущен
    assert json.loads((tmp_path / "state" / "state.json").read_text(encoding="utf-8"))["last_cycle"] == 1
    disp = json.loads((tmp_path / "state" / "display.json").read_text(encoding="utf-8"))
    assert disp["phase"] == "idle" and disp["cycle"] == 1 and len(disp["quotes"]) >= 3
    main_mod.cmd_reprint(svc, 1, render_only=True)
    svc2 = main_mod.Service(cfg, mock=True, scenario="stabilize")
    assert svc2.run_cycle(do_print=False)["cycle"] == 2  # сквозная нумерация после перезапуска


def test_config_extends(tmp_path):
    from apophenia.config import ROOT, load_config
    c = load_config(ROOT / "config.test.yaml")
    assert c["schedule"]["interval_minutes"] == 0 and c["schedule"]["max_cycles"] == 5
    assert c["paths"]["archive"] == "archive_test"
    assert c["llm"]["model"] == load_config()["llm"]["model"]  # унаследовано
    assert c["cycle"]["max_iterations"] == 25                    # унаследовано
    assert "extends" not in c


def test_protocol_pdf_and_protocol_print(cfg, tmp_path, main_mod):
    cfg["printer"]["protocol"] = True
    svc = main_mod.Service(cfg, mock=True, scenario="oscillate")
    rec = svc.run_cycle(do_print=True)
    archive = tmp_path / "archive"
    proto = archive / "cycle_00001_protocol.pdf"
    assert proto.exists() and proto.read_bytes().startswith(b"%PDF")
    assert (tmp_path / "queue" / "cycle_00001_protocol.pdf").exists()
    assert len(rec["iterations"]) == 6


def test_debug_display_includes_evidence(cfg, tmp_path, main_mod):
    cfg["display"]["debug"] = True
    svc = main_mod.Service(cfg, mock=True, scenario="stabilize")
    svc.run_cycle(do_print=False)
    disp = json.loads((tmp_path / "state" / "display.json").read_text(encoding="utf-8"))
    assert disp["test_mode"] is True
    assert disp["iterations"][0]["evidence"]
    assert disp["last_outcome"]["outcome"] == "STABILIZED"


def test_config_local_overrides(tmp_path, monkeypatch):
    from apophenia import config as cmod
    import shutil
    for name in ("config.yaml", "config.test.yaml"):
        shutil.copy(cmod.ROOT / name, tmp_path / name)
    (tmp_path / "config.local.yaml").write_text("extends: config.yaml\nprinter:\n  name: LOCAL\nschedule:\n  open: '09:00'\n", encoding="utf-8")
    c = cmod.load_config(tmp_path / "config.yaml")
    assert c["printer"]["name"] == "LOCAL" and c["schedule"]["open"] == "09:00"
    assert c["llm"]["model"]  # остальное из config.yaml
    t = cmod.load_config(tmp_path / "config.test.yaml")
    assert t["schedule"]["open"] == "00:00"          # тестовый конфиг переопределяет расписание
    assert t["printer"]["name"] != "LOCAL"           # и принтер (свой тестовый)
    assert t["display"]["debug"] is True
    # без локального файла всё по-прежнему
    (tmp_path / "config.local.yaml").unlink()
    assert cmod.load_config(tmp_path / "config.yaml")["printer"]["name"] == "Pantum P2500NW-series"


def test_gdi_backend_error_outside_windows(cfg, tmp_path):
    import os
    from apophenia.printer import PrintQueue
    cfg["printer"].update({"enabled": True, "backend": "gdi", "name": "X"})
    q = PrintQueue(cfg)
    pdf = tmp_path / "a.pdf"; pdf.write_bytes(b"%PDF-1.4\n")
    if os.name != "nt":
        assert q.print_file(pdf) is False and "gdi" in q.last_error


def test_schedule_always(cfg):
    cfg["schedule"].update({"always": True, "open": "11:00", "close": "12:00", "days": [1]})
    s = Schedule(cfg)
    assert s.is_open(datetime(2026, 9, 26, 3, 0)) and s.seconds_until_open(datetime(2026, 9, 26, 3, 0)) == 0.0
