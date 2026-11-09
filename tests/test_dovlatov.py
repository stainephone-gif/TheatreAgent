"""Довлатовская конфигурация: классы из конфига, txt-корпус, цикл на заглушке, лист."""

import pytest

import apophenia
from apophenia.config import ROOT, load_config
from apophenia.corpus import load_corpus, split_chunks
from apophenia.cycle import Engine
from apophenia.llm import MockLLM
from apophenia.moderation import Moderator
from apophenia.prompts import class_definition, class_short_definitions, load_all
from apophenia.sheet import render_sheet

DOV = ["WIT", "GUILT", "HURT", "MASK", "ALIEN", "SPLIT"]


@pytest.fixture
def dcfg(tmp_path):
    c = load_config(ROOT / "config.dovlatov.sample.yaml")
    c["llm"]["provider"] = "mock"
    c["printer"]["enabled"] = False
    c["display"]["enabled"] = False
    c["telegram"]["enabled"] = False
    for k in ("archive", "rejected", "queue", "state", "logs"):
        c["paths"][k] = str(tmp_path / k)
    yield c
    apophenia.configure_classes({})  # вернуть теории сознания для остальных тестов


def test_classes_from_config(dcfg):
    assert apophenia.CLASSES == DOV
    assert apophenia.ALL_CLASSES == DOV + ["UND"]
    assert apophenia.CLASS_NAMES_RU["UND"] == "без позиции"
    assert apophenia.LEVELS == {"direct", "ironic", "lowered"}
    apophenia.configure_classes({})
    assert apophenia.CLASSES[0] == "COMP" and "meta_metaphor" in apophenia.LEVELS


def test_duplicate_or_reserved_code_rejected():
    with pytest.raises(ValueError):
        apophenia.configure_classes({"classes": [{"code": "A"}, {"code": "a"}]})
    with pytest.raises(ValueError):
        apophenia.configure_classes({"classes": [{"code": "UND"}]})
    apophenia.configure_classes({})


def test_codebook_has_every_class(dcfg):
    short = class_short_definitions(dcfg)
    for cls in DOV + ["UND"]:
        assert f"- {cls} (" in short
        assert class_definition(dcfg, cls) != cls, f"нет абзаца ### {cls} в кодбуке"
    assert "рассказчик себя не определяет" in short
    prompts = load_all(dcfg)
    for kind in ("theory", "ghosts"):
        assert "### WIT — свидетель" in prompts[kind].system
        assert "COMP" not in prompts[kind].system


def test_split_chunks():
    para = "Слово " * 50
    text = "\n\n".join([para.strip()] * 10)   # 10 абзацев по 50 слов
    chunks = split_chunks(text, target_words=120, max_words=240)
    assert all(120 <= len(c.split()) <= 240 for c in chunks[:-1])
    assert sum(len(c.split()) for c in chunks) == 500
    long_para = ". ".join(["Длинное предложение номер %d" % i for i in range(100)]) + "."
    chunks = split_chunks(long_para, target_words=100, max_words=150)
    assert len(chunks) >= 3 and all(len(c.split()) <= 150 for c in chunks)


def test_txt_corpus(dcfg):
    corpus = load_corpus(dcfg)
    assert len(corpus) >= 3
    assert all(t["author"] == "Сергей Довлатов" and ", фрагмент " in t["title"] for t in corpus)
    assert all(t["n_words"] >= dcfg["corpus"]["min_words"] for t in corpus)
    assert [t["id"] for t in corpus] == sorted(t["id"] for t in corpus)


@pytest.mark.parametrize("scenario,outcome,extra", [
    ("stabilize", "STABILIZED", {"class": "WIT"}),
    ("drift", "STABILIZED", {"class": "ALIEN"}),
    ("oscillate", "OSCILLATION", {"period": 2}),
    ("und", "UNFINISHED", {"reason": "ceiling"}),
])
def test_cycle_and_sheet(dcfg, tmp_path, scenario, outcome, extra):
    llm = MockLLM(scenario)
    prompts = load_all(dcfg)
    eng = Engine(dcfg, llm, prompts, load_corpus(dcfg), Moderator(dcfg, llm, prompts["moderate"]))
    rec = eng.run(1)
    assert rec["outcome"]["outcome"] == outcome
    for k, v in extra.items():
        assert rec["outcome"][k] == v
    assert set(rec["trajectory"]) <= set(DOV + ["UND"])
    assert all(q["title"].split(",")[0] in ("Экскурсия", "Редакция", "Переезд") for q in rec["ghosts"]["quotes"])
    pdf = render_sheet(rec, dcfg, tmp_path / "s.pdf")
    assert pdf.read_bytes().startswith(b"%PDF")
