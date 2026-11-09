"""Конфигурация «Компромисс»: двенадцать классов, подготовка книги, цикл на заглушке."""

import importlib.util

import pytest

import apophenia
from apophenia.config import ROOT, load_config
from apophenia.corpus import load_corpus
from apophenia.cycle import Engine
from apophenia.llm import MockLLM
from apophenia.moderation import Moderator
from apophenia.prompts import class_definition, class_short_definitions, load_all
from apophenia.sheet import render_sheet

K = [f"K{i}" for i in range(1, 13)]


def _prep():
    spec = importlib.util.spec_from_file_location("prepare_kompromiss", ROOT / "tools" / "prepare_kompromiss.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def kcfg(tmp_path):
    c = load_config(ROOT / "config.kompromiss.sample.yaml")
    c["llm"]["provider"] = "mock"
    c["printer"]["enabled"] = False
    c["display"]["enabled"] = False
    c["telegram"]["enabled"] = False
    for k in ("archive", "rejected", "queue", "state", "logs"):
        c["paths"][k] = str(tmp_path / k)
    yield c
    apophenia.configure_classes({})


def test_twelve_classes_and_codebook(kcfg):
    assert apophenia.CLASSES == K
    assert apophenia.CLASS_NAMES_RU["K11"] == "чужие похороны"
    assert apophenia.LEVELS == {"clipping", "backstage", "reflection"}
    short = class_short_definitions(kcfg)
    for cls in K + ["UND"]:
        assert f"- {cls} (" in short
        assert class_definition(kcfg, cls) != cls, f"нет абзаца ### {cls} в кодбуке"
    # K1 не должен захватывать абзацы K10–K12
    assert "мешают нам жить" not in class_definition(kcfg, "K1")
    prompts = load_all(kcfg)
    for kind in ("theory", "ghosts"):
        assert "### K12 — память" in prompts[kind].system and "COMP" not in prompts[kind].system


def test_moderation_is_theatre_stoplist(kcfg):
    llm = MockLLM("stabilize")
    m = Moderator(kcfg, llm, load_all(kcfg)["moderate"])
    assert m.check("Партийная газета, редактор, ЦК и вытрезвитель.")["ok"]


def _book(tmp_path, n_heads=12, encoding="cp1251"):
    ords = _prep().ORDINALS
    parts = ["Компромисс", "Сергей Довлатов", "© копирайт", "Все права защищены.", "* * *", "Посвящение.",
             "Вступление: я остался без работы."]
    for i in range(n_heads):
        parts += [f"Компромисс {ords[i]}", f"(«Газета». {i + 1})", f"Текст новеллы {i + 1}. Я написал заметку."]
    parts += ["Послесловие", "Это послесловие не должно попасть в корпус.", "notes", "Примечания"]
    p = tmp_path / "book.txt"
    p.write_bytes("\r\n".join(parts).encode(encoding))
    return p


@pytest.mark.parametrize("encoding", ["cp1251", "utf-8"])
def test_prepare_splits_book(tmp_path, encoding):
    prep = _prep()
    parts = prep.split_book(prep.read_text(_book(tmp_path, encoding=encoding)))
    names = [n for n, _ in parts]
    assert names[0] == "00 Вступление" and names[1] == "01 Компромисс первый" and names[-1] == "12 Компромисс двенадцатый"
    assert len(parts) == 13
    body = dict(parts)
    assert "копирайт" not in body["00 Вступление"] and "Посвящение." in body["00 Вступление"]
    assert "послесловие" not in body["12 Компромисс двенадцатый"].lower()
    assert body["05 Компромисс пятый"].startswith("(«Газета». 5)")


def test_prepare_refuses_incomplete_book(tmp_path):
    prep = _prep()
    with pytest.raises(SystemExit):
        prep.split_book(prep.read_text(_book(tmp_path, n_heads=11)))


def test_prepared_titles_without_number(kcfg, tmp_path):
    prep = _prep()
    out = tmp_path / "k"
    out.mkdir()
    for name, body in prep.split_book(prep.read_text(_book(tmp_path))):
        (out / f"{name}.txt").write_text((body + " слово") * 1, encoding="utf-8")
    kcfg["corpus"]["dir"] = str(out)
    kcfg["corpus"]["min_words"] = 0
    titles = [t["title"] for t in load_corpus(kcfg)]
    assert titles[0] == "Вступление, фрагмент 1" and titles[1] == "Компромисс первый, фрагмент 1"


@pytest.mark.parametrize("scenario,outcome,extra", [
    ("stabilize", "STABILIZED", {"class": "K1"}),
    ("drift", "STABILIZED", {"class": "K5"}),
    ("oscillate3", "OSCILLATION", {"period": 3}),
    ("wander", "UNFINISHED", {"reason": "ceiling"}),
])
def test_cycle_and_sheet(kcfg, tmp_path, scenario, outcome, extra):
    llm = MockLLM(scenario)
    prompts = load_all(kcfg)
    eng = Engine(kcfg, llm, prompts, load_corpus(kcfg), Moderator(kcfg, llm, prompts["moderate"]))
    rec = eng.run(13)
    assert rec["outcome"]["outcome"] == outcome
    for k, v in extra.items():
        assert rec["outcome"][k] == v
    assert set(rec["trajectory"]) <= set(K + ["UND"])
    pdf = render_sheet(rec, kcfg, tmp_path / "s.pdf")
    data = pdf.read_bytes()
    assert data.startswith(b"%PDF") and data.count(b"/Type /Page") - data.count(b"/Type /Pages") == 1


def test_pick_by_source_balances_novellas():
    from collections import Counter

    from apophenia.corpus import pick_texts
    corpus = [{"id": i, "source": "long.txt"} for i in range(100)] + [{"id": 100 + i, "source": f"s{i}.txt"} for i in range(3)]
    firsts = Counter(pick_texts(corpus, 1, seed=s, by="source")[0]["source"] for s in range(400))
    assert firsts["long.txt"] < 160          # без балансировки было бы ~390 из 400
    picked = pick_texts(corpus, 8, seed=1, by="source")
    assert len({t["id"] for t in picked}) == 8 and {t["source"] for t in picked} >= {"s0.txt", "s1.txt", "s2.txt", "long.txt"}
    assert len(pick_texts(corpus[:2], 5, seed=1, by="source")) == 2
