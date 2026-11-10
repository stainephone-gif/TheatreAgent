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


def test_moderation_off_for_18plus(kcfg):
    llm = MockLLM("stabilize")
    m = Moderator(kcfg, llm, load_all(kcfg)["moderate"])
    assert kcfg["moderation"]["enabled"] is False
    assert m.check("Партийная газета, редактор, ЦК, вытрезвитель и брань.")["ok"]
    assert not llm.calls
    kcfg["moderation"]["enabled"] = True    # если включить — только театральный стоп-список брани
    assert Moderator(kcfg, llm, load_all(kcfg)["moderate"]).check("Партийная газета и ЦК.")["ok"]


def test_claude_is_configured(kcfg):
    c = load_config(ROOT / "config.kompromiss.yaml")
    assert c["llm"]["provider"] == "claude" and c["llm"]["claude"]["model"] == "claude-opus-5-5"
    assert c["llm"]["claude"]["effort"]["classify"] == "medium"
    p = load_all(kcfg)["explicate"].system
    assert "синтетический журналист" in p and "ВЫРЕЗКА" in p and "КАК БЫЛО" in p


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


def test_ghosts_verified_against_whole_corpus(cfg):
    """verify_against: corpus — цитата из другого фрагмента книги не призрак."""
    from apophenia.ghosts import collect_ghosts
    from apophenia.llm import LLMResponse
    from apophenia.prompts import load_all
    import json
    corpus = [{"id": 0, "title": "А", "source": "a", "text": "Первый фрагмент про ипподром и ставки.", "n_words": 6},
              {"id": 1, "title": "Б", "source": "b", "text": "Второй фрагмент про телеграмму доярки.", "n_words": 6}]

    class Fake:
        def complete(self, role, system, user, temperature):
            ev = [{"class": cfg["classes"][0]["code"] if cfg.get("classes") else "GWT", "span": s, "level": None,
                   "exclusion_checked": True, "mapping": {"source": "a", "target": "b"}}
                  for s in ("фрагмент про ипподром и ставки", "фрагмент про телеграмму доярки", "этого в книге нет вовсе")]
            return LLMResponse(content=json.dumps({"evidence": ev}, ensure_ascii=False))

    theory = load_all(cfg)["theory"]
    cfg["ghosts"] = {**cfg["ghosts"], "min_quotes": 1, "max_texts_per_cycle": 1}
    n_text = len(collect_ghosts(cfg, Fake(), theory, corpus, seed=1)["quotes"])
    cfg["ghosts"]["verify_against"] = "corpus"
    n_corpus = len(collect_ghosts(cfg, Fake(), theory, corpus, seed=1)["quotes"])
    assert (n_text, n_corpus) == (2, 1)


def test_memory_config_is_all_claude_and_main_routes_ghosts():
    main, mem = load_config(ROOT / "config.kompromiss.yaml"), load_config(ROOT / "config.kompromiss.memory.yaml")
    assert main["llm"]["providers"] == {"ghosts": "gigachat"} and main["ghosts"]["verify_against"] == "corpus"
    assert mem["llm"]["providers"] == {"ghosts": "claude"} and mem["prompts"]["ghosts"] == "ghosts_kompromiss_memory_v1.md"
    assert "{title}" in (ROOT / "prompts" / mem["prompts"]["ghosts"]).read_text(encoding="utf-8")


def test_codebook_echo_is_not_a_ghost_and_calibration_is_hidden_from_reader():
    from apophenia.prompts import load_all, load_codebook
    from apophenia.verify_quotes import shares_run
    c = load_config(ROOT / "config.kompromiss.yaml")
    assert c["ghosts"]["codebook_echo_words"] == 3 and c["ghosts"]["codebook_echo_overlap"] == 0.8
    p = load_all(c)
    seen = p["ghosts"].system                      # эхо ищется в том, что читатель видел, а не во всём кодбуке
    assert shares_run("это было переписывание, при котором ничего не добавлено", seen, 4)
    assert not shares_run("трудна дорога от правды к истине", seen, 4)
    assert not shares_run("бескорыстное вранье это не ложь а поэзия", seen, 4)   # справка по новеллам читателю не видна
    assert "Калибровочные примеры" in p["theory"].system and "Калибровочные примеры" not in p["ghosts"].system
    assert "### K12" in p["ghosts"].system and "### UND" in p["ghosts"].system


def test_novellas_reference_is_outside_prompts():
    """Справка по сюжетам двенадцати новелл есть в кодбуке, но ни в один промпт не попадает."""
    from apophenia.prompts import codebook_sections, load_all, load_codebook
    c = load_config(ROOT / "config.kompromiss.yaml")
    ref = codebook_sections(load_codebook(c))["novellas"]
    assert all(f"→ K{i}." in ref for i in range(1, 13)) and ref.count("Как было:") == 12
    assert all("редактор Туронок дважды" not in p.rendered for p in load_all(c).values())
    assert class_definition(c, "K12").startswith("K12 — память") and "Мордовия" not in class_definition(c, "K12")


def test_ghost_defects_from_real_sheets():
    """Дефекты листов 00002 и 00003: формулировка кодбука и пересказ с пометкой — не призраки."""
    from apophenia.prompts import load_all
    from apophenia.verify_quotes import ghost_defect
    c = load_config(ROOT / "config.kompromiss.memory.yaml")
    seen, g = load_all(c)["ghosts"].system, c["ghosts"]
    assert ghost_defect("Речь звучала над чужим гробом", seen, g) == "codebook_echo"
    assert ghost_defect("[пересказ, не цитата] журналист изображает живой интерес к собеседнику", seen, g) == "not_a_quote"
    assert ghost_defect("примерно так: он сказал, что устал", seen, g) == "not_a_quote"
    for ok in ("Главное, чтобы мероприятие прошло организованно", "Жбанков уснул лицом в картофель",
               "Он читал по бумажке, не поднимая глаз"):
        assert ghost_defect(ok, seen, g) is None


def test_defective_ghosts_are_recorded_not_used(cfg):
    from apophenia.ghosts import collect_ghosts
    from apophenia.llm import LLMResponse
    from apophenia.prompts import load_all
    import json

    class Fake:
        def complete(self, role, system, user, temperature):
            ev = [{"class": "GWT", "span": s, "level": None, "exclusion_checked": True, "mapping": {"source": "a", "target": "b"}}
                  for s in ("[пересказ] машина думает о себе", "совсем посторонняя фраза про картофель и венок")]
            return LLMResponse(content=json.dumps({"evidence": ev}, ensure_ascii=False))

    corpus = [{"id": 0, "title": "А", "source": "a", "text": "Текст без этих слов.", "n_words": 4}]
    cfg["ghosts"] = {**cfg["ghosts"], "min_quotes": 1, "max_texts_per_cycle": 1}
    out = collect_ghosts(cfg, Fake(), load_all(cfg)["theory"], corpus, seed=1)
    assert [q["span"] for q in out["quotes"]] == ["совсем посторонняя фраза про картофель и венок"]
    assert out["readings"][0]["not_a_quote"] == ["[пересказ] машина думает о себе"]
