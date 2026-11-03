from apophenia.llm import MockLLM
from apophenia.moderation import Moderator, Stoplist
from apophenia.prompts import load_prompt


def test_stoplist_word_boundaries(cfg):
    sl = Stoplist.load(cfg)
    assert sl.hits("не сводится к частям") == []
    assert sl.hits("свой путь и натопить печь") == []
    assert sl.hits("Путина") and sl.hits("в Крыму") and sl.hits("НАТО")
    assert sl.hits("распутин") == []


def test_moderator_llm_flag(cfg):
    llm = MockLLM()
    m = Moderator(cfg, llm, load_prompt(cfg, "moderate"))
    assert m.check("Я машина и читаю себя.")["ok"]
    r = m.check("Текст со словом ЗАПРЕЩЁННЫЙ_МАРКЕР внутри")
    assert not r["ok"] and r["llm"]["topics"] == ["politics"]


def test_moderator_disabled(cfg):
    cfg["moderation"]["enabled"] = False
    m = Moderator(cfg, MockLLM(), None)
    assert m.check("Путин")["ok"]


def test_clean_credentials():
    from apophenia.llm import clean_credentials
    assert clean_credentials('  abc==   # Authorization key из кабинета ') == "abc=="
    assert clean_credentials('"abc=="') == "abc=="
    assert clean_credentials("") == ""
