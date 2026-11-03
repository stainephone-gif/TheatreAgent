import json

import pytest

from apophenia.corpus import load_corpus
from apophenia.cycle import CycleRejected, Engine
from apophenia.llm import LLMResponse, MockLLM
from apophenia.moderation import Moderator
from apophenia.prompts import load_all
from apophenia.sheet import render_sheet


def make_engine(cfg, scenario):
    llm = MockLLM(scenario)
    prompts = load_all(cfg)
    corpus = load_corpus(cfg)
    return Engine(cfg, llm, prompts, corpus, Moderator(cfg, llm, prompts["moderate"])), llm


@pytest.mark.parametrize("scenario,outcome,extra", [
    ("stabilize", "STABILIZED", {"class": "COMP"}),
    ("drift", "STABILIZED", {"class": "ENACT"}),
    ("oscillate", "OSCILLATION", {"period": 2}),
    ("oscillate3", "OSCILLATION", {"period": 3}),
    ("wander", "UNFINISHED", {"reason": "ceiling"}),
    ("degenerate", "UNFINISHED", {"reason": "degenerate"}),
    ("und", "UNFINISHED", {"reason": "ceiling"}),
])
def test_scenarios(cfg, scenario, outcome, extra):
    eng, llm = make_engine(cfg, scenario)
    rec = eng.run(7)
    assert rec["outcome"]["outcome"] == outcome
    for k, v in extra.items():
        assert rec["outcome"][k] == v
    assert rec["cycle"] == 7
    assert len(rec["ghosts"]["quotes"]) >= cfg["ghosts"]["min_quotes"]
    assert all(q["token_overlap"] is not None for q in rec["ghosts"]["quotes"])
    assert len(rec["trajectory"]) == len(rec["iterations"]) <= cfg["cycle"]["max_iterations"]
    temps = {c["role"]: c["temperature"] for c in llm.calls}
    assert temps["ghosts"] == 1.0 and temps["classify"] == 0.0 and temps["explicate"] == 0.8
    json.dumps(rec, ensure_ascii=False)  # сериализуемо


def test_rejected_when_explication_flagged(cfg, monkeypatch):
    eng, llm = make_engine(cfg, "stabilize")
    orig = llm.complete

    def bad(role, system, user, temperature):
        if role == "explicate":
            return LLMResponse(content="Текст про ЗАПРЕЩЁННЫЙ_МАРКЕР и больше ничего.")
        return orig(role, system, user, temperature)

    monkeypatch.setattr(llm, "complete", bad)
    with pytest.raises(CycleRejected) as ei:
        eng.run(1)
    assert ei.value.reason == "moderation:explicate"
    assert len(ei.value.info["attempts"]) == cfg["moderation"]["max_regenerations"] + 1


def test_sheet_renders_one_page(cfg, tmp_path):
    eng, _ = make_engine(cfg, "wander")
    rec = eng.run(3)
    rec["final_text"] = ("Очень длинный текст экспликации. " * 120).strip()
    pdf = render_sheet(rec, cfg, tmp_path / "s.pdf")
    data = pdf.read_bytes()
    assert data.startswith(b"%PDF") and data.count(b"/Type /Page\n") + data.count(b"/Type /Page>") <= 1
    assert data.count(b"/Type /Page") - data.count(b"/Type /Pages") == 1
