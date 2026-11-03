from apophenia.reading import aggregate, validate_theory_response

TEXT = "Нейросеть здесь выступает как соавтор, который мыслит образами. Машина считает, но ничего не чувствует. " \
       "Каждый пиксель в этой системе обладает зачатком опыта."
VCFG = {"token_overlap_threshold": 0.7, "require_exact": False, "min_span_words": 2, "require_exclusion_checked": True}


def ev(cls, span, **kw):
    d = {"class": cls, "span": span, "mapping": {"source": "машина", "target": "ум"}, "level": "meta_metaphor", "exclusion_checked": True}
    d.update(kw)
    return d


def test_unverified_spans_become_ghosts():
    resp = {"evidence": [ev("COMP", "соавтор, который мыслит образами"), ev("PAN", "алгоритм видит сны о себе"),
                         ev("COMP", "одно"), ev("PRED", "машина считает, но", mapping=None), ev("XXX", "машина считает")]}
    records, stats = validate_theory_response(resp, TEXT, VCFG)
    reasons = [r["drop_reason"] for r in records]
    assert reasons == [None, "unverified", "span_too_short", "no_mapping", "invalid_class"]
    assert stats["n_kept"] == 1 and stats["dropped"]["unverified"] == 1


def test_aggregate_confidence_is_share_of_primary():
    resp = {"evidence": [ev("COMP", "соавтор, который мыслит образами"), ev("COMP", "машина считает, но ничего не чувствует"),
                         ev("PAN", "пиксель в этой системе обладает зачатком опыта")]}
    records, _ = validate_theory_response(resp, TEXT, VCFG)
    agg = aggregate(records, TEXT)
    assert agg["primary"] == "COMP"
    assert abs(agg["confidence"] - 2 / 3) < 1e-3
    assert agg["n_kept"] == 3


def test_aggregate_und_when_nothing_kept():
    records, _ = validate_theory_response({"evidence": []}, TEXT, VCFG)
    agg = aggregate(records, TEXT)
    assert agg["primary"] == "UND" and agg["confidence"] == 0.0


def test_duplicates_of_same_class_are_dropped():
    resp = {"evidence": [ev("COMP", "соавтор, который мыслит образами"), ev("COMP", "который мыслит образами")]}
    records, _ = validate_theory_response(resp, TEXT, VCFG)
    assert [r["drop_reason"] for r in records] == [None, "duplicate"]


def test_extract_json_lenient_and_salvage():
    from apophenia.llm import extract_json, parse_evidence, salvage_evidence
    # перенос строки внутри строки и лишняя запятая
    raw = '{"evidence": [{"class": "COMP", "span": "первая\nстрока", "mapping": {"source": "a", "target": "b"},}], "notes": "x",}'
    assert extract_json(raw)["evidence"][0]["span"] == "первая\nстрока"
    # обрыв ответа: JSON не закрыт → регулярки
    raw2 = '{"evidence": [{"class": "PRED", "span": "машина видит сон", "mapping": {"source": "ум", "target": "машина"}, "level": "meta_metaphor"}, {"class": "PAN", "span": "тонер чувствует'
    data, how = parse_evidence(raw2)
    assert how == "salvaged" and data["evidence"][0]["span"] == "машина видит сон" and data["evidence"][0]["mapping"]["source"] == "ум"
    assert salvage_evidence("нет ничего") is None
    assert parse_evidence("Свидетельств нет.") == (None, "none")


def test_tie_not_always_comp():
    from apophenia.reading import aggregate
    text = "Машина считает, но ничего не чувствует. Каждый пиксель в этой системе обладает зачатком опыта."
    recs = [
        {"kept": True, "class": "COMP", "span": "машина считает но ничего", "drop_reason": None},
        {"kept": True, "class": "PAN", "span": "пиксель обладает зачатком опыта", "drop_reason": None},
    ]
    winners = {aggregate(recs, text + str(i))["primary"] for i in range(40)}
    assert winners == {"COMP", "PAN"}
