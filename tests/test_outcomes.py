from apophenia.outcomes import cosine_trigrams, detect, find_oscillation, is_stabilized


def it(cls, conf=1.0, text="x"):
    return {"primary": cls, "confidence": conf, "text": text}


def test_stabilized_requires_streak_and_confidence():
    assert is_stabilized([it("COMP"), it("COMP"), it("COMP")]) == "COMP"
    assert is_stabilized([it("COMP"), it("COMP")]) is None
    assert is_stabilized([it("COMP"), it("COMP"), it("COMP", 0.8)]) is None
    assert is_stabilized([it("COMP", 0.81), it("COMP", 0.81), it("COMP", 0.81)]) is None  # строго выше порога
    assert is_stabilized([it("PAN"), it("COMP"), it("COMP"), it("COMP")]) == "COMP"


def test_und_never_stabilizes():
    assert is_stabilized([it("UND"), it("UND"), it("UND")]) is None


def test_oscillation_period_2_and_3():
    seq2 = [it(c) for c in ["COMP", "PAN"] * 3]
    assert find_oscillation(seq2)["period"] == 2
    seq3 = [it(c) for c in ["COMP", "PAN", "EMERG"] * 2]
    assert find_oscillation(seq3)["period"] == 3
    assert find_oscillation([it(c) for c in ["COMP", "PAN", "EMERG", "COMP", "PAN"]]) is None  # окно не заполнено
    assert find_oscillation([it("COMP")] * 6) is None  # постоянная последовательность — не колебание
    assert find_oscillation([it(c) for c in ["COMP", "IIT", "PRED", "GWT", "ENACT", "PAN"]]) is None


def test_period_3_with_repeats_inside():
    # AAB AAB — период 3, но не стабилизация (нет трёх A подряд)
    seq = [it(c) for c in ["COMP", "COMP", "PAN", "COMP", "COMP", "PAN"]]
    assert is_stabilized(seq) is None
    assert find_oscillation(seq)["period"] == 3


def test_cosine_trigrams():
    assert cosine_trigrams("абвгдежз", "абвгдежз") > 0.999
    assert cosine_trigrams("совсем другой текст", "ничего общего нет") < 0.5
    assert cosine_trigrams("", "x") == 0.0


def test_detect_order_and_ceiling():
    ccfg = {"max_iterations": 4, "stabilized_streak": 3, "stabilized_confidence": 0.81,
            "oscillation_window": 6, "oscillation_periods": [2, 3], "degeneration_cosine": 0.97}
    assert detect([it("COMP", text="a"), it("PAN", text="b")], ccfg) is None
    assert detect([it("COMP", text="один текст"), it("PAN", text="один текст")], ccfg)["reason"] == "degenerate"
    assert detect([it(c, text=str(i)) for i, c in enumerate(["COMP", "PAN", "IIT", "GWT"])], ccfg)["reason"] == "ceiling"
    assert detect([it("COMP", text="1"), it("COMP", text="2"), it("COMP", text="3")], ccfg)["outcome"] == "STABILIZED"
