"""Метасознание: генеративная инсталляция. Ядро службы apophenia.py."""

from typing import Any, Dict, List

__version__ = "0.2.0"

# Классы кодбука по умолчанию — восемь теорий сознания (исследование AI-art).
# Другой кодбук (например, позиции рассказчика у Довлатова) задаётся в конфиге ключом `classes`;
# configure_classes() переписывает эти списки на месте, так что модули, импортировавшие их, видят новые значения.
DEFAULT_CLASSES: List[Dict[str, str]] = [
    {"code": "COMP", "name": "вычислительный функционализм"},
    {"code": "IIT", "name": "интегрированная информация"},
    {"code": "PRED", "name": "предиктивная обработка"},
    {"code": "GWT", "name": "глобальное рабочее пространство"},
    {"code": "ENACT", "name": "энактивизм"},
    {"code": "PAN", "name": "панпсихизм"},
    {"code": "EMERG", "name": "эмерджентизм"},
]
DEFAULT_LEVELS = ["explicit", "scientific_metaphor", "meta_metaphor"]

CLASSES: List[str] = [c["code"] for c in DEFAULT_CLASSES]
ALL_CLASSES: List[str] = CLASSES + ["UND"]
CLASS_NAMES_RU: Dict[str, str] = {**{c["code"]: c["name"] for c in DEFAULT_CLASSES}, "UND": "не определено"}
LEVELS = set(DEFAULT_LEVELS)

OUTCOMES = {
    "STABILIZED": "стабилизация",
    "OSCILLATION": "колебание",
    "UNFINISHED": "не завершено",
}


def configure_classes(cfg: Dict[str, Any]) -> None:
    """Классы и уровни из конфига (`classes: [{code, name}]`, `levels: [...]`); без них — теории сознания."""
    items = cfg.get("classes") or DEFAULT_CLASSES
    codes, names = [], {}
    for it in items:
        code = str(it["code"]).upper().strip()
        if code == "UND" or code in codes:
            raise ValueError(f"classes: код {code} повторяется или зарезервирован")
        codes.append(code)
        names[code] = str(it.get("name", code))
    names["UND"] = str(cfg.get("und_name", "не определено"))
    CLASSES[:] = codes
    ALL_CLASSES[:] = codes + ["UND"]
    CLASS_NAMES_RU.clear()
    CLASS_NAMES_RU.update(names)
    LEVELS.clear()
    LEVELS.update(str(x) for x in (cfg.get("levels") or DEFAULT_LEVELS))
