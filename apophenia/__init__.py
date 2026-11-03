"""Метасознание: генеративная инсталляция. Ядро службы apophenia.py."""

__version__ = "0.1.0"

CLASSES = ["COMP", "IIT", "PRED", "GWT", "ENACT", "PAN", "EMERG"]
ALL_CLASSES = CLASSES + ["UND"]

CLASS_NAMES_RU = {
    "COMP": "вычислительный функционализм",
    "IIT": "интегрированная информация",
    "PRED": "предиктивная обработка",
    "GWT": "глобальное рабочее пространство",
    "ENACT": "энактивизм",
    "PAN": "панпсихизм",
    "EMERG": "эмерджентизм",
    "UND": "не определено",
}

OUTCOMES = {
    "STABILIZED": "стабилизация",
    "OSCILLATION": "колебание",
    "UNFINISHED": "не завершено",
}
