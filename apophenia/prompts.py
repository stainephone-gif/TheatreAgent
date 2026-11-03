"""Промпты: разбор блоков <!-- system --> / <!-- user --> и подстановка разделов кодбука.

Механика скопирована из AI-art pipeline/common.py, чтобы промпт классификатора
рендерился так же, как в исследовании (тот же hash при том же тексте).
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Any, Dict

from .config import resolve

_SECTION_RE = re.compile(
    r"<!--\s*section:(?P<name>[\w-]+)\s*-->(?P<body>.*?)<!--\s*/section:(?P=name)\s*-->", re.S
)


def sha256(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


@dataclass
class Prompt:
    name: str
    system: str
    user_template: str
    rendered: str
    hash: str
    codebook_hash: str

    def user(self, **fields: str) -> str:
        out = self.user_template
        for k, v in fields.items():
            out = out.replace("{" + k + "}", v)
        return out


def codebook_sections(codebook_text: str) -> Dict[str, str]:
    return {m.group("name"): m.group("body").strip() for m in _SECTION_RE.finditer(codebook_text)}


def load_codebook(cfg: Dict[str, Any]) -> str:
    p = resolve(cfg, cfg["prompts"]["dir"]) / cfg["prompts"]["codebook"]
    return p.read_text(encoding="utf-8")


def render_prompt_text(raw: str, codebook_text: str) -> str:
    sections = codebook_sections(codebook_text)

    def sub(m: re.Match) -> str:
        name = m.group(1)
        if name not in sections:
            raise KeyError(f"В кодбуке нет раздела section:{name}")
        return sections[name]

    return re.sub(r"\{\{\s*codebook:([\w-]+)\s*\}\}", sub, raw)


def _block(text: str, tag: str) -> str:
    m = re.search(rf"<!--\s*{tag}\s*-->(.*?)<!--\s*/{tag}\s*-->", text, re.S)
    if not m:
        raise ValueError(f"В промпте нет блока <!-- {tag} -->")
    return m.group(1).strip()


def load_prompt(cfg: Dict[str, Any], kind: str) -> Prompt:
    """kind: theory | explicate | rewrite | moderate."""
    pdir = resolve(cfg, cfg["prompts"]["dir"])
    name = cfg["prompts"][kind]
    raw = (pdir / name).read_text(encoding="utf-8")
    codebook = load_codebook(cfg)
    rendered = render_prompt_text(raw, codebook)
    return Prompt(
        name=name,
        system=_block(rendered, "system"),
        user_template=_block(rendered, "user"),
        rendered=rendered,
        hash=sha256(rendered)[:16],
        codebook_hash=sha256(codebook)[:16],
    )


def class_definition(cfg: Dict[str, Any], cls: str) -> str:
    """Абзац кодбука про один класс (для промпта переписывания)."""
    theory = codebook_sections(load_codebook(cfg)).get("theory", "")
    m = re.search(rf"### {cls} — (.*?)(?=\n### |\Z)", theory, re.S)
    return (cls + " — " + m.group(1).strip()) if m else cls


def load_all(cfg: Dict[str, Any]) -> Dict[str, Prompt]:
    out = {k: load_prompt(cfg, k) for k in ("theory", "explicate", "rewrite", "moderate")}
    # промпт призрачных цитат; если не задан — тот же промпт, что у классификатора
    out["ghosts"] = load_prompt(cfg, "ghosts") if cfg["prompts"].get("ghosts") else out["theory"]
    return out


def class_short_definitions(cfg: Dict[str, Any]) -> str:
    """Список всех классов по одной строке: код, название и первая строка определения из кодбука."""
    from . import ALL_CLASSES, CLASS_NAMES_RU  # noqa: PLC0415
    theory = codebook_sections(load_codebook(cfg)).get("theory", "")
    lines = []
    for cls in ALL_CLASSES:
        m = re.search(rf"### {cls} — [^\n]*\n([^\n]+)", theory)
        definition = m.group(1).strip() if m else ""
        if cls == "UND":
            definition = "ни одной теории в тексте не найдено."
        lines.append(f"- {cls} ({CLASS_NAMES_RU[cls]}): {definition}")
    return "\n".join(lines)
