"""Загрузка config.yaml и .env. Все пути в конфиге относительны корню репозитория."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict

import yaml

ROOT = Path(__file__).resolve().parent.parent


def _load_plain(path: Path) -> Dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    cfg.pop("extends", None)
    cfg["_root"] = str(path.parent)
    return cfg


def _deep_merge(base: Dict[str, Any], over: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


LOCAL_NAME = "config.local.yaml"


def _prefer_local(path: Path) -> Path:
    """config.yaml → config.local.yaml, если он лежит рядом (локальные настройки, не в git)."""
    if path.name == "config.yaml":
        local = path.with_name(LOCAL_NAME)
        if local.exists():
            return local
    return path


def load_config(path: str | Path | None = None) -> Dict[str, Any]:
    """config.yaml; файл с ключом `extends: другой.yaml` наследует его и переопределяет только указанные поля.

    Без аргумента берётся config.local.yaml, если он есть, иначе config.yaml. Ссылка `extends: config.yaml`
    тоже указывает на config.local.yaml, если тот существует: так локальные настройки (принтер, расписание)
    действуют и в тестовом режиме, а сам config.yaml остаётся нетронутым для git pull.
    """
    path = Path(path) if path else ROOT / "config.yaml"
    path = _prefer_local(path.resolve())
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    parent = cfg.pop("extends", None)
    if parent:
        parent_path = Path(parent)
        if not parent_path.is_absolute():
            parent_path = path.parent / parent_path
        parent_path = parent_path.resolve()
        if parent_path != path:
            parent_path = _prefer_local(parent_path)
        if parent_path == path:  # config.local.yaml extends config.yaml — не зацикливаться
            parent_path = path.with_name("config.yaml")
        base = load_config(parent_path) if parent_path.name != "config.yaml" else _load_plain(parent_path)
        base = {k: v for k, v in base.items() if not k.startswith("_")}
        cfg = _deep_merge(base, cfg)
    cfg["_root"] = str(path.parent)
    cfg["_config_path"] = str(path)
    try:
        from dotenv import load_dotenv
        load_dotenv(path.parent / ".env")
    except ImportError:
        pass
    return cfg


def resolve(cfg: Dict[str, Any], p: str | Path) -> Path:
    p = Path(p)
    if p.is_absolute():
        return p
    return Path(cfg.get("_root", ".")) / p


def path_dir(cfg: Dict[str, Any], key: str) -> Path:
    d = resolve(cfg, cfg["paths"][key])
    d.mkdir(parents=True, exist_ok=True)
    return d


def env(name: str, default: str = "") -> str:
    return os.getenv(name, default) or default
