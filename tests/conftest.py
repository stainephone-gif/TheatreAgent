import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from apophenia.config import load_config  # noqa: E402


@pytest.fixture
def cfg(tmp_path):
    c = load_config(ROOT / "config.yaml")
    c["llm"]["provider"] = "mock"
    c["printer"]["enabled"] = False
    c["display"]["enabled"] = False
    c["telegram"]["enabled"] = False
    for k in ("archive", "rejected", "queue", "state", "logs"):
        c["paths"][k] = str(tmp_path / k)
    return c


@pytest.fixture(scope="session")
def main_mod():
    """Главный скрипт apophenia.py (одноимённый с пакетом apophenia/), загружается по пути."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("apophenia_main", ROOT / "apophenia.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod
