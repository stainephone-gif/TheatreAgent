"""Клиент OpenRouter без сети: параметры запроса по ролям, счётчик стоимости, ошибки."""

import json
from types import SimpleNamespace

import pytest

from apophenia.config import load_config
from apophenia.llm import LLMError, LLMRefusal, OpenRouterClient, make_llm
from tests.conftest import ROOT


def _cfg():
    return {"llm": {"provider": "openrouter", "max_retries": 1, "openrouter": {
        "model": "anthropic/claude-opus-5.5", "models": {"ghosts": "meta-llama/llama-3.1-8b-instruct"},
        "effort": {"classify": "medium"}, "max_tokens_by_role": {"ghosts": 3000}}}}


def _resp(status=200, body=None):
    body = body if body is not None else {
        "id": "gen-1", "model": "m", "choices": [{"message": {"content": "ответ"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "cost": 0.002}}
    return SimpleNamespace(status_code=status, json=lambda: body, text=json.dumps(body))


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    c = make_llm(_cfg())
    assert isinstance(c, OpenRouterClient)
    calls, box = [], {"next": _resp()}

    def post(url, headers=None, data=None, timeout=None):
        calls.append({"url": url, "headers": headers, "body": json.loads(data.decode("utf-8"))})
        return box["next"]

    import requests
    monkeypatch.setattr(requests, "post", post)
    return c, calls, box


def test_roles_go_to_their_models(client):
    c, calls, _ = client
    c.complete("ghosts", "s", "u", 1.0)
    c.complete("classify", "s", "u", 0.0)
    ghosts, classify = calls[0]["body"], calls[1]["body"]
    assert calls[0]["url"].endswith("/chat/completions") and calls[0]["headers"]["Authorization"] == "Bearer sk-or-test"
    assert ghosts["model"] == "meta-llama/llama-3.1-8b-instruct" and ghosts["temperature"] == 1.0 and ghosts["max_tokens"] == 3000
    assert classify["model"] == "anthropic/claude-opus-5.5" and "temperature" not in classify
    assert classify["reasoning"] == {"effort": "medium"} and "reasoning" not in ghosts
    assert c.usage["calls"] == 2 and c.usage["input"] == 20 and c.usage["cost"] == pytest.approx(0.004)


def test_errors_and_refusal(client):
    c, _, box = client
    box["next"] = _resp(401, {"error": {"message": "No auth"}})
    with pytest.raises(LLMError, match="ключ не принят"):
        c.complete("classify", "s", "u", 0.0)
    box["next"] = _resp(200, {"error": {"code": 400, "message": "no such model"}})
    with pytest.raises(LLMError, match="no such model"):
        c.complete("classify", "s", "u", 0.0)
    box["next"] = _resp(200, {"choices": [{"message": {"content": ""}, "finish_reason": "content_filter"}], "usage": {}})
    with pytest.raises(LLMRefusal):
        c.complete("classify", "s", "u", 0.0)


def test_key_required(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY"):
        make_llm(_cfg())


def test_kompromiss_openrouter_config(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    try:
        c = load_config(ROOT / "config.kompromiss.openrouter.yaml")
    finally:
        load_config(ROOT / "config.yaml")                        # вернуть глобальный список классов выставки
    llm = make_llm(c)
    assert isinstance(llm, OpenRouterClient)                     # один клиент на все роли, без GigaChat
    assert llm.models["ghosts"].startswith("meta-llama/") and llm.model.startswith("anthropic/")
    assert c["prompts"]["ghosts"] == "ghosts_kompromiss_v1.md" and c["ghosts"]["codebook_echo_words"] == 4


def test_openrouter_memory_config_reads_from_memory(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    try:
        c = load_config(ROOT / "config.kompromiss.openrouter.memory.yaml")
    finally:
        load_config(ROOT / "config.yaml")
    llm = make_llm(c)
    assert llm.models["ghosts"].startswith("anthropic/") and llm.max_tokens_by_role["ghosts"] == 16000
    assert c["prompts"]["ghosts"] == "ghosts_kompromiss_memory_v1.md" and llm.effort["ghosts"] == "low"
