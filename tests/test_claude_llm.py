"""Клиент Claude API без сети: параметры запроса, отказ модели, ошибки."""

from types import SimpleNamespace

import pytest

anthropic = pytest.importorskip("anthropic")

from apophenia.llm import ClaudeClient, LLMError, LLMRefusal, make_llm  # noqa: E402


def _cfg(**claude):
    return {"llm": {"provider": "claude", "max_retries": 0, "claude": {"effort": {"classify": "medium", "ghosts": "low"}, **claude}}}


def _resp(text="ответ", stop="end_turn", category=None):
    return SimpleNamespace(
        content=[SimpleNamespace(type="thinking", thinking=""), SimpleNamespace(type="text", text=text)],
        stop_reason=stop, stop_details=SimpleNamespace(category=category) if category else None,
        model="claude-opus-5-5", _request_id="req_1",
        usage=SimpleNamespace(input_tokens=10, output_tokens=5, cache_read_input_tokens=7, cache_creation_input_tokens=0))


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    c = make_llm(_cfg())
    assert isinstance(c, ClaudeClient)
    calls = []

    def create(**kw):
        calls.append(kw)
        return c._next

    c._next = _resp()
    c.client = SimpleNamespace(beta=SimpleNamespace(messages=SimpleNamespace(create=create)),
                               messages=SimpleNamespace(create=create))
    return c, calls


def test_request_shape(client):
    c, calls = client
    r = c.complete("classify", "КОДБУК", "ТЕКСТ", 0.0)
    assert r.content == "ответ" and r.finish_reason == "end_turn"
    kw = calls[0]
    assert kw["model"] == "claude-opus-5-5"
    assert "temperature" not in kw and "top_p" not in kw and "thinking" not in kw
    assert kw["output_config"] == {"effort": "medium"}
    assert kw["system"][0]["cache_control"] == {"type": "ephemeral"} and kw["system"][0]["text"] == "КОДБУК"
    assert kw["fallbacks"] == "default" and kw["betas"] == [ClaudeClient.FALLBACK_BETA]
    c.complete("rewrite", "s", "u", 0.8)
    assert "output_config" not in calls[1]          # роль без effort — значение модели по умолчанию
    assert c.usage["calls"] == 2 and c.usage["cache_read"] == 14


def test_refusal_raises(client):
    c, _ = client
    c._next = _resp(text="", stop="refusal", category="cyber")
    with pytest.raises(LLMRefusal, match="cyber"):
        c.complete("ghosts", "s", "u", 1.0)


def test_api_errors_become_llm_error(client):
    c, _ = client
    req = anthropic._base_client.httpx.Request("POST", "https://api.anthropic.com/v1/messages") if hasattr(anthropic._base_client, "httpx") else None

    def boom(**kw):
        raise anthropic.APIConnectionError(request=req)

    c.client.beta.messages.create = boom
    with pytest.raises(LLMError):
        c.complete("classify", "s", "u", 0.0)


def test_no_fallbacks_uses_plain_endpoint(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    c = make_llm(_cfg(fallbacks=None, models={"ghosts": "claude-sonnet-5-5"}))
    calls = []
    c.client = SimpleNamespace(messages=SimpleNamespace(create=lambda **kw: calls.append(kw) or _resp()))
    c.complete("ghosts", "s", "u", 1.0)
    assert calls[0]["model"] == "claude-sonnet-5-5" and "betas" not in calls[0] and calls[0]["output_config"] == {"effort": "low"}


def test_providers_by_role_route_ghosts_to_gigachat(monkeypatch):
    """llm.providers: {ghosts: gigachat} — чтение корпуса идёт в GigaChat, остальные роли в Claude."""
    from apophenia import llm as L
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    monkeypatch.setenv("GIGACHAT_CREDENTIALS", "abc")
    cfg = _cfg()
    cfg["llm"]["providers"] = {"ghosts": "gigachat", "classify": "claude"}
    c = make_llm(cfg)
    assert isinstance(c, L.RoutedLLM) and isinstance(c.default, ClaudeClient)
    assert set(c.by_role) == {"ghosts"} and isinstance(c.by_role["ghosts"], L.GigaChatClient)
    assert c.model == "claude-opus-5-5" and c.usage["calls"] == 0
    seen = []
    c.default.complete = lambda role, *a: seen.append(("claude", role))
    c.by_role["ghosts"].complete = lambda role, *a: seen.append(("gigachat", role))
    c.complete("ghosts", "s", "u", 1.0)
    c.complete("classify", "s", "u", 0.0)
    assert seen == [("gigachat", "ghosts"), ("claude", "classify")]


def test_mock_ignores_providers():
    from apophenia.llm import MockLLM
    assert isinstance(make_llm({"llm": {"provider": "mock", "providers": {"ghosts": "gigachat"}}}), MockLLM)
