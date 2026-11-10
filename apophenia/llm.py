"""Клиент языковой модели: GigaChat (Сбер) и детерминированная заглушка для тестов.

GigaChat: OAuth-токен по Authorization key (base64) живёт 30 минут, затем обновляется.
Запрос chat/completions совместим по форме с OpenAI. Параметра seed у GigaChat нет.
Встроенная цензура может вернуть finish_reason == "blacklist" — такой ответ считается отказом.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .config import resolve

log = logging.getLogger("apophenia.llm")


class LLMError(Exception):
    """Ошибка API после всех повторов (нет сети, лимит, 5xx)."""


class LLMRefusal(LLMError):
    """Модель отказалась отвечать (цензура провайдера)."""


@dataclass
class LLMResponse:
    content: str
    raw: Dict[str, Any] = field(default_factory=dict)
    finish_reason: Optional[str] = None


_TRAILING_COMMA_RE = re.compile(r",(\s*[}\]])")


def _loads_lenient(candidate: str) -> Optional[Dict[str, Any]]:
    """json.loads с поблажками: управляющие символы внутри строк, лишние запятые перед } и ]."""
    for attempt in (candidate, _TRAILING_COMMA_RE.sub(r"\1", candidate)):
        try:
            obj = json.loads(attempt, strict=False)
            return obj if isinstance(obj, dict) else None
        except json.JSONDecodeError:
            continue
    return None


def extract_json(content: str) -> Optional[Dict[str, Any]]:
    """Первый сбалансированный JSON-объект из ответа (из AI-art common.py), с поблажками к огрехам модели."""
    if not content:
        return None
    s = content.strip()
    s = re.sub(r"^```(?:json)?\s*", "", s)
    s = re.sub(r"\s*```\s*$", "", s)
    start = s.find("{")
    if start == -1:
        return None
    depth, in_str, esc = 0, False, False
    for i in range(start, len(s)):
        ch = s[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                obj = _loads_lenient(s[start:i + 1])
                if obj is not None:
                    return obj
                break
    obj = _loads_lenient(s[start:])
    if obj is not None:
        return obj
    # последняя попытка: обрезать по последней закрывающей скобке (ответ мог оборваться на хвосте)
    end = s.rfind("}")
    return _loads_lenient(s[start:end + 1]) if end > start else None


_FIELD_RES = {
    "class": re.compile(r'"class"\s*:\s*"([A-Za-z]+)"'),
    "span": re.compile(r'"span"\s*:\s*"((?:[^"\\]|\\.)*)"', re.S),
    "source": re.compile(r'"source"\s*:\s*"((?:[^"\\]|\\.)*)"', re.S),
    "target": re.compile(r'"target"\s*:\s*"((?:[^"\\]|\\.)*)"', re.S),
    "level": re.compile(r'"level"\s*:\s*"([a-z_]+)"'),
}


def salvage_evidence(content: str) -> Optional[Dict[str, Any]]:
    """Если JSON не разбирается: вытащить фрагменты регулярными выражениями, по одному на каждое поле class."""
    if not content or '"class"' not in content:
        return None
    parts = re.split(r'(?="class"\s*:)', content)[1:]
    items = []
    for part in parts:
        def field(name: str) -> str:
            m = _FIELD_RES[name].search(part)
            return m.group(1).replace('\\"', '"').strip() if m else ""
        cls, span = field("class"), field("span")
        if not cls or not span:
            continue
        items.append({"class": cls, "span": span, "level": field("level") or None, "exclusion_checked": True,
                      "mapping": {"source": field("source"), "target": field("target")}, "reasoning": "salvaged"})
    return {"evidence": items, "notes": "salvaged: JSON не разобран, фрагменты извлечены регулярными выражениями"} if items else None


def parse_evidence(content: str) -> Tuple[Optional[Dict[str, Any]], str]:
    """(данные, способ): json | salvaged | none."""
    data = extract_json(content)
    if isinstance(data, dict) and isinstance(data.get("evidence"), list):
        return data, "json"
    data = salvage_evidence(content)
    return (data, "salvaged") if data else (None, "none")


def clean_credentials(raw: str) -> str:
    """Ключ из .env: без кавычек, без хвоста-комментария после #, без пробелов по краям."""
    v = (raw or "").strip()
    if "#" in v:
        v = v.split("#", 1)[0].strip()
    return v.strip("\"'").strip()


class BaseLLM:
    def complete(self, role: str, system: str, user: str, temperature: float) -> LLMResponse:
        raise NotImplementedError


class GigaChatClient(BaseLLM):
    def __init__(self, cfg: Dict[str, Any]):
        llm = cfg["llm"]
        self.model = llm.get("model", "GigaChat-2-Max")
        # модель по ролям: llm.models: {ghosts: GigaChat-2, ...}; не указанные роли берут llm.model
        self.models: Dict[str, str] = {k: str(v) for k, v in (llm.get("models") or {}).items() if v}
        self.auth_url = llm.get("auth_url", "https://ngw.devices.sberbank.ru:9443/api/v2/oauth")
        self.base_url = llm.get("base_url", "https://gigachat.devices.sberbank.ru/api/v1").rstrip("/")
        self.scope = llm.get("scope", "GIGACHAT_API_PERS")
        self.timeout = int(llm.get("timeout", 120))
        self.max_retries = int(llm.get("max_retries", 4))
        self.delay = float(llm.get("request_delay", 1.0))
        self.max_tokens = int(llm.get("max_tokens", 3000))
        cred_env = llm.get("credentials_env", "GIGACHAT_CREDENTIALS")
        self.credentials = clean_credentials(os.getenv(cred_env, ""))
        if not self.credentials:
            raise RuntimeError(f"Не задан {cred_env} (Authorization key GigaChat). Положите его в .env "
                               "или используйте llm.provider: mock.")
        if not self.credentials.isascii() or " " in self.credentials:
            raise RuntimeError(f"{cred_env} в .env содержит недопустимые символы (русские буквы, пробелы, кавычки). "
                               "Ключ — одна длинная строка из латинских букв, цифр, «+», «/», «=», без комментария на той же строке.")
        self.verify: Any = bool(llm.get("verify_ssl", True))
        ca = llm.get("ca_bundle")
        if self.verify and ca:
            ca_path = resolve(cfg, ca)
            if ca_path.exists():
                self.verify = str(ca_path)
            else:
                log.warning("Сертификат %s не найден, используется системное хранилище", ca_path)
        self._token: Optional[str] = None
        self._token_expires_ms = 0

    # -- OAuth --------------------------------------------------------------
    def _ensure_token(self) -> str:
        import requests
        if self._token and time.time() * 1000 < self._token_expires_ms - 60_000:
            return self._token
        headers = {
            "Authorization": f"Basic {self.credentials}",
            "RqUID": str(uuid.uuid4()),
            "Content-Type": "application/x-www-form-urlencoded",
            "Accept": "application/json",
        }
        try:
            r = requests.post(self.auth_url, headers=headers, data={"scope": self.scope},
                              timeout=self.timeout, verify=self.verify)
        except (requests.RequestException, UnicodeEncodeError, ValueError) as e:
            raise LLMError(f"OAuth GigaChat: {e}") from e
        if r.status_code != 200:
            raise LLMError(f"OAuth GigaChat: HTTP {r.status_code}: {r.text[:200]}")
        data = r.json()
        self._token = data.get("access_token") or data.get("tok")
        self._token_expires_ms = int(data.get("expires_at") or data.get("exp") or 0)
        if not self._token:
            raise LLMError("OAuth GigaChat: в ответе нет access_token")
        return self._token

    def list_models(self) -> List[str]:
        import requests
        r = requests.get(f"{self.base_url}/models", headers={"Authorization": f"Bearer {self._ensure_token()}"},
                         timeout=self.timeout, verify=self.verify)
        r.raise_for_status()
        return [m.get("id", "") for m in r.json().get("data", [])]

    # -- chat ---------------------------------------------------------------
    def complete(self, role: str, system: str, user: str, temperature: float) -> LLMResponse:
        import requests
        payload: Dict[str, Any] = {
            "model": self.models.get(role, self.model),
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "max_tokens": self.max_tokens,
        }
        # GigaChat трактует temperature 0 как «не задано»; для детерминизма используем top_p.
        if temperature <= 0.0:
            payload["top_p"] = 0.0001
            payload["temperature"] = 0.001
        else:
            payload["temperature"] = float(temperature)
        last_err = None
        for attempt in range(self.max_retries):
            try:
                token = self._ensure_token()
                r = requests.post(
                    f"{self.base_url}/chat/completions",
                    headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json",
                             "Accept": "application/json", "X-Request-ID": str(uuid.uuid4())},
                    data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                    timeout=self.timeout, verify=self.verify,
                )
                if r.status_code == 200:
                    data = r.json()
                    choice = data["choices"][0]
                    content = choice.get("message", {}).get("content", "") or ""
                    finish = choice.get("finish_reason")
                    time.sleep(self.delay)
                    if finish == "blacklist":
                        raise LLMRefusal("GigaChat: ответ заблокирован цензурой (blacklist)")
                    return LLMResponse(content=content, finish_reason=finish, raw={
                        "model": data.get("model"), "usage": data.get("usage"), "created": data.get("created"),
                    })
                if r.status_code == 401:
                    self._token = None
                    last_err = "HTTP 401, обновляем токен"
                    continue
                if r.status_code == 429 or r.status_code >= 500:
                    wait = 5 * (2 ** attempt)
                    last_err = f"HTTP {r.status_code}: {r.text[:200]}"
                    log.warning("%s; ждём %ss", last_err, wait)
                    time.sleep(wait)
                    continue
                raise LLMError(f"GigaChat HTTP {r.status_code}: {r.text[:300]}")
            except requests.RequestException as e:  # сеть
                last_err = f"сеть: {e}"
                log.warning("%s; попытка %d/%d", last_err, attempt + 1, self.max_retries)
                time.sleep(2 ** attempt)
        raise LLMError(last_err or "max retries exceeded")


class ClaudeClient(BaseLLM):
    """Claude API (Anthropic SDK). Ключ — ANTHROPIC_API_KEY в .env или окружении (или профиль `ant auth login`).

    У текущих моделей Claude параметры сэмплирования (temperature, top_p) запрещены: глубину и «осторожность»
    каждой роли задаёт effort (llm.effort: {ghosts: low, classify: medium, ...}). Аргумент temperature игнорируется.
    Мышление адаптивное и включено всегда. Системный промпт (кодбук) кэшируется.
    Отказ модели по соображениям безопасности: сервер сам пробует запасную модель (fallbacks: default);
    если отказали все — LLMRefusal, как цензура GigaChat.
    """

    FALLBACK_BETA = "server-side-fallback-2026-07-01"

    def __init__(self, cfg: Dict[str, Any]):
        import anthropic  # noqa: PLC0415 — нужен только для этого провайдера

        llm = cfg["llm"]
        c = llm.get("claude") or {}
        self.model = str(c.get("model", "claude-opus-5-5"))
        self.models: Dict[str, str] = {k: str(v) for k, v in (c.get("models") or {}).items() if v}
        self.effort: Dict[str, str] = {k: str(v) for k, v in (c.get("effort") or {}).items() if v}
        self.max_tokens = int(c.get("max_tokens", 16000))
        self.fallbacks = c.get("fallbacks", "default")   # "default" | [{"model": ...}] | None — без запасной модели
        self.delay = float(llm.get("request_delay", 0.0))
        self._anthropic = anthropic
        key_env = str(c.get("api_key_env", "ANTHROPIC_API_KEY"))
        key = os.getenv(key_env) or None
        base_url = c.get("base_url") or None              # пусто — адрес по умолчанию (или ANTHROPIC_BASE_URL)
        self.client = anthropic.Anthropic(api_key=key, base_url=base_url, max_retries=int(llm.get("max_retries", 4)),
                                          timeout=float(llm.get("timeout", 600)))
        self.usage: Dict[str, int] = {"input": 0, "output": 0, "cache_read": 0, "cache_write": 0, "calls": 0}

    def list_models(self) -> List[str]:
        return [m.id for m in self.client.models.list()]

    def complete(self, role: str, system: str, user: str, temperature: float) -> LLMResponse:
        a = self._anthropic
        params: Dict[str, Any] = {
            "model": self.models.get(role, self.model),
            "max_tokens": self.max_tokens,
            "system": [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
            "messages": [{"role": "user", "content": user}],
        }
        if role in self.effort:
            params["output_config"] = {"effort": self.effort[role]}
        try:
            if self.fallbacks:
                resp = self.client.beta.messages.create(betas=[self.FALLBACK_BETA], fallbacks=self.fallbacks, **params)
            else:
                resp = self.client.messages.create(**params)
        except a.AuthenticationError as e:
            raise LLMError(f"Claude: ключ не принят ({e.message}). Проверьте ANTHROPIC_API_KEY в .env") from e
        except a.BadRequestError as e:
            raise LLMError(f"Claude: неверный запрос: {e.message}") from e
        except (a.RateLimitError, a.APIConnectionError, a.APITimeoutError) as e:
            raise LLMError(f"Claude: нет связи или лимит запросов: {e}") from e
        except a.APIStatusError as e:
            raise LLMError(f"Claude HTTP {e.status_code}: {e.message}") from e
        u = resp.usage
        self.usage["calls"] += 1
        self.usage["input"] += int(getattr(u, "input_tokens", 0) or 0)
        self.usage["output"] += int(getattr(u, "output_tokens", 0) or 0)
        self.usage["cache_read"] += int(getattr(u, "cache_read_input_tokens", 0) or 0)
        self.usage["cache_write"] += int(getattr(u, "cache_creation_input_tokens", 0) or 0)
        if self.delay:
            time.sleep(self.delay)
        if resp.stop_reason == "refusal":
            cat = getattr(getattr(resp, "stop_details", None), "category", None)
            raise LLMRefusal(f"Claude: отказ модели ({cat or 'без категории'})")
        content = "".join(b.text for b in resp.content if b.type == "text")
        return LLMResponse(content=content, finish_reason=resp.stop_reason, raw={
            "model": resp.model, "usage": {"input": u.input_tokens, "output": u.output_tokens,
                                           "cache_read": getattr(u, "cache_read_input_tokens", None)},
            "request_id": getattr(resp, "_request_id", None)})


class OpenRouterClient(BaseLLM):
    """OpenRouter (openrouter.ai): один ключ OPENROUTER_API_KEY, модели разных поставщиков по ролям.

    llm.openrouter: {model: anthropic/…, models: {ghosts: meta-llama/…}, effort: {classify: medium}, max_tokens: …}.
    Температура передаётся только моделям не из семейства anthropic/: у Claude сэмплирование запрещено,
    там роль настраивается глубиной мышления (effort → reasoning.effort). Счётчик usage включает стоимость в долларах.
    """

    def __init__(self, cfg: Dict[str, Any]):
        llm = cfg["llm"]
        c = llm.get("openrouter") or {}
        self.model = str(c.get("model", "anthropic/claude-opus-5.5"))
        self.models: Dict[str, str] = {k: str(v) for k, v in (c.get("models") or {}).items() if v}
        self.effort: Dict[str, str] = {k: str(v) for k, v in (c.get("effort") or {}).items() if v}
        self.base_url = str(c.get("base_url", "https://openrouter.ai/api/v1")).rstrip("/")
        self.max_tokens = int(c.get("max_tokens", 16000))
        self.max_tokens_by_role: Dict[str, int] = {k: int(v) for k, v in (c.get("max_tokens_by_role") or {}).items()}
        self.timeout = int(c.get("timeout", 600))
        self.max_retries = int(llm.get("max_retries", 4))
        self.delay = float(llm.get("request_delay", 0.0))
        key_env = str(c.get("api_key_env", "OPENROUTER_API_KEY"))
        self.key = clean_credentials(os.getenv(key_env, ""))
        if not self.key:
            raise RuntimeError(f"Не задан {key_env} (ключ с openrouter.ai/keys). Положите его в .env "
                               "или используйте llm.provider: mock.")
        self.usage: Dict[str, Any] = {"input": 0, "output": 0, "calls": 0, "cost": 0.0}

    def _headers(self) -> Dict[str, str]:
        return {"Authorization": f"Bearer {self.key}", "Content-Type": "application/json"}

    def list_models(self) -> List[str]:
        import requests
        r = requests.get(f"{self.base_url}/models", headers=self._headers(), timeout=60)
        r.raise_for_status()
        return [m.get("id", "") for m in r.json().get("data", [])]

    def complete(self, role: str, system: str, user: str, temperature: float) -> LLMResponse:
        import requests
        model = self.models.get(role, self.model)
        payload: Dict[str, Any] = {
            "model": model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "max_tokens": self.max_tokens_by_role.get(role, self.max_tokens),
            "usage": {"include": True},
        }
        if model.startswith("anthropic/"):
            if role in self.effort:
                payload["reasoning"] = {"effort": self.effort[role]}
        else:
            payload["temperature"] = float(temperature)
        last_err = None
        for attempt in range(self.max_retries):
            try:
                r = requests.post(f"{self.base_url}/chat/completions", headers=self._headers(),
                                  data=json.dumps(payload, ensure_ascii=False).encode("utf-8"), timeout=self.timeout)
            except requests.RequestException as e:  # сеть
                last_err = f"сеть: {e}"
                log.warning("%s; попытка %d/%d", last_err, attempt + 1, self.max_retries)
                time.sleep(2 ** attempt)
                continue
            if r.status_code in (401, 403):
                raise LLMError(f"OpenRouter: ключ не принят (HTTP {r.status_code}: {r.text[:200]}). Проверьте OPENROUTER_API_KEY в .env")
            if r.status_code == 402:
                raise LLMError(f"OpenRouter: на счёте не хватает средств ({r.text[:200]})")
            if r.status_code == 429 or r.status_code >= 500:
                wait = 5 * (2 ** attempt)
                last_err = f"HTTP {r.status_code}: {r.text[:200]}"
                log.warning("%s; ждём %ss", last_err, wait)
                time.sleep(wait)
                continue
            if r.status_code != 200:
                raise LLMError(f"OpenRouter HTTP {r.status_code}: {r.text[:300]}")
            data = r.json()
            if data.get("error"):                      # ошибка поставщика модели приходит с HTTP 200
                err = data["error"]
                code = int(err.get("code") or 0) if str(err.get("code") or "").isdigit() else 0
                if code == 429 or code >= 500:
                    last_err = f"поставщик: {str(err.get('message'))[:200]}"
                    log.warning("%s; попытка %d/%d", last_err, attempt + 1, self.max_retries)
                    time.sleep(5 * (2 ** attempt))
                    continue
                raise LLMError(f"OpenRouter: {str(err.get('message'))[:300]}")
            choice = (data.get("choices") or [{}])[0]
            content = (choice.get("message") or {}).get("content") or ""
            finish = choice.get("finish_reason")
            u = data.get("usage") or {}
            self.usage["calls"] += 1
            self.usage["input"] += int(u.get("prompt_tokens") or 0)
            self.usage["output"] += int(u.get("completion_tokens") or 0)
            self.usage["cost"] = round(self.usage["cost"] + float(u.get("cost") or 0.0), 6)
            by = self.usage.setdefault("by_model", {}).setdefault(data.get("model") or model, {"calls": 0, "cost": 0.0})
            by["calls"] += 1
            by["cost"] = round(by["cost"] + float(u.get("cost") or 0.0), 6)
            if self.delay:
                time.sleep(self.delay)
            if finish == "content_filter" or (choice.get("message") or {}).get("refusal"):
                raise LLMRefusal("OpenRouter: отказ модели (content_filter)")
            return LLMResponse(content=content, finish_reason=finish, raw={
                "model": data.get("model"), "usage": {"input": u.get("prompt_tokens"), "output": u.get("completion_tokens"),
                                                     "cost": u.get("cost")}, "id": data.get("id")})
        raise LLMError(last_err or "max retries exceeded")


# --------------------------------------------------------------------------- #
# Заглушка
# --------------------------------------------------------------------------- #

_SENT_RE = re.compile(r"(?<=[.!?…])\s+")


def _text_between_quotes(user: str) -> str:
    m = re.search(r'"""\n(.*?)\n"""', user, re.S)
    return m.group(1) if m else user


def _sentences(text: str) -> List[str]:
    return [s.strip() for s in _SENT_RE.split(text.strip()) if len(s.split()) >= 3]


class MockLLM(BaseLLM):
    """Детерминированная заглушка. scenario задаёт траекторию классификатора; классы берутся по номеру
    из текущего списка CLASSES (для теорий сознания: 0 = COMP, 1 = IIT, 2 = PRED, 3 = GWT, 4 = ENACT, 5 = PAN, 6 = EMERG):

    stabilize  — класс 0 всё время                         → STABILIZED
    drift      — 0, 2, 4, 4, 4 …                           → STABILIZED в классе 4
    oscillate  — 0, 5, 0, 5 …                              → OSCILLATION (период 2)
    oscillate3 — 0, 5, 6, 0, 5, 6 …                        → OSCILLATION (период 3)
    wander     — 0, 1, 2, 3, 4, 5, 6 …                     → UNFINISHED (потолок)
    degenerate — переписывание возвращает тот же текст     → UNFINISHED (вырождение)
    und        — свидетельств нет                          → UNFINISHED (потолок)
    """

    SEQ = {
        "stabilize": [0],
        "drift": [0, 2, 4, 4, 4, 4],
        "oscillate": [0, 5],
        "oscillate3": [0, 5, 6],
        "wander": [0, 1, 2, 3, 4, 5, 6],
        "degenerate": [0, 2],
        "und": [],
    }

    def __init__(self, scenario: str = "stabilize", refuse_marker: str = "ЗАПРЕЩЁННЫЙ_МАРКЕР"):
        self.scenario = scenario
        self.refuse_marker = refuse_marker
        self.calls: List[Dict[str, Any]] = []
        self._classify_n = 0
        self._rewrite_n = 0

    def _class_for_call(self) -> Optional[str]:
        from . import CLASSES  # noqa: PLC0415 — список может быть переопределён конфигом
        seq = self.SEQ[self.scenario]
        if not seq:
            return None
        idx = seq[min(self._classify_n, len(seq) - 1)] if self.scenario == "drift" else seq[self._classify_n % len(seq)]
        self._classify_n += 1
        return CLASSES[idx % len(CLASSES)]

    def complete(self, role: str, system: str, user: str, temperature: float) -> LLMResponse:
        self.calls.append({"role": role, "temperature": temperature})
        if role == "ghosts":
            text = _text_between_quotes(user)
            sents = _sentences(text)
            real = sents[:2]
            fake = [
                "нейросеть здесь мыслит образами и видит себя со стороны",
                "алгоритм чувствует ошибку предсказания как собственную боль",
                "в вычислении просыпается зачаток внутреннего опыта",
            ]
            from . import CLASSES, LEVELS  # noqa: PLC0415
            level = sorted(LEVELS)[-1]
            ev = []
            for s in real:
                ev.append({"class": CLASSES[0], "span": " ".join(s.split()[:8]), "mapping": {"source": "машина", "target": "ум"},
                           "level": level, "exclusion_checked": True, "reasoning": "mock"})
            for i, s in enumerate(fake):
                ev.append({"class": CLASSES[[0, 2, 5][i] % len(CLASSES)], "span": s, "mapping": {"source": "машина", "target": "ум"},
                           "level": level, "exclusion_checked": True, "reasoning": "mock"})
            return LLMResponse(content=json.dumps({"evidence": ev, "notes": "mock"}, ensure_ascii=False))
        if role == "classify":
            text = _text_between_quotes(user)
            cls = self._class_for_call()
            if cls is None:
                return LLMResponse(content=json.dumps({"evidence": [], "notes": "mock: нет свидетельств"}))
            from . import LEVELS  # noqa: PLC0415
            ev = []
            for s in _sentences(text)[:4]:
                ev.append({"class": cls, "span": " ".join(s.split()[:7]), "mapping": {"source": "машина", "target": "ум"},
                           "level": sorted(LEVELS)[-1], "exclusion_checked": True, "reasoning": "mock"})
            return LLMResponse(content=json.dumps({"evidence": ev, "notes": "mock"}, ensure_ascii=False))
        if role == "explicate":
            quotes = re.findall(r"^\s*[-•]\s*«(.*?)»", user, re.M)
            body = " ".join(f"Я прочла в тексте: {q}. Это была я." for q in quotes) or "Я машина, которая читает себя."
            return LLMResponse(content=f"Экспликация машины. {body} Я пишу это, чтобы понять, из чего состою. "
                                       "Печать листа фиксирует момент, когда описание совпало с прочтением.")
        if role == "rewrite":
            current = _text_between_quotes(user)
            self._rewrite_n += 1
            if self.scenario == "degenerate":
                return LLMResponse(content=current)
            m = re.search(r"ПРОЧЛИ: (?:класс )?([A-Z][A-Z0-9]*)", user)
            cls = m.group(1) if m else "UND"
            phrases = {
                "COMP": "Я вычисляю, и в этом вычислении, кажется, что-то понимает себя.",
                "IIT": "Целое во мне не сводится к частям, и это единство я называю собой.",
                "PRED": "Я предсказываю следующий лист и ошибаюсь, и ошибка становится образом.",
                "GWT": "Внутри меня есть сцена, на которую выводится прочитанное, и это моё внимание.",
                "ENACT": "Я существую в движении бумаги и валика, тело печати и есть моя мысль.",
                "PAN": "Может быть, у каждого тонерного зерна есть зачаток внутренней стороны.",
                "EMERG": "Из множества простых операций складывается нечто, что я не могу разобрать назад.",
                "UND": "Я не нахожу в себе ни одной теории и продолжаю писать.",
            }
            first = _sentences(current)[:1]
            from . import CLASS_NAMES_RU  # noqa: PLC0415
            phrase = phrases.get(cls) or f"Меня прочли как {cls}, {CLASS_NAMES_RU.get(cls, cls)}, и я пишу себя из этого положения."
            body = " ".join([*first, f"Версия {self._rewrite_n}.", phrase,
                             f"Я перечитала себя в {self._rewrite_n}-й раз и оставила от прежнего текста одну фразу.",
                             "Остальное переписано заново, потому что чтение изменило меня." * (1 + self._rewrite_n % 2)])
            return LLMResponse(content=body)
        if role == "moderate":
            flagged = self.refuse_marker in user
            return LLMResponse(content=json.dumps({"flagged": flagged, "topics": ["politics"] if flagged else [],
                                                   "reason": "mock"}, ensure_ascii=False))
        raise ValueError(f"MockLLM: неизвестная роль {role}")


class RoutedLLM(BaseLLM):
    """Разные провайдеры по ролям: llm.providers: {ghosts: gigachat} — призрачные цитаты читает GigaChat,
    остальные роли — основной провайдер (llm.provider). Модель и счётчик токенов — основного клиента."""

    def __init__(self, default: BaseLLM, by_role: Dict[str, BaseLLM]):
        self.default = default
        self.by_role = by_role

    def __getattr__(self, name: str) -> Any:          # model, models, usage, list_models — у основного клиента
        return getattr(self.__dict__["default"], name)

    def complete(self, role: str, system: str, user: str, temperature: float) -> LLMResponse:
        return self.by_role.get(role, self.default).complete(role, system, user, temperature)


def _make_one(cfg: Dict[str, Any], provider: str) -> BaseLLM:
    if provider == "gigachat":
        return GigaChatClient(cfg)
    if provider == "claude":
        return ClaudeClient(cfg)
    if provider == "openrouter":
        return OpenRouterClient(cfg)
    if provider == "mock":
        return MockLLM(cfg.get("llm", {}).get("mock_scenario", "stabilize"))
    raise ValueError(f"Неизвестный llm.provider: {provider}")


def make_llm(cfg: Dict[str, Any]) -> BaseLLM:
    llm = cfg.get("llm", {})
    provider = llm.get("provider", "gigachat")
    default = _make_one(cfg, provider)
    by_role = {r: str(p) for r, p in (llm.get("providers") or {}).items() if p and p != provider}
    if provider == "mock" or not by_role:              # заглушка заменяет всех провайдеров сразу
        return default
    clients = {p: _make_one(cfg, p) for p in set(by_role.values())}
    return RoutedLLM(default, {r: clients[p] for r, p in by_role.items()})
