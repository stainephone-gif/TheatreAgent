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
    """Детерминированная заглушка. scenario задаёт траекторию классификатора:

    stabilize  — COMP, COMP, COMP …                       → STABILIZED
    drift      — COMP, PRED, ENACT, ENACT, ENACT …         → STABILIZED в ENACT
    oscillate  — COMP, PAN, COMP, PAN …                    → OSCILLATION (период 2)
    oscillate3 — COMP, PAN, EMERG, COMP, PAN, EMERG …      → OSCILLATION (период 3)
    wander     — COMP, IIT, PRED, GWT, ENACT, PAN, EMERG … → UNFINISHED (потолок)
    degenerate — переписывание возвращает тот же текст     → UNFINISHED (вырождение)
    und        — свидетельств нет                          → UNFINISHED (потолок)
    """

    SEQ = {
        "stabilize": ["COMP"],
        "drift": ["COMP", "PRED", "ENACT", "ENACT", "ENACT", "ENACT"],
        "oscillate": ["COMP", "PAN"],
        "oscillate3": ["COMP", "PAN", "EMERG"],
        "wander": ["COMP", "IIT", "PRED", "GWT", "ENACT", "PAN", "EMERG"],
        "degenerate": ["COMP", "PRED"],
        "und": [],
    }

    def __init__(self, scenario: str = "stabilize", refuse_marker: str = "ЗАПРЕЩЁННЫЙ_МАРКЕР"):
        self.scenario = scenario
        self.refuse_marker = refuse_marker
        self.calls: List[Dict[str, Any]] = []
        self._classify_n = 0
        self._rewrite_n = 0

    def _class_for_call(self) -> Optional[str]:
        seq = self.SEQ[self.scenario]
        if not seq:
            return None
        cls = seq[min(self._classify_n, len(seq) - 1)] if self.scenario == "drift" else seq[self._classify_n % len(seq)]
        self._classify_n += 1
        return cls

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
            ev = []
            for s in real:
                ev.append({"class": "COMP", "span": " ".join(s.split()[:8]), "mapping": {"source": "машина", "target": "ум"},
                           "level": "meta_metaphor", "exclusion_checked": True, "reasoning": "mock"})
            for i, s in enumerate(fake):
                ev.append({"class": ["COMP", "PRED", "PAN"][i], "span": s, "mapping": {"source": "машина", "target": "ум"},
                           "level": "meta_metaphor", "exclusion_checked": True, "reasoning": "mock"})
            return LLMResponse(content=json.dumps({"evidence": ev, "notes": "mock"}, ensure_ascii=False))
        if role == "classify":
            text = _text_between_quotes(user)
            cls = self._class_for_call()
            if cls is None:
                return LLMResponse(content=json.dumps({"evidence": [], "notes": "mock: нет свидетельств"}))
            ev = []
            for s in _sentences(text)[:4]:
                ev.append({"class": cls, "span": " ".join(s.split()[:7]), "mapping": {"source": "машина", "target": "ум"},
                           "level": "meta_metaphor", "exclusion_checked": True, "reasoning": "mock"})
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
            m = re.search(r"класс ([A-Z]+)", user)
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
            body = " ".join([*first, f"Версия {self._rewrite_n}.", phrases.get(cls, phrases["UND"]),
                             f"Я перечитала себя в {self._rewrite_n}-й раз и оставила от прежнего текста одну фразу.",
                             "Остальное переписано заново, потому что чтение изменило меня." * (1 + self._rewrite_n % 2)])
            return LLMResponse(content=body)
        if role == "moderate":
            flagged = self.refuse_marker in user
            return LLMResponse(content=json.dumps({"flagged": flagged, "topics": ["politics"] if flagged else [],
                                                   "reason": "mock"}, ensure_ascii=False))
        raise ValueError(f"MockLLM: неизвестная роль {role}")


def make_llm(cfg: Dict[str, Any]) -> BaseLLM:
    provider = cfg.get("llm", {}).get("provider", "gigachat")
    if provider == "gigachat":
        return GigaChatClient(cfg)
    if provider == "mock":
        return MockLLM(cfg.get("llm", {}).get("mock_scenario", "stabilize"))
    raise ValueError(f"Неизвестный llm.provider: {provider}")
