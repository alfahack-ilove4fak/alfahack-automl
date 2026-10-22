"""Клиент LLM (Qwen через OpenAI-совместимый API). Владелец — LLM/agent-инженер.

Принципы:
  * Агент ОБЯЗАН работать без LLM. Если ключа нет, сеть упала или кончился бюджет — available() == False,
    и цикл берёт гипотезы из research/playbook.yaml.
  * Все ответы кэшируются на диск (LLM_CACHE_DIR): повторный запуск на той же задаче не тратит рубли.
  * Считаем потраченные рубли по usage из ответа. Цены — в .env (⚠️ уточнить у организаторов).

  * Запрос знает остаток бюджета (ADR 0008, D3): ждём ответа не дольше LLM_TIME_SHARE оставшегося времени,
    а если это меньше MIN_CALL_SECONDS — не зовём LLM совсем (полный таймаут с повторами — до 6 минут).

  * Выданная Qwen — «рассуждающая» модель: перед ответом она пишет скрытые рассуждения (2–2.5 тыс. токенов
    на промпт гипотез) и тратит на них тот же max_tokens. Выключить нельзя (проверено 21.10: enable_thinking
    в обоих вариантах и /no_think не действуют), поэтому к каждому запросу добавляем запас REASONING_TOKENS.

Переменные окружения: OPENAI_API_KEY, OPENAI_MODEL, OPENAI_BASE_URL (как в baseline.py организаторов),
LLM_MOCK=1 (принудительно без LLM), LLM_CACHE_DIR, LLM_PRICE_RUB_PER_1K_INPUT, LLM_PRICE_RUB_PER_1K_OUTPUT,
LLM_REASONING_TOKENS (запас на рассуждения, по умолчанию 3000).
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from runtime.log import get_logger

log = get_logger("llm")

DEFAULT_REASONING_TOKENS = 3000
MAX_RETRIES = 2
LLM_TIME_SHARE = 0.25  # какую долю оставшегося времени можно ждать LLM: остальное — экспериментам
MIN_CALL_SECONDS = 30.0  # запрос гипотез с рассуждениями идёт 10–20 с; меньше 30 с ждать бессмысленно


class LLMClient:
    def __init__(self, rub_limit: float = 0.0, timeout: float = 120.0):
        self.model = os.getenv("OPENAI_MODEL", "")
        self.timeout = timeout
        self.rub_limit = rub_limit
        self.rub_spent = 0.0
        self.calls = 0
        self.failures = 0
        self.last_error = ""  # причина последнего сбоя — для лога и doctor.py
        self._price_in = float(os.getenv("LLM_PRICE_RUB_PER_1K_INPUT", 0))
        self._price_out = float(os.getenv("LLM_PRICE_RUB_PER_1K_OUTPUT", 0))
        self.reasoning_tokens = int(os.getenv("LLM_REASONING_TOKENS", DEFAULT_REASONING_TOKENS))
        cache = os.getenv("LLM_CACHE_DIR")
        self.cache_dir = Path(cache) if cache else None
        self._client = None
        if os.getenv("LLM_MOCK") == "1" or not os.getenv("OPENAI_API_KEY") or not self.model:
            log.warning("LLM выключена (нет ключа/модели или LLM_MOCK=1) — работаем на playbook")
            return
        from openai import OpenAI  # импорт здесь: без ключа библиотека даже не нужна

        self._client = OpenAI(timeout=timeout, max_retries=MAX_RETRIES)

    def available(self) -> bool:
        if self._client is None or self.failures >= 3:
            return False
        return not (self.rub_limit and self.rub_spent >= self.rub_limit)

    def time_limit(self, seconds_left: float | None) -> float | None:
        """Сколько ждать ответа при таком остатке бюджета: None — без ограничения, 0 — не звать LLM."""
        if seconds_left is None:
            return None
        limit = LLM_TIME_SHARE * seconds_left
        if limit < MIN_CALL_SECONDS:
            return 0.0
        return None if limit >= self.timeout * (1 + MAX_RETRIES) else limit

    def chat(
        self,
        messages: list[dict],
        temperature: float = 0.2,
        max_tokens: int = 2000,
        seconds_left: float | None = None,
    ) -> str | None:
        """Один запрос к модели. Возвращает текст ответа или None, если LLM недоступна, упала или нет времени.

        max_tokens — сколько нужно на сам ответ; к нему добавляется запас на рассуждения модели.
        seconds_left — остаток бюджета агента (budget.seconds_left()); None — не ограничивать.
        """
        key = hashlib.sha256(json.dumps([self.model, messages, temperature], ensure_ascii=False).encode()).hexdigest()
        cached = self._cache_get(key)
        if cached is not None:
            return cached
        if not self.available():
            return None
        limit = self.time_limit(seconds_left)
        if limit == 0:
            log.info("До конца бюджета %.0f с — LLM не зовём, гипотезы из playbook", seconds_left)
            return None
        # мало времени — одна попытка с коротким таймаутом вместо таймаута с повторами
        client = self._client if limit is None else self._client.with_options(timeout=limit, max_retries=0)
        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens + self.reasoning_tokens,
            )
        except Exception as exc:  # noqa: BLE001 — любая ошибка сети/API = «LLM временно нет»
            self.failures += 1
            self.last_error = describe_error(exc)
            log.warning("LLM ошибка (%d подряд): %s", self.failures, self.last_error)
            return None
        self.failures = 0
        self.last_error = ""
        self.calls += 1
        choice = response.choices[0]
        text = choice.message.content or ""
        truncated = getattr(choice, "finish_reason", None) == "length"
        if truncated:
            log.warning(
                "LLM упёрлась в лимит %d токенов (рассуждения + ответ), ответ %s — увеличь LLM_REASONING_TOKENS",
                max_tokens + self.reasoning_tokens,
                "пустой" if not text.strip() else "обрезан",
            )
        if response.usage:
            self.rub_spent += (
                response.usage.prompt_tokens * self._price_in + response.usage.completion_tokens * self._price_out
            ) / 1000
        if not truncated:  # обрезанный ответ не кэшируем: после увеличения запаса повторный запрос должен уйти в LLM
            self._cache_put(key, text)
        return text

    def chat_json(self, messages: list[dict], **kwargs) -> dict | list | None:
        """Запрос, ответ которого должен быть JSON. Достаёт JSON даже из ```json ... ``` обёртки."""
        text = self.chat(messages, **kwargs)
        if not text:
            return None
        start = min((i for i in (text.find("{"), text.find("[")) if i != -1), default=-1)
        end = max(text.rfind("}"), text.rfind("]"))
        if start == -1 or end == -1:
            return None
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            pass
        objects = _complete_objects(text[start:])
        if objects:
            log.warning("LLM вернула оборванный JSON — взяли %d целых объектов", len(objects))
            return objects
        log.warning("LLM вернула невалидный JSON")
        return None

    def _cache_get(self, key: str) -> str | None:
        if self.cache_dir and (self.cache_dir / f"{key}.txt").exists():
            return (self.cache_dir / f"{key}.txt").read_text(encoding="utf-8")
        return None

    def _cache_put(self, key: str, text: str) -> None:
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            (self.cache_dir / f"{key}.txt").write_text(text, encoding="utf-8")


def describe_error(exc: BaseException) -> str:
    """Ошибка с цепочкой причин: openai пишет просто «Connection error.», а настоящая причина
    (прокси, SSL, DNS, таймаут) лежит в __cause__ — без неё сбой не разобрать."""
    parts, seen = [], set()
    while exc is not None and id(exc) not in seen and len(parts) < 4:
        seen.add(id(exc))
        text = f"{type(exc).__name__}: {exc}".strip()
        if not parts or text not in parts[-1]:
            parts.append(text)
        exc = exc.__cause__ or exc.__context__
    return " ← ".join(parts)[:400]


def _complete_objects(text: str) -> list[dict]:
    """Целые JSON-объекты из оборванного списка: «[{…}, {…}, {"id": "x", "ti» → два объекта.

    Если оборвана обёртка ({"hypotheses": [{…}, {…}, …), берутся целые объекты внутри неё.
    """
    decoder = json.JSONDecoder()
    body = text.lstrip()
    pos = 1 if body.startswith("[") else 0
    out = []
    while True:
        start = body.find("{", pos)
        if start == -1:
            return out
        try:
            obj, end = decoder.raw_decode(body, start)
        except json.JSONDecodeError:
            if out:
                return out  # оборван очередной элемент списка — целые уже собраны
            pos = start + 1  # оборвана обёртка {"hypotheses": [{…}, …] — ищем целые объекты внутри
            continue
        if isinstance(obj, dict):
            out.append(obj)
        pos = end
