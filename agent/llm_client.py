"""Клиент LLM (Qwen через OpenAI-совместимый API). Владелец — LLM/agent-инженер.

Принципы:
  * Агент ОБЯЗАН работать без LLM. Если ключа нет, сеть упала или кончился бюджет — available() == False,
    и цикл берёт гипотезы из research/playbook.yaml.
  * Все ответы кэшируются на диск (LLM_CACHE_DIR): повторный запуск на той же задаче не тратит рубли.
  * Считаем потраченные рубли по usage из ответа. Цены — в .env (⚠️ уточнить у организаторов).

Переменные окружения: OPENAI_API_KEY, OPENAI_MODEL, OPENAI_BASE_URL (как в baseline.py организаторов),
LLM_MOCK=1 (принудительно без LLM), LLM_CACHE_DIR, LLM_PRICE_RUB_PER_1K_INPUT, LLM_PRICE_RUB_PER_1K_OUTPUT.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from runtime.log import get_logger

log = get_logger("llm")


class LLMClient:
    def __init__(self, rub_limit: float = 0.0, timeout: float = 120.0):
        self.model = os.getenv("OPENAI_MODEL", "")
        self.rub_limit = rub_limit
        self.rub_spent = 0.0
        self.calls = 0
        self.failures = 0
        self._price_in = float(os.getenv("LLM_PRICE_RUB_PER_1K_INPUT", 0))
        self._price_out = float(os.getenv("LLM_PRICE_RUB_PER_1K_OUTPUT", 0))
        cache = os.getenv("LLM_CACHE_DIR")
        self.cache_dir = Path(cache) if cache else None
        self._client = None
        if os.getenv("LLM_MOCK") == "1" or not os.getenv("OPENAI_API_KEY") or not self.model:
            log.warning("LLM выключена (нет ключа/модели или LLM_MOCK=1) — работаем на playbook")
            return
        from openai import OpenAI  # импорт здесь: без ключа библиотека даже не нужна

        self._client = OpenAI(timeout=timeout, max_retries=2)

    def available(self) -> bool:
        if self._client is None or self.failures >= 3:
            return False
        return not (self.rub_limit and self.rub_spent >= self.rub_limit)

    def chat(self, messages: list[dict], temperature: float = 0.2, max_tokens: int = 2000) -> str | None:
        """Один запрос к модели. Возвращает текст ответа или None, если LLM недоступна или упала."""
        key = hashlib.sha256(json.dumps([self.model, messages, temperature], ensure_ascii=False).encode()).hexdigest()
        cached = self._cache_get(key)
        if cached is not None:
            return cached
        if not self.available():
            return None
        try:
            response = self._client.chat.completions.create(
                model=self.model, messages=messages, temperature=temperature, max_tokens=max_tokens
            )
        except Exception as exc:  # noqa: BLE001 — любая ошибка сети/API = «LLM временно нет»
            self.failures += 1
            log.warning("LLM ошибка (%d подряд): %s", self.failures, exc)
            return None
        self.failures = 0
        self.calls += 1
        text = response.choices[0].message.content or ""
        if response.usage:
            self.rub_spent += (response.usage.prompt_tokens * self._price_in
                               + response.usage.completion_tokens * self._price_out) / 1000
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
