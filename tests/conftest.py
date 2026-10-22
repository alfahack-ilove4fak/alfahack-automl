"""Общее для всех тестов: настоящая LLM недоступна, даже если тесты запущены с `--env-file .env`.

parse_task зовёт LLM-аналитика (agent/task_analyst.py), а он сам создаёт LLMClient: с ключом в окружении
тесты тратили бы рубли на Qwen и зависели бы от сети. Тесты, которым нужен клиент, задают окружение сами
через monkeypatch — их значения применяются после этой фикстуры.
"""

import pytest


@pytest.fixture(autouse=True)
def no_real_llm(monkeypatch):
    for key in ("OPENAI_API_KEY", "OPENAI_BASE_URL", "OPENAI_MODEL", "LLM_CACHE_DIR"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("LLM_MOCK", "1")
