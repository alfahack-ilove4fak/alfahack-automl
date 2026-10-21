"""LLMClient: разбор JSON из ответа модели и устойчивость к сбоям. Настоящая LLM не вызывается."""

from types import SimpleNamespace

import pytest

from agent.llm_client import LLMClient

MESSAGES = [{"role": "user", "content": "предложи гипотезы"}]


@pytest.fixture
def client(monkeypatch):
    """Клиент без ключа и без кэша — ни одного сетевого запроса."""
    monkeypatch.setenv("LLM_MOCK", "1")
    monkeypatch.delenv("LLM_CACHE_DIR", raising=False)
    monkeypatch.setenv("LLM_PRICE_RUB_PER_1K_INPUT", "1")
    monkeypatch.setenv("LLM_PRICE_RUB_PER_1K_OUTPUT", "2")
    return LLMClient()


class FakeOpenAI:
    """Подмена openai.OpenAI: отдаёт заготовленные ответы или бросает исключение."""

    def __init__(self, answers):
        self.answers = list(answers)
        self.requests = 0
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.requests += 1
        answer = self.answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=answer))],
            usage=SimpleNamespace(prompt_tokens=1000, completion_tokens=500),
        )


def answer_with(client, monkeypatch, text):
    monkeypatch.setattr(client, "chat", lambda messages, **kwargs: text)


@pytest.mark.parametrize(
    "text, expected",
    [
        ('{"kind": "model", "params": {"model": "lgbm"}}', {"kind": "model", "params": {"model": "lgbm"}}),
        ('```json\n{"kind": "model"}\n```', {"kind": "model"}),
        (
            'Вот гипотезы:\n```json\n[{"kind": "model"}, {"kind": "features"}]\n```\nУдачи!',
            [
                {"kind": "model"},
                {"kind": "features"},
            ],
        ),
        ('[{"kind": "model"}]', [{"kind": "model"}]),
        ('{"title": "окно [7 дней]"}', {"title": "окно [7 дней]"}),
    ],
)
def test_chat_json_extracts_json(client, monkeypatch, text, expected):
    answer_with(client, monkeypatch, text)
    assert client.chat_json(MESSAGES) == expected


@pytest.mark.parametrize(
    "text",
    [
        None,  # LLM недоступна
        "",  # пустой ответ
        "Не могу предложить гипотезы.",  # текст без JSON
        '```json\n{"kind": "model",\n```',  # оборванный JSON
        "{'kind': 'model'}",  # одинарные кавычки — не JSON
        '{"kind": "model"} и ещё {"kind": "features"}',  # два объекта подряд
    ],
)
def test_chat_json_garbage_returns_none(client, monkeypatch, text):
    answer_with(client, monkeypatch, text)
    assert client.chat_json(MESSAGES) is None


def test_mock_mode_is_unavailable(client):
    assert not client.available()
    assert client.chat(MESSAGES) is None
    assert client.chat_json(MESSAGES) is None


def test_chat_counts_rubles_and_calls(client):
    client._client = FakeOpenAI(['{"ok": true}'])
    assert client.chat_json(MESSAGES) == {"ok": True}
    assert client.calls == 1
    # 1000 токенов запроса по 1 ₽/1K + 500 токенов ответа по 2 ₽/1K
    assert client.rub_spent == pytest.approx(2.0)


def test_api_errors_disable_llm_after_three_in_row(client):
    client._client = FakeOpenAI([TimeoutError("нет сети")] * 3)
    for _ in range(3):
        assert client.chat_json(MESSAGES) is None
    assert client.failures == 3
    assert not client.available()


def test_rub_limit_disables_llm(client):
    client._client = FakeOpenAI(["{}", "{}"])
    client.rub_limit = 1.0
    assert client.chat(MESSAGES) == "{}"
    assert not client.available()  # потратили 2 ₽ при лимите 1 ₽
    assert client.chat([{"role": "user", "content": "ещё"}]) is None


def test_cache_answers_without_request(client, tmp_path):
    client.cache_dir = tmp_path
    fake = client._client = FakeOpenAI(['{"ok": true}'])
    assert client.chat_json(MESSAGES) == {"ok": True}
    assert client.chat_json(MESSAGES) == {"ok": True}
    assert fake.requests == 1
