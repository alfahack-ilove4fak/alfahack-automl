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
        # два объекта подряд — берём оба
        ('{"kind": "model"} и ещё {"kind": "features"}', [{"kind": "model"}, {"kind": "features"}]),
        # оборвана обёртка-объект — спасаем целые гипотезы внутри
        (
            '{"hypotheses": [{"id": "a"}, {"id": "b", "params": {}}, {"id": "c", "ti',
            [
                {"id": "a"},
                {"id": "b", "params": {}},
            ],
        ),
        # ответ оборван лимитом токенов — спасаем целые объекты
        (
            '```json\n[{"id": "a", "params": {"n": [1, 2]}}, {"id": "b"}, {"id": "c", "ti',
            [
                {"id": "a", "params": {"n": [1, 2]}},
                {"id": "b"},
            ],
        ),
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


def test_reasoning_reserve_added_to_max_tokens(client, monkeypatch):
    sent = {}
    fake = FakeOpenAI(["{}"])
    original = fake._create

    def spy(**kwargs):
        sent.update(kwargs)
        return original(**kwargs)

    fake.chat.completions.create = spy
    client._client = fake
    client.chat(MESSAGES, max_tokens=800)
    assert sent["max_tokens"] == 800 + client.reasoning_tokens
    assert client.reasoning_tokens == 3000


def test_reasoning_reserve_from_env(monkeypatch):
    monkeypatch.setenv("LLM_MOCK", "1")
    monkeypatch.setenv("LLM_REASONING_TOKENS", "5000")
    assert LLMClient().reasoning_tokens == 5000


def test_truncated_answer_logged(client, caplog):
    fake = FakeOpenAI([""])
    original = fake._create

    def truncated(**kwargs):
        response = original(**kwargs)
        response.choices[0].finish_reason = "length"
        return response

    fake.chat.completions.create = truncated
    client._client = fake
    assert client.chat_json(MESSAGES) is None
    assert "лимит" in caplog.text and "пустой" in caplog.text


def test_truncated_answer_not_cached(client, tmp_path):
    client.cache_dir = tmp_path
    fake = FakeOpenAI(['[{"id": "a"}, {"id": "b', '[{"id": "a"}, {"id": "b"}]'])
    original = fake._create
    calls = []

    def first_truncated(**kwargs):
        response = original(**kwargs)
        if not calls:
            response.choices[0].finish_reason = "length"
        calls.append(1)
        return response

    fake.chat.completions.create = first_truncated
    client._client = fake
    assert client.chat_json(MESSAGES) == [{"id": "a"}]  # спасли целое из обрезанного
    assert not list(tmp_path.iterdir())  # обрезанный ответ в кэш не попал
    assert client.chat_json(MESSAGES) == [{"id": "a"}, {"id": "b"}]  # повтор ушёл в LLM
    assert fake.requests == 2
