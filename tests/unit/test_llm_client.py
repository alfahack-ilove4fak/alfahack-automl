"""LLMClient: разбор JSON из ответа модели и устойчивость к сбоям. Настоящая LLM не вызывается."""

from types import SimpleNamespace

import pytest

from agent.llm_client import MAX_RETRIES, MIN_CALL_SECONDS, LLMClient

MESSAGES = [{"role": "user", "content": "предложи гипотезы"}]


@pytest.fixture(autouse=True)
def fast_retries(monkeypatch):
    """Повторы при обрыве — без пауз: тесты не ждут по 17 с (тест с длинной паузой задаёт её сам)."""
    monkeypatch.setattr("agent.llm_client.RETRY_BACKOFF", (0.0, 0.0, 0.0))


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
        self.options = None
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def with_options(self, **kwargs):
        self.options = kwargs
        return self

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


@pytest.mark.parametrize(
    "seconds_left, expected",
    [
        (None, None),  # бюджет не передан — как раньше
        (3600, None),  # четверть остатка больше полного таймаута с повторами — не ограничиваем
        (400, 100.0),  # ждём не дольше четверти остатка
        (4 * MIN_CALL_SECONDS - 1, 0.0),  # меньше MIN_CALL_SECONDS на ответ — не зовём
    ],
)
def test_time_limit(client, seconds_left, expected):
    assert client.timeout * (1 + MAX_RETRIES) < 0.25 * 3600
    assert client.time_limit(seconds_left) == expected


def test_no_time_no_request(client):
    fake = client._client = FakeOpenAI(["{}"])
    assert client.chat(MESSAGES, seconds_left=60) is None
    assert fake.requests == 0
    assert client.failures == 0  # нехватка времени — не сбой LLM


def test_short_time_uses_short_timeout(client):
    fake = client._client = FakeOpenAI(["{}"])
    assert client.chat(MESSAGES, seconds_left=400) == "{}"
    assert fake.options["max_retries"] == 0 and fake.options["timeout"] == pytest.approx(100.0, abs=0.5)


def test_plenty_of_time_default_timeout(client):
    fake = client._client = FakeOpenAI(["{}"])
    assert client.chat(MESSAGES, seconds_left=3600) == "{}"
    assert fake.options is None


def test_cache_served_even_without_time(client, tmp_path):
    client.cache_dir = tmp_path
    fake = client._client = FakeOpenAI(['{"ok": true}'])
    assert client.chat_json(MESSAGES) == {"ok": True}
    assert client.chat_json(MESSAGES, seconds_left=10) == {"ok": True}  # ответ из кэша бесплатный и мгновенный
    assert fake.requests == 1


def test_success_after_failure_resets_counter(client):
    client._client = FakeOpenAI([TimeoutError("нет сети"), "{}"])
    assert client.chat(MESSAGES) is None
    assert client.chat(MESSAGES) == "{}"  # следующий запрос прошёл — счётчик сбоев обнулился
    assert client.failures == 0 and client.available()


@pytest.fixture
def live_env(monkeypatch):
    """Окружение для настоящего openai-клиента; сеть наружу не нужна — только 127.0.0.1."""
    monkeypatch.delenv("LLM_MOCK", raising=False)
    monkeypatch.delenv("LLM_CACHE_DIR", raising=False)
    for key in ("HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy", "ALL_PROXY", "all_proxy"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("OPENAI_MODEL", "qwen-test")
    return monkeypatch


@pytest.mark.parametrize(
    "key, base_url, cause",
    [
        ("sk-test-123", "http://127.0.0.1:9/v1", "ConnectError"),  # сервер недоступен
        ("sk-test-123", "127.0.0.1:9/v1", "UnsupportedProtocol"),  # URL без схемы
    ],
)
def test_connection_error_logs_real_cause(live_env, caplog, monkeypatch, key, base_url, cause):
    monkeypatch.setattr("agent.llm_client.RETRY_BACKOFF", (0.0, 0.0, 0.0))  # повторы без пауз — тест быстрый
    live_env.setenv("OPENAI_API_KEY", key)
    live_env.setenv("OPENAI_BASE_URL", base_url)
    llm = LLMClient()
    assert llm.chat(MESSAGES, seconds_left=400) is None  # все попытки упали — один сбой клиента
    assert llm.failures == 1
    assert "Connection error" in caplog.text and cause in caplog.text  # видно настоящую причину


@pytest.mark.parametrize(
    "raw_key, raw_url",
    [
        ("sk-test-123\r", "http://127.0.0.1:9/v1\r"),  # .env, сохранённый в Windows (CRLF)
        (" sk-test-123\n", " http://127.0.0.1:9/v1 "),
        ('"sk-test-123"', "'http://127.0.0.1:9/v1'"),  # кавычки, попавшие в значение
    ],
)
def test_env_values_cleaned(live_env, caplog, raw_key, raw_url):
    live_env.setenv("OPENAI_API_KEY", raw_key)
    live_env.setenv("OPENAI_BASE_URL", raw_url)
    llm = LLMClient()
    assert llm._client.api_key == "sk-test-123"
    assert str(llm._client.base_url) == "http://127.0.0.1:9/v1/"
    llm.chat(MESSAGES, seconds_left=400)
    assert "ConnectError" in caplog.text  # дошли до сети, а не упали на заголовке или URL


def test_key_hidden_in_error_log(client, caplog):
    client._api_key = "sk-secret-777"
    header_error = ValueError("Illegal header value b'Bearer sk-secret-777\\r'")
    wrapped = RuntimeError("Connection error.")
    wrapped.__cause__ = header_error
    client._client = FakeOpenAI([wrapped])
    assert client.chat(MESSAGES) is None
    assert "Illegal header value" in caplog.text
    assert "sk-secret-777" not in caplog.text


# ---------------------------------------------------------------- учёт токенов, стоимость, журнал вызовов


def _usage(prompt=100, completion=50, reasoning=30, cost=None):
    usage = SimpleNamespace(
        prompt_tokens=prompt,
        completion_tokens=completion,
        completion_tokens_details=SimpleNamespace(reasoning_tokens=reasoning),
    )
    if cost is not None:
        usage.cost = cost
    return usage


class UsageOpenAI(FakeOpenAI):
    """Как FakeOpenAI, но с usage в формате прокси LiteLLM (cost, reasoning_tokens)."""

    def __init__(self, answers, usage):
        super().__init__(answers)
        self.usage = usage

    def _create(self, **kwargs):
        self.requests += 1
        answer = self.answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=answer), finish_reason="stop")],
            usage=self.usage,
        )


@pytest.fixture
def live_client(monkeypatch, tmp_path):
    """Клиент «с ключом», но с подменённым OpenAI; журнал — в tmp_path."""
    monkeypatch.setenv("LLM_MOCK", "0")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-1234567890")
    monkeypatch.setenv("OPENAI_MODEL", "qwen/test")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://example.test/v1")
    monkeypatch.delenv("LLM_CACHE_DIR", raising=False)
    monkeypatch.setenv("AGENT_RUN_DIR", str(tmp_path))
    monkeypatch.setenv("LLM_PRICE_RUB_PER_1K_INPUT", "0")
    monkeypatch.setenv("LLM_PRICE_RUB_PER_1K_OUTPUT", "0")
    return LLMClient()


def _ledger(tmp_path):
    import json

    return [json.loads(line) for line in (tmp_path / "llm_calls.jsonl").read_text(encoding="utf-8").splitlines()]


def test_proxy_cost_is_used_instead_of_zero_prices(live_client, tmp_path):
    live_client._client = UsageOpenAI(["ok"], _usage(cost=0.0156))
    assert live_client.chat(MESSAGES, purpose="hypotheses") == "ok"
    s = live_client.summary()
    assert s["rub"] == pytest.approx(0.0156)
    assert (s["prompt_tokens"], s["completion_tokens"], s["reasoning_tokens"], s["total_tokens"]) == (100, 50, 30, 150)
    assert s["calls"] == 1 and s["errors"] == 0


def test_rub_limit_works_with_proxy_cost(live_client):
    live_client.rub_limit = 0.02
    live_client._client = UsageOpenAI(["a", "b"], _usage(cost=0.015))
    assert live_client.chat(MESSAGES) == "a"
    assert live_client.chat(MESSAGES + [{"role": "user", "content": "ещё"}]) == "b"
    assert not live_client.available()  # 0.03 ₽ ≥ лимита 0.02 ₽ — дальше без LLM


def test_ledger_records_purpose_tokens_and_answer(live_client, tmp_path):
    live_client._client = UsageOpenAI(["гипотезы"], _usage(cost=0.01))
    live_client.chat(MESSAGES, purpose="critic")
    row = _ledger(tmp_path)[0]
    assert row["purpose"] == "critic" and row["ok"] is True
    assert row["prompt"] == 100 and row["reasoning"] == 30 and row["cost"] == pytest.approx(0.01)
    assert row["answer"] == "гипотезы" and row["seconds"] >= 0


def test_ledger_records_errors_without_key(live_client, tmp_path):
    live_client._client = UsageOpenAI([ConnectionError("boom sk-test-1234567890")], _usage())
    assert live_client.chat(MESSAGES, purpose="reflect") is None
    row = _ledger(tmp_path)[0]
    assert row["ok"] is False and "ConnectionError" in row["error"]
    assert "sk-test-1234567890" not in row["error"]
    assert live_client.summary()["errors"] == 1


def test_no_ledger_without_run_dir(monkeypatch, live_client, tmp_path):
    monkeypatch.delenv("AGENT_RUN_DIR")
    client = LLMClient()
    client._client = UsageOpenAI(["ok"], _usage())
    client.chat(MESSAGES)
    assert client.ledger is None


def test_log_summary_mentions_tokens(live_client, caplog):
    import logging

    live_client._client = UsageOpenAI(["ok"], _usage(cost=0.5))
    live_client.chat(MESSAGES)
    with caplog.at_level(logging.INFO, logger="agent.llm"):
        live_client.log_summary()
    assert "LLM итог: 1 запросов" in caplog.text and "токены 150" in caplog.text


# ---------------------------------------------------------------- устойчивость к перегруженному серверу (22.10)


def _openai_errors():
    import httpx
    import openai

    request = httpx.Request("POST", "https://example.test/v1/chat/completions")
    disconnect = openai.APIConnectionError(message="Connection error.", request=request)

    def status(code, text):
        return openai.APIStatusError(text, response=httpx.Response(code, request=request), body=None)

    return disconnect, status


@pytest.fixture
def no_pause(monkeypatch):
    monkeypatch.setattr("agent.llm_client.RETRY_BACKOFF", (0.0, 0.0, 0.0))


def test_disconnect_is_retried_and_succeeds(client, no_pause):
    disconnect, _ = _openai_errors()
    fake = client._client = FakeOpenAI([disconnect, disconnect, "{}"])
    assert client.chat(MESSAGES) == "{}"
    assert fake.requests == 3 and client.failures == 0


def test_bad_request_is_not_retried(client, no_pause):
    _, status = _openai_errors()
    fake = client._client = FakeOpenAI([status(401, "Unauthorized"), "{}"])
    assert client.chat(MESSAGES) is None
    assert fake.requests == 1


def test_server_error_is_retried(client, no_pause):
    _, status = _openai_errors()
    fake = client._client = FakeOpenAI([status(503, "overloaded"), "{}"])
    assert client.chat(MESSAGES) == "{}"
    assert fake.requests == 2


def test_gives_up_after_all_retries(client, no_pause):
    disconnect, _ = _openai_errors()
    fake = client._client = FakeOpenAI([disconnect] * 4)
    assert client.chat(MESSAGES) is None
    assert fake.requests == 4 and client.failures == 1  # 1 + 3 повтора = один сбой клиента


def test_no_retry_when_time_is_short(client, monkeypatch):
    monkeypatch.setattr("agent.llm_client.RETRY_BACKOFF", (120.0, 120.0, 120.0))
    disconnect, _ = _openai_errors()
    fake = client._client = FakeOpenAI([disconnect, "{}"])
    assert client.chat(MESSAGES, seconds_left=400) is None  # пауза 120 с не влезает в 100 с ожидания
    assert fake.requests == 1


def test_pause_after_failures_then_llm_comes_back(client, monkeypatch):
    clock = [1000.0]
    monkeypatch.setattr("agent.llm_client.time.monotonic", lambda: clock[0])
    client._client = FakeOpenAI([TimeoutError("нет сети")] * 3 + ["{}"])
    for _ in range(3):
        client.chat(MESSAGES)
    assert not client.available()  # пауза после трёх сбоев подряд
    clock[0] += 130  # пауза COOLDOWN_SECONDS (120 с) прошла
    assert client.available()
    assert client.chat(MESSAGES) == "{}"
    assert client.failures == 0


def test_reasoning_effort_sent_in_extra_body(client):
    seen = {}

    class Spy(FakeOpenAI):
        def _create(self, **kwargs):
            seen.update(kwargs)
            return super()._create(**kwargs)

    client._client = Spy(["{}"])
    client.chat(MESSAGES)
    assert seen["extra_body"] == {"reasoning_effort": "low"}


def test_reasoning_effort_dropped_if_server_rejects(client, no_pause):
    _, status = _openai_errors()
    calls = []

    class Picky(FakeOpenAI):
        def _create(self, **kwargs):
            calls.append("extra_body" in kwargs)
            if "extra_body" in kwargs:
                raise status(400, "unsupported parameter: reasoning_effort")
            return super()._create(**kwargs)

    client._client = Picky(["{}", "{}"])
    assert client.chat(MESSAGES) == "{}"
    assert calls == [True, False] and client.reasoning_effort == ""
    client.chat(MESSAGES + [{"role": "user", "content": "ещё"}])
    assert calls[-1] is False  # дальше без параметра


def test_reasoning_effort_can_be_disabled(monkeypatch):
    monkeypatch.setenv("LLM_MOCK", "1")
    monkeypatch.setenv("LLM_REASONING_EFFORT", "")
    assert LLMClient().reasoning_effort == ""
