"""Клиент LLM (Qwen через OpenAI-совместимый API). Владелец — LLM/agent-инженер.

Принципы:
  * Агент ОБЯЗАН работать без LLM. Если ключа нет, сеть упала или кончился бюджет — available() == False,
    и цикл берёт гипотезы из research/playbook.yaml.
  * Все ответы кэшируются на диск (LLM_CACHE_DIR): повторный запуск на той же задаче не тратит рубли.
  * Считаем потраченные рубли по usage из ответа. Цены — в .env (⚠️ уточнить у организаторов).

  * Запрос знает остаток бюджета (ADR 0008, D3): ждём ответа не дольше LLM_TIME_SHARE оставшегося времени
    и не дольше MAX_CALL_SECONDS на весь запрос с повторами; если это меньше MIN_CALL_SECONDS — не зовём LLM совсем.
    Срок проверяется и внутри потока (медленная «струйка» не рвётся таймаутом чтения), а после TIMEOUTS_TO_DISABLE
    запросов подряд, упёршихся в срок, LLM выключается до конца прогона (#239).

  * Выданная Qwen — «рассуждающая» модель: перед ответом она пишет скрытые рассуждения (2–2.5 тыс. токенов
    на промпт гипотез) и тратит на них тот же max_tokens. Выключить нельзя (проверено 21.10: enable_thinking
    в обоих вариантах и /no_think не действуют), но можно сократить: reasoning_effort=low в extra_body
    (22.10: рассуждения ≈ в 2.5 раза короче). К каждому запросу добавляем запас REASONING_TOKENS — с запасом
    на случай, когда сервер не принял reasoning_effort и рассуждения длинные (22.10: > 3 тыс. токенов).

  * Сервер под нагрузкой рвёт соединение без ответа через 10–17 с. Повторяем с паузами (RETRY_BACKOFF) в пределах
    времени, а после серии сбоев делаем паузу COOLDOWN_SECONDS вместо отключения LLM до конца прогона.

  * openai любую ошибку отправки (кривой ключ, URL без схемы, отказ сервера, прокси) показывает одинаково —
    «Connection error.». Поэтому в лог пишем всю цепочку причин (ключ в ней скрываем), а ключ и URL из
    окружения чистим от пробелов, \\r\\n и кавычек: .env из Windows даёт «Illegal header value b'Bearer …\\r'».

Переменные окружения: OPENAI_API_KEY, OPENAI_MODEL, OPENAI_BASE_URL (как в baseline.py организаторов),
LLM_MOCK=1 (принудительно без LLM), LLM_CACHE_DIR, LLM_PRICE_RUB_PER_1K_INPUT, LLM_PRICE_RUB_PER_1K_OUTPUT,
LLM_REASONING_EFFORT (low по умолчанию; пусто — не передавать), LLM_COOLDOWN_SEC (пауза после 3 сбоев, 120 с),
LLM_STREAM (1 по умолчанию: потоковый ответ — прокси рвёт обычный запрос через ~10 с),
LLM_REASONING_TOKENS (запас на рассуждения, по умолчанию 6000), LLM_MAX_CALL_SEC (потолок на запрос, 90 с).
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from pathlib import Path
from urllib.parse import urlsplit

from runtime.log import get_logger

log = get_logger("llm")

DEFAULT_REASONING_TOKENS = 6000  # 22.10 без reasoning_effort 3000 не хватило: ответ пустой, ≈ 1 ₽ впустую
MAX_RETRIES = 2
LLM_TIME_SHARE = 0.25  # какую долю оставшегося времени можно ждать LLM: остальное — экспериментам
LEDGER_ANSWER_CHARS = 4000
# Сервер организаторов под нагрузкой рвёт соединение через 10–17 с («Server disconnected without sending a
# response», проверено 22.10: 5 запросов из 6). Поэтому повторяем сами — с паузами и в пределах времени, а после
# FAILURES_TO_PAUSE сбоев подряд не выключаем LLM навсегда, а делаем паузу COOLDOWN_SECONDS и пробуем снова.
RETRY_BACKOFF = (2.0, 5.0, 10.0)  # паузы между попытками одного запроса, с
FAILURES_TO_PAUSE = 3
COOLDOWN_SECONDS = float(os.getenv("LLM_COOLDOWN_SEC", 120))
# Короткие рассуждения: reasoning_effort=low через extra_body режет рассуждения ≈ в 2.5 раза (22.10: 672–856
# токенов против 2–3.8 тыс., 0.33 ₽ против 1.02 ₽ за запрос гипотез) — дешевле, быстрее и реже попадаем под обрыв.
# Пустая строка — не передавать параметр.
DEFAULT_REASONING_EFFORT = "low"
# Потоковый ответ: прокси рвёт НЕпотоковый запрос, если ответа нет ~10 с («Server disconnected without sending
# a response»), а запрос гипотез с рассуждениями идёт 15–60 с. С stream=True байты идут сразу — соединение живёт
# (22.10: обычный — обрыв на 11 с, потоковый — OK за 28–64 с). LLM_STREAM=0 — выключить.
DEFAULT_STREAM = "1"  # сколько символов ответа модели хранить в журнале
# Запрос гипотез идёт 6–64 с (low / полные рассуждения / поток, замеры 22.10), плюс повторы 2/5/10 с:
# меньше минуты на ответ — запрос не успеет.
MIN_CALL_SECONDS = 60.0
# Потолок на весь запрос вместе с повторами (#239): ночной аудит (#230) — зависший сервер держал запрос 497 с,
# а медленный поток («струйка», символ раз в 3 с) — до сторожа, без даже тривиального ответа. LLM_MAX_CALL_SEC.
MAX_CALL_SECONDS = float(os.getenv("LLM_MAX_CALL_SEC", 90))
# Между кусками потока: нормальный поток Qwen идёт без пауз дольше 2.3 с (22.10), тишина дольше — сервер завис
STREAM_STALL_SECONDS = 30.0
# Столько запросов подряд упёрлись в потолок времени — сервер не успевает отвечать, дальше без LLM (#193 п. 3)
TIMEOUTS_TO_DISABLE = 3


class LLMTimeout(TimeoutError):
    """Ответ не уложился в потолок времени запроса — повторять бессмысленно, время вышло."""


def _env(name: str) -> str:
    """Значение переменной без пробелов, \\r\\n и обрамляющих кавычек (частые ошибки при правке .env)."""
    return os.getenv(name, "").strip().strip("\"'").strip()


class LLMClient:
    def __init__(self, rub_limit: float = 0.0, timeout: float = 120.0):
        self.model = _env("OPENAI_MODEL")
        self.timeout = timeout
        self.rub_limit = rub_limit
        self.rub_spent = 0.0
        self.calls = 0
        self.failures = 0
        # Учёт для тай-брейка (вебинар: при равном балле решают токены, затем время) и журнала решений LLM
        self.errors = 0
        self.cache_hits = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.reasoning_used = 0
        self.seconds = 0.0
        run_dir = os.getenv("AGENT_RUN_DIR")
        self.ledger = Path(run_dir) / "llm_calls.jsonl" if run_dir else None
        self._price_in = float(os.getenv("LLM_PRICE_RUB_PER_1K_INPUT", 0))
        self._price_out = float(os.getenv("LLM_PRICE_RUB_PER_1K_OUTPUT", 0))
        self.reasoning_tokens = int(os.getenv("LLM_REASONING_TOKENS", DEFAULT_REASONING_TOKENS))
        self.reasoning_effort = os.getenv("LLM_REASONING_EFFORT", DEFAULT_REASONING_EFFORT).strip()
        self.stream = os.getenv("LLM_STREAM", DEFAULT_STREAM).strip() not in ("0", "false", "")
        self.stream_usage = True  # просить usage в потоке; сервер не понял stream_options — дальше без него (#224)
        self._paused_until = 0.0  # после серии сбоев LLM «отдыхает» до этого момента (time.monotonic)
        self.max_call_seconds = MAX_CALL_SECONDS
        self.timeouts = 0  # запросов подряд, упёршихся в потолок времени
        self.disabled_reason = ""  # непусто — LLM выключена до конца прогона
        cache = os.getenv("LLM_CACHE_DIR")
        self.cache_dir = Path(cache) if cache else None
        self._client = None
        self._api_key = _env("OPENAI_API_KEY")
        if os.getenv("LLM_MOCK") == "1" or not self._api_key or not self.model:
            log.warning("LLM выключена (нет ключа/модели или LLM_MOCK=1) — работаем на playbook")
            return
        from openai import OpenAI  # импорт здесь: без ключа библиотека даже не нужна

        base_url = _env("OPENAI_BASE_URL") or None  # None — openai возьмёт адрес по умолчанию
        if base_url and urlsplit(base_url).scheme not in ("http", "https"):
            log.warning("OPENAI_BASE_URL без http(s):// — запросы упадут с «Connection error»")
        proxy = any(os.getenv(k) for k in ("HTTPS_PROXY", "https_proxy", "ALL_PROXY", "all_proxy"))
        log.info(
            "LLM: модель %s, сервер %s%s",
            self.model,
            urlsplit(base_url).netloc or base_url if base_url else "по умолчанию",
            ", через прокси из окружения" if proxy else "",
        )
        # повторы делаем сами (_create_with_retries): с паузами, в пределах времени и с понятным логом
        self._client = OpenAI(api_key=self._api_key, base_url=base_url, timeout=timeout, max_retries=0)

    def available(self) -> bool:
        if self._client is None or self.disabled_reason:
            return False
        if self.failures >= FAILURES_TO_PAUSE and time.monotonic() < self._paused_until:
            return False  # серия сбоев — пауза; после неё пробуем снова (сервер часто оживает через минуту-две)
        return not (self.rub_limit and self.rub_spent >= self.rub_limit)

    def time_limit(self, seconds_left: float | None) -> float:
        """Сколько ждать ответа (весь запрос с повторами) при таком остатке бюджета: 0 — не звать LLM.

        Не больше max_call_seconds, даже если бюджета много: один запрос не должен съедать время экспериментов.
        """
        if seconds_left is None:
            return self.max_call_seconds
        limit = LLM_TIME_SHARE * seconds_left
        if limit < MIN_CALL_SECONDS:
            return 0.0
        return min(limit, self.max_call_seconds)

    def chat(
        self,
        messages: list[dict],
        temperature: float = 0.2,
        max_tokens: int = 2000,
        seconds_left: float | None = None,
        max_wait: float | None = None,
        purpose: str = "chat",
    ) -> str | None:
        """Один запрос к модели. Возвращает текст ответа или None, если LLM недоступна, упала или нет времени.

        max_tokens — сколько нужно на сам ответ; к нему добавляется запас на рассуждения модели.
        seconds_left — остаток бюджета агента (budget.seconds_left()); None — не ограничивать.
        max_wait — сколько секунд готовы ждать этот запрос (одна попытка); None — по seconds_left.
        purpose — зачем запрос (hypotheses, critic, reflect…): попадает в журнал llm_calls.jsonl.
        """
        key = hashlib.sha256(json.dumps([self.model, messages, temperature], ensure_ascii=False).encode()).hexdigest()
        cached = self._cache_get(key)
        if cached is not None:
            self.cache_hits += 1
            self._record(purpose, ok=True, cached=True, text=cached)
            return cached
        if not self.available():
            return None
        limit = self.time_limit(seconds_left)
        if max_wait is not None and limit != 0:
            limit = min(limit, max_wait)
        if limit == 0:
            log.info("До конца бюджета %.0f с — LLM не зовём, гипотезы из playbook", seconds_left)
            return None
        started = time.monotonic()
        try:
            response = self._create_with_retries(
                limit,
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens + self.reasoning_tokens,
            )
        except Exception as exc:  # noqa: BLE001 — любая ошибка сети/API = «LLM временно нет»
            self.failures += 1
            if _timed_out(exc):
                self.timeouts += 1
                if self.timeouts >= TIMEOUTS_TO_DISABLE:
                    self.disabled_reason = f"{self.timeouts} запроса подряд не уложились в {limit:.0f} с"
                    log.warning("LLM выключена до конца прогона: %s — дальше playbook и правила", self.disabled_reason)
            if self.failures >= FAILURES_TO_PAUSE:
                self._paused_until = time.monotonic() + COOLDOWN_SECONDS
                log.warning("LLM: %d сбоев подряд — пауза %.0f с, потом пробуем снова", self.failures, COOLDOWN_SECONDS)
            self.errors += 1
            self.seconds += time.monotonic() - started
            error = self._describe_error(exc)
            log.warning("LLM ошибка (%d подряд): %s", self.failures, error)
            self._record(purpose, ok=False, seconds=time.monotonic() - started, error=error)
            return None
        elapsed = time.monotonic() - started
        self.seconds += elapsed
        self.failures = 0
        self.timeouts = 0
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
        usage = response.usage
        cost = 0.0
        if usage:
            prompt, completion = int(usage.prompt_tokens or 0), int(usage.completion_tokens or 0)
            details = getattr(usage, "completion_tokens_details", None)
            reasoning = int(getattr(details, "reasoning_tokens", 0) or 0)
            self.prompt_tokens += prompt
            self.completion_tokens += completion
            self.reasoning_used += reasoning
            # Прокси организаторов (LiteLLM) сам отдаёт стоимость запроса в ₽ — это точнее цен из .env
            proxy_cost = getattr(usage, "cost", None)
            if proxy_cost is None and isinstance(getattr(usage, "model_extra", None), dict):
                proxy_cost = usage.model_extra.get("cost")
            cost = (
                float(proxy_cost)
                if proxy_cost is not None
                else (prompt * self._price_in + completion * self._price_out) / 1000
            )
            self.rub_spent += cost
            self._record(
                purpose, ok=True, seconds=elapsed, prompt=prompt, completion=completion, reasoning=reasoning,
                cost=cost, text=text, truncated=truncated,
            )  # fmt: skip
        else:
            self._record(purpose, ok=True, seconds=elapsed, text=text, truncated=truncated)
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

    def _create_with_retries(self, limit: float, **request):
        """Запрос с повторами при обрыве/таймауте/перегрузке сервера; всё — в пределах limit секунд.

        Ошибки запроса (400/401/404) не повторяем: повтор их не исправит. Если сервер не понял reasoning_effort
        или stream_options, один раз повторяем без параметра и дальше его не шлём.
        """
        deadline = time.monotonic() + limit
        if self.reasoning_effort:
            request["extra_body"] = {"reasoning_effort": self.reasoning_effort}
        if self.stream:
            request["stream"] = True
            if self.stream_usage:
                request["stream_options"] = {"include_usage": True}  # usage (токены, cost) — в последнем чанке
        attempt = 0
        while True:
            attempt += 1
            remaining = max(deadline - time.monotonic(), 1.0)
            # timeout httpx — на каждое чтение, а не на весь ответ: для потока ограничиваем паузу между кусками,
            # а общий срок проверяет _collect_stream (иначе «струйка» читается бесконечно — #239)
            timeout = _stream_timeout(remaining) if self.stream else remaining
            client = self._client.with_options(timeout=timeout, max_retries=0)
            try:
                response = client.chat.completions.create(**request)
                # обрыв посреди потока — тоже исключение здесь, и его повторяем так же
                return response if hasattr(response, "choices") else _collect_stream(response, deadline)
            except Exception as exc:  # noqa: BLE001
                if "extra_body" in request and _rejects_param(exc, "reasoning_effort"):
                    log.warning("Сервер не принимает reasoning_effort — дальше без него")
                    self.reasoning_effort = ""
                    request.pop("extra_body")
                    continue
                if "stream_options" in request and _rejects_param(exc, "stream_options"):
                    log.warning("Сервер не принимает stream_options — дальше без него (токены в журнале — 0)")
                    self.stream_usage = False
                    request.pop("stream_options")
                    continue
                if isinstance(exc, LLMTimeout) or time.monotonic() >= deadline:
                    raise
                pause = RETRY_BACKOFF[attempt - 1] if attempt <= len(RETRY_BACKOFF) else None
                out_of_time = pause is not None and time.monotonic() + pause + 5 > deadline
                if not _retryable(exc) or pause is None or out_of_time:
                    raise
                log.info("LLM: попытка %d не удалась (%s) — повтор через %.0f с", attempt, _short(exc), pause)
                time.sleep(pause)

    def summary(self) -> dict:
        """Итог по клиенту: для лога прогона, харнесса и тай-брейка (токены, время)."""
        return {
            "calls": self.calls,
            "errors": self.errors,
            "cache_hits": self.cache_hits,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "reasoning_tokens": self.reasoning_used,
            "total_tokens": self.prompt_tokens + self.completion_tokens,
            "seconds": round(self.seconds, 1),
            "rub": round(self.rub_spent, 4),
        }

    def log_summary(self) -> None:
        s = self.summary()
        if not (s["calls"] or s["errors"] or s["cache_hits"]):
            log.info("LLM итог: запросов не было%s", "" if self.available() else " (LLM выключена)")
            return
        log.info(
            "LLM итог: %d запросов, %d ошибок, %d из кэша; токены %d (вход %d, выход %d, из них рассуждения %d); "
            "%.1f с ожидания; %.2f ₽",
            s["calls"], s["errors"], s["cache_hits"], s["total_tokens"], s["prompt_tokens"],
            s["completion_tokens"], s["reasoning_tokens"], s["seconds"], s["rub"],
        )  # fmt: skip

    def _record(self, purpose: str, ok: bool, **fields) -> None:
        """Строка в журнал AGENT_RUN_DIR/llm_calls.jsonl: что спросили, сколько стоило, что ответила модель.

        Это доказательство агентности (решения LLM видны) и данные для экономии токенов. Сбой записи не роняет агента.
        """
        if self.ledger is None:
            return
        text = fields.pop("text", None)
        row = {"t": round(time.time(), 3), "purpose": purpose, "ok": ok, "model": self.model, **fields}
        if text is not None:
            row["answer"] = text[:LEDGER_ANSWER_CHARS]
        try:
            self.ledger.parent.mkdir(parents=True, exist_ok=True)
            with self.ledger.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        except OSError as exc:
            log.debug("журнал LLM не записан: %s", exc)

    def _describe_error(self, exc: BaseException) -> str:
        """«APIConnectionError: Connection error. ← ConnectError: [Errno 111] …» — вся цепочка, ключ скрыт."""
        parts, seen = [], set()
        while exc is not None and id(exc) not in seen and len(parts) < 5:
            seen.add(id(exc))
            parts.append(f"{type(exc).__name__}: {exc}")
            exc = exc.__cause__ or exc.__context__
        text = " ← ".join(parts)
        if self._api_key:
            text = text.replace(self._api_key, "***")
        return re.sub(r"Bearer\s+[^\s'\"]+", "Bearer ***", text)  # httpx печатает заголовок целиком

    def _cache_get(self, key: str) -> str | None:
        if self.cache_dir and (self.cache_dir / f"{key}.txt").exists():
            return (self.cache_dir / f"{key}.txt").read_text(encoding="utf-8")
        return None

    def _cache_put(self, key: str, text: str) -> None:
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            (self.cache_dir / f"{key}.txt").write_text(text, encoding="utf-8")


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


def _retryable(exc: BaseException) -> bool:
    """Обрыв, таймаут, 429 и 5xx — временные: имеет смысл повторить. Остальное (400/401/404) — нет."""
    try:
        import openai
    except ImportError:  # pragma: no cover
        return False
    if isinstance(exc, openai.APIConnectionError | openai.RateLimitError):  # таймаут — подкласс обрыва
        return True
    try:  # обрыв посреди потока приходит «сырым» исключением httpx, без обёртки openai
        import httpx

        if isinstance(exc, httpx.TransportError):
            return True
    except ImportError:  # pragma: no cover
        pass
    return isinstance(exc, openai.APIStatusError) and exc.status_code >= 500


def _rejects_param(exc: BaseException, param: str) -> bool:
    """400 от сервера, где в тексте упомянут параметр, — сервер его не поддерживает."""
    return getattr(exc, "status_code", None) == 400 and param in str(exc)


def _short(exc: BaseException) -> str:
    root = exc
    while root.__cause__ is not None:
        root = root.__cause__
    return f"{type(root).__name__}: {str(root)[:120]}"


def _timed_out(exc: BaseException) -> bool:
    """Запрос упёрся во время: наш потолок или таймаут чтения (openai/httpx)."""
    if isinstance(exc, TimeoutError):
        return True
    try:
        import httpx
        import openai
    except ImportError:  # pragma: no cover
        return False
    return isinstance(exc, openai.APITimeoutError | httpx.TimeoutException)


def _stream_timeout(remaining: float):
    """Таймаут потокового запроса: соединение и первый кусок — в пределах остатка, пауза между кусками ≤ 30 с."""
    import httpx

    read = min(remaining, STREAM_STALL_SECONDS)
    return httpx.Timeout(remaining, connect=min(remaining, 10.0), read=read)


def _collect_stream(stream, deadline: float | None = None):
    """Собрать потоковый ответ в объект как у обычного: choices[0].message.content, finish_reason, usage.

    deadline (time.monotonic) — общий срок на весь ответ: «струйка» по символу раз в 3 с не обрывается таймаутом
    чтения, поэтому проверяем срок между кусками и закрываем поток (#239).
    """
    from types import SimpleNamespace

    parts: list[str] = []
    finish_reason = None
    usage = None
    for chunk in stream:
        if deadline is not None and time.monotonic() > deadline:
            close = getattr(stream, "close", None)
            if callable(close):
                close()
            raise LLMTimeout(f"ответ не уложился в срок, получено {sum(map(len, parts))} символов")
        if getattr(chunk, "usage", None) is not None:
            usage = chunk.usage
        for choice in getattr(chunk, "choices", None) or []:
            delta = getattr(choice, "delta", None)
            if delta is not None and getattr(delta, "content", None):
                parts.append(delta.content)
            if getattr(choice, "finish_reason", None):
                finish_reason = choice.finish_reason
    message = SimpleNamespace(content="".join(parts))
    return SimpleNamespace(choices=[SimpleNamespace(message=message, finish_reason=finish_reason)], usage=usage)
