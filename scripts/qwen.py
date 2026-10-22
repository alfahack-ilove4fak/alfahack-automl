"""Работа с Qwen локально: проверка связи, остаток бюджета, прогон агента с LLM, сравнение «без LLM / с Qwen».

    uv run python scripts/qwen.py check                     # связь + длинный ответ, как у гипотез (≈ 30 с, ≈ 1 ₽)
    uv run python scripts/qwen.py check --quick             # только короткий запрос (копейки)
    uv run python scripts/qwen.py budget                    # сколько ₽ осталось на сегодня (бесплатно)
    uv run python scripts/qwen.py run tasks/public/regression/task.md --minutes 10
    uv run python scripts/qwen.py compare --minutes 5 --tasks tasks/public/regression/task.md

Ключ берётся из окружения или из .env в корне репозитория (уже заданные переменные не перезаписываются).
Ключ никогда не печатается: только «задан, длина N».
Бюджет 600 ₽/сутки — общий на команду, сброс в 00:00 UTC = 03:00 МСК. Коды выхода: 0 — хорошо, 1 — ошибка
(нет ключа, нет связи, упал запуск), 2 — бюджета меньше порога --min-rub.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

OK, FAIL, WARN = "✅", "❌", "⚠️"
REQUIRED = ("OPENAI_API_KEY", "OPENAI_MODEL", "OPENAI_BASE_URL")
MSK = timezone(timedelta(hours=3), "МСК")
HTTP_TIMEOUT = 15.0
CHECK_MAX_TOKENS = 600  # Qwen рассуждает: при max_tokens < ~300 content бывает пустым
CHECK_PROMPT = "Ответь одним словом: готов"
# Длинная проверка: запрос гипотез идёт ≈ 35 с, а некоторые сети рвут HTTPS-ответы дольше ≈ 10 с (#193)
LONG_PROMPT = (
    "Предложи 5 гипотез улучшения LightGBM для табличной задачи классификации. "
    "На каждую — абзац из 3–4 предложений с обоснованием."
)
LONG_MAX_TOKENS = 2500
LONG_MIN_SECONDS = 15.0  # ответ быстрее — обрыв на ~10 с этим запросом не проверить
LONG_TIMEOUT = 90.0
DEFAULT_MIN_RUB = 30.0
NO_KEY_HINT = "впиши OPENAI_API_KEY, OPENAI_MODEL, OPENAI_BASE_URL в .env (ключ даёт капитан лично, не через чат)"


# ------------------------------------------------------------------ окружение и секреты


def load_env(path: Path = ROOT / ".env") -> set[str]:
    """Подхватить .env, не перезаписывая заданное. Возвращает имена из REQUIRED, заданные ДО .env (из оболочки)."""
    from_shell = {name for name in REQUIRED if os.environ.get(name)}
    if path.exists():
        from dotenv import load_dotenv

        load_dotenv(path, override=False)
    return from_shell


def clean(value: str | None) -> str:
    """Как agent.llm_client._env: без пробелов, \\r\\n и обрамляющих кавычек."""
    return (value or "").strip().strip("\"'").strip()


def mask(text: str, key: str | None = None) -> str:
    """Убрать ключ из текста: сам ключ, заголовок Bearer и всё похожее на sk-…"""
    if key:
        text = text.replace(key, "***")
        if clean(key) and clean(key) != key:
            text = text.replace(clean(key), "***")
    text = re.sub(r"Bearer\s+[^\s'\"]+", "Bearer ***", text)
    return re.sub(r"\bsk-[A-Za-z0-9_\-]{6,}", "sk-***", text)


def error_chain(exc: BaseException, key: str | None = None, limit: int = 6) -> str:
    """«APIConnectionError: Connection error. ← ConnectError: [Errno 111] …» — вся цепочка причин, ключ скрыт."""
    parts, seen = [], set()
    while exc is not None and id(exc) not in seen and len(parts) < limit:
        seen.add(id(exc))
        parts.append(f"{type(exc).__name__}: {exc}")
        exc = exc.__cause__ or exc.__context__
    return mask(" ← ".join(parts), key)


@dataclass
class Line:
    status: str
    name: str
    detail: str = ""


def env_lines(env: Mapping[str, str], from_shell: set[str] = frozenset()) -> list[Line]:
    """Переменные LLM: есть ли, нет ли пробелов/кавычек/\\r, правильный ли URL. Значение ключа не показываем."""
    out = []
    for name in REQUIRED:
        raw = env.get(name)
        where = " (из окружения оболочки — перекрывает .env)" if name in from_shell else ""
        if not raw or not clean(raw):
            out.append(Line(FAIL, name, "не задана"))
            continue
        problems = []
        if "\r" in raw or "\n" in raw:
            problems.append("есть \\r/\\n (файл из Windows?)")
        if raw != raw.strip():
            problems.append("пробелы по краям")
        if raw.strip()[:1] in "\"'" or raw.strip()[-1:] in "\"'":
            problems.append("кавычки вокруг значения")
        if re.search(r"\s", clean(raw)):
            problems.append("пробел внутри значения")
        value = clean(raw)
        if name == "OPENAI_API_KEY":
            if value.lower().startswith("bearer "):
                problems.append("начинается с «Bearer» — нужен только сам ключ")
            shown = f"задан, длина {len(value)}"
        elif name == "OPENAI_BASE_URL":
            parts = urlsplit(value)
            if parts.scheme not in ("http", "https"):
                problems.append("нет http(s):// — openai упадёт с «Connection error»")
            elif not parts.path.rstrip("/").endswith("/v1"):
                problems.append("путь не заканчивается на /v1")
            shown = value
        else:
            shown = value
        # агент чистит пробелы, \r и кавычки сам (llm_client._env), поэтому это ⚠️, а не ❌
        fatal = [p for p in problems if "http" in p or "Bearer" in p or "внутри" in p]
        status = FAIL if fatal else (WARN if problems else OK)
        out.append(Line(status, name, shown + where + (f" — {'; '.join(problems)}" if problems else "")))
    return out


def settings() -> tuple[str, str, str]:
    """(ключ, модель, base_url) уже очищенные — как их увидит агент."""
    return clean(os.getenv("OPENAI_API_KEY")), clean(os.getenv("OPENAI_MODEL")), clean(os.getenv("OPENAI_BASE_URL"))


def server_root(base_url: str) -> str:
    """https://host/v1 → https://host (там лежит /key/info LiteLLM)."""
    parts = urlsplit(base_url)
    return f"{parts.scheme}://{parts.netloc}"


# ------------------------------------------------------------------ HTTP


def http_get(url: str, key: str, timeout: float = HTTP_TIMEOUT) -> tuple[int, bytes]:
    """GET с ключом. Код ответа и тело; HTTP-ошибка — тоже ответ (код 4xx/5xx), сетевая — исключение."""
    request = urllib.request.Request(url, headers={"Authorization": f"Bearer {key}", "Accept": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read()


@dataclass
class Budget:
    spent: float
    limit: float
    reset_at: datetime | None = None

    @property
    def left(self) -> float:
        return self.limit - self.spent

    def reset_text(self, now: datetime | None = None) -> str:
        if self.reset_at is None:
            return "время сброса неизвестно"
        now = now or datetime.now(UTC)
        minutes = max(0, int((self.reset_at - now).total_seconds() // 60))
        return f"сброс {self.reset_at.astimezone(MSK):%d.%m %H:%M} МСК (через {minutes // 60} ч {minutes % 60:02d} мин)"


def parse_key_info(payload: dict) -> Budget:
    """Ответ LiteLLM /key/info → Budget. Поля key/info.key — секрет, их не трогаем."""
    info = payload.get("info", payload)
    reset = info.get("budget_reset_at")
    reset_at = None
    if reset:
        reset_at = datetime.fromisoformat(str(reset).replace("Z", "+00:00"))
        if reset_at.tzinfo is None:
            reset_at = reset_at.replace(tzinfo=UTC)
    limit = info.get("max_budget")
    return Budget(spent=float(info.get("spend") or 0.0), limit=float(limit) if limit else 0.0, reset_at=reset_at)


def fetch_budget(base_url: str, key: str) -> Budget:
    code, body = http_get(server_root(base_url) + "/key/info", key)
    if code != 200:
        raise RuntimeError(f"/key/info ответил {code}: {mask(body.decode('utf-8', 'replace')[:200], key)}")
    return parse_key_info(json.loads(body))


def budget_line(budget: Budget, min_rub: float) -> Line:
    if not budget.limit:
        return Line(WARN, "бюджет", f"потрачено {budget.spent:.2f} ₽, лимит не задан; {budget.reset_text()}")
    status = OK if budget.left >= min_rub else FAIL
    detail = (
        f"потрачено {budget.spent:.2f} из {budget.limit:.0f} ₽, осталось {budget.left:.2f} ₽; {budget.reset_text()}"
    )
    return Line(status, "бюджет", detail)


# ------------------------------------------------------------------ вывод


def show(lines: list[Line]) -> None:
    for line in lines:
        print(f"  {line.status} {line.name}" + (f" — {line.detail}" if line.detail else ""), flush=True)


# ------------------------------------------------------------------ check


def check_network(base_url: str) -> list[Line]:
    parts = urlsplit(base_url)
    host, port = parts.hostname or "", parts.port or (443 if parts.scheme == "https" else 80)
    try:
        addresses = sorted({a[4][0] for a in socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)})
    except OSError as exc:
        return [Line(FAIL, "DNS", f"{host}: {exc} — нет сети или DNS; проверь VPN/прокси")]
    out = [Line(OK, "DNS", f"{host} → {', '.join(addresses[:3])}")]
    started = time.time()
    try:
        socket.create_connection((host, port), timeout=10).close()
        out.append(Line(OK, "TCP", f"{host}:{port} за {1000 * (time.time() - started):.0f} мс"))
    except OSError as exc:
        out.append(Line(FAIL, "TCP", f"{host}:{port}: {exc} — фаервол/прокси режет соединение"))
    return out


def check_models(base_url: str, key: str) -> Line:
    try:
        code, body = http_get(base_url.rstrip("/") + "/models", key)
    except Exception as exc:  # noqa: BLE001
        return Line(FAIL, "GET /v1/models", error_chain(exc, key))
    if code == 200:
        return Line(OK, "GET /v1/models", "200")
    hint = {401: "ключ не принят", 403: "ключу запрещено", 404: "неверный путь — нужен …/v1"}.get(code, "")
    return Line(FAIL, "GET /v1/models", f"{code} {hint}".strip())


@dataclass
class ChatResult:
    ok: bool
    detail: str
    seconds: float = 0.0
    cost: float | None = None


def check_chat(key: str, prompt: str = CHECK_PROMPT, max_tokens: int = CHECK_MAX_TOKENS) -> ChatResult:
    """Один запрос через LLMClient — тот же клиент (очистка env, base_url, таймауты), что у агента."""
    saved = {name: os.environ.get(name) for name in ("LLM_MOCK", "LLM_CACHE_DIR")}
    os.environ.update(LLM_MOCK="0", LLM_CACHE_DIR="")  # проверяем связь, а не playbook и не кэш
    started = time.time()
    try:
        from agent.llm_client import LLMClient, _collect_stream

        client = LLMClient(timeout=LONG_TIMEOUT)
        if client._client is None:  # noqa: SLF001
            return ChatResult(False, "клиент не создан — нет ключа или модели")
        started = time.time()
        # сырой ответ нужен ради usage (cost, reasoning_tokens) — chat() отдаёт только текст;
        # без повторов: обрыв должен быть виден сразу, а не через 3 попытки
        request = {
            "model": client.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
            "max_tokens": max_tokens,
        }
        if client.stream:  # тот же путь, что у агента (#209): иначе прокси рвёт «молчащий» запрос через ~10 с
            request["stream"] = True
            request["stream_options"] = {"include_usage": True}
        response = client._client.with_options(max_retries=0).chat.completions.create(**request)  # noqa: SLF001
        if client.stream:
            response = _collect_stream(response)
        seconds = time.time() - started
    except Exception as exc:  # noqa: BLE001
        return ChatResult(False, error_chain(exc, key), time.time() - started)
    finally:
        for name, value in saved.items():  # run/compare дальше решают про кэш сами
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
    choice = response.choices[0]
    text = (choice.message.content or "").strip()
    usage = response.usage
    details = getattr(usage, "completion_tokens_details", None) if usage else None
    reasoning = getattr(details, "reasoning_tokens", None) if details else None
    cost = (usage.model_extra or {}).get("cost") if usage is not None else None
    parts = [f"{seconds:.1f} с", f"ответ «{text[:40]}»" if text else "ответ ПУСТОЙ"]
    if usage is not None:
        parts.append(f"токены {usage.prompt_tokens}→{usage.completion_tokens}")
    if reasoning is not None:
        parts.append(f"из них рассуждения {reasoning}")
    if cost is not None:
        parts.append(f"цена {float(cost):.4f} ₽")
    if not text and choice.finish_reason == "length":
        parts.append(f"упёрлись в max_tokens={max_tokens}: рассуждения съели лимит")
    return ChatResult(bool(text), ", ".join(parts), seconds, float(cost) if cost is not None else None)


def _stream_enabled() -> bool:
    """Как у агента (agent/llm_client.py): LLM_STREAM по умолчанию включён."""
    from agent.llm_client import DEFAULT_STREAM

    return os.getenv("LLM_STREAM", DEFAULT_STREAM).strip() not in ("0", "false", "")


def check_long(key: str) -> tuple[Line, float]:
    """Длинный ответ (≈ 20–40 с), как запрос гипотез. Ловит сеть, которая рвёт ответы дольше N секунд:
    короткий запрос при этом проходит, а агент не получает ни одной гипотезы (#193)."""
    result = check_chat(key, LONG_PROMPT, LONG_MAX_TOKENS)
    if result.ok and result.seconds >= LONG_MIN_SECONDS:
        return Line(OK, "длинный ответ", result.detail), result.seconds
    if result.ok:
        return Line(
            WARN, "длинный ответ", f"{result.detail} — пришёл быстрее {LONG_MIN_SECONDS:g} с, обрыв не проверен"
        ), result.seconds
    mode = "потоком" if _stream_enabled() else "без потока"
    hint = (
        ""
        if _stream_enabled()
        else " — поток выключен (LLM_STREAM=0): включи его, агент так обходит обрыв прокси (#209)"
    )
    return Line(
        FAIL,
        "длинный ответ",
        f"в режиме агента ({mode}) не дошёл, обрыв через {result.seconds:.1f} с: {result.detail}{hint}",
    ), result.seconds


def cmd_check(args: argparse.Namespace, from_shell: set[str]) -> int:
    print("🔎 Qwen: проверка связи\n── Переменные окружения")
    lines = env_lines(os.environ, from_shell)
    show(lines)
    key, _, base_url = settings()
    if not key or not base_url or any(line.status == FAIL for line in lines):
        print(f"\n{FAIL} Дальше не проверяю: исправь переменные. Подсказка: {NO_KEY_HINT}")
        return 1
    if os.getenv("HTTPS_PROXY") or os.getenv("https_proxy") or os.getenv("ALL_PROXY") or os.getenv("all_proxy"):
        show([Line(WARN, "прокси", "в окружении задан HTTPS_PROXY/ALL_PROXY — запросы идут через него")])

    print("── Сеть")
    net = check_network(base_url)
    show(net)
    if any(line.status == FAIL for line in net):
        print(f"\n{FAIL} Сервер недоступен по сети. Проверь интернет, VPN, прокси; агент уйдёт на playbook.")
        return 1

    print("── API")
    models = check_models(base_url, key)
    chat = check_chat(key)
    show([models, Line(OK if chat.ok else FAIL, "chat/completions через LLMClient", chat.detail)])
    long, long_seconds = None, 0.0
    if chat.ok and not args.quick:
        long, long_seconds = check_long(key)
        show([long])
    try:
        show([budget_line(fetch_budget(base_url, key), args.min_rub)])
    except Exception as exc:  # noqa: BLE001
        show([Line(WARN, "бюджет", error_chain(exc, key))])

    if chat.ok and long is not None and long.status == FAIL:
        print(
            f"\n{FAIL} Короткие ответы доходят, длинные — нет: сеть (точка доступа, провайдер, VPN) рвёт ответы "
            f"дольше ≈ {long_seconds:.0f} с. Запрос гипотез идёт ≈ 35 с — агент не получит ни одной и уйдёт "
            "на playbook. Живые прогоны с Qwen запускай с другой сети или в Actions (#193)."
        )
        return 1
    if chat.ok:
        print(f"\n{OK} Qwen отвечает — агент будет брать гипотезы у LLM.")
        if long is None:
            print(f"{WARN} Длинный ответ не проверялся (--quick): обрыв долгих ответов так не увидеть.")
        return 0
    if models.status == FAIL and models.detail.startswith("401"):
        hint = "ключ не принят: сверь OPENAI_API_KEY с выданным капитаном"
    elif "Connection error" in chat.detail or "Connect" in chat.detail:
        hint = "сеть до сервера есть, а SDK не соединяется: смотри цепочку причин выше (прокси, SSL, заголовок)"
    else:
        hint = "смотри цепочку причин выше; повтори через минуту, не помогло — пиши капитану с этим выводом"
    print(f"\n{FAIL} Qwen не отвечает. Что делать: {hint}.")
    return 1


# ------------------------------------------------------------------ budget


def cmd_budget(args: argparse.Namespace, from_shell: set[str]) -> int:
    key, _, base_url = settings()
    if not key or not base_url:
        print(f"{FAIL} Нет OPENAI_API_KEY или OPENAI_BASE_URL. {NO_KEY_HINT}")
        return 1
    try:
        budget = fetch_budget(base_url, key)
    except Exception as exc:  # noqa: BLE001
        print(f"{FAIL} Не удалось узнать бюджет: {error_chain(exc, key)}")
        return 1
    line = budget_line(budget, args.min_rub)
    show([line])
    if line.status == FAIL:
        print(f"{FAIL} Осталось меньше {args.min_rub:g} ₽ — прогоны с LLM отложи до сброса или согласуй с капитаном.")
        return 2
    return 0


# ------------------------------------------------------------------ сводка из лога агента


LOG_PROPOSED = re.compile(r"LLM предложила (\d+) гипотез \(потрачено ([\d.]+) ₽ за (\d+) запросов\)")
LOG_ERROR = re.compile(r"LLM ошибка \((\d+) подряд\): (.*)")
LOG_ITERATION = re.compile(r"Итерация \d+: (\S+) — .*?(?:\[(llm|playbook|baseline)\])?\s*$")
LOG_ACCEPTED = "✓ лучше:"
LOG_TRUNCATED = "LLM упёрлась в лимит"
LOG_OFF = "LLM выключена"
DISCONNECT = "Server disconnected without sending a response"


@dataclass
class LogSummary:
    llm_off: bool = False
    batches: int = 0
    proposed: int = 0
    calls: int = 0  # успешных запросов (llm.calls: рефлексия и критик тоже); упавшие — в errors
    errors: list[str] = field(default_factory=list)
    truncated: int = 0
    accepted: list[str] = field(default_factory=list)
    accepted_llm: list[str] = field(default_factory=list)


def playbook_ids() -> set[str]:
    try:
        from agent.hypotheses import load_playbook

        return {str(item["id"]) for item in load_playbook()} | {"baseline"}
    except Exception:  # noqa: BLE001 — без playbook просто не отличим источники
        return {"baseline"}


def parse_agent_log(text: str, known_ids: set[str] | None = None) -> LogSummary:
    """Что делала LLM в прогоне: пачки и гипотезы, запросы, ошибки, какие принятые гипотезы — от LLM.

    Источник гипотезы — метка `[llm]`/`[playbook]` в строке «Итерация …»; в старых логах без метки
    принятая гипотеза считается «от LLM», если её id не из playbook.
    """
    known_ids = playbook_ids() if known_ids is None else known_ids
    summary, current, source = LogSummary(), None, None
    for line in text.splitlines():
        if LOG_OFF in line:
            summary.llm_off = True
        if m := LOG_PROPOSED.search(line):
            summary.batches += 1
            summary.proposed += int(m.group(1))
            summary.calls = max(summary.calls, int(m.group(3)))
        elif m := LOG_ERROR.search(line):
            summary.errors.append(m.group(2).strip())
        elif LOG_TRUNCATED in line:
            summary.truncated += 1
        elif m := LOG_ITERATION.search(line):
            current, source = m.group(1), m.group(2)
        elif LOG_ACCEPTED in line and current:
            summary.accepted.append(current)
            if source == "llm" or (source is None and current not in known_ids):
                summary.accepted_llm.append(current)
    return summary


def render_summary(summary: LogSummary, spent_rub: float | None) -> str:
    lines = ["── Что делала LLM"]
    if summary.llm_off:
        lines.append(f"  {FAIL} агент считал LLM выключенной (нет ключа/модели или LLM_MOCK=1)")
    lines += [
        f"  запросов к LLM: {summary.calls + len(summary.errors)} (успешных {summary.calls}, "
        f"с ошибкой {len(summary.errors)}), пачек гипотез: {summary.batches}",
        f"  гипотез от LLM предложено: {summary.proposed}, принято: {len(summary.accepted_llm)}"
        + (f" ({', '.join(summary.accepted_llm)})" if summary.accepted_llm else ""),
        f"  всего принятых гипотез: {len(summary.accepted)}"
        + (f" ({', '.join(summary.accepted)})" if summary.accepted else ""),
        f"  ошибок LLM: {len(summary.errors)}" + (f", последняя: {summary.errors[-1][:300]}" if summary.errors else ""),
    ]
    if any(DISCONNECT in error for error in summary.errors):
        lines.append(
            f"  {WARN} сервер обрывал ответ («{DISCONNECT}»): похоже, сеть рвёт долгие ответы — "
            "проверь `qwen.py check` (длинный ответ) или запускай с другой сети (#193)"
        )
    if summary.truncated:
        lines.append(f"  {WARN} ответ упёрся в лимит токенов {summary.truncated} раз — увеличь LLM_REASONING_TOKENS")
    if spent_rub is not None:
        lines.append(f"  потрачено по /key/info: {spent_rub:.2f} ₽ (бюджет общий: сюда попали и запросы команды)")
    return "\n".join(lines)


# ------------------------------------------------------------------ run / compare


def preflight(args: argparse.Namespace, from_shell: set[str]) -> tuple[int, float | None]:
    """check + budget перед дорогим прогоном. (код, spend до запуска)."""
    code = cmd_check(args, from_shell)
    if code != 0:
        return code, None
    key, _, base_url = settings()
    try:
        budget = fetch_budget(base_url, key)
    except Exception as exc:  # noqa: BLE001
        print(f"{FAIL} Не удалось узнать бюджет: {error_chain(exc, key)}")
        return 1, None
    if budget.limit and budget.left < args.min_rub:
        print(f"{FAIL} Осталось {budget.left:.2f} ₽ < {args.min_rub:g} ₽ — прогон с LLM не запускаю.")
        return 2, None
    return 0, budget.spent


def spend_now() -> float | None:
    key, _, base_url = settings()
    try:
        return fetch_budget(base_url, key).spent
    except Exception:  # noqa: BLE001
        return None


def llm_env(cache: bool) -> dict[str, str]:
    env = {**os.environ, "LLM_MOCK": "0"}
    if not cache:
        env["LLM_CACHE_DIR"] = ""  # пустое значение .env в дочернем процессе не перезапишет
    return env


def cmd_run(args: argparse.Namespace, from_shell: set[str]) -> int:
    task = args.task.resolve()
    if not task.exists():
        print(f"{FAIL} Нет файла {args.task}")
        return 1
    code, spent_before = preflight(args, from_shell)
    if code:
        return code
    runs = ROOT / "runs"
    before = set(runs.glob("*/agent.log")) if runs.exists() else set()
    print(f"\n▶ Агент с Qwen: {task.relative_to(ROOT) if task.is_relative_to(ROOT) else task}, {args.minutes:g} мин")
    cmd = [sys.executable, str(ROOT / "scripts" / "run_local.py"), str(task), "--minutes", str(args.minutes)]
    proc = subprocess.run(cmd, env=llm_env(args.cache), cwd=ROOT)
    new_logs = sorted(set(runs.glob("*/agent.log")) - before, key=lambda p: p.stat().st_mtime) if runs.exists() else []
    spent_after = spend_now()
    spent = spent_after - spent_before if spent_after is not None and spent_before is not None else None
    if new_logs:
        print(render_summary(parse_agent_log(new_logs[-1].read_text(encoding="utf-8", errors="replace")), spent))
    else:
        print(f"{WARN} Лог агента не найден в runs/ — сводки по LLM нет")
    return proc.returncode


def last_by_task(path: Path):
    import pandas as pd

    if not path.exists():
        return pd.DataFrame(columns=["score", "status", "metric", "value"])
    return pd.read_csv(path).groupby("task", sort=True).tail(1).set_index("task")


def render_compare(mock, qwen) -> str:
    """Таблица «без LLM / с Qwen / разница» по двум CSV харнесса (последняя строка по задаче)."""
    import pandas as pd

    def cell(frame, task: str, column: str):
        return frame.loc[task, column] if task in frame.index and column in frame.columns else None

    def fmt(value, spec: str = ".5g") -> str:
        return "—" if value is None or pd.isna(value) else format(float(value), spec)

    lines = [
        "| Задача | Метрика | Без LLM | С Qwen | Балл без LLM | Балл с Qwen | Δ балла | Статус с Qwen |",
        "|---|---|---|---|---|---|---|---|",
    ]
    deltas = []
    for task in sorted(set(mock.index) | set(qwen.index)):
        m_score, q_score = cell(mock, task, "score"), cell(qwen, task, "score")
        delta = None if m_score is None or q_score is None else float(q_score) - float(m_score)
        if delta is not None:
            deltas.append(delta)
        metric = cell(qwen, task, "metric") or cell(mock, task, "metric")
        lines.append(
            f"| {task} | {metric if isinstance(metric, str) else '—'} | {fmt(cell(mock, task, 'value'))} | "
            f"{fmt(cell(qwen, task, 'value'))} | {fmt(m_score, '.1f')} | {fmt(q_score, '.1f')} | "
            f"{'—' if delta is None else f'{delta:+.1f}'} | {cell(qwen, task, 'status') or '—'} |"
        )
    m_mean = mock["score"].mean() if len(mock) else float("nan")
    q_mean = qwen["score"].mean() if len(qwen) else float("nan")
    lines.append(f"\nСредний балл: без LLM **{fmt(m_mean, '.1f')}**, с Qwen **{fmt(q_mean, '.1f')}**")
    if deltas:
        lines.append(f"Δ в среднем {sum(deltas) / len(deltas):+.1f} (разница меньше ~2 баллов на одном прогоне — шум)")
    return "\n".join(lines)


def harness_cmd(tasks: list[Path] | None, args: argparse.Namespace, out: Path, use_llm: bool) -> list[str]:
    cmd = [sys.executable, "-m", "evaluation.harness", "--minutes", str(args.minutes), "--out", str(out)]
    cmd += ["--all"] if not tasks else [x for t in tasks for x in ("--task", str(t))]
    if args.rows:
        cmd += ["--rows", str(args.rows)]
    if use_llm:
        cmd.append("--llm")
    return cmd


def cmd_compare(args: argparse.Namespace, from_shell: set[str]) -> int:
    tasks = [t.resolve() for t in args.tasks] if args.tasks else None
    missing = [str(t) for t in tasks or [] if not t.exists()]
    if missing:
        print(f"{FAIL} Нет файлов: {', '.join(missing)}")
        return 1
    count = len(tasks) if tasks else len(list(ROOT.glob("tasks/*/*/task.md")))
    print(f"ℹ️ {count} задач × 2 прогона × {args.minutes:g} мин ≈ {2 * count * args.minutes:.0f} мин без учёта оценки")
    code, spent_before = preflight(args, from_shell)
    if code:
        return code
    out_dir = ROOT / "runs" / "qwen-compare" / time.strftime("%m%d-%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    mock_csv, qwen_csv = out_dir / "without_llm.csv", out_dir / "with_qwen.csv"
    print("\n▶ Прогон 1/2: без LLM (LLM_MOCK=1)")
    subprocess.run(harness_cmd(tasks, args, mock_csv, False), env={**os.environ, "LLM_MOCK": "1"}, cwd=ROOT)
    print("\n▶ Прогон 2/2: с Qwen (LLM_MOCK=0)")
    subprocess.run(harness_cmd(tasks, args, qwen_csv, True), env=llm_env(args.cache), cwd=ROOT)
    table = render_compare(last_by_task(mock_csv), last_by_task(qwen_csv))
    spent_after = spend_now()
    print("\n" + table)
    if spent_after is not None and spent_before is not None:
        print(f"Потрачено по /key/info: {spent_after - spent_before:.2f} ₽ (бюджет общий на команду)")
    (out_dir / "compare.md").write_text(table + "\n", encoding="utf-8")
    print(f"CSV и таблица: {out_dir.relative_to(ROOT)}")
    return 0


# ------------------------------------------------------------------ CLI


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="qwen.py", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--min-rub",
        type=float,
        default=DEFAULT_MIN_RUB,
        help=f"порог остатка бюджета, ₽ (по умолчанию {DEFAULT_MIN_RUB:g})",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    check = sub.add_parser("check", parents=[common], help="переменные, сеть, /v1/models, запросы chat, бюджет")
    check.add_argument(
        "--quick", action="store_true", help="без длинного ответа (≈ 30 с и ≈ 1 ₽): только короткий запрос"
    )
    sub.add_parser("budget", parents=[common], help="потрачено/лимит/остаток ₽ и время сброса (код 2 — мало)")
    run = sub.add_parser("run", parents=[common], help="агент на задаче с Qwen + сводка по LLM и ₽")
    run.add_argument("task", type=Path, help="путь к task.md")
    run.add_argument("--minutes", type=float, default=10, help="бюджет агента, минут (по умолчанию 10)")
    compare = sub.add_parser("compare", parents=[common], help="харнесс без LLM и с Qwen, таблица разницы")
    compare.add_argument("--minutes", type=float, default=5, help="бюджет на задачу, минут (по умолчанию 5)")
    compare.add_argument("--tasks", type=Path, nargs="+", help="task.md (по умолчанию — все задачи харнесса)")
    compare.add_argument("--rows", type=int, help="урезать train до N строк (быстрее)")
    for p in (run, compare):
        p.add_argument("--cache", action="store_true", help="разрешить кэш ответов LLM_CACHE_DIR (по умолчанию выкл.)")
    return parser


COMMANDS = {"check": cmd_check, "budget": cmd_budget, "run": cmd_run, "compare": cmd_compare}


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    from_shell = load_env()
    return COMMANDS[args.command](args, from_shell)


if __name__ == "__main__":
    for _stream in (sys.stdout, sys.stderr):  # Windows/Git Bash: не UTF-8 → ✅ и кириллица роняют вывод
        _stream.reconfigure(encoding="utf-8", errors="replace")
    sys.exit(main())
