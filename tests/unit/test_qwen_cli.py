"""scripts/qwen.py: бюджет, маскировка ключа, проверка env, сводка из лога агента — без сети."""

import sys
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

import qwen  # noqa: E402

KEY = "sk-TestSecretKey1234567890"
URL = "https://changellenge.up.railway.app/v1"
KEY_INFO = {
    "key": KEY,
    "info": {
        "key": "hashed-secret",
        "spend": 41.4,
        "max_budget": 600.0,
        "budget_duration": "1d",
        "budget_reset_at": "2026-10-23T00:00:00+00:00",
        "models": ["qwen/qwen3.8-27b"],
        "rpm_limit": None,
    },
}

# Реальные строки agent.log (runtime/log.py: «[elapsed] LEVEL name: message»)
AGENT_LOG = """\
[    0.4s] INFO  agent.solution: Старт: task/task.md, бюджет 600 с, RAM 30000 МБ
[    1.2s] INFO  agent.llm: LLM: модель qwen/qwen3.8-27b, сервер changellenge.up.railway.app
[   20.1s] INFO  agent.solution: Шаг 1: бейзлайн rmse=0.51 ± 0.01
[   35.7s] INFO  agent.llm_hypotheses: LLM предложила 3 гипотез (потрачено 0.00 ₽ за 1 запросов)
[   35.8s] INFO  agent.loop: Итерация 1: lgbm_deeper_trees — Глубже деревья
[   60.2s] INFO  agent.loop:   ✓ лучше: 0.51000 → 0.50000 — t=3.1, 5/5 фолдов
[   60.3s] INFO  agent.loop: Итерация 2: balanced_classes — Веса классов
[   80.0s] INFO  agent.loop:   ✓ лучше: 0.50000 → 0.49000 — t=2.5, 4/5 фолдов
[   80.1s] INFO  agent.loop: Итерация 3: llm_target_log — Логарифм таргета
[   95.0s] INFO  agent.loop:   · не лучше: 0.49500 (лучший 0.49000) — t=0.2
[  120.0s] WARNING agent.llm: LLM ошибка (1 подряд): APIConnectionError: Connection error. ← ConnectError: [Errno 111]
[  150.0s] WARNING agent.llm: LLM упёрлась в лимит 3600 токенов (рассуждения + ответ), ответ пустой — увеличь X
[  180.0s] INFO  agent.llm_hypotheses: LLM предложила 2 гипотез (потрачено 0.00 ₽ за 3 запросов)
"""


def test_parse_key_info_reads_budget_and_reset_in_msk():
    budget = qwen.parse_key_info(KEY_INFO)
    assert budget.spent == pytest.approx(41.4)
    assert budget.limit == 600.0
    assert budget.left == pytest.approx(558.6)
    text = budget.reset_text(now=datetime(2026, 10, 22, 21, 30, tzinfo=UTC))
    assert "23.10 03:00 МСК" in text and "через 2 ч 30 мин" in text


def test_parse_key_info_without_limit_or_reset():
    budget = qwen.parse_key_info({"info": {"spend": None, "max_budget": None}})
    assert budget.spent == 0.0 and budget.limit == 0.0 and budget.reset_at is None
    assert qwen.budget_line(budget, 30).status == qwen.WARN


@pytest.fixture
def llm_env(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", KEY)
    monkeypatch.setenv("OPENAI_BASE_URL", URL)
    monkeypatch.setenv("OPENAI_MODEL", "qwen/qwen3.8-27b")
    monkeypatch.setattr(qwen, "load_env", lambda *a, **kw: set())  # настоящий .env в тесты не тянем


@pytest.mark.parametrize(("spend", "code"), [(41.4, 0), (571.0, 2), (599.9, 2)])
def test_budget_exit_code_by_threshold(llm_env, monkeypatch, capsys, spend, code):
    payload = {"info": {**KEY_INFO["info"], "spend": spend}}
    calls = []

    def fake_get(url, key, timeout=qwen.HTTP_TIMEOUT):
        calls.append(url)
        return 200, qwen.json.dumps(payload).encode()

    monkeypatch.setattr(qwen, "http_get", fake_get)
    assert qwen.main(["budget"]) == code
    assert calls == ["https://changellenge.up.railway.app/key/info"]
    out = capsys.readouterr().out
    assert KEY not in out and "hashed-secret" not in out
    assert "осталось" in out


def test_budget_custom_threshold(llm_env, monkeypatch):
    monkeypatch.setattr(qwen, "http_get", lambda *a, **kw: (200, qwen.json.dumps(KEY_INFO).encode()))
    assert qwen.main(["budget", "--min-rub", "600"]) == 2
    assert qwen.main(["budget", "--min-rub", "100"]) == 0


def test_budget_http_error_masks_key(llm_env, monkeypatch, capsys):
    body = f'{{"error": "invalid key {KEY}", "header": "Bearer {KEY}"}}'.encode()
    monkeypatch.setattr(qwen, "http_get", lambda *a, **kw: (401, body))
    assert qwen.main(["budget"]) == 1
    out = capsys.readouterr().out
    assert "401" in out and KEY not in out


@pytest.mark.parametrize("command", ["check", "budget"])
def test_without_key_exit_1_with_hint(monkeypatch, capsys, command):
    for name in qwen.REQUIRED:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(qwen, "load_env", lambda *a, **kw: set())
    monkeypatch.setattr(qwen, "http_get", lambda *a, **kw: pytest.fail("без ключа в сеть не ходим"))
    assert qwen.main([command]) == 1
    assert ".env" in capsys.readouterr().out


def test_error_chain_shows_causes_and_hides_key():
    try:
        try:
            try:
                raise ConnectionRefusedError(111, "Connection refused")
            except OSError as low:
                raise ValueError(f"Illegal header value b'Bearer {KEY}\\r'") from low
        except ValueError:
            raise RuntimeError("Connection error.")  # noqa: B904 — неявная цепочка через __context__
    except RuntimeError as exc:
        text = qwen.error_chain(exc, KEY)
    assert KEY not in text
    assert text.index("RuntimeError: Connection error.") < text.index("ValueError") < text.index("ConnectionRefused")
    assert "Bearer ***" in text


def test_mask_hides_key_like_tokens_even_without_key():
    assert "sk-AnotherSecret999" not in qwen.mask("token sk-AnotherSecret999 leaked")
    assert qwen.mask(f"x {KEY} y", KEY) == "x *** y"


def _by_name(lines):
    return {line.name: line for line in lines}


def test_env_lines_clean_values_are_ok():
    lines = _by_name(
        qwen.env_lines({"OPENAI_API_KEY": KEY, "OPENAI_MODEL": "qwen/qwen3.8-27b", "OPENAI_BASE_URL": URL})
    )
    assert all(line.status == qwen.OK for line in lines.values())
    assert lines["OPENAI_API_KEY"].detail == f"задан, длина {len(KEY)}"


def test_env_lines_catch_crooked_values():
    env = {
        "OPENAI_API_KEY": f'"{KEY}"\r',
        "OPENAI_MODEL": " qwen/qwen3.8-27b ",
        "OPENAI_BASE_URL": "changellenge.up.railway.app/v1",
    }
    lines = _by_name(qwen.env_lines(env, from_shell={"OPENAI_MODEL"}))
    key = lines["OPENAI_API_KEY"]
    assert key.status == qwen.WARN and "\\r" in key.detail and "кавычки" in key.detail
    assert KEY not in key.detail
    assert lines["OPENAI_MODEL"].status == qwen.WARN and "пробелы" in lines["OPENAI_MODEL"].detail
    assert "оболочки" in lines["OPENAI_MODEL"].detail
    assert lines["OPENAI_BASE_URL"].status == qwen.FAIL and "http" in lines["OPENAI_BASE_URL"].detail


def test_env_lines_missing_bearer_and_path():
    env = {"OPENAI_API_KEY": f"Bearer {KEY}", "OPENAI_BASE_URL": "https://changellenge.up.railway.app"}
    lines = _by_name(qwen.env_lines(env))
    assert lines["OPENAI_MODEL"].status == qwen.FAIL
    assert lines["OPENAI_API_KEY"].status == qwen.FAIL and "Bearer" in lines["OPENAI_API_KEY"].detail
    assert lines["OPENAI_BASE_URL"].status == qwen.WARN and "/v1" in lines["OPENAI_BASE_URL"].detail


def test_check_stops_on_bad_env_without_network(monkeypatch, capsys):
    monkeypatch.setattr(qwen, "load_env", lambda *a, **kw: set())
    monkeypatch.setenv("OPENAI_API_KEY", KEY)
    monkeypatch.setenv("OPENAI_MODEL", "qwen/qwen3.8-27b")
    monkeypatch.setenv("OPENAI_BASE_URL", "changellenge.up.railway.app/v1")
    monkeypatch.setattr(qwen, "check_network", lambda *a: pytest.fail("с кривым URL в сеть не ходим"))
    assert qwen.main(["check"]) == 1
    assert KEY not in capsys.readouterr().out


def test_parse_agent_log_counts_llm_activity():
    summary = qwen.parse_agent_log(AGENT_LOG, known_ids={"baseline", "balanced_classes"})
    assert summary.batches == 2 and summary.proposed == 5 and summary.calls == 3
    assert summary.accepted == ["lgbm_deeper_trees", "balanced_classes"]
    assert summary.accepted_llm == ["lgbm_deeper_trees"]  # balanced_classes — из playbook
    assert len(summary.errors) == 1 and summary.errors[0].startswith("APIConnectionError: Connection error.")
    assert summary.truncated == 1 and not summary.llm_off
    text = qwen.render_summary(summary, spent_rub=0.37)
    assert "предложено: 5, принято: 1" in text and "0.37 ₽" in text


def test_parse_agent_log_detects_llm_off():
    log = "[    0.1s] WARNING agent.llm: LLM выключена (нет ключа/модели или LLM_MOCK=1) — работаем на playbook\n"
    summary = qwen.parse_agent_log(log, known_ids=set())
    assert summary.llm_off and summary.calls == 0 and summary.proposed == 0


def test_playbook_ids_include_real_playbook():
    ids = qwen.playbook_ids()
    assert "baseline" in ids and "balanced_classes" in ids


def _board(path, rows):
    pd.DataFrame(rows).to_csv(path, index=False)
    return qwen.last_by_task(path)


def test_render_compare_table(tmp_path):
    mock = _board(
        tmp_path / "m.csv",
        [
            {"task": "tasks/public/regression", "metric": "rmse", "value": 0.6, "score": 10.0, "status": "OK"},
            {"task": "tasks/public/regression", "metric": "rmse", "value": 0.5, "score": 20.0, "status": "OK"},
            {"task": "tasks/public/multiclass", "metric": "f1", "value": 0.4, "score": 30.0, "status": "OK"},
        ],
    )
    llm = _board(
        tmp_path / "q.csv",
        [{"task": "tasks/public/regression", "metric": "rmse", "value": 0.45, "score": 25.5, "status": "OK"}],
    )
    table = qwen.render_compare(mock, llm)
    assert "| tasks/public/regression | rmse | 0.5 | 0.45 | 20.0 | 25.5 | +5.5 | OK |" in table  # последняя строка
    assert "| tasks/public/multiclass | f1 | 0.4 | — | 30.0 | — | — | — |" in table
    assert "Δ в среднем +5.5" in table


def test_harness_cmd_uses_real_cli(tmp_path):
    args = SimpleNamespace(minutes=2.0, rows=5000)
    task = Path("/x/task.md")
    cmd = qwen.harness_cmd([task], args, tmp_path / "o.csv", use_llm=True)
    assert cmd[1:3] == ["-m", "evaluation.harness"]
    assert ["--task", str(task)] == cmd[cmd.index("--task") : cmd.index("--task") + 2]
    assert "--llm" in cmd and "--rows" in cmd
    assert "--all" in qwen.harness_cmd(None, args, tmp_path / "o.csv", use_llm=False)
    source = (Path(qwen.__file__).resolve().parents[1] / "evaluation" / "harness.py").read_text(encoding="utf-8")
    for flag in ("--task", "--all", "--minutes", "--rows", "--llm", "--out"):  # флаги есть в настоящем харнессе
        assert f'"{flag}"' in source


def test_run_preflight_failure_does_not_start_agent(llm_env, monkeypatch, tmp_path):
    task = tmp_path / "task.md"
    task.write_text("x", encoding="utf-8")
    monkeypatch.setattr(qwen, "preflight", lambda *a: (2, None))
    monkeypatch.setattr(qwen.subprocess, "run", lambda *a, **kw: pytest.fail("агент не должен запускаться"))
    assert qwen.main(["run", str(task)]) == 2


def test_run_summarises_new_log_and_spend(llm_env, monkeypatch, tmp_path, capsys):
    task = tmp_path / "task.md"
    task.write_text("x", encoding="utf-8")
    monkeypatch.setattr(qwen, "ROOT", tmp_path)
    monkeypatch.setattr(qwen, "preflight", lambda *a: (0, 41.4))
    monkeypatch.setattr(qwen, "spend_now", lambda: 41.9)
    monkeypatch.setattr(qwen, "playbook_ids", lambda: {"baseline", "balanced_classes"})
    seen = {}

    def fake_run(cmd, env, cwd):
        seen["env"] = env
        log = tmp_path / "runs" / "1022-120000-task" / "agent.log"
        log.parent.mkdir(parents=True)
        log.write_text(AGENT_LOG, encoding="utf-8")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(qwen.subprocess, "run", fake_run)
    assert qwen.main(["run", str(task), "--minutes", "3"]) == 0
    assert seen["env"]["LLM_MOCK"] == "0" and seen["env"]["LLM_CACHE_DIR"] == ""
    out = capsys.readouterr().out
    assert "предложено: 5, принято: 1" in out and "0.50 ₽" in out


def test_parse_agent_log_uses_source_tag():
    """Метка [llm]/[playbook] в строке итерации важнее угадывания по id."""
    from qwen import parse_agent_log

    log = (
        "[  10.0s] INFO  agent.loop: Итерация 1: lgbm_shallow — Неглубокие деревья [llm]\n"
        "[  20.0s] INFO  agent.loop:   ✓ лучше: 0.50000 → 0.51000 — t=2.5\n"
        "[  30.0s] INFO  agent.loop: Итерация 2: my_new_idea — Что-то новое [playbook]\n"
        "[  40.0s] INFO  agent.loop:   ✓ лучше: 0.51000 → 0.52000 — t=2.1\n"
    )
    summary = parse_agent_log(log, known_ids={"lgbm_shallow"})
    assert summary.accepted == ["lgbm_shallow", "my_new_idea"]
    assert summary.accepted_llm == ["lgbm_shallow"]


def _check_env(monkeypatch, chats):
    """check без сети: переменные, DNS/TCP, /v1/models и бюджет — заглушки; ответы chat — по очереди из chats."""
    monkeypatch.setattr(qwen, "check_network", lambda *a: [qwen.Line(qwen.OK, "TCP")])
    monkeypatch.setattr(qwen, "check_models", lambda *a: qwen.Line(qwen.OK, "GET /v1/models", "200"))
    monkeypatch.setattr(qwen, "http_get", lambda *a, **kw: (200, qwen.json.dumps(KEY_INFO).encode()))
    calls = []

    def fake_chat(key, prompt=qwen.CHECK_PROMPT, max_tokens=qwen.CHECK_MAX_TOKENS):
        calls.append(prompt)
        return chats[len(calls) - 1]

    monkeypatch.setattr(qwen, "check_chat", fake_chat)
    return calls


def test_check_catches_network_cutting_long_answers(llm_env, monkeypatch, capsys):
    # #193: короткий ответ за 3 с доходит, длинный рвётся на 11 с — раньше check был зелёным
    cut = "APIConnectionError: Connection error. ← RemoteProtocolError: Server disconnected without sending a response."
    calls = _check_env(
        monkeypatch, [qwen.ChatResult(True, "3.1 с, ответ «Готов»", 3.1), qwen.ChatResult(False, cut, 11.1)]
    )
    assert qwen.main(["check"]) == 1
    assert calls == [qwen.CHECK_PROMPT, qwen.LONG_PROMPT]
    out = capsys.readouterr().out
    assert "обрыв через 11.1 с" in out and "дольше ≈ 11 с" in out and "#193" in out


def test_check_long_answer_ok(llm_env, monkeypatch, capsys):
    _check_env(monkeypatch, [qwen.ChatResult(True, "3 с", 3.0), qwen.ChatResult(True, "34 с, ответ «1. …»", 34.0)])
    assert qwen.main(["check"]) == 0
    out = capsys.readouterr().out
    assert "длинный ответ" in out and "Qwen отвечает" in out


def test_check_long_answer_too_fast_is_only_a_warning(llm_env, monkeypatch, capsys):
    _check_env(monkeypatch, [qwen.ChatResult(True, "3 с", 3.0), qwen.ChatResult(True, "6 с", 6.0)])
    assert qwen.main(["check"]) == 0
    assert "не проверен" in capsys.readouterr().out


def test_check_quick_skips_long_answer(llm_env, monkeypatch, capsys):
    calls = _check_env(monkeypatch, [qwen.ChatResult(True, "3 с", 3.0)])
    assert qwen.main(["check", "--quick"]) == 0
    assert calls == [qwen.CHECK_PROMPT]
    assert "--quick" in capsys.readouterr().out


def test_summary_counts_failed_requests_and_hints_disconnect():
    # реальный хвост лога regression 22.10: 5 коротких запросов прошли, 6 запросов гипотез оборвались
    cut = (
        "[  29.4s] WARNING agent.llm: LLM ошибка (1 подряд): APIConnectionError: Connection error. "
        "← RemoteProtocolError: Server disconnected without sending a response.\n"
    )
    log = cut * 6 + "[ 393.7s] INFO  agent.llm_hypotheses: LLM предложила 0 гипотез (потрачено 0.00 ₽ за 5 запросов)\n"
    text = qwen.render_summary(qwen.parse_agent_log(log, known_ids=set()), spent_rub=None)
    assert "запросов к LLM: 11 (успешных 5, с ошибкой 6)" in text
    assert "рвёт долгие ответы" in text
