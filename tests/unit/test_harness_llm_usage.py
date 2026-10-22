"""Харнесс считает расход LLM за прогон (тай-брейк платформы: токены, затем время) из журнала llm_calls.jsonl."""

import json

from evaluation.harness import llm_usage, render


def _ledger(run_dir, calls, tail=""):
    run_dir.mkdir()
    lines = [json.dumps(call, ensure_ascii=False) for call in calls]
    (run_dir / "llm_calls.jsonl").write_text("\n".join(lines) + "\n" + tail, encoding="utf-8")


def test_counts_tokens_rub_and_errors_but_not_cache(tmp_path):
    _ledger(
        tmp_path / "run",
        [
            {"purpose": "hypotheses", "ok": True, "prompt": 1200, "completion": 800, "reasoning": 500, "cost": 0.4},
            {"purpose": "critic", "ok": True, "prompt": 300, "completion": 100, "cost": 0.05},
            {"purpose": "hypotheses", "ok": False, "error": "APIConnectionError"},
            {"purpose": "hypotheses", "ok": True, "cached": True, "answer": "…"},  # на платформе кэша нет
            {"purpose": "reflect", "ok": True, "seconds": 2.0},  # ответ без usage — токены неизвестны
        ],
    )
    assert llm_usage(tmp_path / "run") == {"llm_calls": 3, "llm_errors": 1, "llm_tokens": 2400, "llm_rub": 0.45}


def test_no_ledger_means_no_llm(tmp_path):
    # прогон без LLM (LLM_MOCK=1) или агент упал до первого запроса — нули, а не ошибка
    assert llm_usage(tmp_path / "missing") == {"llm_calls": 0, "llm_errors": 0, "llm_tokens": 0, "llm_rub": 0.0}


def test_broken_last_line_from_killed_agent_is_skipped(tmp_path):
    _ledger(tmp_path / "run", [{"ok": True, "prompt": 10, "completion": 5, "cost": 0.01}], tail='{"ok": tr')
    assert llm_usage(tmp_path / "run")["llm_tokens"] == 15


def test_render_shows_llm_columns_and_total():
    rows = [
        {
            "task": "a",
            "score": 10.0,
            "status": "OK",
            "llm_calls": 4,
            "llm_errors": 1,
            "llm_tokens": 8000,
            "llm_rub": 1.2,
        },
        {"task": "b", "score": 20.0, "status": "OK", "llm_calls": 0, "llm_errors": 0, "llm_tokens": 0, "llm_rub": 0.0},
    ]
    text = render(rows)
    assert "| 4 (1) | 8000 | 1.20 |" in text
    assert "LLM за прогон: 5 запросов, **8000 токенов** (4000 на задачу), 1.20 ₽" in text


def test_render_without_llm_has_no_total_line():
    text = render([{"task": "a", "score": 10.0, "status": "OK"}])
    assert "LLM за прогон" not in text and "| 0 (0) | 0 | 0.00 |" in text
