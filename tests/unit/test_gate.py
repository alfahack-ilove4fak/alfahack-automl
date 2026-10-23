"""Гейт перед сабмитом (scripts/gate.py): что считаем нулём и когда не пускаем."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import gate  # noqa: E402


def _row(task, strong=10.0, **extra):
    return {
        "task": task,
        "exit_code": 0,
        "problems": [],
        "value": 1.0,
        "score_vs_strong": strong,
        "seconds": 60,
        "minutes": 5,
        **extra,
    }


def test_alien_zero_cases():
    assert not gate.alien_zero(_row("a"))
    assert gate.alien_zero(_row("a", strong=0))
    assert gate.alien_zero(_row("a", strong=None))
    assert gate.alien_zero(_row("a", exit_code=1))
    assert gate.alien_zero(_row("a", problems=["нет колонки"]))
    assert gate.alien_zero(_row("a", value=None))


def test_summarize_alien_takes_last_row():
    out = gate.summarize_alien([_row("a", strong=0), _row("a", strong=5)])
    assert out["a"]["zero"] is False


def _public(score=20.0, status="OK", seconds=100.0):
    return {"regression": {"score": score, "status": status, "seconds": seconds, "minutes": 5.0}}


def _base(zero=False, score=20.0):
    return {"alien": {"a": {"zero": zero, "score": 5}}, "public": {"regression": {"score": score}}}


def _alien(zero=False):
    return {"a": {"zero": zero, "score": 5, "seconds": 60, "minutes": 5}}


def test_passes_when_not_worse():
    fails, _ = gate.compare(_base(), _alien(), _public(score=19.0))
    assert fails == []


def test_fails_on_new_zero():
    fails, _ = gate.compare(_base(), _alien(zero=True), _public())
    assert any("нулей" in f for f in fails) and any("теперь ноль" in f for f in fails)


def test_zero_already_in_base_is_allowed():
    fails, _ = gate.compare(_base(zero=True), _alien(zero=True), _public())
    assert fails == []


def test_fails_on_public_drop_beyond_noise():
    fails, _ = gate.compare(_base(score=20.0), _alien(), _public(score=17.0))
    assert any("public" in f for f in fails)


def test_fails_on_time_overrun():
    fails, _ = gate.compare(_base(), _alien(), _public(seconds=5 * 60 * 2))
    assert any("время" in f for f in fails)


def test_no_baseline_is_note_not_fail():
    fails, notes = gate.compare(None, _alien(), _public())
    assert fails == [] and notes


def test_summarize_public_and_render():
    board = pd.DataFrame(
        [{"task": "tasks/public/regression", "model_score": 12.0, "status": "OK", "seconds": 30.0, "minutes": 1.0}]
    )
    public = gate.summarize_public(board)
    assert public["regression"]["score"] == 12.0
    assert "проходит" in gate.render(_base(), _alien(), public, [], [])


def test_llm_problem_lists_missing_variables():
    assert gate.llm_problem({"OPENAI_API_KEY": "k", "OPENAI_MODEL": "m"}) is None
    problem = gate.llm_problem({"OPENAI_API_KEY": "k"})
    assert problem and "OPENAI_MODEL" in problem and "OPENAI_API_KEY" not in problem


def test_llm_totals_sum_alien_and_public():
    board = pd.DataFrame(
        [
            {"task": "a", "llm_calls": 1, "llm_errors": 0, "llm_tokens": 10, "llm_rub": 0.5},
            {"task": "a", "llm_calls": 3, "llm_errors": 1, "llm_tokens": 30, "llm_rub": 1.5},  # берём последнюю строку
        ]
    )
    total = gate.llm_totals({"x": {"llm_calls": 2, "llm_errors": 0, "llm_tokens": 20, "llm_rub": 1.0}}, board)
    assert total == {"llm_calls": 5, "llm_errors": 1, "llm_tokens": 50, "llm_rub": 2.5}


def test_llm_checks_fail_when_qwen_never_answered():
    assert gate.llm_checks(None) == []
    assert gate.llm_checks({"llm_calls": 4, "llm_errors": 1, "llm_tokens": 1, "llm_rub": 0.1}) == []
    fails = gate.llm_checks({"llm_calls": 0, "llm_errors": 7, "llm_tokens": 0, "llm_rub": 0})
    assert fails and "Qwen" in fails[0]


def test_render_shows_llm_usage():
    llm = {"llm_calls": 5, "llm_errors": 1, "llm_tokens": 50, "llm_rub": 2.5}
    report = gate.render(None, _alien(), _public(), [], [], llm)
    assert "Qwen" in report and "2.50 ₽" in report
