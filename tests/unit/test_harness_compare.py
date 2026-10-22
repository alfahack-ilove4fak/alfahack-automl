"""scripts/harness_compare.py: таблица «было/стало» для комментария в PR (Q-L3-1)."""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

from harness_compare import MARKER, main, render  # noqa: E402


def _board(rows, commit):
    df = pd.DataFrame(rows)
    df["commit"] = commit
    return df.set_index("task")


def _row(task, score, value, status="OK", error=""):
    return {
        "task": task,
        "metric": "rmse",
        "value": value,
        "score": score,
        "status": status,
        "seconds": 30,
        "peak_ram_mb": 200,
        "error": error,
    }


def test_render_marks_better_worse_noise_and_failed():
    base = _board([_row("a", 10, 50), _row("b", 40, 30), _row("c", 20, 40), _row("d", 30, 35)], "base1")
    head = _board(
        [_row("a", 20, 45), _row("b", 30, 32), _row("c", 21, 39.5), _row("d", 0, None, "FAILED", "KeyError: x")],
        "head2",
    )
    text = render(base, head, minutes=1, rows=5000)
    assert text.startswith(MARKER)
    assert "`base1`" in text and "`head2`" in text
    lines = {line.split("|")[1].strip(): line for line in text.splitlines() if line.startswith("| ") and "→" in line}
    assert "🟢 лучше" in lines["a"] and "+10.0" in lines["a"]
    assert "🔴 хуже" in lines["b"]
    assert "⚪ в пределах шума" in lines["c"]
    assert "🔴 упал" in lines["d"]
    assert "Средний балл: 25.0 → 17.8 (-7.2)" in text
    assert "`d`: KeyError: x" in text
    assert "бюджет 1 мин на задачу, train урезан до 5000 строк" in text


def test_new_task_only_in_head():
    text = render(_board([_row("a", 10, 50)], "b"), _board([_row("a", 10, 50), _row("new", 5, 1)], "h"))
    assert "| new |" in text and "0.0 → 5.0" in text


def test_llm_tokens_line_and_setup_say_llm_was_used():
    # тай-брейк: токены «было → стало» видны в комментарии PR, а подпись не врёт «без LLM»
    base = _board([_row("a", 10, 50) | {"llm_calls": 3, "llm_tokens": 9000, "llm_rub": 1.5}], "b")
    head = _board([_row("a", 10, 50) | {"llm_calls": 2, "llm_tokens": 6000, "llm_rub": 1.0}], "h")
    text = render(base, head)
    assert "Токены LLM: 9000 → 6000 (-3000), 1.50 → 1.00 ₽" in text
    assert "с LLM" in text and "без LLM" not in text


def test_old_board_without_llm_columns_has_no_tokens_line():
    text = render(_board([_row("a", 10, 50)], "b"), _board([_row("a", 12, 48)], "h"))
    assert "Токены LLM" not in text and "без LLM" in text


def test_main_takes_last_row_per_task(tmp_path):
    base, head, out = tmp_path / "base.csv", tmp_path / "head.csv", tmp_path / "comment.md"
    pd.DataFrame([_row("a", 1, 9) | {"commit": "old"}, _row("a", 10, 50) | {"commit": "b"}]).to_csv(base, index=False)
    pd.DataFrame([_row("a", 15, 45) | {"commit": "h"}]).to_csv(head, index=False)
    assert main([str(base), str(head), "--out", str(out)]) == 0
    assert "10.0 → 15.0" in out.read_text(encoding="utf-8")


def test_append_rows_migrates_old_leaderboard_header(tmp_path):
    """Старый лидерборд без колонок модельного бейзлайна — не ломается, старые строки сохраняются."""
    from evaluation.harness import COLUMNS, append_rows

    path = tmp_path / "leaderboard.csv"
    old = [c for c in COLUMNS if c not in ("model_baseline", "model_score")]
    pd.DataFrame([{c: 1 for c in old} | {"task": "old"}], columns=old).to_csv(path, index=False)
    append_rows([{"task": "new", "score": 5.0, "model_score": 2.0}], path)
    board = pd.read_csv(path)
    assert list(board.columns) == COLUMNS
    assert board.task.tolist() == ["old", "new"] and board.model_score.isna().tolist() == [True, False]
