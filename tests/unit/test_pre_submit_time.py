"""pre_submit: сколько бюджета агент реально использовал и не ушёл ли за него (#240)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

from pre_submit import TaskRun, render_report, time_usage  # noqa: E402

LONG = 10800


def test_idle_budget_warns_with_reason():
    end, warnings = time_usage(400, LONG, "x\nГипотезы закончились — завершаю цикл\n")
    assert end == "гипотезы закончились"
    assert len(warnings) == 1 and "простаивает" in warnings[0] and "4%" in warnings[0]


def test_full_use_is_quiet():
    end, warnings = time_usage(LONG - 100, LONG, "Время на исходе — завершаю цикл после 40 итераций")
    assert end == "время вышло" and warnings == []


def test_overrun_warns():
    _, warnings = time_usage(LONG + 30, LONG, "")
    assert len(warnings) == 1 and "за бюджет" in warnings[0]


def test_short_run_idle_is_not_a_signal():
    _, warnings = time_usage(30, 300, "Гипотезы закончились")
    assert warnings == []


def test_report_has_budget_columns():
    run = TaskRun(name="t", ok=True, seconds=5400, budget_sec=LONG, loop_end="время вышло")
    args = type("A", (), {"minutes": 180, "rows": None, "llm": False})()
    report = render_report(Path("x.zip"), [], [], [run], args)
    assert "Из бюджета" in report and "50% (10800 с)" in report and "время вышло" in report
