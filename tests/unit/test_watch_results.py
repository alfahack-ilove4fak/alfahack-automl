"""research/night/watch_results.py: разбор agent.log, проверки К1–К8 и код выхода монитора."""

import json
import os
import time

import pandas as pd
import pytest

from research.night import watch_results as wr

HEAD = (
    "[    1.0s] INFO  agent: План этапов: бейзлайн до +60 с, AutoResearch до +255 с, лимит платформы +300 с\n"
    "[    4.0s] INFO  agent: Шаг 1: бейзлайн rmse=48.0366214 ± 0.5\n"
)
FINAL = "[  290.0s] INFO  agent: Финал (конец работы): submission.csv валиден, 10 строк\n"


def _log(tmp_path, name, body, final=True, age=0.0):
    path = tmp_path / name / "run" / "agent.log"
    path.parent.mkdir(parents=True)
    path.write_text(HEAD + body + (FINAL if final else ""), encoding="utf-8")
    if age:
        old = time.time() - age
        os.utime(path, (old, old))
    return path


def _iter(n, hyp="object_history"):
    return f"[   10.0s] INFO  agent.loop: Итерация {n}: {hyp} — История объекта [playbook]\n"


@pytest.mark.parametrize(
    ("body", "decision", "detail"),
    [
        (
            _iter(3) + "[   12.0s] INFO  agent.loop:   ✓ лучше: 48.03662 → 45.92986 — улучшение 2.1, лучше на 1/3 "
            "фолдах, t=0.44 (порог 2.0)\n",
            "принята",
            "t=0.44, лучше на 1/3 фолдах",
        ),
        (
            _iter(3) + "[   12.0s] INFO  agent.loop:   · не лучше: 48.03662 (лучший 47.1) — среднее улучшение 0\n",
            "отклонена",
            "операция ничего не добавила",
        ),
        (
            _iter(3) + "[   12.0s] INFO  agent.loop:   · не лучше: 49.5 (лучший 47.1) — среднее улучшение -1.4\n",
            "отклонена",
            "CV 49.5 против лучшего 47.1",
        ),
        (
            _iter(3) + "[   12.0s] INFO  agent.loop:   ⊘ пропуск критиком: не успеет: оценка 8 с, осталось 7 с\n",
            "пропущена",
            "не успеет",
        ),
        (_iter(3) + "[   12.0s] INFO  agent.loop:   ✗ failed: ValueError: x\n", "упала", "ValueError"),
        (_iter(3, "pair_ops"), "не дошла", ""),
    ],
)
def test_parse_log_decisions(tmp_path, body, decision, detail):
    run = wr.parse_log(_log(tmp_path, "t", body), "object_history", "t")
    assert run.decision == decision
    assert detail in run.detail
    assert run.finished and run.budget == 300 and run.elapsed == 290.0


def test_decision_waits_for_lines_between_iteration_and_verdict(tmp_path):
    # между «Итерация N» и вердиктом бывают строки эксперимента (proba_weights и т. п.)
    body = _iter(3) + "[   11.0s] INFO  agent.experiment: proba_weights: 0.1 → 0.5\n"
    body += "[   12.0s] INFO  agent.loop:   · не лучше: 49.5 (лучший 47.1) — мало\n"
    assert wr.parse_log(_log(tmp_path, "t", body), "object_history", "t").decision == "отклонена"


def test_running_and_stale_runs(tmp_path):
    live = wr.parse_log(_log(tmp_path, "a", _iter(1, "x"), final=False), "object_history", "a")
    stale = wr.parse_log(_log(tmp_path, "b", _iter(1, "x"), final=False, age=3600), "object_history", "b")
    assert live.state(180) == "идёт" and live.decision == "ещё не дошла"
    assert stale.state(180) == "брошен?"


def test_same_as_logged_uses_precision_of_the_log():
    assert wr.same_as_logged("0.76961", 0.7696078695714696)
    assert wr.same_as_logged("1234.5", 1234.53)
    assert not wr.same_as_logged("0.76961", 0.7697)
    assert wr.same_as_logged("5.9691e+06", 5969124.1)


def _csv(path, rows):
    pd.DataFrame(rows).to_csv(path, index=False)
    return str(path)


def _row(task, value, score, metric="rmse", status="OK", seconds=100, minutes=5):
    return {
        "task": task,
        "metric": metric,
        "value": value,
        "model_score": score,
        "status": status,
        "seconds": seconds,
        "minutes": minutes,
    }


def _plan(tmp_path, before, after, **extra):
    plan = {"hypothesis": "object_history", "runs": [], "criteria": {"min_tasks": 2}}
    plan["pairs"] = [
        {"name": "15 мин", "before": _csv(tmp_path / "b.csv", before), "after": _csv(tmp_path / "a.csv", after)}
    ]
    return {**plan, **extra}


def _status(snapshot, code):
    return next(c["status"] for c in snapshot["checks"] if c["code"] == code)


def test_gain_on_one_task_of_three_fails_criterion_without_losses(tmp_path):
    before = [_row("tasks/public/regression", 44.138, 22.3), _row("sales", 0.47, 2.7), _row("util", 0.8, 3.5)]
    after = [_row("tasks/public/regression", 38.271, 32.6), _row("sales", 0.47, 2.7), _row("util", 0.8, 3.5)]
    snap = wr.evaluate(_plan(tmp_path, before, after))
    assert _status(snap, "К3") == wr.OK
    assert _status(snap, "К4") == wr.FAIL  # 1 из 3, нужно 2
    regression = next(r for r in snap["pairs"][0]["rows"] if r["task"] == "tasks/public/regression")
    assert regression["gain"] == pytest.approx(13.29, abs=0.01)
    assert regression["verdict"] == "🟢 лучше"


def test_loss_beyond_noise_and_direction_for_greater_is_better(tmp_path):
    before = [_row("mc", 0.60, 27.4, metric="f1_macro"), _row("r", 10.0, 20.0)]
    after = [_row("mc", 0.55, 13.1, metric="f1_macro"), _row("r", 9.7, 21.0)]
    snap = wr.evaluate(_plan(tmp_path, before, after))
    assert _status(snap, "К3") == wr.FAIL
    mc = next(r for r in snap["pairs"][0]["rows"] if r["task"] == "mc")
    assert mc["gain"] == pytest.approx(-8.33, abs=0.01)  # F1 растёт вверх: упал — минус


def test_missing_after_file_waits(tmp_path):
    plan = _plan(tmp_path, [_row("r", 10.0, 20.0)], [_row("r", 9.0, 25.0)])
    plan["pairs"][0]["after"] = str(tmp_path / "нет.csv")
    snap = wr.evaluate(plan)
    assert _status(snap, "К3") == wr.WAIT and _status(snap, "К4") == wr.WAIT
    assert wr.exit_code(snap) == 2


def _gate(tmp_path, names):
    lines = ["# Гейт перед сабмитом", "", "**❌ НЕ ПРОХОДИТ**", ""]
    lines += [f"- ❌ public «{n}»: балл к сильному 27.4 → 13.1 (шум ±2.0)" for n in names]
    (tmp_path / "gate.md").write_text("\n".join(lines), encoding="utf-8")
    return str(tmp_path / "gate.md")


def test_gate_failure_reproduced_on_dev_is_warning_not_failure(tmp_path):
    rows = [_row("tasks/public/multiclass", 0.58, 13.8, metric="f1_macro")]
    plan = _plan(tmp_path, rows, rows)
    plan["gate"] = {"report": _gate(tmp_path, ["multiclass"])}
    plan["rechecks"] = [
        {"name": "mc", "before": _csv(tmp_path / "dev.csv", rows), "after": _csv(tmp_path / "br.csv", rows)}
    ]
    assert _status(wr.evaluate(plan), "К6") == wr.WARN


def test_gate_failure_better_on_dev_is_failure_and_without_recheck_waits(tmp_path):
    rows = [_row("tasks/public/multiclass", 0.58, 13.1, metric="f1_macro")]
    plan = _plan(tmp_path, rows, rows)
    plan["gate"] = {"report": _gate(tmp_path, ["multiclass"])}
    assert _status(wr.evaluate(plan), "К6") == wr.WAIT
    dev = [_row("tasks/public/multiclass", 0.62, 27.4, metric="f1_macro")]
    plan["rechecks"] = [
        {"name": "mc", "before": _csv(tmp_path / "dev.csv", dev), "after": _csv(tmp_path / "br.csv", rows)}
    ]
    snap = wr.evaluate(plan)
    assert _status(snap, "К6") == wr.FAIL
    assert wr.exit_code(snap) == 1


def _jsonl(path, rows):
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")
    return str(path)


def _trap(task, value, score, exit_code=0, problems=()):
    return {
        "task": task,
        "metric": "rmse",
        "value": value,
        "score_vs_strong": score,
        "exit_code": exit_code,
        "problems": list(problems),
        "seconds": 250,
        "minutes": 5.0,
    }


def test_leak_traps_same_as_dev_pass_and_worse_fail(tmp_path):
    rows = [_row("r", 10.0, 20.0)]
    plan = _plan(tmp_path, rows, rows)
    dev = _jsonl(tmp_path / "dev.jsonl", [_trap("leak_a", 16.0, 0.0), _trap("leak_b", 0.77, 5.3)])
    plan["leak"] = {
        "before": dev,
        "after": _jsonl(tmp_path / "br.jsonl", [_trap("leak_a", 16.0, 0.0), _trap("leak_b", 0.77, 5.3)]),
    }
    assert _status(wr.evaluate(plan), "К7") == wr.OK
    plan["leak"]["after"] = _jsonl(tmp_path / "br2.jsonl", [_trap("leak_a", 16.0, 0.0), _trap("leak_b", 0.6, 0.0)])
    assert _status(wr.evaluate(plan), "К7") == wr.FAIL


def test_overtime_and_failed_status_are_failures(tmp_path):
    before = [_row("r", 10.0, 20.0)]
    after = [_row("r", 9.0, 25.0, seconds=500, minutes=5), _row("x", None, 0.0, status="FAILED")]
    snap = wr.evaluate(_plan(tmp_path, before, after))
    assert _status(snap, "К8") == wr.FAIL
    assert _status(snap, "К2") == wr.FAIL


def test_weak_acceptance_is_warning(tmp_path):
    body = _iter(3) + "[   12.0s] INFO  agent.loop:   ✓ лучше: 48.0 → 45.9 — лучше на 1/3 фолдах, t=0.44 (порог 2.0)\n"
    _log(tmp_path / "runs", "regression", body)
    rows = [_row("r", 10.0, 20.0)]
    plan = _plan(tmp_path, rows, rows, runs=[str(tmp_path / "runs")])
    snap = wr.evaluate(plan)
    assert _status(snap, "К5") == wr.WARN
    assert snap["runs"][0].name == "runs/regression"


def test_once_markdown_renders_all_checks_and_returns_code(tmp_path, capsys):
    before = [_row("r", 10.0, 20.0), _row("s", 5.0, 10.0)]
    after = [_row("r", 9.0, 25.0), _row("s", 4.8, 13.0)]
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(_plan(tmp_path, before, after), ensure_ascii=False), encoding="utf-8")
    code = wr.main(["--plan", str(plan_path), "--once", "--markdown"])
    text = capsys.readouterr().out
    assert code == 0  # К4: 2 из 2 задач лучше на ≥ 2%, потерь нет; остальных частей в плане нет
    for check in ("К1", "К2", "К3", "К4", "К5", "К6", "К7", "К8"):
        assert f"| {check} |" in text
    assert "## До/после: 15 мин" in text
