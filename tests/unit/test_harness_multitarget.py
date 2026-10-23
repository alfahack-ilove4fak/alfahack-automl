"""Харнесс на задаче с несколькими таргетами (#220) и устойчивость к журналу LLM (ревью #217)."""

import json

import numpy as np
import pandas as pd
import pytest

from evaluation.harness import llm_usage, make_holdout_task, score_targets
from evaluation.local_score import score_submission


def _multitarget_task(folder, n=200):
    rng = np.random.default_rng(0)
    x = rng.normal(size=n)
    train = pd.DataFrame({"id": range(n), "x": x, "heat": 10 * x + 50, "water": x + 5})
    train.to_csv(folder / "train.csv", index=False)
    pd.DataFrame({"id": range(n, n + 20), "x": rng.normal(size=20)}).to_csv(folder / "test.csv", index=False)
    pd.DataFrame({"id": range(n, n + 20), "heat": 0.0, "water": 0.0}).to_csv(
        folder / "sample_submission.csv", index=False
    )
    (folder / "task.md").write_text(
        "Предскажите `heat` и `water`. Метрика — MAE, среднее по таргетам.", encoding="utf-8"
    )
    return folder / "task.md"


def test_holdout_keeps_all_targets(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    prepared = make_holdout_task(_multitarget_task(src), tmp_path / "out")
    spec = prepared["spec"]
    assert sorted([spec.target, *spec.extra_targets]) == ["heat", "water"]
    task_dir = tmp_path / "out" / "task"
    assert list(pd.read_csv(task_dir / "test.csv").columns) == ["id", "x"]  # ни один таргет не утёк в test
    assert sorted(pd.read_csv(task_dir / "sample_submission.csv").columns) == ["heat", "id", "water"]
    answers = pd.read_csv(tmp_path / "out" / "answers.csv")
    assert sorted(answers.columns) == ["heat", "id", "water"] and len(answers) > 0
    assert sorted(prepared["y_train"].columns) == ["heat", "water"]


def test_score_is_mean_over_targets(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    prepared = make_holdout_task(_multitarget_task(src), tmp_path / "out")
    spec, y_train = prepared["spec"], prepared["y_train"]
    answers = pd.read_csv(tmp_path / "out" / "answers.csv")
    submission = answers.copy()
    submission[spec.target] += 1.0  # основной таргет ошибается на 1, второй — точный
    merged = answers.merge(submission, on="id", suffixes=("_true", "_pred"))
    scores = score_targets(spec, merged, y_train, x_train=None, x_test=None)
    assert scores["value"] == pytest.approx(0.5)  # MAE (1 + 0) / 2
    per_target = [
        score_submission(spec.model_copy(update={"target": t}), merged[f"{t}_true"], merged[f"{t}_pred"], y_train[t])
        for t in ("heat", "water")
    ]
    assert scores["baseline"] == pytest.approx(np.mean([p["baseline"] for p in per_target]))
    assert 0 < scores["score"] <= 100


def test_single_target_score_unchanged(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    task_md = _multitarget_task(src)
    pd.read_csv(src / "train.csv").drop(columns="water").to_csv(src / "train.csv", index=False)
    pd.read_csv(src / "sample_submission.csv").drop(columns="water").to_csv(src / "sample_submission.csv", index=False)
    prepared = make_holdout_task(task_md, tmp_path / "out")
    spec, y_train = prepared["spec"], prepared["y_train"]
    answers = pd.read_csv(tmp_path / "out" / "answers.csv")
    merged = answers.merge(answers, on="id", suffixes=("_true", "_pred"))
    expected = score_submission(spec, merged["heat_true"], merged["heat_pred"], y_train["heat"])
    assert score_targets(spec, merged, y_train, x_train=None, x_test=None) == expected


def test_llm_ledger_with_garbage_fields_does_not_fail_run(tmp_path):
    # ревью #217: нечисловое поле журнала не должно превращать успешный прогон в FAILED
    run = tmp_path / "run"
    run.mkdir()
    rows = [
        {"ok": True, "prompt": "abc", "completion": 5, "cost": 0.1},
        {"ok": True, "prompt": [1], "completion": 1},
        {"ok": True, "prompt": 10, "completion": 5, "cost": float("nan")},
        {"ok": True, "prompt": 100, "completion": 50, "cost": 0.2},
    ]
    (run / "llm_calls.jsonl").write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    assert llm_usage(run) == {"llm_calls": 2, "llm_errors": 0, "llm_tokens": 165, "llm_rub": 0.2}
