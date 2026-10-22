"""Задача с несколькими таргетами (#174, контракт 1.1.1): разбор, тривиальный ответ и ответ модели по каждому."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import solution
from agent.task_parser import parse_task, spec_for_target
from evaluation.submission import validate_submission
from interfaces.schemas import TaskType
from runtime.budget import make_budget, plan_stages


@pytest.fixture
def task_md(tmp_path, monkeypatch) -> Path:
    """Два таргета: число (cost) и класс (risk). В test их нет, в sample_submission — оба."""
    for key in ("OPENAI_API_KEY", "OPENAI_MODEL", "OPENAI_BASE_URL"):
        monkeypatch.delenv(key, raising=False)
    rng = np.random.default_rng(0)
    n = 600
    x = pd.DataFrame({"row_id": range(n), "a": rng.normal(size=n), "b": rng.normal(size=n), "c": rng.integers(0, 5, n)})
    x["cost"] = 3 * x["a"] - x["b"] + rng.normal(scale=0.1, size=n)
    x["risk"] = np.where(x["a"] + x["b"] > 0, "high", "low")
    train, test = x.iloc[:450], x.iloc[450:].drop(columns=["cost", "risk"])
    train.to_csv(tmp_path / "train.csv", index=False)
    test.to_csv(tmp_path / "test.csv", index=False)
    test[["row_id"]].assign(cost=0.0, risk="low").to_csv(tmp_path / "sample_submission.csv", index=False)
    (tmp_path / "task.md").write_text("# Два ответа\n\nПредскажите `cost` и `risk`. Метрика — MAE.\n", encoding="utf-8")
    return tmp_path / "task.md"


def test_parse_task_fills_extra_targets(task_md):
    spec = parse_task(task_md)
    assert spec.target == "cost" and spec.extra_targets == ["risk"]  # основной — тот, к которому подходит MAE
    assert any("Несколько таргетов" in n for n in spec.notes)


def test_single_target_task_has_no_extra(tmp_path):
    pd.DataFrame({"id": [1, 2, 3], "x": [1, 2, 3], "y": [1.0, 2.0, 3.0]}).to_csv(tmp_path / "train.csv", index=False)
    pd.DataFrame({"id": [4], "x": [4]}).to_csv(tmp_path / "test.csv", index=False)
    pd.DataFrame({"id": [4], "y": [0.0]}).to_csv(tmp_path / "sample_submission.csv", index=False)
    (tmp_path / "task.md").write_text("Метрика — RMSE.", encoding="utf-8")
    assert parse_task(tmp_path / "task.md").extra_targets == []


def test_spec_for_target_types(task_md):
    spec = parse_task(task_md)
    train = pd.read_csv(spec.train_path)
    cost = spec_for_target(spec, "cost", train["cost"])
    assert cost.task_type == TaskType.REGRESSION and cost.class_labels is None
    assert cost.submission_columns == ["row_id", "cost"] and cost.extra_targets == []


def test_trivial_answer_covers_all_targets(task_md):
    spec = parse_task(task_md)
    train, test = pd.read_csv(spec.train_path), pd.read_csv(spec.test_path)
    answer = solution.trivial_answer(spec, train, test)
    assert list(answer.columns) == ["row_id", "cost", "risk"]
    assert answer["cost"].iloc[0] == pytest.approx(train["cost"].mean())
    assert set(answer["risk"]) <= {"high", "low"}


def test_pipeline_writes_model_answer_for_every_target(task_md, monkeypatch):
    # цикл подменён — проверяем шаги 0–1: бейзлайн основного таргета + модель на дополнительный
    monkeypatch.setattr(solution, "run_autoresearch", lambda *a, **kw: a[5])
    budget = make_budget()
    keeper = solution.AnswerKeeper(task_md)
    solution.run_pipeline(task_md, budget, plan_stages(budget), keeper, budget.started_at)
    sub = pd.read_csv(task_md.parent / "submission.csv")
    spec = parse_task(task_md)
    assert validate_submission(task_md.parent / "submission.csv", spec, pd.read_csv(spec.test_path)).ok
    assert list(sub.columns) == ["row_id", "cost", "risk"]
    assert sub["cost"].nunique() > 10  # модель, а не среднее
    truth = 3 * pd.read_csv(spec.test_path)["a"] - pd.read_csv(spec.test_path)["b"]
    assert np.mean(np.abs(sub["cost"] - truth)) < 1.0  # модель уловила зависимость
    assert sub["risk"].nunique() == 2
