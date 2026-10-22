"""Шаг 1 solution.py: бейзлайн в выбранной схеме валидации и запасной путь на обычную CV (#151, ADR 0013)."""

from pathlib import Path

import pytest

import solution
from interfaces.schemas import DataProfile, ExperimentResult, ExperimentStatus
from interfaces.stubs import stub_task_spec


def _result(hyp, ok=True):
    validation = hyp.params.get("validation", "kfold")
    if not ok:
        return ExperimentResult(
            hypothesis_id=hyp.id, status=ExperimentStatus.FAILED, metric_name="rmse", error="во фолде один класс"
        )
    return ExperimentResult(
        hypothesis_id=hyp.id,
        status=ExperimentStatus.OK,
        metric_name="rmse",
        fold_scores=[40.0] * 4,
        mean_score=40.0,
        std_score=0.0,
        validation=f"{validation}-4",
    )


@pytest.fixture
def spec():
    return stub_task_spec(Path("tasks/demo"))


@pytest.fixture
def profile():
    return DataProfile(n_train=500, n_test=200, numeric_features=["year"], categorical_features=[])


def _run(monkeypatch, spec, profile, scheme, fail_in_scheme=False):
    seen = []

    def fake_experiment(spec, profile, hyp, train, budget):
        seen.append(hyp.params.get("validation"))
        return _result(hyp, ok=not (fail_in_scheme and "validation" in hyp.params))

    def fake_choose(*args):
        if isinstance(scheme, Exception):
            raise scheme
        return scheme

    monkeypatch.setattr(solution, "run_experiment", fake_experiment)
    monkeypatch.setattr(solution, "choose_validation", fake_choose)
    base, result = solution.baseline_in_scheme(spec, profile, None, None, None)
    return base, result, seen


def test_baseline_runs_in_chosen_scheme(monkeypatch, spec, profile):
    base, result, seen = _run(monkeypatch, spec, profile, "time:year")
    assert seen == ["time:year"]
    assert base.params["validation"] == "time:year"  # цикл возьмёт схему из params бейзлайна (agent/loop.py)
    assert result.validation == "time:year-4"


def test_failed_scheme_falls_back_to_plain_cv(monkeypatch, spec, profile):
    """Схема роняет бейзлайн — пересчитываем в kfold, а не остаёмся с тривиальным ответом."""
    base, result, seen = _run(monkeypatch, spec, profile, "time:year", fail_in_scheme=True)
    assert seen == ["time:year", None]
    assert "validation" not in base.params
    assert result.mean_score == 40.0 and result.validation == "kfold-4"


def test_choose_validation_error_means_plain_cv(monkeypatch, spec, profile):
    base, result, seen = _run(monkeypatch, spec, profile, RuntimeError("сломался выбор схемы"))
    assert seen == [None] and "validation" not in base.params
    assert result.mean_score == 40.0


def test_plain_scheme_failure_is_not_retried(monkeypatch, spec, profile):
    """Бейзлайн упал и без схемы — второй раз то же самое не считаем, решает вызывающий код."""

    def always_fail(spec, profile, hyp, train, budget):
        always_fail.calls += 1
        return _result(hyp, ok=False)

    always_fail.calls = 0
    monkeypatch.setattr(solution, "run_experiment", always_fail)
    monkeypatch.setattr(solution, "choose_validation", lambda *a: None)
    _, result = solution.baseline_in_scheme(spec, profile, None, None, None)
    assert result.mean_score is None and always_fail.calls == 1
