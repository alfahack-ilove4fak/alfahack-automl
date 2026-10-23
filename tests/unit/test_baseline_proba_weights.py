"""#191: бейзлайн и гипотезы подбирают множители классов (proba_weights), если метрика считается по меткам."""

from pathlib import Path

import pytest

from agent.hypotheses import baseline_hypothesis, propose_hypotheses, with_task_defaults
from interfaces.schemas import DataProfile, ExperimentResult, ExperimentStatus, MetricSpec, TaskType
from interfaces.stubs import stub_task_spec


def spec_for(task_type: TaskType, metric: str, greater: bool = True):
    return stub_task_spec(Path("tasks/demo")).model_copy(
        update={"task_type": task_type, "metric": MetricSpec(name=metric, greater_is_better=greater)}
    )


@pytest.mark.parametrize(
    "task_type, metric, expected",
    [
        (TaskType.BINARY, "f1", {"model": "lightgbm", "proba_weights": True}),
        (TaskType.MULTICLASS, "accuracy", {"model": "lightgbm"}),  # accuracy — argmax, множители лишь шумят
        (TaskType.REGRESSION, "rmse", {"model": "lightgbm"}),
    ],
)
def test_baseline_params(task_type, metric, expected):
    assert baseline_hypothesis(spec_for(task_type, metric)).params == expected
    assert baseline_hypothesis().params == {"model": "lightgbm"}  # без задачи — как раньше


@pytest.mark.parametrize(
    "task_type, metric, expected",
    [
        (TaskType.BINARY, "f1", True),  # редкий класс + F1: без подбора порога — лотерея
        (TaskType.MULTICLASS, "f1_macro", True),
        (TaskType.MULTICLASS, "qwk", True),
        (TaskType.MULTICLASS, "accuracy", False),
        (TaskType.BINARY, "roc_auc", False),  # метрика по вероятностям — множители не нужны
        (TaskType.MULTICLASS, "logloss", False),
        (TaskType.REGRESSION, "rmse", False),
    ],
)
def test_with_task_defaults(task_type, metric, expected):
    params = with_task_defaults({"model": "xgboost"}, spec_for(task_type, metric))
    assert ("proba_weights" in params) is expected


def test_explicit_value_kept():
    spec = spec_for(TaskType.BINARY, "f1")
    assert with_task_defaults({"proba_weights": False}, spec) == {"proba_weights": False}


def _history(spec):
    base = baseline_hypothesis(spec)
    result = ExperimentResult(
        hypothesis_id="baseline", status=ExperimentStatus.OK, metric_name=spec.metric.name, mean_score=0.5
    )
    return [(base, result)]


@pytest.mark.parametrize("metric, expected", [("f1", True), ("roc_auc", False)])
def test_playbook_hypotheses_get_label_defaults(metric, expected):
    """Гипотезы playbook не наследуют params лучшего — без этого они проигрывали бы бейзлайну с множителями."""
    spec = spec_for(TaskType.BINARY, metric)
    profile = DataProfile(n_train=500, n_test=200, numeric_features=["a"], categorical_features=["c"])
    hyps = propose_hypotheses(spec, profile, _history(spec), k=5)
    assert hyps
    assert all(("proba_weights" in h.params) is expected for h in hyps)
    # повтор бейзлайна (только proba_weights поверх LightGBM по умолчанию) не предлагается
    assert all(h.params != {"model": "lightgbm", "proba_weights": True} for h in hyps)
