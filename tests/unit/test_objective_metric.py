"""Цель обучения под метрику — params.objective="metric" (#186, #237): выбор цели, α квантиля, обучение по моделям."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from interfaces.schemas import (
    DataProfile,
    ExperimentStatus,
    Hypothesis,
    HypothesisKind,
    MetricSpec,
    RunBudget,
    TaskSpec,
    TaskType,
)
from ml.experiment import (
    _objective_params,
    fit_predict_final,
    has_metric_objective,
    metric_objective_kind,
    run_experiment,
)


def _spec(metric: str, task_type: TaskType = TaskType.REGRESSION) -> TaskSpec:
    root = Path("/nonexistent")
    return TaskSpec(
        task_md=root / "task.md",
        task_dir=root,
        train_path=root / "train.csv",
        test_path=root / "test.csv",
        submission_path=root / "submission.csv",
        task_type=task_type,
        target="y",
        id_column="id",
        metric=MetricSpec(name=metric, greater_is_better=False),
        submission_columns=["id", "y"],
        class_labels=["a", "b"] if task_type != TaskType.REGRESSION else None,
    )


def _hyp(**params) -> Hypothesis:
    return Hypothesis(id="h", kind=HypothesisKind.HYPERPARAMS, title="тест", params={"n_folds": 3, **params})


def _skewed() -> tuple[pd.DataFrame, pd.DataFrame, DataProfile]:
    """Регрессия со скошенным шумом: верхний квантиль заметно выше среднего — L2 и квантиль 0.9 расходятся."""
    rng = np.random.default_rng(0)
    n = 1200
    df = pd.DataFrame({"id": range(n), "a": rng.normal(size=n), "b": rng.normal(size=n)})
    df["y"] = 10 + 3 * df["a"] + rng.exponential(scale=4.0, size=n)
    profile = DataProfile(n_train=n, n_test=200, numeric_features=["a", "b"], categorical_features=[])
    return df, df.drop(columns="y").head(200), profile


@pytest.mark.parametrize(
    "metric, expected",
    [
        ("pinball@0.9", ("quantile", 0.9)),
        ("pinball", ("quantile", 0.5)),
        ("mae", ("l1", 0.5)),
        ("wape", ("l1", 0.5)),
        ("huber", ("huber", 0.5)),
        ("mape", ("mape", 0.5)),
        ("tweedie_deviance", ("tweedie", 0.5)),
        ("rmse", None),  # L2 по умолчанию и так под RMSE
        ("rmsle", None),  # log_target по умолчанию — метрика уже на другой шкале
    ],
)
def test_metric_objective_kind(metric, expected):
    assert metric_objective_kind(_spec(metric), {"objective": "metric"}) == expected


def test_no_key_no_objective_and_classification_skipped():
    assert metric_objective_kind(_spec("mae"), {}) is None
    assert metric_objective_kind(_spec("logloss", TaskType.BINARY), {"objective": "metric"}) is None
    assert metric_objective_kind(_spec("mae"), {"objective": "metric", "log_target": True}) is None


def test_unknown_objective_value_is_error():
    with pytest.raises(ValueError, match="только 'metric'"):
        metric_objective_kind(_spec("mae"), {"objective": "l1"})


@pytest.mark.parametrize(
    "model, expected",
    [
        ("lightgbm", {"objective": "quantile", "alpha": 0.9}),
        ("xgboost", {"objective": "reg:quantileerror", "quantile_alpha": 0.9}),
        ("catboost", {"loss_function": "Quantile:alpha=0.9"}),
        ("hist_gb", {"loss": "quantile", "quantile": 0.9}),
        ("linear", {}),  # у Ridge квантильной цели нет — L2
    ],
)
def test_objective_params_per_model(model, expected):
    assert _objective_params(_spec("pinball@0.9"), _hyp(objective="metric"), model, None) == expected


def test_explicit_model_objective_wins():
    hyp = _hyp(objective="metric", model_params={"objective": "huber"})
    assert _objective_params(_spec("pinball@0.9"), hyp, "lightgbm", None) == {}


def test_target_unfit_for_objective_falls_back_to_l2():
    hyp = _hyp(objective="metric")
    assert _objective_params(_spec("poisson_deviance"), hyp, "lightgbm", np.array([-1.0, 2.0])) == {}
    assert _objective_params(_spec("gamma_deviance"), hyp, "lightgbm", np.array([0.0, 2.0])) == {}
    assert _objective_params(_spec("poisson_deviance"), hyp, "lightgbm", np.array([0.0, 2.0])) == {
        "objective": "poisson"
    }


@pytest.mark.parametrize("model", ["lightgbm", "hist_gb"])
def test_quantile_objective_beats_l2_on_pinball(model):
    """pinball@0.9 на скошенном шуме: цель-квантиль лучше L2 в CV, а прогноз сдвинут вверх (верхняя граница)."""
    train, test, profile = _skewed()
    spec = _spec("pinball@0.9")
    params = {"model": model, "model_params": {"n_estimators": 200} if model == "lightgbm" else {}}
    plain, tuned = _hyp(**params), _hyp(**params, objective="metric")

    plain_result = run_experiment(spec, profile, plain, train, RunBudget())
    tuned_result = run_experiment(spec, profile, tuned, train, RunBudget())

    assert tuned_result.status == ExperimentStatus.OK, tuned_result.error
    assert tuned_result.mean_score < plain_result.mean_score * 0.8
    plain_answer = fit_predict_final(spec, profile, plain, train, test)
    tuned_answer = fit_predict_final(spec, profile, tuned, train, test)
    assert tuned_answer["y"].mean() > plain_answer["y"].mean() + 1.0


@pytest.mark.parametrize("model", ["xgboost", "catboost"])
def test_quantile_objective_heavy_models(model):
    pytest.importorskip(model)  # из группы full; без неё тест пропускается
    train, test, profile = _skewed()
    spec = _spec("pinball@0.9")
    plain, tuned = _hyp(model=model), _hyp(model=model, objective="metric")

    plain_result = run_experiment(spec, profile, plain, train, RunBudget())
    tuned_result = run_experiment(spec, profile, tuned, train, RunBudget())

    assert tuned_result.status == ExperimentStatus.OK, tuned_result.error
    assert tuned_result.mean_score < plain_result.mean_score


def test_wrong_objective_value_is_failed_not_crash():
    train, _, profile = _skewed()
    result = run_experiment(_spec("mae"), profile, _hyp(objective="l1"), train, RunBudget())
    assert result.status == ExperimentStatus.FAILED and "objective" in result.error


@pytest.mark.parametrize(
    "metric, model, expected",
    [
        ("pinball@0.1", "xgboost", True),
        ("pinball@0.1", "catboost", True),
        ("mape", "xgboost", False),  # у XGBoost нет цели MAPE — был бы повтор на L2
        ("mape", "catboost", True),
        ("rmse", "lightgbm", False),
        ("rmsle", "catboost", False),
        ("mae", "linear", False),
    ],
)
def test_has_metric_objective(metric, model, expected):
    assert has_metric_objective(_spec(metric), {"model": model, "objective": "metric"}) is expected


def test_playbook_metric_objective_only_when_it_changes_something():
    """requires: metric_objective — гипотеза берётся, только если у метрики есть своя цель для её модели (#237)."""
    from agent.hypotheses import _applies

    profile = DataProfile(n_train=10, n_test=5, numeric_features=["a"], categorical_features=[])
    item = {"task_types": ["regression"], "requires": "metric_objective", "params": {"model": "xgboost"}}
    assert _applies(item, _spec("pinball@0.9"), profile)
    assert not _applies(item, _spec("mape"), profile)
    assert not _applies(item, _spec("rmse"), profile)
