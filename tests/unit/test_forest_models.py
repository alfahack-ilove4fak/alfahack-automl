"""RandomForest и ExtraTrees (#261): модели реестра ml/, запасные направления агента, участие в ансамбле."""

from pathlib import Path

import numpy as np
import pytest

from agent.fallback import fallback_hypotheses
from interfaces.schemas import (
    ExperimentResult,
    ExperimentStatus,
    Hypothesis,
    HypothesisKind,
    RunBudget,
    TaskType,
)
from interfaces.stubs import stub_task_spec
from ml.experiment import MODEL_FITTERS, fit_predict_final, run_experiment
from tests.unit.test_playbook import _toy_task


@pytest.fixture
def fallback_on(monkeypatch):
    monkeypatch.setenv("AGENT_FALLBACK", "1")  # conftest выключает запасные направления для остальных тестов цикла


def _hyp(**params) -> Hypothesis:
    params = {"n_folds": 3, **params}
    return Hypothesis(id="h", kind=HypothesisKind.MODEL, title="лес", params=params)


def test_forest_models_registered():
    assert {"random_forest", "extra_trees"} <= set(MODEL_FITTERS)


@pytest.mark.parametrize("model", ["random_forest", "extra_trees"])
@pytest.mark.parametrize("task_type", [TaskType.REGRESSION, TaskType.BINARY, TaskType.MULTICLASS])
def test_forest_cv_and_final(model, task_type):
    """Пропуски (x2) и категория (cat) не мешают; CV считается, финал даёт ответ нужной формы."""
    spec, profile, train = _toy_task(task_type)
    hyp = _hyp(model=model, model_params={"n_estimators": 30})
    result = run_experiment(spec, profile, hyp, train, RunBudget())
    assert result.status == ExperimentStatus.OK, result.error
    answer = fit_predict_final(spec, profile, hyp, train, train.drop(columns="y").head(50))
    assert len(answer) == 50 and answer["y"].notna().all()


def test_forest_final_ignores_tree_count_from_cv():
    """У леса нет ранней остановки: _best_iteration = 0 из CV не превращается в 50 деревьев (как у linear)."""
    spec, profile, train = _toy_task(TaskType.REGRESSION)
    hyp = _hyp(model="random_forest", model_params={"n_estimators": 40})
    assert run_experiment(spec, profile, hyp, train, RunBudget()).status == ExperimentStatus.OK
    assert hyp.params["_best_iteration"] == 0
    from ml.experiment import _fit_model

    x = train[["x1", "x2"]].assign(cat=train["cat"].astype("category"))
    model, _ = _fit_model(spec, hyp, x, train["y"].to_numpy(), n_estimators=999)
    assert model[-1].n_estimators == 40


def test_forest_in_ensemble_with_boosting():
    spec, profile, train = _toy_task(TaskType.BINARY)
    members = ["lightgbm", {"model": "random_forest", "model_params": {"n_estimators": 30}}]
    hyp = _hyp(ensemble=members)
    assert run_experiment(spec, profile, hyp, train, RunBudget()).status == ExperimentStatus.OK
    answer = fit_predict_final(spec, profile, hyp, train, train.drop(columns="y").head(40))
    assert len(answer) == 40


def _best(model="lightgbm", **params):
    spec = stub_task_spec(Path("tasks/demo"))
    hyp = Hypothesis(id="best", kind=HypothesisKind.MODEL, title="best", params={"model": model, **params})
    res = ExperimentResult(
        hypothesis_id="best",
        status=ExperimentStatus.OK,
        metric_name="rmse",
        fold_scores=[40.0] * 5,
        mean_score=40.0,
        std_score=0.0,
    )
    return spec, (hyp, res)


def test_fallback_has_forest_directions(fallback_on):
    spec, best = _best()
    hyps = fallback_hypotheses(spec, [best], best, RunBudget(total_seconds=3600))
    models = [h.params.get("model") for h in hyps if "ensemble" not in h.params]
    assert "random_forest" in models and "extra_trees" in models
    forest_ensembles = [
        h
        for h in hyps
        if "ensemble" in h.params and "random_forest" in h.params["ensemble"] and len(h.params["ensemble"]) == 2
    ]
    assert len(forest_ensembles) == 1 and forest_ensembles[0].params["ensemble"][0] == "lightgbm"


def test_no_optuna_for_forest_best(fallback_on):
    """У леса нет пространства поиска Optuna (ml/tuning.py) — направление не добавляем, гипотеза упала бы."""
    spec, best = _best("random_forest")
    hyps = fallback_hypotheses(spec, [best], best, RunBudget(total_seconds=3600))
    assert not any(h.params.get("optuna") and h.params.get("model") == "random_forest" for h in hyps)
    assert not any("ensemble" in h.params and h.params["ensemble"] == ["random_forest", "random_forest"] for h in hyps)


def test_frame_dtypes_survive():
    """Категории после _category_codes → числа: лес не видит строк."""
    spec, profile, train = _toy_task(TaskType.REGRESSION)
    x = train[["x1", "x2"]].assign(cat=train["cat"].astype("category"))
    model, _ = MODEL_FITTERS["extra_trees"](
        spec, _hyp(model="extra_trees", model_params={"n_estimators": 10}), x, train["y"].to_numpy()
    )
    assert np.isfinite(model.predict(x.head(5))).all()
