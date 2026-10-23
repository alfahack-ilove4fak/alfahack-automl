"""#261 E3: финальная модель — среднее нескольких сидов (AGENT_FINAL_SEEDS), CV не меняется."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import ml.experiment as experiment
from interfaces.schemas import DataProfile, Hypothesis, HypothesisKind, MetricSpec, RunBudget, TaskSpec, TaskType
from ml.experiment import SEED, final_seeds, fit_predict_final, run_experiment, seeded


def _spec(task_type: TaskType, metric: str, greater: bool) -> TaskSpec:
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
        metric=MetricSpec(name=metric, greater_is_better=greater),
        submission_columns=["id", "y"],
        class_labels=["neg", "pos"] if task_type == TaskType.BINARY else None,
    )


def _data(task_type: TaskType) -> tuple[pd.DataFrame, pd.DataFrame, DataProfile]:
    rng = np.random.default_rng(0)
    n = 300
    df = pd.DataFrame({"id": range(n), "a": rng.normal(size=n), "b": rng.normal(size=n)})
    signal = df["a"] + 0.5 * df["b"] + rng.normal(scale=0.5, size=n)
    df["y"] = (signal > 0).map({True: "pos", False: "neg"}) if task_type == TaskType.BINARY else signal
    profile = DataProfile(n_train=n, n_test=30, numeric_features=["a", "b"], categorical_features=[])
    return df, df.drop(columns="y").head(30), profile


def _hyp(**params) -> Hypothesis:
    # _best_iteration — как после CV: финал без него взял бы 3000 деревьев
    base = {"model_params": {"n_estimators": 40}, "_best_iteration": 40}
    return Hypothesis(id="h", kind=HypothesisKind.MODEL, title="тест", params={**base, **params})


@pytest.fixture
def seeds_used(monkeypatch):
    """Сиды, с которыми финал обучал модели."""
    used = []
    fit = experiment._fit_model

    def spy(*args, **kwargs):
        used.append(experiment._model_seed.get())
        return fit(*args, **kwargs)

    monkeypatch.setattr(experiment, "_fit_model", spy)
    return used


@pytest.mark.parametrize("value, expected", [(None, 3), ("1", 1), ("5", 5), ("0", 1), ("мусор", 3), ("50", 10)])
def test_final_seeds_from_env(monkeypatch, value, expected):
    if value is None:
        monkeypatch.delenv("AGENT_FINAL_SEEDS", raising=False)
    else:
        monkeypatch.setenv("AGENT_FINAL_SEEDS", value)
    assert final_seeds() == expected


def test_one_seed_is_old_behaviour(monkeypatch, seeds_used):
    """AGENT_FINAL_SEEDS=1 — одна модель с SEED, как до #261."""
    monkeypatch.setenv("AGENT_FINAL_SEEDS", "1")
    train, test, profile = _data(TaskType.REGRESSION)
    fit_predict_final(_spec(TaskType.REGRESSION, "rmse", False), profile, _hyp(), train, test)
    assert seeds_used == [SEED]


def test_regression_answer_is_mean_of_seeds(monkeypatch, seeds_used):
    spec = _spec(TaskType.REGRESSION, "rmse", False)
    train, test, profile = _data(TaskType.REGRESSION)
    monkeypatch.setenv("AGENT_FINAL_SEEDS", "3")
    averaged = fit_predict_final(spec, profile, _hyp(), train, test)["y"].to_numpy()
    assert seeds_used == [SEED, SEED + 1, SEED + 2]

    monkeypatch.setenv("AGENT_FINAL_SEEDS", "1")
    single = []
    for i in range(3):
        with seeded(SEED + i):
            single.append(fit_predict_final(spec, profile, _hyp(), train, test)["y"].to_numpy())
    assert not np.allclose(single[0], single[1])  # подвыборка строк/признаков — сиды действительно разные
    assert np.allclose(averaged, np.mean(single, axis=0))


def test_binary_proba_is_mean_of_seeds(monkeypatch):
    spec = _spec(TaskType.BINARY, "roc_auc", True)
    train, test, profile = _data(TaskType.BINARY)
    monkeypatch.setenv("AGENT_FINAL_SEEDS", "3")
    averaged = fit_predict_final(spec, profile, _hyp(), train, test)["y"].to_numpy()

    monkeypatch.setenv("AGENT_FINAL_SEEDS", "1")
    single = []
    for i in range(3):
        with seeded(SEED + i):
            single.append(fit_predict_final(spec, profile, _hyp(), train, test)["y"].to_numpy())
    assert ((averaged > 0) & (averaged < 1)).all()
    assert np.allclose(averaged, np.mean(single, axis=0))


def test_linear_model_fitted_once(monkeypatch, seeds_used):
    """У Ridge сид ничего не меняет — копии только тратили бы время."""
    monkeypatch.setenv("AGENT_FINAL_SEEDS", "3")
    train, test, profile = _data(TaskType.REGRESSION)
    hyp = _hyp(model="linear", model_params={})
    fit_predict_final(_spec(TaskType.REGRESSION, "rmse", False), profile, hyp, train, test)
    assert seeds_used == [SEED]


def test_ensemble_with_boosting_gets_seeds(monkeypatch, seeds_used):
    monkeypatch.setenv("AGENT_FINAL_SEEDS", "2")
    train, test, profile = _data(TaskType.REGRESSION)
    hyp = _hyp(ensemble=["lightgbm", "linear"], _best_iteration=[40, 0])
    answer = fit_predict_final(_spec(TaskType.REGRESSION, "rmse", False), profile, hyp, train, test)
    assert seeds_used == [SEED, SEED + 1]
    assert answer["y"].notna().all()


def test_cv_does_not_depend_on_final_seeds(monkeypatch):
    """Сиды касаются только финала: скор CV тот же при любом AGENT_FINAL_SEEDS."""
    spec = _spec(TaskType.REGRESSION, "rmse", False)
    train, _, profile = _data(TaskType.REGRESSION)
    scores = []
    for value in ("1", "3"):
        monkeypatch.setenv("AGENT_FINAL_SEEDS", value)
        hyp = Hypothesis(id="h", kind=HypothesisKind.MODEL, title="тест", params={"n_folds": 3})
        scores.append(run_experiment(spec, profile, hyp, train, RunBudget()).fold_scores)
    assert scores[0] == scores[1]
