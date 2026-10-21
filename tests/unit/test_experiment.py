"""run_experiment и fit_predict_final: выбор модели по params.model."""

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
from ml.experiment import fit_predict_final, run_experiment


def _spec(task_type: TaskType, metric: str, greater: bool, labels=None) -> TaskSpec:
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
        class_labels=labels,
    )


def _data(task_type: TaskType) -> tuple[pd.DataFrame, pd.DataFrame, DataProfile]:
    rng = np.random.default_rng(42)
    n = 300
    df = pd.DataFrame({"id": range(n), "a": rng.normal(size=n), "c": rng.choice(["x", "y", "z"], size=n)})
    signal = df["a"] + (df["c"] == "x") * 2
    df["y"] = (signal > signal.median()).map({True: "pos", False: "neg"}) if task_type == TaskType.BINARY else signal
    test = df.drop(columns="y").head(20)
    profile = DataProfile(n_train=n, n_test=20, numeric_features=["a"], categorical_features=["c"])
    return df, test, profile


def _hyp(**params) -> Hypothesis:
    return Hypothesis(id="h", kind=HypothesisKind.MODEL, title="тест", params={"n_folds": 3, **params})


@pytest.mark.parametrize("model", [None, "lightgbm", " LightGBM "])
def test_lightgbm_regression_ok(model):
    spec = _spec(TaskType.REGRESSION, "rmse", greater=False)
    train, test, profile = _data(TaskType.REGRESSION)
    hyp = _hyp(model=model, model_params={"n_estimators": 50})

    result = run_experiment(spec, profile, hyp, train, RunBudget())

    assert result.status == ExperimentStatus.OK, result.error
    assert len(result.fold_scores) == 3
    assert 1 <= hyp.params["_best_iteration"] <= 50
    answer = fit_predict_final(spec, profile, hyp, train, test)
    assert list(answer.columns) == ["id", "y"] and len(answer) == 20
    assert np.isfinite(answer["y"]).all()


def test_lightgbm_binary_answer_uses_class_labels():
    spec = _spec(TaskType.BINARY, "f1_macro", greater=True, labels=["neg", "pos"])
    train, test, profile = _data(TaskType.BINARY)
    hyp = _hyp(model="lightgbm", model_params={"n_estimators": 50})

    assert run_experiment(spec, profile, hyp, train, RunBudget()).status == ExperimentStatus.OK
    answer = fit_predict_final(spec, profile, hyp, train, test)
    assert set(answer["y"]) <= {"neg", "pos"}


def test_unknown_model_fails_instead_of_silent_lightgbm():
    spec = _spec(TaskType.REGRESSION, "rmse", greater=False)
    train, test, profile = _data(TaskType.REGRESSION)

    result = run_experiment(spec, profile, _hyp(model="random_forest"), train, RunBudget())

    assert result.status == ExperimentStatus.FAILED
    assert "random_forest" in result.error
    with pytest.raises(ValueError, match="неизвестная модель"):
        fit_predict_final(spec, profile, _hyp(model="random_forest"), train, test)


# Модель → (пакет, который нужен; гиперпараметры для быстрого теста; ждём ли ≥ 1 дерева)
MODELS = {
    "catboost": ("catboost", {"iterations": 50, "depth": 4}, True),
    "xgboost": ("xgboost", {"n_estimators": 50, "max_depth": 3}, True),
    "linear": (None, {}, False),
}


@pytest.mark.parametrize("task_type", [TaskType.REGRESSION, TaskType.BINARY])
@pytest.mark.parametrize("model", list(MODELS))
def test_extra_models_ok_and_write_no_files(model, task_type, tmp_path, monkeypatch):
    package, model_params, has_trees = MODELS[model]
    if package:
        pytest.importorskip(package)  # catboost/xgboost — из группы full; без неё тест пропускается
    monkeypatch.chdir(tmp_path)  # на платформе рабочая папка — каталог задачи, мусор туда писать нельзя
    if task_type == TaskType.REGRESSION:
        spec = _spec(task_type, "rmse", greater=False)
    else:
        spec = _spec(task_type, "f1_macro", greater=True, labels=["neg", "pos"])
    train, test, profile = _data(task_type)
    hyp = _hyp(model=model, class_weight="balanced", model_params=model_params)

    result = run_experiment(spec, profile, hyp, train, RunBudget())

    assert result.status == ExperimentStatus.OK, result.error
    if has_trees:
        assert 1 <= hyp.params["_best_iteration"] <= 50
    answer = fit_predict_final(spec, profile, hyp, train, test)
    assert len(answer) == 20
    if task_type == TaskType.BINARY:
        assert set(answer["y"]) <= {"neg", "pos"}
    else:
        assert np.isfinite(answer["y"]).all()
    assert list(tmp_path.iterdir()) == []  # нет папки catboost_info и прочего мусора


def test_linear_handles_missing_values_and_new_categories():
    spec = _spec(TaskType.REGRESSION, "rmse", greater=False)
    train, test, profile = _data(TaskType.REGRESSION)
    train.loc[::7, "a"] = np.nan
    test = test.assign(c=["new"] * len(test))  # категории, которой не было в train

    hyp = _hyp(model="linear", model_params={"alpha": 10.0})
    assert run_experiment(spec, profile, hyp, train, RunBudget()).status == ExperimentStatus.OK
    assert np.isfinite(fit_predict_final(spec, profile, hyp, train, test)["y"]).all()


def test_catboost_rejects_lightgbm_params():
    pytest.importorskip("catboost")
    spec = _spec(TaskType.REGRESSION, "rmse", greater=False)
    train, _, profile = _data(TaskType.REGRESSION)

    result = run_experiment(
        spec, profile, _hyp(model="catboost", model_params={"colsample_bytree": 0.8}), train, RunBudget()
    )

    assert result.status == ExperimentStatus.FAILED
    assert "colsample_bytree" in result.error


@pytest.mark.parametrize("model", ["lightgbm", *MODELS])
def test_models_accept_special_column_names(model):
    package = MODELS.get(model, (None,))[0]
    if package:
        pytest.importorskip(package)
    spec = _spec(TaskType.REGRESSION, "rmse", greater=False)
    train, _, _ = _data(TaskType.REGRESSION)
    train = train.rename(columns={"a": "income[usd]", "c": 'city<"x">'})
    profile = DataProfile(
        n_train=len(train), n_test=0, numeric_features=["income[usd]"], categorical_features=['city<"x">']
    )

    result = run_experiment(
        spec, profile, _hyp(model=model, model_params=MODELS.get(model, (None, {}))[1]), train, RunBudget()
    )

    assert result.status == ExperimentStatus.OK, result.error


def _multiclass_with_rare_class() -> tuple[TaskSpec, pd.DataFrame, DataProfile]:
    rng = np.random.default_rng(0)
    n = 200
    df = pd.DataFrame({"id": range(n), "a": rng.normal(size=n), "c": rng.choice(["x", "y"], size=n)})
    df["y"] = np.where(df["a"] > 0, "up", "down")
    df.loc[0, "y"] = "rare"  # класс с одним объектом
    spec = _spec(TaskType.MULTICLASS, "f1_macro", greater=True, labels=["down", "rare", "up"])
    profile = DataProfile(n_train=n, n_test=0, numeric_features=["a"], categorical_features=["c"])
    return spec, df, profile


@pytest.mark.parametrize("model", ["lightgbm", *MODELS])
def test_class_with_one_object(model):
    package, model_params, _ = MODELS.get(model, (None, {}, True))
    if package:
        pytest.importorskip(package)
    spec, train, profile = _multiclass_with_rare_class()

    result = run_experiment(spec, profile, _hyp(model=model, model_params=model_params), train, RunBudget())

    assert result.status == ExperimentStatus.OK, result.error


def test_split_keeps_rare_class_in_train_and_old_folds_otherwise():
    from sklearn.model_selection import StratifiedKFold

    from ml.experiment import SEED, _split

    spec, _, _ = _multiclass_with_rare_class()
    y = np.array([0] + [1, 2] * 50)  # объект 0 — единственный в своём классе
    for tr, va in _split(spec, y, 5):
        assert 0 in tr and 0 not in va
        assert len(np.intersect1d(tr, va)) == 0

    y_common = np.array([1, 2] * 50)
    old = StratifiedKFold(5, shuffle=True, random_state=SEED).split(np.zeros(len(y_common)), y_common)
    for (tr, va), (tr_old, va_old) in zip(_split(spec, y_common, 5), old, strict=True):
        assert (tr == tr_old).all() and (va == va_old).all()
