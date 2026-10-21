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
    "hist_gb": (None, {"max_iter": 50}, True),
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


def test_params_doc_keys_parse_after_dedent():
    """Ключи params в docstring разбирает agent (prompting.allowed_param_keys): текст после «Поддерживаемые params»,
    textwrap.dedent, ключ — в начале строки. Абзац без отступа в этом куске ломает разбор у всех ключей."""
    import re
    import textwrap

    import ml.experiment

    section = ml.experiment.__doc__.split("Поддерживаемые params", 1)[1].split("\n", 1)[-1]
    keys = set(re.findall(r"^(\w+)\s+—", textwrap.dedent(section), re.MULTILINE))
    assert {"model", "model_params", "n_folds", "log_target", "class_weight"} <= keys


def test_cv_subsample_when_too_many_rows(monkeypatch):
    import ml.experiment

    monkeypatch.setattr(ml.experiment, "MAX_CV_ROWS", 150)
    monkeypatch.setattr(ml.experiment, "MIN_CV_ROWS", 10)
    spec = _spec(TaskType.REGRESSION, "rmse", greater=False)
    train, _, profile = _data(TaskType.REGRESSION)
    y = train["y"].to_numpy()

    rows = ml.experiment._cv_rows(spec, profile, y, RunBudget())
    assert rows is not None and len(rows) == 150
    assert (rows == ml.experiment._cv_rows(spec, profile, y, RunBudget())).all()  # детерминированно

    result = run_experiment(spec, profile, _hyp(model_params={"n_estimators": 50}), train, RunBudget())
    assert result.status == ExperimentStatus.OK, result.error


def test_cv_subsample_keeps_rare_class():
    from ml.experiment import _sample_rows

    spec, _, _ = _multiclass_with_rare_class()
    y = np.array([0] * 3 + [1] * 5000 + [2] * 5000)
    rows = _sample_rows(spec, y, 1000)
    assert set(y[rows]) == {0, 1, 2} and (y[rows] == 0).sum() == 3
    assert 950 <= len(rows) <= 1050


def test_not_enough_memory_fails_cleanly(monkeypatch):
    from runtime.budget import used_ram_mb

    spec = _spec(TaskType.REGRESSION, "rmse", greater=False)
    train, _, profile = _data(TaskType.REGRESSION)
    budget = RunBudget(ram_limit_mb=used_ram_mb())  # свободной памяти нет

    result = run_experiment(spec, profile, _hyp(), train, budget)

    assert result.status == ExperimentStatus.FAILED
    assert "MemoryError" in result.error


def test_first_fold_forecast_stops_early(monkeypatch):
    import time

    import ml.experiment

    class ShortBudget(RunBudget):
        def seconds_left(self) -> float:
            return 0.5  # время ещё есть, но на 4 фолда по 0.3 с — нет

    real_fit = ml.experiment._fit_model

    def slow_fit(*args, **kwargs):
        time.sleep(0.3)
        return real_fit(*args, **kwargs)

    monkeypatch.setattr(ml.experiment, "_fit_model", slow_fit)
    spec = _spec(TaskType.REGRESSION, "rmse", greater=False)
    train, _, profile = _data(TaskType.REGRESSION)

    result = run_experiment(spec, profile, _hyp(model_params={"n_estimators": 20}), train, ShortBudget())

    assert result.status == ExperimentStatus.TIMEOUT
    assert len(result.fold_scores) == 1 and "прогноз" in result.error


@pytest.mark.skipif(not __import__("os").getenv("ML_BIG_TEST"), reason="долгий: ML_BIG_TEST=1 uv run pytest …")
def test_big_synthetic_fits_budget():
    """M-L1-3: 1.5 млн строк, 6 признаков — не падает и укладывается в бюджет 10 минут."""
    import time

    rng = np.random.default_rng(1)
    n = 1_500_000
    train = pd.DataFrame(rng.normal(size=(n, 5)).astype(np.float32), columns=[f"x{i}" for i in range(5)])
    train["c"] = rng.choice(list("abcdefghij"), size=n)
    train["y"] = train["x0"] * 2 + train["x1"] ** 2 + (train["c"] == "a") + rng.normal(size=n)
    profile = DataProfile(n_train=n, n_test=0, numeric_features=[f"x{i}" for i in range(5)], categorical_features=["c"])
    spec = _spec(TaskType.REGRESSION, "rmse", greater=False)
    budget = RunBudget(total_seconds=600, safety_seconds=0)

    started = time.time()
    result = run_experiment(spec, profile, _hyp(n_folds=5), train, budget)

    print(f"\n1.5 млн строк: {result.status.value}, {time.time() - started:.0f} с, RAM {result.peak_ram_mb:.0f} МБ")
    assert result.status == ExperimentStatus.OK, result.error
    assert time.time() - started < budget.total_seconds


def test_split_when_all_classes_are_rare():
    from ml.experiment import _split

    spec, _, _ = _multiclass_with_rare_class()
    y = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])  # по 3 объекта, а фолдов 5

    folds = _split(spec, y, 5)

    assert len(folds) == 3
    for tr, va in folds:
        assert set(y[tr]) == {0, 1, 2} and len(va) == 3


@pytest.mark.parametrize("task_type", [TaskType.REGRESSION, TaskType.BINARY])
def test_feature_ops_in_experiment_and_final(task_type):
    if task_type == TaskType.REGRESSION:
        spec = _spec(task_type, "rmse", greater=False)
    else:
        spec = _spec(task_type, "f1_macro", greater=True, labels=["neg", "pos"])
    train, test, profile = _data(task_type)
    train.loc[::9, "a"] = np.nan
    ops = ["missing_flags", "freq_encoding", "target_encoding", "pair_ops"]
    hyp = _hyp(feature_ops=ops, model_params={"n_estimators": 50})

    result = run_experiment(spec, profile, hyp, train, RunBudget())

    assert result.status == ExperimentStatus.OK, result.error
    answer = fit_predict_final(spec, profile, hyp, train, test)
    assert len(answer) == 20


def test_target_encoding_no_leak_on_id_like_category():
    """Категория уникальна для каждой строки, target — шум. С утечкой CV было бы «отличным»; без неё — не лучше шума."""
    rng = np.random.default_rng(3)
    n = 400
    train = pd.DataFrame({"id": range(n), "a": rng.normal(size=n), "c": [f"u{i}" for i in range(n)]})
    train["y"] = rng.normal(size=n)
    profile = DataProfile(n_train=n, n_test=0, numeric_features=["a"], categorical_features=["c"])
    spec = _spec(TaskType.REGRESSION, "rmse", greater=False)

    result = run_experiment(
        spec, profile, _hyp(feature_ops=["target_encoding"], model_params={"n_estimators": 50}), train, RunBudget()
    )

    assert result.status == ExperimentStatus.OK, result.error
    assert result.mean_score > 0.9 * train["y"].std()


def test_unknown_feature_op_fails_cleanly():
    spec = _spec(TaskType.REGRESSION, "rmse", greater=False)
    train, _, profile = _data(TaskType.REGRESSION)
    result = run_experiment(spec, profile, _hyp(feature_ops=["magic"]), train, RunBudget())
    assert result.status == ExperimentStatus.FAILED and "magic" in result.error


@pytest.mark.parametrize(
    "bad", [{"feature_ops": 5}, {"feature_ops": [{"op": "pair_ops", "depth": 3}]}, {"n_folds": None}]
)
def test_malformed_params_fail_hypothesis_not_agent(bad):
    """Кривые params от LLM — FAILED, а не исключение наверх: цикл агента его не ловит (раньше падали до try)."""
    spec = _spec(TaskType.REGRESSION, "rmse", greater=False)
    train, _, profile = _data(TaskType.REGRESSION)
    result = run_experiment(spec, profile, _hyp(**bad), train, RunBudget())
    assert result.status == ExperimentStatus.FAILED


def test_feature_ops_with_params_in_experiment():
    spec = _spec(TaskType.REGRESSION, "rmse", greater=False)
    train, test, profile = _data(TaskType.REGRESSION)
    ops = [{"op": "target_encoding", "smoothing": 5}, {"op": "pair_ops", "top_k": 2}]
    hyp = _hyp(feature_ops=ops, model_params={"n_estimators": 50})
    result = run_experiment(spec, profile, hyp, train, RunBudget())
    assert result.status == ExperimentStatus.OK, result.error
    assert len(fit_predict_final(spec, profile, hyp, train, test)) == 20


def _three_classes() -> tuple[pd.DataFrame, pd.DataFrame, DataProfile]:
    rng = np.random.default_rng(7)
    n = 300
    train = pd.DataFrame({"id": range(n), "a": rng.normal(size=n), "b": rng.normal(size=n)})
    train["y"] = np.where(train["a"] > 0.5, "x", np.where(train["b"] > 0, "y", "z"))
    profile = DataProfile(n_train=n, n_test=20, numeric_features=["a", "b"], categorical_features=[])
    return train, train.drop(columns="y").head(20), profile


@pytest.mark.parametrize("columns", [["x", "y", "z"], ["z", "x", "y"], ["class_1", "class_2", "class_3"]])
def test_per_class_probability_answer(columns):
    """#117: шаблон «колонка на класс» — вероятности по колонкам шаблона (по имени, иначе по порядку)."""
    train, test, profile = _three_classes()
    spec = _spec(TaskType.MULTICLASS, "logloss", greater=False, labels=["x", "y", "z"])
    spec = spec.model_copy(update={"submission_columns": ["id", *columns]})
    hyp = _hyp(model_params={"n_estimators": 30})

    assert run_experiment(spec, profile, hyp, train, RunBudget()).status == ExperimentStatus.OK
    answer = fit_predict_final(spec, profile, hyp, train, test)

    assert list(answer.columns) == ["id", *columns]
    proba = answer[columns].to_numpy()
    assert ((proba > 0) & (proba < 1)).all()
    assert np.allclose(proba.sum(axis=1), 1.0)
    if columns[0] == "z":  # по имени, а не по порядку: колонка «z» — вероятность класса z
        plain = fit_predict_final(
            spec.model_copy(update={"submission_columns": ["id", "x", "y", "z"]}), profile, hyp, train, test
        )
        assert np.allclose(answer["z"], plain["z"])


def test_per_class_column_count_mismatch_is_error():
    train, test, profile = _three_classes()
    spec = _spec(TaskType.MULTICLASS, "logloss", greater=False, labels=["x", "y", "z"])
    spec = spec.model_copy(update={"submission_columns": ["id", "x", "y"]})
    hyp = _hyp(model_params={"n_estimators": 30})
    run_experiment(spec, profile, hyp, train, RunBudget())
    with pytest.raises(ValueError, match="колонок ответа 2"):
        fit_predict_final(spec, profile, hyp, train, test)
