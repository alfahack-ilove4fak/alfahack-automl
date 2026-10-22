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


@pytest.mark.parametrize("model", ["lightgbm", *MODELS])
def test_optuna_tunes_each_model(model):
    package = MODELS.get(model, (None,))[0]
    if package:
        pytest.importorskip(package)
    spec = _spec(TaskType.REGRESSION, "rmse", greater=False)
    train, test, profile = _data(TaskType.REGRESSION)
    small = {"lightgbm": {"n_estimators": 30}, "catboost": {"iterations": 30}, "xgboost": {"n_estimators": 30}}
    hyp = _hyp(model=model, model_params=small.get(model, {}), optuna={"trials": 3, "timeout": 60, "folds": 2})

    result = run_experiment(spec, profile, hyp, train, RunBudget())

    assert result.status == ExperimentStatus.OK, result.error
    assert hyp.params["_optuna_best"]  # что-то подобрано
    assert hyp.params["_optuna_best"].items() <= hyp.params["model_params"].items()
    assert len(fit_predict_final(spec, profile, hyp, train, test)) == 20


def test_optuna_timeout_is_hard(monkeypatch):
    import time

    import ml.experiment

    real_fit = ml.experiment._fit_model

    def slow_fit(*args, **kwargs):
        time.sleep(0.4)
        return real_fit(*args, **kwargs)

    monkeypatch.setattr(ml.experiment, "_fit_model", slow_fit)
    spec = _spec(TaskType.REGRESSION, "rmse", greater=False)
    train, _, profile = _data(TaskType.REGRESSION)
    hyp = _hyp(model_params={"n_estimators": 20}, optuna={"trials": 100, "timeout": 1, "folds": 3})

    started = time.time()
    from ml.tuning import tune

    x, _ = ml.experiment.make_matrix(profile, train, train.head(0))
    y = train["y"].to_numpy()
    tune(spec, hyp, x, y, y, [], False, RunBudget())

    assert time.time() - started < 1 + 0.4 * 2 + 1  # таймаут + не больше одного фолда сверху + запас


def test_optuna_respects_agent_budget():
    from ml.tuning import tune

    class AlmostOver(RunBudget):
        def seconds_left(self) -> float:
            return 0.0

    spec = _spec(TaskType.REGRESSION, "rmse", greater=False)
    train, _, profile = _data(TaskType.REGRESSION)
    import ml.experiment

    x, _ = ml.experiment.make_matrix(profile, train, train.head(0))
    y = train["y"].to_numpy()
    assert tune(spec, _hyp(optuna={"trials": 10, "timeout": 600}), x, y, y, [], False, AlmostOver()) == {}


def test_optuna_not_repeated_for_same_hypothesis(monkeypatch):
    import ml.tuning

    calls = []
    monkeypatch.setattr(ml.tuning, "tune", lambda *a, **k: calls.append(1) or {"num_leaves": 20})
    spec = _spec(TaskType.REGRESSION, "rmse", greater=False)
    train, _, profile = _data(TaskType.REGRESSION)
    hyp = _hyp(model_params={"n_estimators": 20}, optuna={"trials": 2})

    run_experiment(spec, profile, hyp, train, RunBudget())
    run_experiment(spec, profile, hyp, train, RunBudget())

    assert calls == [1] and hyp.params["model_params"]["num_leaves"] == 20


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


def test_proba_weights_in_experiment_and_final():
    rng = np.random.default_rng(5)
    n = 600
    train = pd.DataFrame({"id": range(n), "a": rng.normal(size=n), "b": rng.normal(size=n)})
    train["y"] = np.where(train["a"] > 1.4, "rare", np.where(train["b"] > 0, "up", "down"))
    test = train.drop(columns="y").head(30)
    profile = DataProfile(n_train=n, n_test=30, numeric_features=["a", "b"], categorical_features=[])
    spec = _spec(TaskType.MULTICLASS, "f1_macro", greater=True, labels=["down", "rare", "up"])
    hyp = _hyp(proba_weights=True, model_params={"n_estimators": 50})

    result = run_experiment(spec, profile, hyp, train, RunBudget())

    assert result.status == ExperimentStatus.OK, result.error
    assert len(hyp.params["_proba_weights"]) == 3
    answer = fit_predict_final(spec, profile, hyp, train, test)
    assert set(answer["y"]) <= {"down", "rare", "up"}


def test_proba_weights_ignored_where_not_applicable():
    spec = _spec(TaskType.REGRESSION, "rmse", greater=False)
    train, _, profile = _data(TaskType.REGRESSION)
    hyp = _hyp(proba_weights=True, model_params={"n_estimators": 20})
    assert run_experiment(spec, profile, hyp, train, RunBudget()).status == ExperimentStatus.OK
    assert "_proba_weights" not in hyp.params

    spec = _spec(TaskType.BINARY, "roc_auc", greater=True, labels=["neg", "pos"])
    train, _, profile = _data(TaskType.BINARY)
    hyp = _hyp(proba_weights=True, model_params={"n_estimators": 20})
    assert run_experiment(spec, profile, hyp, train, RunBudget()).status == ExperimentStatus.OK
    assert "_proba_weights" not in hyp.params


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


def test_time_validation_in_experiment_and_result_label():
    spec = _spec(TaskType.REGRESSION, "rmse", greater=False)
    train, _, profile = _data(TaskType.REGRESSION)
    train["year"] = np.repeat([1, 2, 3, 4, 5, 6], len(train) // 6)
    profile = profile.model_copy(update={"numeric_features": [*profile.numeric_features, "year"]})
    hyp = _hyp(validation="time:year", model_params={"n_estimators": 30})

    result = run_experiment(spec, profile, hyp, train, RunBudget())

    assert result.status == ExperimentStatus.OK, result.error
    assert result.validation == "time:year-3"  # n_folds=3 в _hyp
    assert len(result.fold_scores) == 3


def test_time_validation_class_appears_only_late():
    """Ранний период не знает поздний класс — вероятности дополняются нулём, эксперимент не падает."""
    spec, train, profile = _multiclass_with_rare_class()
    train["period"] = np.arange(len(train)) // 50
    train.loc[train["y"] == "rare", "y"] = "up"  # в ранних периодах класса rare нет совсем
    late = train["period"] >= 3
    train.loc[late, "y"] = np.where(train.loc[late, "a"] > 1.0, "rare", "up")
    assert (train.loc[train["period"] < 3, "y"] != "rare").all()
    profile = profile.model_copy(update={"numeric_features": [*profile.numeric_features, "period"]})
    result = run_experiment(
        spec, profile, _hyp(validation="time:period", model_params={"n_estimators": 20}), train, RunBudget()
    )
    assert result.status == ExperimentStatus.OK, result.error


def test_default_validation_label_and_bad_scheme():
    spec = _spec(TaskType.REGRESSION, "rmse", greater=False)
    train, _, profile = _data(TaskType.REGRESSION)
    assert (
        run_experiment(spec, profile, _hyp(model_params={"n_estimators": 20}), train, RunBudget()).validation
        == "kfold-3"
    )
    bad = run_experiment(spec, profile, _hyp(validation="magic"), train, RunBudget())
    assert bad.status == ExperimentStatus.FAILED and "magic" in bad.error


def test_fold_proba_has_all_classes_when_train_part_misses_one():
    """Обучение без класса 1: вероятности всё равно по 3 классам, у класса 1 — нули (иначе argmax сдвинет номера)."""
    from ml.experiment import _fold_score
    from ml.profiling import make_matrix

    spec = _spec(TaskType.MULTICLASS, "f1_macro", greater=True, labels=["a", "b", "c"])
    rng = np.random.default_rng(3)
    train = pd.DataFrame({"x": rng.normal(size=120)})
    y = np.where(train["x"] > 0, 2, 0)
    y[100:110] = 1  # класс 1 только в валидации
    profile = DataProfile(n_train=120, n_test=0, numeric_features=["x"], categorical_features=[])
    x, _ = make_matrix(profile, train, train.head(0))
    tr, va = np.arange(100), np.arange(100, 120)

    _, _, proba = _fold_score(spec, _hyp(model_params={"n_estimators": 20}), x, y, y, tr, va, None, False)

    assert proba.shape == (20, 3)
    assert (proba[:, 1] == 0).all()
    assert np.allclose(proba.sum(axis=1), 1.0)


def test_optuna_tunes_on_the_same_validation_scheme(monkeypatch):
    """Ревью #115: подбор на KFold оптимизирует утечку — фолды подбора должны идти по схеме гипотезы (здесь время)."""
    import ml.tuning

    seen = {}

    def fake_tune(spec, hypothesis, x, y, y_fit, ops, log_target, budget, split=None):
        seen["folds"] = split(np.arange(len(y)), 3)
        return {}

    monkeypatch.setattr(ml.tuning, "tune", fake_tune)
    spec = _spec(TaskType.REGRESSION, "rmse", greater=False)
    train, _, profile = _data(TaskType.REGRESSION)
    train["year"] = np.repeat([1, 2, 3, 4, 5, 6], len(train) // 6)
    profile = profile.model_copy(update={"numeric_features": [*profile.numeric_features, "year"]})
    hyp = _hyp(validation="time:year", optuna={"trials": 2}, model_params={"n_estimators": 20})

    assert run_experiment(spec, profile, hyp, train, RunBudget()).status == ExperimentStatus.OK
    for tr, va in seen["folds"]:
        assert train["year"].iloc[tr].max() < train["year"].iloc[va].min()


def test_optuna_hidden_from_agent_until_honest_validation():
    """Ревью #115: подбор на KFold честно хуже бейзлайна — LLM и playbook не должны видеть ключ optuna (до #29)."""
    import re
    import textwrap

    import ml.experiment

    section = ml.experiment.__doc__.split("Поддерживаемые params", 1)[1].split("\n", 1)[-1]
    keys = set(re.findall(r"^(\w+)\s+—", textwrap.dedent(section), re.MULTILINE))
    assert "optuna" not in keys


def test_unseen_category_becomes_missing_but_catboost_keeps_it():
    """#145: категория без строк в обучающей части → пропуск; известные значения и пропуски не трогаем."""
    from ml.experiment import _hide_unseen

    cats = ["A", "B", "C"]
    x_fit = pd.DataFrame({"c": pd.Categorical(["A", "B", "A"], categories=cats), "n": [1.0, 2.0, 3.0]})
    x_new = pd.DataFrame({"c": pd.Categorical(["A", "C", None, "B"], categories=cats), "n": [1.0, 2.0, 3.0, 4.0]})

    hidden = _hide_unseen(_hyp(model="xgboost"), x_fit, x_new)

    assert hidden["c"].tolist()[0] == "A" and hidden["c"].tolist()[3] == "B"
    assert hidden["c"].isna().tolist() == [False, True, True, False]
    assert list(hidden["c"].cat.categories) == cats  # словарь общий, коды известных категорий не сдвигаются
    assert x_new["c"].tolist()[1] == "C"  # исходная таблица не изменена
    assert _hide_unseen(_hyp(model="catboost"), x_fit, x_new) is x_new


def test_unseen_category_reaches_model_as_missing(monkeypatch):
    """#145: и в CV (валидация фолда), и в финальном обучении (test) модель видит новую категорию как пропуск."""
    import ml.experiment

    seen = []
    real_fit, real_predict = ml.experiment._fit_model, ml.experiment._predict

    def spy_fit(spec, hypothesis, x, y, eval_set=None, n_estimators=None):
        if eval_set is not None:
            seen.append(("cv", x["c"], eval_set[0]["c"]))
        return real_fit(spec, hypothesis, x, y, eval_set, n_estimators)

    def spy_predict(spec, model, x, log_target):
        seen.append(("final", None, x["c"]))
        return real_predict(spec, model, x, log_target)

    monkeypatch.setattr(ml.experiment, "_fit_model", spy_fit)
    monkeypatch.setattr(ml.experiment, "_predict", spy_predict)
    spec = _spec(TaskType.REGRESSION, "rmse", greater=False)
    train, _, profile = _data(TaskType.REGRESSION)
    train.loc[train.index[-1], "c"] = "solo"  # одна строка: в своём фолде валидации, в его обучении — нет
    test = train.drop(columns="y").head(3).copy()
    test["c"] = ["new", "x", "solo"]  # "new" нет в train
    hyp = _hyp(model_params={"n_estimators": 20})

    assert run_experiment(spec, profile, hyp, train, RunBudget()).status == ExperimentStatus.OK
    fit_predict_final(spec, profile, hyp, train, test)

    cv = [(fit_c, valid_c) for kind, fit_c, valid_c in seen if kind == "cv"]
    assert cv and all(set(valid_c.dropna()) <= set(fit_c.dropna()) for fit_c, valid_c in cv)
    assert sum(int(valid_c.isna().sum()) for _, valid_c in cv) == 1  # "solo" в своём фолде — пропуск
    final = [c for kind, _, c in seen if kind == "final"][-1]  # последний _predict — test в fit_predict_final
    assert final.isna().tolist() == [True, False, False]


@pytest.mark.parametrize(
    ("task_type", "metric", "params", "expected"),
    [
        (TaskType.REGRESSION, "rmsle", {}, True),  # метрика на лог-шкале — учимся сразу под неё (#206)
        (TaskType.REGRESSION, "msle", {}, True),
        (TaskType.REGRESSION, "rmsle", {"log_target": False}, False),  # явный выбор гипотезы важнее умолчания
        (TaskType.REGRESSION, "rmse", {}, False),
        (TaskType.REGRESSION, "rmse", {"log_target": True}, True),
        (TaskType.BINARY, "rmsle", {"log_target": True}, False),  # на классификации log_target не действует
    ],
)
def test_log_target_default_follows_metric(task_type, metric, params, expected):
    from ml.experiment import _use_log_target

    assert _use_log_target(_spec(task_type, metric, greater=False), params) is expected


def test_rmsle_default_log_target_gives_non_negative_answer():
    # без log_target прогноз регрессии бывает < 0 (на practice/sales — до −0.5), а RMSLE ждёт y ≥ 0
    spec = _spec(TaskType.REGRESSION, "rmsle", greater=False)
    train, test, profile = _data(TaskType.REGRESSION)
    train["y"] = np.exp(train["y"]) * 0.1  # положительный скошенный target, часть значений около нуля
    hyp = _hyp(model_params={"n_estimators": 50})

    result = run_experiment(spec, profile, hyp, train, RunBudget())

    assert result.status == ExperimentStatus.OK, result.error
    answer = fit_predict_final(spec, profile, hyp, train, test)
    assert (answer["y"] >= 0).all()


def test_ensemble_averages_members_and_keeps_their_tree_counts():
    """params.ensemble: прогноз — среднее участников, число деревьев из CV — своё у каждого."""
    from ml.experiment import _fit_ensemble

    spec = _spec(TaskType.REGRESSION, "rmse", greater=False)
    train, test, profile = _data(TaskType.REGRESSION)
    hyp = _hyp(ensemble=["lightgbm", {"model": "linear"}])

    result = run_experiment(spec, profile, hyp, train, RunBudget())
    assert result.status == ExperimentStatus.OK, result.error
    iters = hyp.params["_best_iteration"]
    assert isinstance(iters, list) and len(iters) == 2 and iters[1] == 0  # у линейной модели деревьев нет
    answer = fit_predict_final(spec, profile, hyp, train, test)
    assert answer["y"].notna().all()

    from ml.profiling import make_matrix

    x, x_test = make_matrix(profile, train, test)
    model, _ = _fit_ensemble(spec, hyp, x, train["y"].to_numpy(), n_estimators=[30, 0])
    members = [m.predict(x_test) for m in model.models]
    np.testing.assert_allclose(model.predict(x_test), np.mean(members, axis=0))


def test_ensemble_classification_with_catboost_and_unseen_category():
    """CatBoost в ансамбле получает новую категорию как отдельное значение, а не NaN (иначе CatBoost падает)."""
    pytest.importorskip("catboost")
    pytest.importorskip("xgboost")
    spec = _spec(TaskType.BINARY, "roc_auc", greater=True, labels=["neg", "pos"])
    train, test, profile = _data(TaskType.BINARY)
    test = test.copy()
    test.loc[test.index[:3], "c"] = "new"  # категории нет в train
    hyp = _hyp(ensemble=["lightgbm", "xgboost", {"model": "catboost", "model_params": {"iterations": 30}}])

    result = run_experiment(spec, profile, hyp, train, RunBudget())
    assert result.status == ExperimentStatus.OK, result.error
    answer = fit_predict_final(spec, profile, hyp, train, test)
    assert len(answer) == len(test)


@pytest.mark.parametrize(
    "params, message",
    [
        ({"ensemble": ["lightgbm"]}, "хотя бы из 2"),
        ({"ensemble": ["lightgbm", "magic"]}, "magic"),
        ({"ensemble": ["lightgbm", "linear"], "optuna": {"trials": 2}}, "optuna"),
    ],
)
def test_bad_ensemble_is_failed_not_exception(params, message):
    spec = _spec(TaskType.REGRESSION, "rmse", greater=False)
    train, _, profile = _data(TaskType.REGRESSION)
    result = run_experiment(spec, profile, _hyp(**params), train, RunBudget())
    assert result.status == ExperimentStatus.FAILED and message in result.error
