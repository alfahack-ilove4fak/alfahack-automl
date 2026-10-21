"""ml/features.py: операции params.feature_ops — каждая учится только на train-части, без утечки."""

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
from ml.features import FEATURE_OPS, FeatureOps, parse_feature_ops
from ml.profiling import make_matrix

ALL_OPS = sorted(FEATURE_OPS)


def _spec(task_type: TaskType, metric: str = "rmse", greater: bool = False, labels=None) -> TaskSpec:
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


def _matrix(df: pd.DataFrame, numeric: list[str], categorical: list[str]) -> pd.DataFrame:
    profile = DataProfile(n_train=len(df), n_test=0, numeric_features=numeric, categorical_features=categorical)
    x, _ = make_matrix(profile, df, df.head(0))
    return x


def _toy(task_type: TaskType, n: int = 300) -> tuple[pd.DataFrame, pd.DataFrame, DataProfile]:
    """Числа с пропусками, нулями в знаменателе и категория, от которой зависит target."""
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            "id": np.arange(n),
            "a": rng.normal(size=n),
            "b": np.where(rng.random(n) < 0.2, np.nan, rng.normal(size=n)),
            "z": np.where(rng.random(n) < 0.3, 0.0, rng.normal(size=n)),
            "c": np.where(rng.random(n) < 0.1, None, rng.choice(["x", "y", "w"], size=n)),
        }
    )
    signal = df["a"] + (df["c"] == "x") * 2
    if task_type == TaskType.REGRESSION:
        df["y"] = signal
    elif task_type == TaskType.BINARY:
        df["y"] = np.where(signal > signal.median(), "pos", "neg")
    else:
        df["y"] = pd.cut(signal, 3, labels=["k0", "k1", "k2"]).astype(str)
    test = df.drop(columns="y").head(30)
    profile = DataProfile(n_train=n, n_test=30, numeric_features=["a", "b", "z"], categorical_features=["c"])
    return df, test, profile


CASES = {
    TaskType.REGRESSION: ("rmse", False, None),
    TaskType.BINARY: ("roc_auc", True, ["neg", "pos"]),
    TaskType.MULTICLASS: ("f1_macro", True, ["k0", "k1", "k2"]),
}


# ── Разбор params.feature_ops


def test_parse_accepts_name_list_and_dicts():
    assert parse_feature_ops(None) == []
    assert parse_feature_ops("pair_diffs") == [("pair_diffs", {})]
    assert parse_feature_ops([" Target_Encoding ", {"op": "pair_ratios", "top_k": 3}]) == [
        ("target_encoding", {}),
        ("pair_ratios", {"top_k": 3}),
    ]


@pytest.mark.parametrize("raw", ["polynomial", [{"top_k": 3}], [42]])
def test_parse_rejects_unknown(raw):
    with pytest.raises(ValueError):
        parse_feature_ops(raw)


def test_bad_option_fails_before_training():
    with pytest.raises(TypeError):
        FeatureOps(_spec(TaskType.REGRESSION), [{"op": "pair_diffs", "depth": 3}])


def test_no_ops_returns_matrix_unchanged():
    x = _matrix(_toy(TaskType.REGRESSION)[0], ["a"], ["c"])
    ops = FeatureOps(_spec(TaskType.REGRESSION), [])
    assert ops.fit_transform(x, np.zeros(len(x))) is x
    assert ops.transform(x) is x


# ── Каждая операция: учится только на train-части


def test_missing_indicators_columns_come_from_train_part():
    df = pd.DataFrame({"a": [1.0, np.nan, 3.0, 4.0], "b": [1.0, 2.0, 3.0, np.nan], "c": ["x", None, "y", "x"]})
    x = _matrix(df, ["a", "b"], ["c"])
    ops = FeatureOps(_spec(TaskType.REGRESSION), "missing_indicators")
    train = ops.fit_transform(x.iloc[:3], np.zeros(3))
    # В train-части пропуски есть только в a и c (в c пропуск — строка «nan» после make_matrix)
    assert {"a__isna", "c__isna", "n_missing"} <= set(train.columns)
    assert "b__isna" not in train.columns
    assert train["a__isna"].tolist() == [0, 1, 0]
    assert train["n_missing"].tolist() == [0, 2, 0]
    valid = ops.transform(x.iloc[3:])
    assert list(valid.columns) == list(train.columns)  # пропуск в b на валидации колонку не добавляет


def test_frequency_uses_train_counts_and_unseen_is_zero():
    df = pd.DataFrame({"c": ["x", "x", "x", "y", "y", "w"]})
    x = _matrix(df, [], ["c"])
    ops = FeatureOps(_spec(TaskType.REGRESSION), "frequency_encoding")
    train = ops.fit_transform(x.iloc[:4], np.zeros(4))
    assert train["c__freq"].tolist() == pytest.approx([0.75, 0.75, 0.75, 0.25])
    valid = ops.transform(x.iloc[4:])
    # Частоты — из train-части, а не из валидации («y» там 1 из 2); «w» в train нет → 0
    assert valid["c__freq"].tolist() == pytest.approx([0.25, 0.0])


def test_target_encoding_train_rows_do_not_see_own_target():
    """Классическая утечка: у «id-подобной» категории наивное среднее равно target строки. Вне фолда — prior."""
    rng = np.random.default_rng(1)
    n = 200
    df = pd.DataFrame({"c": [f"u{i}" for i in range(n)]})
    y = rng.normal(size=n)
    x = _matrix(df, [], ["c"])
    ops = FeatureOps(_spec(TaskType.REGRESSION), "target_encoding")
    enc = ops.fit_transform(x, y)["c__te"].to_numpy()
    assert np.corrcoef(enc, y)[0, 1] < 0.2
    assert np.ptp(enc) < 0.5  # все строки получили примерно среднее по остальным строкам


def test_target_encoding_own_target_change_keeps_own_encoding():
    rng = np.random.default_rng(2)
    df = pd.DataFrame({"c": rng.choice(["x", "y", "w"], size=100)})
    x = _matrix(df, [], ["c"])
    y = rng.normal(size=100)
    before = FeatureOps(_spec(TaskType.REGRESSION), "target_encoding").fit_transform(x, y)["c__te"].to_numpy()
    y2 = y.copy()
    y2[0] += 1000
    after = FeatureOps(_spec(TaskType.REGRESSION), "target_encoding").fit_transform(x, y2)["c__te"].to_numpy()
    assert after[0] == pytest.approx(before[0])  # строка 0 кодируется по другим внутренним фолдам


def test_target_encoding_valid_uses_train_means_and_prior_for_unseen():
    df = pd.DataFrame({"c": ["x", "x", "y", "y", "x", "w"]})
    x = _matrix(df, [], ["c"])
    y = np.array([10.0, 10.0, 0.0, 0.0])
    ops = FeatureOps(_spec(TaskType.REGRESSION), [{"op": "target_encoding", "smoothing": 0}])
    ops.fit_transform(x.iloc[:4], y)
    valid = ops.transform(x.iloc[4:])["c__te"].to_numpy()
    assert valid == pytest.approx([10.0, 5.0])  # «x» — среднее из train-части, «w» не было → prior 5


def test_target_encoding_multiclass_column_per_class():
    df = pd.DataFrame({"c": ["x", "y"] * 30})
    x = _matrix(df, [], ["c"])
    y = np.tile([0, 1, 2], 20)
    out = FeatureOps(_spec(TaskType.MULTICLASS), "target_encoding").fit_transform(x, y)
    assert [c for c in out.columns if "__te" in c] == ["c__te_0", "c__te_1", "c__te_2"]
    assert out[["c__te_0", "c__te_1", "c__te_2"]].sum(axis=1).to_numpy() == pytest.approx(1.0, abs=1e-5)


def test_pairs_pick_informative_features_on_train_part():
    rng = np.random.default_rng(3)
    n = 300
    df = pd.DataFrame({f"n{i}": rng.normal(size=n) for i in range(5)})
    y = (df["n3"] + df["n4"]).to_numpy()
    x = _matrix(df, list(df.columns), [])
    ops = FeatureOps(_spec(TaskType.REGRESSION), [{"op": "pair_diffs", "top_k": 2}])
    out = ops.fit_transform(x, y)
    assert [c for c in out.columns if "__minus__" in c] == ["n3__minus__n4"]
    # На валидации пары не пересчитываются, даже если там другая зависимость
    assert list(ops.transform(x.iloc[:10]).columns) == list(out.columns)


def test_ratio_by_zero_is_nan_not_inf():
    df = pd.DataFrame({"a": [1.0, 2.0, 1e30, 4.0], "b": [0.0, 1.0, 1e-20, 2.0]})
    x = _matrix(df, ["a", "b"], [])
    out = FeatureOps(_spec(TaskType.REGRESSION), "pair_ratios").fit_transform(x, np.arange(4.0))
    ratio = out["a__div__b"].to_numpy()
    assert np.isnan(ratio[0]) and np.isnan(ratio[2])  # деление на 0 и переполнение float32
    assert not np.isinf(ratio).any()


def test_new_names_do_not_clash_with_existing():
    df = pd.DataFrame({"a": [1.0, np.nan, 3.0], "n_missing": [1.0, 2.0, 3.0]})
    x = _matrix(df, ["a", "n_missing"], [])
    out = FeatureOps(_spec(TaskType.REGRESSION), "missing_indicators").fit_transform(x, np.zeros(3))
    assert out.columns.is_unique
    assert "n_missing_1" in out.columns


# ── Сквозь run_experiment и fit_predict_final


@pytest.mark.parametrize("task_type", list(CASES))
@pytest.mark.parametrize("model", ["lightgbm", "linear", "hist_gb"])
def test_all_ops_run_through_experiment(task_type, model):
    metric, greater, labels = CASES[task_type]
    spec = _spec(task_type, metric, greater, labels)
    train, test, profile = _toy(task_type)
    fast = {"lightgbm": {"n_estimators": 50}, "hist_gb": {"max_iter": 30}, "linear": {}}
    params = {"model": model, "n_folds": 3, "feature_ops": ALL_OPS, "model_params": fast[model]}
    hyp = Hypothesis(id="feats", kind=HypothesisKind.FEATURES, title="все операции", params=params)

    result = run_experiment(spec, profile, hyp, train, RunBudget())
    assert result.status == ExperimentStatus.OK, result.error
    answer = fit_predict_final(spec, profile, hyp, train, test)
    assert len(answer) == len(test) and answer["y"].notna().all()


def test_unknown_op_fails_hypothesis_instead_of_agent():
    spec = _spec(TaskType.REGRESSION)
    train, _, profile = _toy(TaskType.REGRESSION)
    hyp = Hypothesis(id="bad", kind=HypothesisKind.FEATURES, title="нет такой", params={"feature_ops": ["magic"]})
    result = run_experiment(spec, profile, hyp, train, RunBudget())
    assert result.status == ExperimentStatus.FAILED
    assert "magic" in result.error


def test_target_encoding_helps_on_high_cardinality_category():
    """Смысл операции: категория с сотнями значений и сильным эффектом — TE не хуже голой модели."""
    rng = np.random.default_rng(4)
    n = 2000
    cats = rng.integers(0, 300, size=n)
    effect = rng.normal(size=300) * 3
    df = pd.DataFrame({"id": np.arange(n), "a": rng.normal(size=n), "c": [f"k{c}" for c in cats]})
    df["y"] = effect[cats] + df["a"] + rng.normal(size=n) * 0.5
    profile = DataProfile(n_train=n, n_test=0, numeric_features=["a"], categorical_features=["c"])
    spec = _spec(TaskType.REGRESSION)

    def score(params):
        hyp = Hypothesis(id="h", kind=HypothesisKind.FEATURES, title="t", params={"n_folds": 3, **params})
        result = run_experiment(spec, profile, hyp, df, RunBudget())
        assert result.status == ExperimentStatus.OK, result.error
        return result.mean_score

    # Линейная модель не умеет one-hot для 300 редких значений (min_frequency=10) — TE даёт ей эффект категории
    assert score({"model": "linear", "feature_ops": ["target_encoding"]}) < score({"model": "linear"})
