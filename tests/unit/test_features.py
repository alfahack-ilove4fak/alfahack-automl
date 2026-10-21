"""Преобразования признаков (ml/features.py): поведение и отсутствие утечки."""

import numpy as np
import pandas as pd
import pytest

from ml.features import (
    FEATURE_OPS,
    MAX_FLAGS,
    TE_MAX_CLASSES,
    FreqEncoding,
    MissingFlags,
    PairOps,
    TargetEncoding,
    apply_feature_ops,
    parse_feature_ops,
)


def _frame(n: int = 200, seed: int = 0) -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = pd.DataFrame(
        {
            "a": rng.normal(size=n).astype(np.float32),
            "b": rng.normal(size=n).astype(np.float32),
            "noise": rng.normal(size=n).astype(np.float32),
            "c": pd.Categorical(rng.choice(["x", "y", "z"], size=n), categories=["x", "y", "z", "new"]),
        }
    )
    x.loc[::10, "a"] = np.nan
    y = (x["b"] * 3 + (x["c"] == "x") * 2).to_numpy(dtype=float)
    return x, y


def test_missing_flags_only_for_columns_with_gaps_in_train():
    x, y = _frame()
    out = MissingFlags().fit(x, y, classification=False).transform(x)
    assert list(out.columns) == ["a__isna", "n_missing"]
    assert out["a__isna"].sum() == x["a"].isna().sum()


def test_freq_encoding_unseen_category_is_zero():
    x, y = _frame()
    enc = FreqEncoding().fit(x, y, classification=False)
    test = x.head(2).copy()
    test["c"] = pd.Categorical(["new", "x"], categories=x["c"].cat.categories)
    out = enc.transform(test)
    assert out["c__freq"].iloc[0] == 0.0
    assert out["c__freq"].iloc[1] == pytest.approx((x["c"] == "x").mean())


def test_target_encoding_train_rows_do_not_see_own_target():
    """Out-of-fold: значение строки train не зависит от её собственного target."""
    x, y = _frame()
    before = TargetEncoding().fit(x, y, classification=False).transform(x)
    y2 = y.copy()
    y2[5] += 1000.0
    after = TargetEncoding().fit(x, y2, classification=False).transform(x)
    assert before.iloc[5, 0] == after.iloc[5, 0]
    assert not np.allclose(before.to_numpy(), after.to_numpy())  # на других строках изменение видно


def test_target_encoding_other_frame_uses_full_table_even_with_same_index():
    x, y = _frame()
    enc = TargetEncoding().fit(x, y, classification=False)
    same_index_copy = x.copy()  # как test с индексом 0..n-1 той же длины
    assert not enc.transform(same_index_copy).equals(enc.transform(x))
    smoothed_x = enc.tables["c"][0].loc["x", "mean"]
    assert enc.transform(same_index_copy).loc[x["c"] == "x", "c__te_mean"].nunique() == 1
    assert enc.transform(same_index_copy).loc[x["c"] == "x", "c__te_mean"].iloc[0] == pytest.approx(smoothed_x)


def test_target_encoding_multiclass_one_column_per_class():
    x, _ = _frame()
    y = np.arange(len(x)) % 3
    out = TargetEncoding().fit(x, y, classification=True).transform(x)
    assert list(out.columns) == ["c__te_p0", "c__te_p1", "c__te_p2"]


def test_pair_ops_pick_informative_features_and_no_inf():
    x, y = _frame()
    x.loc[0, "b"] = 0.0
    out = PairOps().fit(x, y, classification=False).transform(x)
    assert "b__minus__a" in out.columns or "a__minus__b" in out.columns
    assert not np.isinf(out.to_numpy()).any()


def test_apply_feature_ops_unknown_op():
    x, y = _frame()
    with pytest.raises(ValueError, match="неизвестные feature_ops"):
        apply_feature_ops(["magic"], x, y, False)


def test_apply_feature_ops_fit_only_on_train_part():
    """Валидация не влияет на обучение операций: её target в fit не передаётся, а её строки — только в transform."""
    x, y = _frame()
    tr, va = x.iloc[:150], x.iloc[150:]
    ops = list(FEATURE_OPS)
    _, va_out = apply_feature_ops(ops, tr, y[:150], False, va)
    va_changed = va.copy()
    va_changed["c"] = pd.Categorical(["x"] * len(va), categories=x["c"].cat.categories)
    _, va_out2 = apply_feature_ops(ops, tr, y[:150], False, va_changed)
    # Кодировки «x» берутся из train-части, а не пересчитываются по валидации
    te_x = va_out.loc[va["c"] == "x", "c__te_mean"].iloc[0]
    assert (va_out2["c__te_mean"] == te_x).all()


# ── Формат params.feature_ops


def test_parse_accepts_name_list_and_dicts_with_params():
    assert parse_feature_ops(None) == []
    assert parse_feature_ops("pair_ops") == [("pair_ops", {})]
    assert parse_feature_ops([" Target_Encoding ", {"op": "pair_ops", "top_k": 3}]) == [
        ("target_encoding", {}),
        ("pair_ops", {"top_k": 3}),
    ]


@pytest.mark.parametrize(
    ("raw", "error"),
    [
        (5, ValueError),  # не список — раньше list(5) ронял агента TypeError до try
        ([{"top_k": 3}], ValueError),  # словарь без op
        ([{"op": "pair_ops", "depth": 3}], TypeError),  # чужой параметр
        ([{"op": "target_encoding", "folds": 1}], ValueError),  # плохое значение
        ([{"op": "pair_ops", "top_k": 1}], ValueError),
    ],
)
def test_parse_rejects_bad_input_before_training(raw, error):
    with pytest.raises(error):
        parse_feature_ops(raw)


def test_single_name_is_not_split_into_letters():
    x, y = _frame()
    (out,) = apply_feature_ops("freq_encoding", x, y, False)
    assert "c__freq" in out.columns


# ── Улучшения операций


def test_missing_flags_see_categorical_gaps_and_count_per_row():
    """В категориях make_matrix превращает пропуск в строку «nan» — это тоже пропуск."""
    x, y = _frame()
    x["c"] = pd.Categorical(np.where(np.arange(len(x)) % 4 == 0, "nan", x["c"].astype(str)))
    out = MissingFlags().fit(x, y, classification=False).transform(x)
    assert out["c__isna"].sum() == len(range(0, len(x), 4))
    assert (out["n_missing"] == out["a__isna"] + out["c__isna"]).all()


def test_missing_flags_capped_on_wide_data():
    rng = np.random.default_rng(1)
    x = pd.DataFrame({f"f{i}": np.where(rng.random(50) < 0.1 + i / 1000, np.nan, 1.0) for i in range(MAX_FLAGS + 20)})
    out = MissingFlags().fit(x, np.zeros(50), classification=False).transform(x)
    assert len(out.columns) == MAX_FLAGS + 1  # флаги + n_missing


def test_target_encoding_smoothing_param():
    x, y = _frame()
    raw = TargetEncoding(smoothing=0).fit(x, y, classification=False)
    assert raw.tables["c"][0].loc["x", "mean"] == pytest.approx(y[(x["c"] == "x").to_numpy()].mean())
    smoothed = TargetEncoding(smoothing=1000).fit(x, y, classification=False)
    assert abs(smoothed.tables["c"][0].loc["x", "mean"] - y.mean()) < abs(
        raw.tables["c"][0].loc["x", "mean"] - y.mean()
    )


def test_target_encoding_multiclass_keeps_most_frequent_classes():
    x, _ = _frame(n=600)
    # Классы 0…4 — редкие (по 2 объекта), 5…29 — частые: колонки должны достаться частым, а не первым по номеру
    y = np.concatenate([np.repeat(np.arange(5), 2), np.arange(5, 30).repeat(24)[: len(x) - 10]])
    out = TargetEncoding().fit(x, y, classification=True).transform(x.copy())
    classes = {int(c.rsplit("_p", 1)[1]) for c in out.columns}
    assert len(classes) == TE_MAX_CLASSES and min(classes) >= 5


def test_pair_ops_top_k_param():
    x, y = _frame()
    out = PairOps(top_k=2).fit(x, y, classification=False).transform(x)
    assert len(out.columns) == 2  # одна пара: разность и отношение


def test_new_columns_never_clash_with_existing_names():
    x, y = _frame()
    x["n_missing"] = 1.0  # своя колонка с таким именем в данных
    (out,) = apply_feature_ops(["missing_flags", "freq_encoding", "freq_encoding"], x, y, False)
    assert out.columns.is_unique
    assert {"n_missing_1", "c__freq", "c__freq_1"} <= set(out.columns)
