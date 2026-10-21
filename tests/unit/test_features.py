"""Преобразования признаков (ml/features.py): поведение и отсутствие утечки."""

import numpy as np
import pandas as pd
import pytest

from ml.features import FEATURE_OPS, FreqEncoding, MissingFlags, PairOps, TargetEncoding, apply_feature_ops


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
    assert list(out.columns) == ["a__isna"]
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
