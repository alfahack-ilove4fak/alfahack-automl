"""Задача без id в test (контракт 1.1.2): шаблон ответа — одна колонка, строки в порядке test.

Раньше `id_column` брался из первой колонки шаблона («price» = target), и ответ получался мусорным (0 баллов).
"""

import numpy as np
import pandas as pd

from agent.hypotheses import baseline_hypothesis
from agent.task_parser import parse_task
from evaluation.submission import validate_submission
from interfaces.schemas import ROW_ID_COLUMN
from ml.experiment import fit_predict_final
from ml.profiling import load_data, profile_data
from solution import trivial_answer


def make_task(folder, sample_columns=("price",), n=200, n_test=50):
    """Таблица без id: x0..x2 и price. sample_columns — колонки шаблона ответа."""
    rng = np.random.default_rng(1)
    x = pd.DataFrame(rng.normal(size=(n + n_test, 3)), columns=["x0", "x1", "x2"])
    y = 10 + 3 * x["x0"] + rng.normal(scale=0.1, size=len(x))
    train = x.iloc[:n].assign(price=y.iloc[:n].to_numpy())
    test = x.iloc[n:].reset_index(drop=True)
    sample = pd.DataFrame({c: (np.zeros(n_test) if c == "price" else np.arange(n_test) + 1000) for c in sample_columns})
    train.to_csv(folder / "train.csv", index=False)
    test.to_csv(folder / "test.csv", index=False)
    sample.to_csv(folder / "sample_submission.csv", index=False)
    (folder / "task.md").write_text("Предскажи price. Метрика: RMSE.", encoding="utf-8")
    return folder / "task.md"


def test_template_without_id_gives_row_number_id(tmp_path):
    spec = parse_task(make_task(tmp_path), use_analyst=False)
    assert spec.id_column == ROW_ID_COLUMN
    assert spec.target == "price"
    assert spec.submission_columns == ["price"]


def test_load_data_adds_row_numbers_and_features_exclude_them(tmp_path):
    spec = parse_task(make_task(tmp_path), use_analyst=False)
    train, test = load_data(spec)
    assert list(test[ROW_ID_COLUMN]) == [str(i) for i in range(len(test))]
    assert ROW_ID_COLUMN not in train.columns
    profile = profile_data(spec, train, test)
    assert "price" not in profile.numeric_features and ROW_ID_COLUMN not in profile.numeric_features


def test_answers_have_only_template_columns_and_pass_validation(tmp_path):
    spec = parse_task(make_task(tmp_path), use_analyst=False)
    train, test = load_data(spec)
    path = tmp_path / "submission.csv"
    for answer in (
        trivial_answer(spec, train, test),
        fit_predict_final(spec, profile_data(spec, train, test), baseline_hypothesis(), train, test),
    ):
        answer[spec.submission_columns].to_csv(path, index=False)
        report = validate_submission(path, spec, pd.read_csv(spec.test_path))  # test как есть, без колонки-номера
        assert report.ok, report.problems
        assert list(pd.read_csv(path).columns) == ["price"]


def test_id_in_template_but_not_in_test_is_taken_from_sample(tmp_path):
    spec = parse_task(make_task(tmp_path, sample_columns=("id", "price")), use_analyst=False)
    assert spec.id_column == "id" and spec.submission_columns == ["id", "price"]
    train, test = load_data(spec)
    assert list(test["id"]) == [str(1000 + i) for i in range(len(test))]
    answer = trivial_answer(spec, train, test)
    assert list(answer.columns) == ["id", "price"] and len(answer) == len(test)


def test_unknown_first_name_in_template_is_id_taken_from_sample(tmp_path):
    """index,price / row,price: первая колонка шаблона — не таргет из train, значит это id; значения — из образца."""
    for name in ("index", "row"):
        folder = tmp_path / name
        folder.mkdir()
        spec = parse_task(make_task(folder, sample_columns=(name, "price")), use_analyst=False)
        assert spec.id_column == name and spec.submission_columns == [name, "price"]
        train, test = load_data(spec)
        answer = trivial_answer(spec, train, test)
        assert list(answer.columns) == [name, "price"] and len(answer) == len(test)


def make_task_with_train_id(folder, id_name, sample_columns, targets=("price",), n=60, n_test=20):
    """id_name лежит в train (номера строк), в test его нет; sample_columns — шаблон ответа."""
    rng = np.random.default_rng(2)
    train = pd.DataFrame({id_name: np.arange(n), "a": rng.normal(size=n)})
    for t in targets:
        train[t] = train["a"] * 3 + rng.normal(scale=0.1, size=n)
    train.to_csv(folder / "train.csv", index=False)
    pd.DataFrame({"a": rng.normal(size=n_test)}).to_csv(folder / "test.csv", index=False)
    sample = pd.DataFrame({c: (np.arange(n, n + n_test) if c == id_name else np.zeros(n_test)) for c in sample_columns})
    sample.to_csv(folder / "sample_submission.csv", index=False)
    (folder / "task.md").write_text("Предскажи price. Метрика RMSE.", encoding="utf-8")
    return folder / "task.md"


def test_id_in_train_and_template_but_not_in_test_stays_id_not_second_target(tmp_path):
    """Ревью artem в #268: id, index, row есть в train и в шаблоне, нет в test — это id, а не таргет «extra»."""
    for name in ("id", "index", "row"):
        folder = tmp_path / name
        folder.mkdir()
        spec = parse_task(make_task_with_train_id(folder, name, (name, "price")), use_analyst=False)
        assert spec.id_column == name and spec.target == "price"
        assert spec.extra_targets == [] and spec.submission_columns == [name, "price"]
        train, test = load_data(spec)
        assert list(test[name]) == [str(60 + i) for i in range(len(test))]  # id — из образца


def test_multitarget_template_without_id_still_uses_row_numbers(tmp_path):
    """Шаблон из двух таргетов (без id): первая колонка — вещественный таргет из train, а не id."""
    spec = parse_task(
        make_task_with_train_id(tmp_path, "id", ("price", "price2"), targets=("price", "price2")), use_analyst=False
    )
    assert spec.id_column == ROW_ID_COLUMN
    assert spec.submission_columns == ["price", "price2"]
