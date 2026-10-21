"""trivial_answer для шаблона «колонка на класс» (#117): шаг 0 не должен падать с KeyError."""

import numpy as np
import pandas as pd
import pytest

from agent.task_parser import parse_task
from evaluation.submission import validate_submission
from solution import trivial_answer


def _task(tmp_path, labels, columns):
    train = pd.DataFrame({"id": range(100), "x": np.arange(100), "label": labels})
    test = pd.DataFrame({"id": range(100, 120), "x": np.arange(20)})
    train.to_csv(tmp_path / "train.csv", index=False)
    test.to_csv(tmp_path / "test.csv", index=False)
    pd.DataFrame({"id": test["id"], **{c: 1 / len(columns) for c in columns}}).to_csv(
        tmp_path / "sample_submission.csv", index=False
    )
    (tmp_path / "task.md").write_text(
        "# Задача\n\nПредсказать `label`. Файлы: `train.csv`, `test.csv`. Метрика — log loss: вероятность каждого класса.\n",
        encoding="utf-8",
    )
    return parse_task(tmp_path / "task.md"), train, test


def test_columns_named_as_classes_get_class_shares(tmp_path):
    labels = ["a"] * 50 + ["b"] * 30 + ["c"] * 20
    spec, train, test = _task(tmp_path, labels, ["a", "b", "c"])
    answer = trivial_answer(spec, train, test)
    assert list(answer.columns) == ["id", "a", "b", "c"]
    assert answer.iloc[0][["a", "b", "c"]].tolist() == pytest.approx([0.5, 0.3, 0.2], abs=1e-4)
    answer.to_csv(tmp_path / "submission.csv", index=False)
    assert validate_submission(tmp_path / "submission.csv", spec, test).ok


def test_rows_sum_to_one_and_stay_in_unit_interval(tmp_path):
    spec, train, test = _task(tmp_path, ["a"] * 99 + ["b"], ["a", "b", "c"])  # класс c в train отсутствует
    probs = trivial_answer(spec, train, test)[["a", "b", "c"]].to_numpy()
    assert np.allclose(probs.sum(axis=1), 1) and (probs > 0).all() and (probs < 1).all()


def test_other_column_names_match_by_order(tmp_path):
    spec, train, test = _task(tmp_path, ["x"] * 80 + ["y"] * 20, ["p0", "p1"] + ["p2"])
    answer = trivial_answer(spec, train, test)
    assert list(answer.columns) == ["id", "p0", "p1", "p2"]
    assert np.allclose(answer[["p0", "p1", "p2"]].sum(axis=1), 1)
