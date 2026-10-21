"""validate_submission и id: test из load_data (id строкой) против числовых id в submission.csv.

Регрессия 21.10: int из CSV ≠ str из load_data → любой ответ агента «невалиден», AnswerKeeper не запоминал
последний хороший ответ и не мог откатиться к нему.
"""

import pandas as pd
import pytest

from agent.task_parser import parse_task
from evaluation.submission import validate_submission
from ml.profiling import load_data


def _task(tmp_path, ids):
    train = pd.DataFrame({"id": ids[:6], "x": range(6), "label": ["a", "b"] * 3})
    test = pd.DataFrame({"id": ids[6:], "x": range(4)})
    train.to_csv(tmp_path / "train.csv", index=False)
    test.to_csv(tmp_path / "test.csv", index=False)
    test[["id"]].assign(label="a").to_csv(tmp_path / "sample_submission.csv", index=False)
    (tmp_path / "task.md").write_text("# Задача\n\nПредсказать `label`. Метрика — accuracy.\n", encoding="utf-8")
    spec = parse_task(tmp_path / "task.md")
    _, test_loaded = load_data(spec)
    return spec, test_loaded


@pytest.mark.parametrize(
    "ids",
    [
        list(range(10)),  # числовые id — как в публичных задачах
        [f"{i:03d}" for i in range(10)],  # строковые с ведущими нулями — ноль не должен потеряться
    ],
)
def test_answer_written_from_loaded_test_is_valid(tmp_path, ids):
    spec, test = _task(tmp_path, ids)
    test[[spec.id_column]].assign(label="b").to_csv(spec.submission_path, index=False)
    report = validate_submission(spec.submission_path, spec, test)
    assert report.ok, report.problems


def test_wrong_order_is_still_caught(tmp_path):
    spec, test = _task(tmp_path, list(range(10)))
    test[[spec.id_column]].iloc[::-1].assign(label="b").to_csv(spec.submission_path, index=False)
    report = validate_submission(spec.submission_path, spec, test)
    assert not report.ok and any("id" in p for p in report.problems)


def test_lost_leading_zero_is_caught(tmp_path):
    spec, test = _task(tmp_path, [f"{i:03d}" for i in range(10)])
    pd.DataFrame({"id": [int(i) for i in test["id"]], "label": "b"}).to_csv(spec.submission_path, index=False)
    assert not validate_submission(spec.submission_path, spec, test).ok  # «007» превратился в «7» — это ошибка
