"""AnswerKeeper из solution.py: submission.csv восстанавливается, что бы ни случилось с файлом."""

import pandas as pd
import pytest
from sklearn.datasets import load_breast_cancer

from agent.task_parser import parse_task
from evaluation.submission import validate_submission
from solution import AnswerKeeper, trivial_answer


@pytest.fixture
def task(tmp_path):
    data = load_breast_cancer(as_frame=True).frame.rename(columns=lambda c: c.replace(" ", "_"))
    data.insert(0, "row_id", range(len(data)))
    data["diagnosis"] = data.pop("target").map({0: "malignant", 1: "benign"})
    train, test = data.iloc[:300], data.iloc[300:350].drop(columns="diagnosis")
    train.to_csv(tmp_path / "train.csv", index=False)
    test.to_csv(tmp_path / "test.csv", index=False)
    test[["row_id"]].assign(diagnosis=0.5).to_csv(tmp_path / "sample_submission.csv", index=False)
    (tmp_path / "task.md").write_text(
        "# Diagnosis\n\nPredict `diagnosis`. Files: `train.csv`, `test.csv`.\n\nМетрика — ROC-AUC.\n", encoding="utf-8"
    )
    return tmp_path / "task.md", train, test


def _ready_keeper(task):
    task_md, train, test = task
    keeper = AnswerKeeper(task_md)
    keeper.spec = parse_task(task_md)
    keeper.test = test
    return keeper, trivial_answer(keeper.spec, train, test)


def test_write_valid_answer_becomes_last_good(task):
    keeper, answer = _ready_keeper(task)
    assert keeper.write(answer) == []
    assert keeper.last_good is answer
    assert validate_submission(keeper.path, keeper.spec, keeper.test).ok


def test_invalid_answer_is_rolled_back_to_last_good(task):
    keeper, answer = _ready_keeper(task)
    keeper.write(answer)
    broken = answer.assign(diagnosis=float("nan"))
    assert keeper.write(broken)  # проблемы возвращены
    assert keeper.last_good is answer
    assert validate_submission(keeper.path, keeper.spec, keeper.test).ok


def test_finalize_restores_corrupted_file_and_removes_tmp(task):
    keeper, answer = _ready_keeper(task)
    keeper.write(answer)
    keeper.path.write_text("row_id,diagnosis\n1,", encoding="utf-8")  # оборванный файл
    keeper.path.with_suffix(".tmp.csv").write_text("мусор", encoding="utf-8")
    keeper.finalize("тест")
    assert validate_submission(keeper.path, keeper.spec, keeper.test).ok
    assert not keeper.path.with_suffix(".tmp.csv").exists()


def test_finalize_without_spec_copies_sample_submission(task):
    task_md, _, _ = task
    keeper = AnswerKeeper(task_md)  # даже задачу разобрать не успели
    keeper.finalize("тест")
    assert pd.read_csv(keeper.path).equals(pd.read_csv(task_md.parent / "sample_submission.csv"))


def test_finalize_keeps_existing_valid_answer(task):
    keeper, answer = _ready_keeper(task)
    keeper.write(answer)
    before = keeper.path.read_bytes()
    keeper.finalize("тест")
    assert keeper.path.read_bytes() == before
