"""Запасной ответ без разбора задачи (#187): parse_task упал, sample_submission нет — файл всё равно появляется."""

import pandas as pd

import solution


def make_task(tmp_path, text, extra=("y1", "y2"), id_col="id"):
    train = pd.DataFrame({id_col: [1, 2, 3, 4], "x": [1.0, 2.0, 3.0, 4.0]})
    for i, col in enumerate(extra):
        train[col] = [1.0, 2.0, 3.0, 6.0] if i == 0 else ["a", "b", "b", "c"]
    train.to_csv(tmp_path / "train.csv", index=False)
    pd.DataFrame({id_col: [10, 11], "x": [5.0, 6.0]}).to_csv(tmp_path / "test.csv", index=False)
    (tmp_path / "task.md").write_text(text, encoding="utf-8")
    return tmp_path / "task.md"


def test_ambiguous_task_without_sample_still_gets_answer(tmp_path):
    # воспроизведение из #187: две колонки только в train, нет sample и словаря → parse_task падает
    task_md = make_task(tmp_path, "Предскажите `y1` и `y2`.")
    assert solution.main(["solution.py", str(task_md)]) == 0
    sub = pd.read_csv(tmp_path / "submission.csv")
    assert list(sub.columns) == ["id", "y1", "y2"] and sub["id"].tolist() == [10, 11]
    assert sub["y1"].tolist() == [3.0, 3.0] and sub["y2"].tolist() == ["b", "b"]  # среднее и самое частое


def test_unmentioned_columns_take_last(tmp_path):
    task_md = make_task(tmp_path, "Предскажите значение.")
    assert list(solution.blind_answer(task_md).columns) == ["id", "y2"]


def test_short_name_matched_as_word_not_substring(tmp_path):
    # «y» не должна находиться внутри слов («yes», «yearly»)
    task_md = make_task(tmp_path, "Yearly report, yes. Целевая — `y`.", extra=("aux", "y"))
    assert list(solution.blind_answer(task_md).columns) == ["id", "y"]


def test_no_id_column_uses_first(tmp_path):
    task_md = make_task(tmp_path, "`y1`", extra=("y1",), id_col="row")
    assert list(solution.blind_answer(task_md).columns) == ["row", "y1"]


def test_nothing_to_build(tmp_path):
    pd.DataFrame({"id": [1], "x": [1]}).to_csv(tmp_path / "train.csv", index=False)
    pd.DataFrame({"id": [2], "x": [2]}).to_csv(tmp_path / "test.csv", index=False)
    (tmp_path / "task.md").write_text("?", encoding="utf-8")
    assert solution.blind_answer(tmp_path / "task.md") is None
