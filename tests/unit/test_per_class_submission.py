"""Шаблон ответа «колонка на класс» (id, класс_1, …, класс_k): parse_task не падает и находит настоящий target."""

import pandas as pd

from agent.task_parser import parse_task


def make_task(tmp_path, sample_columns, labels=("a", "b", "c", "a", "b", "c")):
    pd.DataFrame({"id": range(6), "x": range(6), "label": list(labels)}).to_csv(tmp_path / "train.csv", index=False)
    pd.DataFrame({"id": range(3), "x": range(3)}).to_csv(tmp_path / "test.csv", index=False)
    sample = pd.DataFrame({"id": range(3)})
    for col in sample_columns:
        sample[col] = 1 / len(sample_columns)
    sample.to_csv(tmp_path / "sample_submission.csv", index=False)
    (tmp_path / "task.md").write_text("Предскажите вероятности классов. Метрика: log loss.", encoding="utf-8")
    return parse_task(tmp_path / "task.md")


def test_per_class_layout_parsed(tmp_path):
    spec = make_task(tmp_path, ["a", "b", "c"])
    assert spec.target == "label"
    assert spec.id_column == "id"
    assert spec.submission_columns == ["id", "a", "b", "c"]
    assert spec.task_type.value == "multiclass"
    assert spec.class_labels == ["a", "b", "c"]
    assert any("колонка на класс" in note for note in spec.notes)
    assert not any("не совпадают" in note for note in spec.notes)


def test_per_class_names_mismatch_noted(tmp_path):
    spec = make_task(tmp_path, ["class_a", "class_b", "class_c"])
    assert spec.target == "label"
    assert any("не совпадают" in note for note in spec.notes)


def test_usual_layout_unchanged(tmp_path):
    pd.DataFrame({"id": range(4), "x": range(4), "label": ["a", "b", "a", "b"]}).to_csv(
        tmp_path / "train.csv", index=False
    )
    pd.DataFrame({"id": range(2), "x": range(2)}).to_csv(tmp_path / "test.csv", index=False)
    pd.DataFrame({"id": range(2), "label": ["a", "a"]}).to_csv(tmp_path / "sample_submission.csv", index=False)
    (tmp_path / "task.md").write_text("Метрика: accuracy.", encoding="utf-8")
    spec = parse_task(tmp_path / "task.md")
    assert spec.target == "label" and spec.submission_columns == ["id", "label"]
    assert not any("колонка на класс" in note for note in spec.notes)
