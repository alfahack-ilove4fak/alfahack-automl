"""Имена и форматы файлов скрытых задач (#276, приоритет 2): data_train, TRAIN.CSV, .gz/.zip, learn/holdout, «"10,5"»."""

import gzip
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from agent import csv_dialect
from agent.csv_dialect import plain_name, sniff, standard_csv
from agent.task_parser import file_role, find_task_files, parse_task
from ml.profiling import load_data


@pytest.fixture(autouse=True)
def cache_in_tmp(tmp_path, monkeypatch):
    monkeypatch.setattr(csv_dialect, "CACHE_DIR", tmp_path / "cache")


@pytest.mark.parametrize(
    "name, role",
    [
        ("train.csv", "train"),
        ("TRAIN.CSV", "train"),
        ("data_train.csv", "train"),
        ("trainData.csv", "train"),
        ("learn.csv.gz", "train"),
        ("training_set.tsv", "train"),
        ("test.csv", "test"),
        ("x_test.csv", "test"),
        ("holdout.txt", "test"),
        ("test_data.zip", "test"),
        ("sample_submission.csv", "sample_submission"),
        ("sampleSubmission.csv", "sample_submission"),
        ("submission_example.csv.gz", "sample_submission"),
        ("data_dictionary.csv", "data_dictionary"),
        ("submission.csv", None),  # наш ответ
        ("submission.tmp.csv", None),  # огрызок записи
        ("train_test.csv", None),  # непонятно — не угадываем
        ("latest.csv", None),
        ("constraints.csv", None),
        ("task.md", None),
        ("train.parquet", None),
    ],
)
def test_file_role(name, role):
    assert file_role(name) == role


@pytest.mark.parametrize(
    "name, plain",
    [("train.csv.gz", "train.csv"), ("TEST.tsv", "TEST.csv"), ("learn.zip", "learn.csv"), ("x.txt.xz", "x.csv")],
)
def test_plain_name(name, plain):
    assert plain_name(Path(name)) == plain


def write_task(folder: Path, n=60, n_test=20) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(0)
    train = pd.DataFrame({"id": range(n), "a": rng.normal(size=n)})
    train["y"] = train["a"] * 2
    test = pd.DataFrame({"id": range(n, n + n_test), "a": rng.normal(size=n_test)})
    (folder / "task.md").write_text("Предскажи y. Метрика — MAE.", encoding="utf-8")
    return train, test


def test_parse_task_upper_case_gz_and_zip(tmp_path):
    """TRAIN.CSV, data_test.csv.gz и sampleSubmission.zip: агент находит всё и читает как обычный CSV."""
    train, test = write_task(tmp_path)
    train.to_csv(tmp_path / "TRAIN.CSV", index=False)
    with gzip.open(tmp_path / "data_test.csv.gz", "wt", encoding="utf-8") as f:
        test.to_csv(f, index=False)
    with zipfile.ZipFile(tmp_path / "sampleSubmission.zip", "w") as z:
        z.writestr("__MACOSX/._sample.csv", b"\x00\x05garbage")
        z.writestr("sample.csv", test[["id"]].assign(y=0).to_csv(index=False))
    spec = parse_task(tmp_path / "task.md", use_analyst=False)
    assert (spec.target, spec.id_column, spec.submission_columns) == ("y", "id", ["id", "y"])
    assert spec.test_path.suffix == ".csv" and spec.sample_submission_path.suffix == ".csv"
    loaded_train, loaded_test = load_data(spec)
    assert len(loaded_train) == 60 and len(loaded_test) == 20


def test_learn_holdout_in_subfolder(tmp_path):
    """Файлы в подпапке data/ с именами learn/holdout в .tsv."""
    train, test = write_task(tmp_path)
    (tmp_path / "data").mkdir()
    train.to_csv(tmp_path / "data" / "learn.tsv", sep="\t", index=False)
    test.to_csv(tmp_path / "data" / "holdout.tsv", sep="\t", index=False)
    spec = parse_task(tmp_path / "task.md", use_analyst=False)
    assert spec.target == "y"
    assert list(pd.read_csv(spec.train_path).columns) == ["id", "a", "y"]


def test_prefers_nearest_and_canonical_names(tmp_path):
    train, test = write_task(tmp_path)
    for name in ("train.csv", "train_extra.csv", "test.csv"):
        (train if name.startswith("train") else test).to_csv(tmp_path / name, index=False)
    (tmp_path / "old").mkdir()
    test.to_csv(tmp_path / "old" / "test.csv", index=False)
    (tmp_path / "__MACOSX").mkdir()
    train.to_csv(tmp_path / "__MACOSX" / "train.csv", index=False)
    files = find_task_files(tmp_path / "task.md")
    assert files["train"] == tmp_path / "train.csv"
    assert files["test"] == tmp_path / "test.csv"


def test_plain_csv_preferred_over_compressed_copy(tmp_path):
    train, test = write_task(tmp_path)
    train.to_csv(tmp_path / "train.csv", index=False)
    train.to_csv(tmp_path / "train.csv.gz", index=False)
    test.to_csv(tmp_path / "test.csv", index=False)
    assert find_task_files(tmp_path / "task.md")["train"] == tmp_path / "train.csv"


def test_path_from_text_wins(tmp_path):
    """Путь в тексте в бэктиках — даже с нестандартным именем — берётся первым."""
    train, test = write_task(tmp_path)
    train.to_csv(tmp_path / "train.csv", index=False)
    train.head(10).to_csv(tmp_path / "train_small.csv", index=False)
    test.to_csv(tmp_path / "test.csv", index=False)
    (tmp_path / "task.md").write_text("Обучение — `train_small.csv`. Метрика — MAE.", encoding="utf-8")
    assert find_task_files(tmp_path / "task.md")["train"] == tmp_path / "train_small.csv"


def test_decimal_comma_in_quotes(tmp_path):
    """Разделитель «,», число в кавычках «"10,5"»: колонка становится числовой."""
    path = tmp_path / "train.csv"
    path.write_text('id,x,city\n1,"10,5","Москва, центр"\n2,"-3,25",Тула\n3,7,Орёл\n', encoding="utf-8")
    assert sniff(path) == {"encoding": "utf-8-sig", "sep": ",", "decimal": ","}
    df = pd.read_csv(standard_csv(path))
    assert df["x"].tolist() == [10.5, -3.25, 7.0]
    assert df["city"].tolist() == ["Москва, центр", "Тула", "Орёл"]  # текст с запятой не трогаем


def test_quoted_text_with_comma_keeps_standard_file(tmp_path):
    path = tmp_path / "train.csv"
    path.write_text('id,city\n1,"Москва, центр"\n2,"1,5 км от центра"\n', encoding="utf-8")
    assert standard_csv(path) == path
