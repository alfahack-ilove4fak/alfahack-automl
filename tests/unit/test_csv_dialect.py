"""Файлы задачи «как в жизни» (#238, ночной аудит #230): cp1251, «;», десятичная запятая, id без шаблона."""

from pathlib import Path

import pandas as pd
import pytest

from agent import csv_dialect
from agent.csv_dialect import read_text_any, sniff, standard_csv
from agent.task_parser import _guess_id, _looks_like_id, parse_task


@pytest.fixture(autouse=True)
def cache_in_tmp(tmp_path, monkeypatch):
    monkeypatch.setattr(csv_dialect, "CACHE_DIR", tmp_path / "cache")


def write(path: Path, text: str, encoding: str = "utf-8") -> Path:
    path.write_bytes(text.encode(encoding))
    return path


def test_standard_file_is_used_as_is(tmp_path):
    path = write(tmp_path / "train.csv", "id,x,y\n1,2.5,3\n")
    assert sniff(path) == {"encoding": "utf-8-sig", "sep": ",", "decimal": "."}
    assert standard_csv(path) == path


def test_cp1251_converted_to_utf8(tmp_path):
    path = write(tmp_path / "train.csv", "ид,выручка\n1,10.5\n2,20\n", "cp1251")
    out = standard_csv(path)
    assert out != path
    df = pd.read_csv(out)
    assert list(df.columns) == ["ид", "выручка"] and df["выручка"].tolist() == [10.5, 20.0]


def test_semicolon_and_decimal_comma(tmp_path):
    path = write(tmp_path / "train.csv", "id;площадь;район;цена\n007;79,5;Юг;7,404\n2;132,0;Север, центр;14\n")
    df = pd.read_csv(standard_csv(path), dtype={"id": str})
    assert list(df.columns) == ["id", "площадь", "район", "цена"]
    assert df["площадь"].tolist() == [79.5, 132.0] and df["цена"].tolist() == [7.404, 14.0]
    assert df["id"].tolist() == ["007", "2"]  # значения не трогаем: id «007» остаётся «007»
    assert df["район"].tolist() == ["Юг", "Север, центр"]  # запятая в тексте — не десятичная


def test_decimal_comma_column_with_missing_tokens(tmp_path):
    """Ревью #246: «NA», «-», пусто среди чисел не мешают признать колонку числовой."""
    path = write(tmp_path / "train.csv", "id;x\n1;1,5\n2;NA\n3;-2,25\n4;\n5;-\n")
    df = pd.read_csv(standard_csv(path), na_values=["-"])
    assert df["x"].dtype.kind == "f"
    assert df["x"].iloc[[0, 2]].tolist() == [1.5, -2.25] and df["x"].isna().sum() == 3


def test_commas_inside_quotes_do_not_change_separator(tmp_path):
    path = write(tmp_path / "train.csv", '"Номер заявки","Доход, руб.","Город; район"\n1,"10,5",A\n')
    assert sniff(path)["sep"] == ","


def test_single_column_template(tmp_path):
    path = write(tmp_path / "sample_submission.csv", "price\n0\n0\n")
    assert standard_csv(path) == path


def test_converted_once(tmp_path, monkeypatch):
    path = write(tmp_path / "train.csv", "a;b\n1;2\n")
    first = standard_csv(path)
    monkeypatch.setattr(pd, "read_csv", lambda *a, **k: pytest.fail("второй раз не читаем"))
    assert standard_csv(path) == first


def test_broken_file_falls_back_to_original(tmp_path):
    path = tmp_path / "missing.csv"
    assert standard_csv(path) == path  # сбой приведения — не падение разбора


def test_task_md_in_cp1251(tmp_path):
    path = write(tmp_path / "task.md", "Метрика — R2", "cp1251")
    assert read_text_any(path) == "Метрика — R2"


def test_parse_task_on_cp1251_semicolon_task(tmp_path):
    write(tmp_path / "task.md", "# Цена\nПредскажите цену. Метрика — MAE.\n", "cp1251")
    write(tmp_path / "train.csv", "id;площадь;цена\n" + "".join(f"{i};{50 + i},5;{i},25\n" for i in range(30)))
    write(tmp_path / "test.csv", "id;площадь\n" + "".join(f"{i};{40 + i},5\n" for i in range(100, 110)))
    write(tmp_path / "sample_submission.csv", "id;цена\n" + "".join(f"{i};0,0\n" for i in range(100, 110)))
    spec = parse_task(tmp_path / "task.md")
    assert (spec.target, spec.id_column, spec.metric.name) == ("цена", "id", "mae")
    assert pd.read_csv(spec.train_path)["цена"].dtype.kind == "f"


@pytest.mark.parametrize(
    "name, expected",
    [
        ("txn_id", True),
        ("clientID", True),
        ("Номер заявки", True),
        ("row-key", True),
        ("ид", True),
        ("paid", False),
        ("tenure", False),
        ("idea_score", False),
    ],
)
def test_looks_like_id(name, expected):
    assert _looks_like_id(name) is expected


def test_guess_id_without_template_prefers_unique_id_named_column(tmp_path):
    """Аудит #238: без шаблона и без колонки «id» раньше бралась первая колонка test («tenure»)."""
    test = tmp_path / "test.csv"
    pd.DataFrame({"tenure": [1, 1, 2], "group_id": [5, 5, 6], "txn_id": [10, 11, 12]}).to_csv(test, index=False)
    assert _guess_id(["tenure", "group_id", "txn_id"], {}, test) == "txn_id"  # group_id не уникален
    assert _guess_id(["tenure", "x"], {}, test) == "tenure"  # кандидатов по имени нет — как раньше
    assert _guess_id(["tenure", "id", "txn_id"], {}, test) == "id"
