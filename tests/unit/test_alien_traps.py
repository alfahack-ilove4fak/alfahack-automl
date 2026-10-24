"""Ловушки research/night: наборы leak и uncovered не попадают в гейт, утечки описаны в meta, MAP@k по запросам."""

import json

import numpy as np
import pandas as pd
import pytest

from research.night import make_alien_tasks as mat
from research.night import run_alien


def test_gate_set_stays_15_tasks():
    # гейт (scripts/gate.py) сравнивает число нулей на 15 задачах набора alien — новые наборы туда не попадают
    assert len(mat.TASKS) == 15
    assert not set(mat.LEAK) & set(mat.TASKS)
    assert not set(mat.UNCOVERED) & set(mat.TASKS)
    assert set(mat.SETS) == {"alien", "leak", "uncovered"}


@pytest.mark.parametrize("name", sorted(mat.LEAK))
def test_leak_trap_is_filled_in_test_and_not_announced(name, tmp_path, monkeypatch):
    monkeypatch.setattr(mat, "OUT", tmp_path)
    mat.LEAK[name]()
    meta = json.loads((tmp_path / "_answers" / f"{name}.json").read_text(encoding="utf-8"))
    train = pd.read_csv(tmp_path / name / "train.csv")
    test = pd.read_csv(tmp_path / name / "test.csv")
    md = (tmp_path / name / "task.md").read_text(encoding="utf-8").lower()
    assert meta["leak_columns"]
    for col in meta["leak_columns"]:
        assert col in train.columns and col in test.columns
        assert test[col].notna().all()  # не «пусто в test», как у leak_column_auc: другая природа
    assert "утечк" not in md and "leak" not in md


def test_frames_drop_leak_columns():
    train = pd.DataFrame({"id": [1, 2], "x": [0.1, 0.2], "leak": [1, 0], "y": [1, 0]})
    test = pd.DataFrame({"id": [3], "x": [0.3], "leak": [1]})
    meta = {"targets": ["y"], "id": "id", "leak_columns": ["leak"]}
    xtr, xte = run_alien._frames(train, test, meta)
    assert list(xtr.columns) == list(xte.columns) == ["x_0"]


def test_map_at_k_per_query():
    rel = np.array([1, 0, 0, 0, 1, 1, 0, 0, 0])
    score = np.array([0.9, 0.5, 0.1, 0.2, 0.8, 0.1, 0.3, 0.7, 0.6])
    groups = np.array([1, 1, 1, 2, 2, 2, 3, 3, 3])  # в запросе 3 нет релевантных — не считается
    # запрос 1: релевантный первым → AP 1; запрос 2: порядок 4(1), 3(0), 5(1) → (1 + 2/3) / 2
    assert run_alien._map_at_k(rel, score, groups, 5) == pytest.approx((1 + (1 + 2 / 3) / 2) / 2)


def test_true_metric_multilabel_and_ranking():
    answers = pd.DataFrame({"a": [0, 1, 0, 1], "b": [1, 0, 0, 1]})
    sub = pd.DataFrame({"a": [0.1, 0.9, 0.2, 0.8], "b": [0.9, 0.1, 0.8, 0.2]})
    meta = {"metric": "mean_roc_auc", "targets": ["a", "b"]}
    assert run_alien.true_metric(meta, answers, sub, None) == pytest.approx(
        (1.0 + 0.75) / 2
    )  # b: 3 из 4 пар упорядочены верно
    answers = pd.DataFrame({"relevant": [0, 1], "query_id": [7, 7]})
    meta = {"metric": "map@5", "targets": ["relevant"], "group": "query_id", "answer_column": "score"}
    assert run_alien.true_metric(meta, answers, pd.DataFrame({"score": [0.2, 0.9]}), None) == 1.0
