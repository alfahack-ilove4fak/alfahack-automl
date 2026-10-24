"""Генератор внешних задач (research/external): без сети — данные подменены, проверяем формат и разбор агентом."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from agent.task_parser import parse_task

ROOT = Path(__file__).resolve().parents[2]


def _load_module(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "research" / "external" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


make = _load_module("make_external_tasks")


def fake_frame(kind: str) -> tuple[pd.DataFrame, str]:
    rng = np.random.default_rng(0)
    frame = pd.DataFrame({"a": rng.normal(size=400), "b": rng.choice(["x", "y", "z"], 400)})
    if kind == "regression":
        frame["target"] = frame["a"] * 3 + rng.normal(size=400)
    elif kind == "binary":
        frame["target"] = np.where(frame["a"] + rng.normal(size=400) > 0.6, "yes", "no")
    else:
        frame["target"] = rng.choice(["c1", "c2", "c3"], 400)
    return frame, "target"


def test_manifest_is_valid():
    tasks = json.loads(make.MANIFEST.read_text(encoding="utf-8"))["tasks"]
    assert len(tasks) >= 24 and len({t["data_id"] for t in tasks}) == len(tasks)
    assert {t["task_type"] for t in tasks} == {"regression", "binary", "multiclass"}
    assert sum(bool(t.get("core")) for t in tasks) >= 9


@pytest.mark.parametrize("name", ["alpha one", "bank marketing", "gamma", "delta", "epsilon", "zeta"])
@pytest.mark.parametrize("kind", ["regression", "binary", "multiclass"])
def test_generated_task_is_parsed_like_meta(tmp_path, monkeypatch, name, kind):
    monkeypatch.setattr(make, "OUT", tmp_path)
    monkeypatch.setattr(make, "load", lambda data_id: fake_frame(kind))
    entry = {"data_id": 1, "name": name, "task_type": kind, "suites": ["test"]}
    slug = make.make(entry)
    assert slug is not None
    meta = json.loads((tmp_path / "_answers" / f"{slug}.json").read_text(encoding="utf-8"))
    answers = pd.read_csv(tmp_path / "_answers" / f"{slug}.csv")
    test = pd.read_csv(tmp_path / slug / "test.csv")
    assert len(answers) == len(test) and "target" not in test.columns
    assert meta["id"] is None or (meta["id"] in test.columns and meta["id"] in answers.columns)
    spec = parse_task(tmp_path / slug / "task.md", use_analyst=False)
    assert spec.target == "target"
    assert spec.metric.name == meta["metric"]
    assert (spec.id_column == meta["id"]) or (meta["id"] is None and spec.id_column in (make_row_id(),))


def make_row_id() -> str:
    from interfaces.schemas import ROW_ID_COLUMN

    return ROW_ID_COLUMN
