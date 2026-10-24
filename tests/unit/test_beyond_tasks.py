"""Генератор BeyondArena (research/external/make_beyond_tasks.py): без сети — данные подменены.

Проверяем временное разбиение (test строго позже train, дата не делится), групповое (группы test не встречаются
в train, ≈ TEST_SHARE строк), формат задачи и разбор агентом, а также проверку схемы валидации в report.py.
"""

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from agent.task_parser import parse_task

ROOT = Path(__file__).resolve().parents[2]
EXTERNAL = ROOT / "research" / "external"


def _load_module(name: str):
    sys.path.insert(0, str(EXTERNAL))  # make_beyond_tasks импортирует соседний make_external_tasks
    spec = importlib.util.spec_from_file_location(name, EXTERNAL / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


beyond = _load_module("make_beyond_tasks")
report = _load_module("report")


def fake_frame(kind: str, numeric_time: bool = False) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    n = 600
    days = pd.Timestamp("2020-01-01") + pd.to_timedelta(rng.integers(0, 120, n), unit="D")
    frame = pd.DataFrame({"a": rng.normal(size=n), "b": rng.choice(["x", "y", "z"], n)})
    frame["when"] = rng.integers(1990, 2010, n) if numeric_time else days
    frame["patient"] = [f"p{i}" for i in rng.zipf(1.6, n) % 90]  # группы очень разного размера, как в musk
    if kind == "regression":
        frame["target"] = frame["a"] * 3 + rng.normal(size=n)
    elif kind == "binary":
        frame["target"] = np.where(frame["a"] + rng.normal(size=n) > 0.6, "1", "0")
    else:
        frame["target"] = rng.choice(["c1", "c2", "c3"], n)
    return frame


def test_manifest_is_valid():
    manifest = json.loads(beyond.MANIFEST.read_text(encoding="utf-8"))
    tasks = manifest["tasks"]
    assert manifest["revision"] == beyond.REVISION and len(beyond.REVISION) == 40
    assert len(tasks) >= 5 and len({t["name"] for t in tasks}) == len(tasks)
    assert all(t["target"] and t["folder"].startswith(t["name"]) for t in tasks)
    assert all(t["time_on"] if t["split"] == "temporal" else t["group_on"] for t in tasks)
    for split in beyond.SETS:
        core = [t for t in tasks if t.get("core") and t["split"] == split]
        assert 5 <= len(core) <= 6 and {t["task_type"] for t in core} == {"regression", "binary", "multiclass"}


def test_readme_rows_parses_table():
    readme = (
        "| Dataset | Domain | Source | Year | Age | N | d | C | Prob. | Task | BibKey(s) |\n"
        "|---|---|---|---|---|---|---|---|---|---|---|\n"
        "| kick | B & M | Kaggle | 2011 | 15 | 72,983 | 32 | 2 | Binary | Temporal | DontGetKicked |\n"
        "| musk | C & M | UCI | 1994 | 32 | 6,598 | 166 | 2 | Binary | Grouped | dietterich1993comparison |\n"
        "| garments | I & M | UCI | 2020 | 6 | 1,197 | 15 | — | Reg | Temporal | imran2021mining |\n"
    )
    rows = beyond.readme_rows(readme)
    assert [(r["name"], r["rows"], r["task_type"], r["regime"]) for r in rows] == [
        ("kick", 72983, "binary", "temporal"),
        ("musk", 6598, "binary", "grouped"),
        ("garments", 1197, "regression", "temporal"),
    ]


@pytest.mark.parametrize("numeric_time", [False, True])
def test_time_split_keeps_order_and_dates_whole(numeric_time):
    frame = fake_frame("regression", numeric_time)
    train, test, _ = beyond.time_split(frame, "when")
    assert len(train) + len(test) == len(frame)
    assert beyond.MIN_TEST_SHARE <= len(test) / len(frame) <= 0.5
    assert train["when"].max() < test["when"].min()  # одна дата целиком по одну сторону


@pytest.mark.parametrize("kind", ["regression", "binary", "multiclass"])
def test_group_split_keeps_groups_whole(kind):
    frame = fake_frame(kind)
    train, test, info = beyond.group_split(frame, "patient", "target", kind, seed=7)
    assert len(train) + len(test) == len(frame) and info.startswith("групп в test")
    assert not set(train["patient"]) & set(test["patient"])
    assert abs(len(test) / len(frame) - beyond.TEST_SHARE) < 0.05  # подбор разбиения по доле строк
    again, _, _ = beyond.group_split(frame, "patient", "target", kind, seed=7)
    assert again.equals(train)  # детерминировано


def test_time_split_with_many_rows_on_cutoff_date():
    frame = pd.DataFrame({"when": pd.to_datetime(["2020-01-01"] * 20 + ["2020-02-01"] * 80), "x": range(100)})
    train, test, _ = beyond.time_split(frame, "when")
    assert len(test) == 80 and train["when"].max() < test["when"].min()


@pytest.mark.parametrize("name", ["alpha one", "kick", "gamma", "delta"])
@pytest.mark.parametrize("kind", ["regression", "binary", "multiclass"])
@pytest.mark.parametrize(("split", "column"), [("temporal", "when"), ("grouped", "patient")])
def test_generated_task_is_split_and_parsed_like_meta(tmp_path, monkeypatch, name, kind, split, column):
    monkeypatch.setattr(beyond, "OUT", tmp_path)
    monkeypatch.setattr(beyond, "load", lambda entry: fake_frame(kind))
    key = "time_on" if split == "temporal" else "group_on"
    entry = {"name": name, "task_type": kind, "target": "target", "split": split, key: column, "folder": f"{name}/u"}
    slug = beyond.make(entry)
    assert slug is not None
    out = tmp_path / beyond.SETS[split]
    meta = json.loads((out / "_answers" / f"{slug}.json").read_text(encoding="utf-8"))
    train = pd.read_csv(out / slug / "train.csv", parse_dates=["when"])
    test = pd.read_csv(out / slug / "test.csv", parse_dates=["when"])
    answers = pd.read_csv(out / "_answers" / f"{slug}.csv")
    scheme = "time" if split == "temporal" else "group"
    assert meta["expected_validation"] == f"{scheme}:{column}" and meta["split"] == split
    if split == "temporal":
        assert train["when"].max() < test["when"].min()
    else:
        assert not set(train["patient"]) & set(test["patient"])
    assert len(answers) == len(test) and "target" not in test.columns
    text = (out / slug / "task.md").read_text(encoding="utf-8")
    assert any(h.format(col=column) in text for h in beyond.HINTS[split]) == meta["hint_in_text"]
    spec = parse_task(out / slug / "task.md", use_analyst=False)
    assert spec.target == "target"
    assert spec.metric.name == meta["metric"]


@pytest.mark.parametrize(
    ("log_line", "expected"),
    [
        ("по времени (when) — test лежит после train; кандидаты: ['when']", True),
        ("по времени (year) — test лежит после train; кандидаты: ['year', 'when']", False),
        ("stratified — времени и заметных дубликатов нет", False),
        (None, False),
    ],
)
def test_report_checks_validation_scheme(log_line, expected):
    row = {"log": {"validation": log_line}}
    assert report.validation_ok(row, {"expected_validation": "time:when"}) is expected
    assert report.validation_ok(row, {}) is None


@pytest.mark.parametrize(
    ("log_line", "expected"),
    [
        ("по группам (patient) — значения test не встречаются в train", True),
        ("group:patient-5", True),
        ("группы дубликатов — 45% строк train повторяются", False),
        ("по группам (doctor) — значения test не встречаются в train", False),
        ("по времени (patient) — test лежит после train", False),
        ("stratified — времени и заметных дубликатов нет", False),
    ],
)
def test_report_checks_group_scheme(log_line, expected):
    assert report.validation_ok({"log": {"validation": log_line}}, {"expected_validation": "group:patient"}) is expected
