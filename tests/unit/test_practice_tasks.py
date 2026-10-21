"""Тренировочные задачи tasks/practice/: агент верно разбирает каждую, ответы согласованы с test."""

import json
from pathlib import Path

import pandas as pd
import pytest

from agent.task_parser import parse_task

PRACTICE = Path(__file__).resolve().parents[2] / "tasks" / "practice"
MANIFEST = json.loads((PRACTICE / "_answers" / "manifest.json").read_text(encoding="utf-8"))
# Известные проблемы агента, которые тренировочные задачи и должны ловить. Починили — убрать отсюда.
KNOWN_PARSER_BUGS: dict[str, str] = {}  # ловушку с метрикой в network_binary_logloss_crooked починили в #104


def _cases():
    for name in sorted(MANIFEST):
        marks = [pytest.mark.xfail(reason=KNOWN_PARSER_BUGS[name], strict=True)] if name in KNOWN_PARSER_BUGS else []
        yield pytest.param(name, marks=marks, id=name)


def test_every_task_has_answers():
    tasks = {p.parent.name for p in PRACTICE.glob("*/task.md")}
    assert len(tasks) >= 4
    assert tasks == set(MANIFEST)


@pytest.mark.parametrize("name", _cases())
def test_parser_understands_task(name):
    expected = MANIFEST[name]
    spec = parse_task(PRACTICE / name / "task.md")
    assert (spec.metric.name, spec.target, spec.id_column) == (
        expected["metric"],
        expected["target"],
        expected["id_column"],
    )


@pytest.mark.parametrize("name", sorted(MANIFEST))
def test_answers_match_test(name):
    expected = MANIFEST[name]
    test_csv = next((PRACTICE / name).rglob("test*.csv"))
    test = pd.read_csv(test_csv, usecols=[expected["id_column"]])
    answers = pd.read_csv(PRACTICE / "_answers" / f"{name}.csv")
    assert list(answers.columns) == [expected["id_column"], expected["target"]]
    assert answers[expected["id_column"]].tolist() == test[expected["id_column"]].tolist()
    assert answers[expected["target"]].notna().all()
