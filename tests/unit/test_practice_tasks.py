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


# Схема валидации, которую должен выбрать агент: задачи-ловушки ревью #150 (ADR 0013)
EXPECTED_VALIDATION = {
    "payments_binary_f1_rare": "time:date",  # test — следующий месяц, редкий класс во time-фолдах не роняет CV
    "claims_regression_mape_rowno": "kfold",  # record_no — номер записи, а не время
    "batches_ordinal_qwk": "stratified",
}


@pytest.mark.parametrize("name", sorted(EXPECTED_VALIDATION))
def test_trap_tasks_get_expected_validation(name):
    from ml.profiling import load_data, profile_data
    from ml.validation import choose_validation

    spec = parse_task(PRACTICE / name / "task.md")
    train, test = load_data(spec)
    assert choose_validation(spec, profile_data(spec, train, test), train, test) == EXPECTED_VALIDATION[name]


def test_trap_tasks_keep_their_traps():
    """Перегенерировали задачи — ловушки должны остаться: иначе задача проверяет не то, что написано в README."""
    pay_train = pd.read_csv(PRACTICE / "payments_binary_f1_rare" / "train.csv")
    pay_test = pd.read_csv(PRACTICE / "payments_binary_f1_rare" / "test.csv")
    assert 0.01 < pay_train["is_fraud"].mean() < 0.05
    assert pay_test["date"].min() > pay_train["date"].max()
    assert 0.15 < (~pay_test["merchant_id"].isin(pay_train["merchant_id"])).mean() < 0.35

    batches_md = (PRACTICE / "batches_ordinal_qwk" / "task.md").read_text(encoding="utf-8")
    batches_cols = set(pd.read_csv(PRACTICE / "batches_ordinal_qwk" / "train.csv", nrows=1).columns)
    assert "`vsc`" in batches_md and "additive_mg" not in batches_md  # ошибка в описании — намеренная
    assert {"viscosity", "additive_mg"} <= batches_cols

    search_train = pd.read_csv(PRACTICE / "search_ranking_ndcg" / "train.csv")
    search_test = pd.read_csv(PRACTICE / "search_ranking_ndcg" / "test.csv")
    assert not set(search_train["query_id"]) & set(search_test["query_id"])  # в test — новые запросы
    assert set(search_train["relevance"]) == {0, 1, 2, 3}

    claims_train = pd.read_csv(PRACTICE / "claims_regression_mape_rowno" / "train.csv")
    claims_test = pd.read_csv(PRACTICE / "claims_regression_mape_rowno" / "test.csv")
    assert claims_test["record_no"].min() > claims_train["record_no"].max()
    assert (claims_train["claim_amount"] > 0).all()
