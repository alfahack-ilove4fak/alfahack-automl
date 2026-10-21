"""Сборка промпта гипотез из agent/prompts/*.md на заглушках. LLM не вызывается."""

import json
import re
from pathlib import Path

import pytest

from agent import prompting
from interfaces.schemas import (
    DataProfile,
    ExperimentResult,
    ExperimentStatus,
    Hypothesis,
    HypothesisKind,
    MetricSpec,
    TaskType,
)
from interfaces.stubs import stub_task_spec


@pytest.fixture
def spec():
    return stub_task_spec(Path("tasks/demo")).model_copy(update={"description": "Предсказать потребление энергии"})


@pytest.fixture
def profile():
    return DataProfile(
        n_train=1000,
        n_test=400,
        numeric_features=[f"num_{i}" for i in range(50)],
        categorical_features=["city"],
        missing_share={"num_1": 0.3, "num_2": 0.05},
        target_summary={"mean": 81.5, "skew": 3.2},
        warnings=["Сильно скошенный target — стоит попробовать log1p-преобразование"],
    )


def make_step(i: int, status: ExperimentStatus = ExperimentStatus.OK):
    hyp = Hypothesis(id=f"h{i}", kind=HypothesisKind.HYPERPARAMS, title=f"Гипотеза {i}", params={"model": "lightgbm"})
    ok = status == ExperimentStatus.OK
    res = ExperimentResult(
        hypothesis_id=hyp.id,
        status=status,
        metric_name="rmse",
        mean_score=40.0 + i if ok else None,
        std_score=0.5 if ok else None,
        error=None if ok else "кончилось время на фолде 4",
    )
    return hyp, res


def user_prompt(spec, profile, history=(), k=3) -> str:
    messages = prompting.build_hypothesis_messages(spec, profile, list(history), k=k)
    assert [m["role"] for m in messages] == ["system", "user"]
    return messages[1]["content"]


def test_all_placeholders_filled(spec, profile):
    text = user_prompt(spec, profile, [make_step(1)], k=4)
    assert not re.search(r"\$\w", text), "в промпте остался незаполненный плейсхолдер"
    assert "JSON-список из 4 новых гипотез" in text


def test_no_emoji_in_prompt(spec, profile):
    messages = prompting.build_hypothesis_messages(spec, profile, [make_step(1)])
    for message in messages:
        assert not re.search(r"[☀-➿\U0001F300-\U0001FAFF]", message["content"])


def test_task_and_profile_in_prompt(spec, profile):
    text = user_prompt(spec, profile)
    assert "rmse (меньше — лучше)" in text
    assert "site_eui" in text
    assert "Предсказать потребление энергии" in text
    assert "num_1 0.30" in text
    assert "skew=3.2" in text
    assert "log1p" in text
    assert "и ещё 20" in text  # 50 признаков сжаты до 30
    assert "num_49" not in text


def test_params_doc_matches_executor(spec, profile):
    text = user_prompt(spec, profile)
    for key in ("model ", "model_params", "n_folds", "log_target", "class_weight"):
        assert f"\n{key}" in text, f"ключ {key!r} из docstring ml/experiment.py не попал в промпт"


def test_examples_are_valid_hypotheses(spec, profile):
    text = user_prompt(spec, profile)
    block = re.search(r"```json\n(.*?)\n```", text, re.S).group(1)
    examples = json.loads(block)
    assert examples
    for item in examples:
        Hypothesis.model_validate(item)


def test_examples_respect_task_type(spec, profile):
    classification = spec.model_copy(
        update={"task_type": TaskType.MULTICLASS, "metric": MetricSpec(name="f1_macro", greater_is_better=True)}
    )
    ids = {item["id"] for item in json.loads(prompting.playbook_examples(classification, profile, n=100))}
    assert "log_target" not in ids


def test_history_table(spec, profile):
    text = user_prompt(spec, profile, [make_step(1), make_step(2, ExperimentStatus.TIMEOUT)])
    assert "| h1 | hyperparams |" in text
    assert "41.00000 ± 0.50000" in text
    assert "timeout: кончилось время на фолде 4" in text


def test_empty_and_long_history(spec, profile):
    assert "ни одного эксперимента" in user_prompt(spec, profile)
    text = user_prompt(spec, profile, [make_step(i) for i in range(25)])
    assert "| h24 |" in text and "| h14 |" not in text
    assert "ранее было ещё 15" in text


def test_missing_prompt_returns_none(spec, profile, monkeypatch, tmp_path):
    monkeypatch.setattr(prompting, "PROMPTS_DIR", tmp_path)
    assert prompting.build_hypothesis_messages(spec, profile, []) is None


def test_unknown_placeholder_returns_none(spec, profile, monkeypatch, tmp_path):
    (tmp_path / "system.md").write_text("system", encoding="utf-8")
    (tmp_path / "hypotheses.md").write_text("$task $опечатка_в_имени", encoding="utf-8")
    monkeypatch.setattr(prompting, "PROMPTS_DIR", tmp_path)
    assert prompting.build_hypothesis_messages(spec, profile, []) is None
