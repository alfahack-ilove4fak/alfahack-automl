"""Qwen — инженер данных (agent/data_engineer.py): план → код в песочнице → CV → вывод; без LLM — как раньше."""

import numpy as np
import pandas as pd
import pytest

from agent import data_engineer as de
from interfaces.schemas import (
    ExperimentResult,
    ExperimentStatus,
    Hypothesis,
    HypothesisKind,
    MetricSpec,
    RunBudget,
    TaskSpec,
    TaskType,
)
from ml.experiment import run_experiment
from ml.profiling import profile_data

GOOD_CODE = (
    "def prepare(df):\n    out = df.drop(columns=['noise_text'])\n"
    "    out['ratio'] = out['a'] / (out['b'].abs() + 1)\n    return out"
)


STOP = {"stop": True}


class FakeLLM:
    """Отвечает по очереди заготовленными ответами; список кончился — None."""

    def __init__(self, answers, available=True):
        self.answers = list(answers)
        self.calls = []
        self._available = available

    def available(self):
        return self._available

    def chat_json(self, messages, **kwargs):
        self.calls.append(messages)
        return self.answers.pop(0) if self.answers else None


def _spec(path) -> TaskSpec:
    return TaskSpec(
        task_md=path / "task.md",
        task_dir=path,
        train_path=path / "train.csv",
        test_path=path / "test.csv",
        submission_path=path / "submission.csv",
        task_type=TaskType.REGRESSION,
        target="y",
        id_column="id",
        metric=MetricSpec(name="rmse", greater_is_better=False),
        submission_columns=["id", "y"],
        description="Предсказать y по a и b.",
    )


@pytest.fixture
def task(tmp_path):
    rng = np.random.default_rng(0)
    n = 500
    df = pd.DataFrame(
        {
            "id": np.arange(n),
            "a": rng.normal(size=n),
            "b": rng.normal(size=n),
            "noise_text": [f"row{i}" for i in range(n)],
            "year": np.repeat(np.arange(5), n // 5),
        }
    )
    df["y"] = 3 * df["a"] / (df["b"].abs() + 1) + rng.normal(0, 0.1, n)
    train, test = df.iloc[:400].reset_index(drop=True), df.iloc[400:].drop(columns="y").reset_index(drop=True)
    spec = _spec(tmp_path)
    profile = profile_data(spec, train, test)
    base = Hypothesis(
        id="baseline", kind=HypothesisKind.BASELINE, title="b", params={"model": "lightgbm"}, source="baseline"
    )
    base_result = run_experiment(spec, profile, base, train, RunBudget())
    return spec, profile, train, test, base, base_result


def _plan(code=GOOD_CODE, model="lightgbm", **extra):
    return {"plan": "убрать текст-идентификатор, добавить отношение a/b", "code": code, "model": model, **extra}


def test_plan_runs_in_sandbox_and_wins(task):
    spec, profile, train, test, base, base_result = task
    llm = FakeLLM([_plan(), STOP, {"choice": "qwen_data_1", "conclusion": "отношение a/b — главный признак"}])
    outcome = de.run_data_engineer(spec, profile, train, test, base, base_result, RunBudget(), llm)
    chosen = outcome.chosen
    assert chosen is not None and chosen.hyp.id == "qwen_data_1" and "отношение" in outcome.conclusion
    assert "ratio" in chosen.columns and "noise_text" not in chosen.columns
    assert chosen.result.mean_score < base_result.mean_score  # признак, которого не было, честно помог
    assert list(chosen.test.columns[:1]) == ["id"] and "y" in chosen.train.columns and "y" not in chosen.test.columns
    assert len(chosen.test) == len(test)


def test_broken_code_goes_back_to_qwen_and_is_fixed(task):
    spec, profile, train, test, base, base_result = task
    broken = _plan(code="def prepare(df):\n    return df['нет_колонки']")
    llm = FakeLLM([broken, _plan(), STOP, {"choice": "qwen_data_2", "conclusion": "исправил"}])
    outcome = de.run_data_engineer(spec, profile, train, test, base, base_result, RunBudget(), llm)
    assert "упал" in outcome.attempts[0].error and outcome.chosen.hyp.id == "qwen_data_2"
    second_prompt = llm.calls[1][-1]["content"]
    assert "нет_колонки" in second_prompt  # ошибка первого раунда — в промпте второго


def test_unsafe_or_wrong_plans_are_rejected(task):
    spec, profile, train, test, base, base_result = task
    plans = [
        _plan(code="def prepare(df):\n    import os\n    return df"),
        _plan(model="superboost"),
        _plan(code="def prepare(df):\n    return df.iloc[:10]"),
    ]
    outcome = de.run_data_engineer(spec, profile, train, test, base, base_result, RunBudget(), FakeLLM(plans))
    assert outcome.chosen is None
    errors = [a.error for a in outcome.attempts]
    assert "запрещённый импорт" in errors[0] and "неизвестная модель" in errors[1] and "изменил строки" in errors[2]


def test_verdict_guard_keeps_baseline_when_choice_is_much_worse(task):
    spec, profile, train, test, base, base_result = task
    junk = _plan(
        code="def prepare(df):\n    out = df[['noise_text']].copy()\n    out['z'] = 1.0\n    out['w'] = df['year']\n    return out"
    )
    llm = FakeLLM([junk, STOP, {"choice": "qwen_data_1", "conclusion": "беру"}])
    outcome = de.run_data_engineer(spec, profile, train, test, base, base_result, RunBudget(), llm)
    assert outcome.chosen is None and "страховка" in outcome.conclusion


def test_no_verdict_picks_best_cv_only_if_better(task):
    spec, profile, train, test, base, base_result = task
    outcome = de.run_data_engineer(spec, profile, train, test, base, base_result, RunBudget(), FakeLLM([_plan()]))
    assert outcome.chosen is not None and outcome.chosen.hyp.id == "qwen_data_1"


def test_off_without_llm_or_switch(task, monkeypatch):
    spec, profile, train, test, base, base_result = task
    assert de.run_data_engineer(spec, profile, train, test, base, base_result, RunBudget(), None) is None
    off = FakeLLM([_plan()], available=False)
    assert de.run_data_engineer(spec, profile, train, test, base, base_result, RunBudget(), off) is None
    monkeypatch.setenv(de.DE_ENV, "0")
    assert de.run_data_engineer(spec, profile, train, test, base, base_result, RunBudget(), FakeLLM([_plan()])) is None


def test_time_column_dropped_by_qwen_stays_for_folds(task):
    """Схема time:year, Qwen убрал year из признаков — фолды строятся по служебной колонке, в признаки она не идёт."""
    spec, profile, train, test, base, base_result = task
    base_t = base.model_copy(update={"params": {**base.params, "validation": "time:year"}})
    test = test.assign(year=5)
    base_res = run_experiment(spec, profile_data(spec, train, test), base_t, train, RunBudget())
    code = "def prepare(df):\n    return df[['a', 'b']]"
    llm = FakeLLM([_plan(code=code), STOP, {"choice": "qwen_data_1", "conclusion": "ок"}])
    outcome = de.run_data_engineer(spec, profile, train, test, base_t, base_res, RunBudget(), llm)
    chosen = outcome.chosen or outcome.attempts[0]
    assert chosen.result.status == ExperimentStatus.OK and chosen.result.validation == base_res.validation
    assert "year" in chosen.train.columns and "year" not in chosen.profile.numeric_features


def test_column_named_like_id_is_dropped(task):
    """Колонка с именем id в ответе prepare — не признак: id и target в train берутся из исходных данных."""
    spec, profile, train, test, base, base_result = task
    code = "def prepare(df):\n    out = df[['a', 'b']].copy()\n    out['id'] = 1.0\n    return out"
    attempt = de._round(1, de.parse_plan(_plan(code=code), spec)[0], spec, train, test, None, RunBudget())
    assert not attempt.error and attempt.train["id"].equals(train["id"]) and attempt.train["y"].equals(train["y"])


def test_code_touching_target_is_rejected(task):
    spec = task[0]
    plan, reason = de.parse_plan(_plan(code="def prepare(df):\n    df['y'] = 0\n    return df"), spec)
    assert plan is None and "target" in reason


def test_stop_ends_rounds_early(task):
    spec, profile, train, test, base, base_result = task
    llm = FakeLLM([_plan(), STOP, {"choice": "baseline", "conclusion": "бейзлайн надёжнее"}])
    outcome = de.run_data_engineer(spec, profile, train, test, base, base_result, RunBudget(), llm)
    assert len(outcome.attempts) == 1 and outcome.chosen is None and len(llm.calls) == 3


def test_prompt_has_task_columns_and_feedback(task):
    spec, profile, train, test, base, base_result = task
    failed = de.Attempt(round=1, plan="план", error="код упал: KeyError")
    messages = de.build_plan_messages(spec, profile, de.raw_features(spec, train, test), base_result, [failed])
    text = messages[-1]["content"]
    assert "Предсказать y" in text and "noise_text" in text and "KeyError" in text and "random_forest" in text
    assert "$task" not in text and "$columns" not in text and "$feedback" not in text


def test_plan_params_respect_task_type(task):
    spec = task[0]
    params = de.plan_params(
        {
            "model": "xgboost",
            "model_params": {"max_depth": 4},
            "log_target": True,
            "objective_metric": True,
            "class_weight": "balanced",
        },
        spec,
        "kfold",
    )
    assert params == {"model": "xgboost", "model_params": {"max_depth": 4}, "log_target": True, "validation": "kfold"}


def test_too_worse_rule():
    def res(mean, std=1.0):
        return ExperimentResult(
            hypothesis_id="h", status=ExperimentStatus.OK, metric_name="rmse", fold_scores=[mean] * 3,
            mean_score=mean, std_score=std, validation="kfold-3",
        )  # fmt: skip

    assert not de.too_worse(res(10.5), res(10.0), greater_is_better=False)  # хуже, но в пределах разброса
    assert de.too_worse(res(11.5), res(10.0), greater_is_better=False)
    assert not de.too_worse(res(0.85), res(0.80, 0.01), greater_is_better=True)
