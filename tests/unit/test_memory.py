"""Память экспериментов (A-L2-2): сжатая история, решения, выводы LLM и размер промпта."""

from pathlib import Path

import pytest

from agent import loop, prompting
from agent.llm_hypotheses import reflect
from agent.memory import MAX_LESSON_CHARS, MAX_LESSONS, ResearchMemory
from interfaces.schemas import (
    DataProfile,
    ExperimentResult,
    ExperimentStatus,
    Hypothesis,
    HypothesisKind,
    RunBudget,
)
from interfaces.stubs import stub_task_spec


@pytest.fixture
def spec():
    return stub_task_spec(Path("tasks/demo"))


@pytest.fixture
def profile():
    return DataProfile(n_train=500, n_test=200, numeric_features=["a"], categorical_features=[])


def step(i, score=None, status=ExperimentStatus.OK, error=None):
    hyp = Hypothesis(id=f"h{i}", kind=HypothesisKind.HYPERPARAMS, title="t", params={"n_folds": i})
    res = ExperimentResult(
        hypothesis_id=hyp.id, status=status, metric_name="rmse", mean_score=score, std_score=0.1, error=error
    )
    return hyp, res


class FakeLLM:
    def __init__(self, hyps=None, lesson="Регуляризация не помогла: данных много, пробуй ёмкость модели."):
        self.hyps, self.lesson = hyps, lesson
        self.json_calls, self.text_calls, self.rub_spent, self.calls = 0, 0, 0.0, 0
        self.last_messages = None

    def available(self):
        return True

    def chat_json(self, messages, **kwargs):
        self.json_calls += 1
        self.last_messages = messages
        return self.hyps

    def chat(self, messages, **kwargs):
        self.text_calls += 1
        return self.lesson


def test_memory_verdicts_and_lesson_limits():
    memory = ResearchMemory()
    memory.record("a", True)
    memory.record("b", False, "не лучше")
    assert memory.verdicts == {"a": "принята", "b": "отклонена: не лучше"}
    for i in range(MAX_LESSONS + 2):
        memory.add_lesson(f"вывод {i}\n" + "x" * 1000)
    memory.add_lesson(None)
    memory.add_lesson("   ")
    assert len(memory.lessons) == MAX_LESSONS
    assert memory.lessons[-1].startswith(f"вывод {MAX_LESSONS + 1} x")
    assert all(len(lesson) <= MAX_LESSON_CHARS for lesson in memory.lessons)


def test_history_shows_verdicts_and_best_for_both_directions():
    history = [step(1, 0.5), step(2, 0.9), step(3, status=ExperimentStatus.FAILED, error="упала")]
    memory = ResearchMemory()
    memory.record("h2", True)
    memory.record("h3", False, "failed")
    text = prompting.describe_history(history, greater_is_better=True, memory=memory)
    assert '| h2 | hyperparams | {"n_folds": 2} | 0.90000 ± 0.10000 (лучший) | принята |' in text
    assert "| h1 |" in text and "| — |" in text  # без решения — прочерк
    assert "failed: упала" in text and "отклонена: failed" in text
    low = prompting.describe_history(history, greater_is_better=False)
    assert "0.50000 ± 0.10000 (лучший)" in low


def test_prompt_size_bounded_on_long_run(spec, profile):
    history = [step(i, 40.0 + i % 7, error="e" * 500) for i in range(300)]
    memory = ResearchMemory()
    for i in range(300):
        memory.record(f"h{i}", False, "не лучше лучшего с учётом шума")
        memory.add_lesson("длинный вывод " * 50)
    messages = prompting.build_hypothesis_messages(spec, profile, history, k=3, memory=memory)
    size = sum(len(m["content"]) for m in messages)
    assert size < 8000, size


def test_lessons_in_hypothesis_prompt(spec, profile):
    memory = ResearchMemory()
    assert "Пока нет." in prompting.build_hypothesis_messages(spec, profile, [], memory=memory)[1]["content"]
    memory.add_lesson("Пробуй log_target")
    assert "- Пробуй log_target" in prompting.build_hypothesis_messages(spec, profile, [], memory=memory)[1]["content"]


def test_reflection_prompt_and_call(spec):
    memory = ResearchMemory()
    llm = FakeLLM()
    reflect(spec, step(0, 40.0), [step(1, 41.0), step(2, status=ExperimentStatus.TIMEOUT, error="время")], llm, memory)
    assert llm.text_calls == 1
    assert memory.lessons == ["Регуляризация не помогла: данных много, пробуй ёмкость модели."]
    messages = prompting.build_reflection_messages(spec, step(0, 40.0), [step(1, 41.0)])
    assert "$" not in messages[1]["content"] and "| h1 |" in messages[1]["content"]


@pytest.mark.parametrize("lesson", [None, "", RuntimeError("сбой")])
def test_reflection_failure_is_harmless(spec, lesson):
    memory = ResearchMemory()

    class Broken(FakeLLM):
        def chat(self, messages, **kwargs):
            if isinstance(lesson, Exception):
                raise lesson
            return lesson

    reflect(spec, step(0, 40.0), [step(1, 41.0)], Broken(), memory)
    assert memory.lessons == []


class FreshLLM(FakeLLM):
    """Каждый запрос — 3 новые гипотезы с уникальными params."""

    def chat_json(self, messages, **kwargs):
        super().chat_json(messages)
        n = self.json_calls * 10
        return [
            {"id": f"llm{n + i}", "kind": "hyperparams", "title": "t", "params": {"n_folds": n + i}} for i in range(3)
        ]


def test_loop_reflects_only_after_fully_failed_batch(spec, profile, monkeypatch):
    scores = iter([39.0, 41.0, 41.0, 41.0, 41.0, 41.0, 41.0])  # пачка 1: первая принята; пачка 2: все мимо

    def fake_experiment(spec, profile, hyp, train, budget):
        return ExperimentResult(
            hypothesis_id=hyp.id, status=ExperimentStatus.OK, metric_name="rmse", mean_score=next(scores), std_score=0.0
        )

    monkeypatch.setattr(loop, "run_experiment", fake_experiment)
    base = step(0, 40.0)
    history = [base]
    memory = ResearchMemory()
    llm = FreshLLM()
    best = loop.run_autoresearch(
        spec, profile, None, RunBudget(), history, base, lambda h, r: None, max_iterations=7, llm=llm, memory=memory
    )
    assert best[1].mean_score == 39.0
    assert llm.json_calls == 3 and llm.text_calls == 1  # рефлексия — только перед третьей пачкой
    assert list(memory.verdicts.values()).count("принята") == 1
    assert len(memory.verdicts) == 7
    assert "- Регуляризация не помогла" in llm.last_messages[1]["content"]  # вывод попал в промпт
