"""Итог решений LLM в agent.log: что предложила LLM и что из этого вышло (агентность видна в логе)."""

from agent.loop import llm_decisions
from agent.memory import ResearchMemory
from interfaces.schemas import ExperimentResult, ExperimentStatus, Hypothesis, HypothesisKind


def step(hid, source, status=ExperimentStatus.OK):
    hyp = Hypothesis(id=hid, kind=HypothesisKind.HYPERPARAMS, title=hid, source=source)
    return hyp, ExperimentResult(hypothesis_id=hid, status=status, metric_name="rmse", mean_score=1.0)


def test_counts_only_llm_hypotheses():
    memory = ResearchMemory()
    history = [
        step("baseline", "baseline"),
        step("pb", "playbook"),
        step("l1", "llm"),
        step("l2", "llm"),
        step("l3", "llm", ExperimentStatus.SKIPPED),
        step("l4", "llm", ExperimentStatus.FAILED),
        step("l5", "llm", ExperimentStatus.TIMEOUT),
    ]
    memory.record("pb", True, "t=3")
    memory.record("l1", True, "t=5.18, 5/5 фолдов")
    memory.record("l2", False, "t=0.4")
    memory.record("l3", False, "не успеет")
    line = llm_decisions(history, memory)
    assert line == "Решения LLM: предложено 5 → принято 1 (l1), не лучше 1, пропущено критиком 1, упало 2"


def test_no_llm_hypotheses():
    line = llm_decisions([step("baseline", "baseline"), step("pb", "playbook")], ResearchMemory())
    assert "гипотез от LLM не было" in line
