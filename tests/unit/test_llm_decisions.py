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
    line = llm_decisions(history, memory).splitlines()[0]
    assert line == "Решения LLM: предложено 5 → принято 1 (l1), не лучше 1, пропущено критиком 1, упало 2"


def test_no_llm_hypotheses():
    line = llm_decisions([step("baseline", "baseline"), step("pb", "playbook")], ResearchMemory())
    assert "гипотез от LLM не было" in line


def test_each_llm_hypothesis_has_its_outcome():
    """В agent.log по каждой гипотезе LLM видно, чем кончилось: принята / не лучше / пропущена / упала (#200, п. 4)."""
    memory = ResearchMemory()
    history = [step("l1", "llm"), step("l2", "llm"), step("l3", "llm", ExperimentStatus.SKIPPED)]
    history.append(
        (
            Hypothesis(id="l4", kind=HypothesisKind.FEATURES, title="код признаков", source="llm"),
            ExperimentResult(hypothesis_id="l4", status=ExperimentStatus.FAILED, metric_name="rmse", error="KeyError"),
        )
    )
    memory.record("l1", True, "t=5.18, 5/5 фолдов")
    memory.record("l2", False, "t=0.4")
    memory.record("l4", False, "failed")
    history[2] = (history[2][0], history[2][1].model_copy(update={"error": "не успеет: оценка 900 с"}))
    memory.llm_rejected = ["bad: неизвестные params=['n_estimators_x']"]
    lines = llm_decisions(history, memory).splitlines()
    assert lines[0].endswith("упало 1; отброшено при разборе ответа 1")
    assert lines[1] == "  l1 «l1» → принята: t=5.18, 5/5 фолдов"
    assert lines[2] == "  l2 «l2» → не лучше: t=0.4"
    assert lines[3] == "  l3 «l3» → пропущена: не успеет: оценка 900 с"
    assert lines[4] == "  l4 «код признаков» → упала (failed): KeyError"
    assert lines[5] == "  bad: неизвестные params=['n_estimators_x'] → отброшена при разборе"


def test_rejected_only():
    memory = ResearchMemory(llm_rejected=["x: повтор проверенных params"])
    assert "отброшено при разборе ответа 1" in llm_decisions([step("baseline", "baseline")], memory)
