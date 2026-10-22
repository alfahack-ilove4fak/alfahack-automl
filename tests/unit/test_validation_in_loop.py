"""Схема валидации одна на весь прогон (#151): цикл передаёт её каждой гипотезе, повторы не зависят от неё."""

from pathlib import Path

import pytest

from agent import loop
from agent.hypotheses import params_key
from interfaces.schemas import (
    DataProfile,
    ExperimentResult,
    ExperimentStatus,
    Hypothesis,
    HypothesisKind,
    RunBudget,
)
from interfaces.stubs import stub_task_spec


def _hyp(hyp_id: str, **params) -> Hypothesis:
    return Hypothesis(id=hyp_id, kind=HypothesisKind.HYPERPARAMS, title=hyp_id, params={"model": "lightgbm", **params})


def _result(hyp_id: str, score: float, validation: str) -> ExperimentResult:
    return ExperimentResult(
        hypothesis_id=hyp_id,
        status=ExperimentStatus.OK,
        metric_name="rmse",
        mean_score=score,
        std_score=0.1,
        fold_scores=[score] * 5,
        validation=validation,
    )


@pytest.fixture
def run_loop(monkeypatch):
    """Запускает цикл с одной гипотезой и возвращает params, с которыми она ушла в run_experiment."""

    def go(base_params: dict, hyp: Hypothesis) -> dict:
        seen = {}
        queue = [hyp]

        def fake_propose(spec, profile, history, k=3, budget=None, llm=None, memory=None):
            out, queue[:] = list(queue), []
            return out

        def fake_run(spec, profile, h, train, budget):
            seen.update(h.params)
            return _result(h.id, 2.0, base_validation)

        base_validation = "time:Year-4"
        monkeypatch.setattr(loop, "propose_hypotheses", fake_propose)
        monkeypatch.setattr(loop, "run_experiment", fake_run)
        spec = stub_task_spec(Path("tasks/demo"))
        profile = DataProfile(n_train=500, n_test=200, numeric_features=["a"], categorical_features=[])
        base = Hypothesis(id="baseline", kind=HypothesisKind.BASELINE, title="база", params=base_params)
        history = [(base, _result("baseline", 1.0, base_validation))]
        loop.run_autoresearch(spec, profile, None, RunBudget(), history, history[0], lambda h, r: None, llm=None)
        assert seen, "гипотеза не дошла до run_experiment (отклонена критиком?)"
        return seen

    return go


def test_hypothesis_gets_validation_of_best(run_loop):
    seen = run_loop({"model": "lightgbm", "validation": "time:Year"}, _hyp("h1", model_params={"reg_lambda": 5.0}))
    assert seen["validation"] == "time:Year"
    assert seen["model_params"] == {"reg_lambda": 5.0}  # остальное не тронуто


def test_own_validation_is_not_overwritten(run_loop):
    seen = run_loop({"model": "lightgbm", "validation": "time:Year"}, _hyp("h1", validation="kfold"))
    assert seen["validation"] == "kfold"  # accept() сам отклонит сравнение разных схем


def test_no_validation_in_best_leaves_hypothesis_alone(run_loop):
    seen = run_loop({"model": "lightgbm"}, _hyp("h1", model_params={"reg_lambda": 5.0}))
    assert "validation" not in seen


def test_validation_is_not_part_of_repeat_key():
    # в истории у всех гипотез есть validation, у новых кандидатов playbook/LLM её нет — повтор должен находиться
    assert params_key({"model": "lightgbm", "validation": "time:Year"}) == params_key({"model": "lightgbm"})
    assert params_key({"model": "lightgbm", "class_weight": "balanced"}) != params_key({"model": "lightgbm"})
