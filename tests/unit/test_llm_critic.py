"""LLM-критик (A-L3-2): проверка пачки гипотез до запуска; при любом сбое гипотезы проходят."""

from pathlib import Path

import pytest

from agent import loop, prompting
from agent.llm_critic import CRITIC_ENV, candidates_json, critique
from agent.memory import ResearchMemory
from interfaces.schemas import (
    DataProfile,
    ExperimentResult,
    ExperimentStatus,
    Hypothesis,
    HypothesisKind,
    RunBudget,
)
from interfaces.stubs import stub_task_spec


def hyp(hid, source="llm", params=None):
    return Hypothesis(
        id=hid, kind=HypothesisKind.HYPERPARAMS, title=f"t {hid}", params=params or {"n_folds": 5}, source=source
    )


class FakeLLM:
    def __init__(self, answer, available=True):
        self.answer, self._available = answer, available
        self.requests, self.rub_spent, self.calls = 0, 0.0, 0
        self.last_messages = None

    def available(self):
        return self._available

    def chat_json(self, messages, **kwargs):
        self.requests += 1
        self.last_messages = messages
        if isinstance(self.answer, Exception):
            raise self.answer
        return self.answer

    def chat(self, messages, **kwargs):
        return None


@pytest.fixture
def spec():
    return stub_task_spec(Path("tasks/demo"))


@pytest.fixture
def profile():
    return DataProfile(n_train=500, n_test=200, numeric_features=["a"], categorical_features=[])


@pytest.fixture
def on(monkeypatch):
    monkeypatch.setenv(CRITIC_ENV, "1")


REJECT_B = [{"id": "a", "verdict": "ok", "reason": ""}, {"id": "b", "verdict": "reject", "reason": "3: log1p при RMSE"}]


def test_disabled_by_default(spec, profile, monkeypatch):
    monkeypatch.delenv(CRITIC_ENV, raising=False)
    llm = FakeLLM(REJECT_B)
    assert critique(spec, profile, [], [hyp("a"), hyp("b")], llm, RunBudget()) == {}
    assert llm.requests == 0


def test_rejects_with_reason(spec, profile, on):
    llm = FakeLLM(REJECT_B)
    assert critique(spec, profile, [], [hyp("a"), hyp("b")], llm, RunBudget()) == {"b": "3: log1p при RMSE"}
    assert llm.requests == 1


def test_prompt_contains_candidates_and_checklist(spec, profile, on):
    llm = FakeLLM([])
    critique(spec, profile, [], [hyp("a", params={"log_target": True})], llm, RunBudget())
    user = llm.last_messages[1]["content"]
    assert "$" not in user
    assert '"id": "a"' in user and '"log_target": true' in user
    assert "Утечка" in user and "Метрика" in user


@pytest.mark.parametrize(
    "answer",
    [
        None,
        "мусор",
        {"id": "a"},
        [{"id": "zzz", "verdict": "reject"}],
        [{"id": "a", "verdict": "maybe"}],
        RuntimeError(),
    ],
)
def test_garbage_lets_everything_pass(spec, profile, on, answer):
    assert critique(spec, profile, [], [hyp("a")], FakeLLM(answer), RunBudget()) == {}


def test_playbook_hypotheses_not_sent(spec, profile, on):
    llm = FakeLLM([{"id": "p", "verdict": "reject", "reason": "x"}])
    assert critique(spec, profile, [], [hyp("p", source="playbook")], llm, RunBudget()) == {}
    assert llm.requests == 0


def test_skipped_when_llm_unavailable_or_no_time(spec, profile, on):
    llm = FakeLLM(REJECT_B, available=False)
    assert critique(spec, profile, [], [hyp("b")], llm, RunBudget()) == {}
    llm = FakeLLM(REJECT_B)
    assert critique(spec, profile, [], [hyp("b")], llm, RunBudget(total_seconds=100, safety_seconds=0)) == {}
    assert llm.requests == 0


def test_candidates_one_per_line():
    assert candidates_json([hyp("a")]).startswith("[\n  {")


def test_loop_skips_rejected_without_running(spec, profile, on, monkeypatch):
    ran = []

    def fake_experiment(spec, profile, h, train, budget):
        ran.append(h.id)
        return ExperimentResult(
            hypothesis_id=h.id, status=ExperimentStatus.OK, metric_name="rmse", mean_score=50.0, std_score=0.0
        )

    monkeypatch.setattr(loop, "run_experiment", fake_experiment)
    batches = [[hyp("a"), hyp("b", params={"n_folds": 6})]]
    monkeypatch.setattr(loop, "propose_hypotheses", lambda *a, **kw: batches.pop(0) if batches else [])
    base = (
        Hypothesis(id="baseline", kind=HypothesisKind.BASELINE, title="b", params={"model": "lightgbm"}),
        ExperimentResult(
            hypothesis_id="baseline", status=ExperimentStatus.OK, metric_name="rmse", mean_score=40.0, std_score=0.1
        ),
    )
    history, memory = [base], ResearchMemory()
    loop.run_autoresearch(
        spec, profile, None, RunBudget(), history, base, lambda h, r: None, llm=FakeLLM(REJECT_B), memory=memory
    )
    assert ran == ["a"]
    skipped = {h.id: r for h, r in history if r.status == ExperimentStatus.SKIPPED}
    assert "критик LLM: 3: log1p при RMSE" in skipped["b"].error
    assert memory.verdicts["b"].startswith("отклонена: критик LLM")
    # причина попадает в историю, которую видит генератор
    text = prompting.describe_history(history, False, memory)
    assert "критик LLM" in text
