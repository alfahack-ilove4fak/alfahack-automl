"""#276: при сдвиге train → test (AUC в DataProfile.warnings, #294) множители классов не подбираются."""

from pathlib import Path

import pytest

from agent import hypotheses
from agent.hypotheses import baseline_hypothesis, propose_hypotheses, shift_auc, wants_proba_weights, with_task_defaults
from interfaces.schemas import DataProfile, MetricSpec, TaskType
from interfaces.stubs import stub_task_spec


def spec_for(task_type=TaskType.MULTICLASS, metric="f1_macro"):
    return stub_task_spec(Path("tasks/demo")).model_copy(
        update={"task_type": task_type, "metric": MetricSpec(name=metric, greater_is_better=True)}
    )


def profile(*warnings):
    return DataProfile(n_train=100, n_test=50, numeric_features=["a"], categorical_features=[], warnings=list(warnings))


SHIFTED = profile("пропуски в a", "adversarial_auc=0.88; сдвиг train→test, сильнее всего: ct_srv_src")
SIMILAR = profile("adversarial_auc=0.52; сдвига нет")


@pytest.fixture(autouse=True)
def default_limit(monkeypatch):
    monkeypatch.delenv(hypotheses.PROBA_SHIFT_ENV, raising=False)


def test_shift_auc_parsed_from_warnings():
    assert shift_auc(SHIFTED) == pytest.approx(0.88)
    assert shift_auc(SIMILAR) == pytest.approx(0.52)
    assert shift_auc(profile("пропуски в a")) is None
    assert shift_auc(profile("сдвиг: adversarial_auc=0.99")) is None  # только в начале строки
    assert shift_auc(None) is None


def test_no_weights_under_shift():
    spec = spec_for()
    assert not wants_proba_weights(spec, SHIFTED)
    assert baseline_hypothesis(spec, SHIFTED).params == {"model": "lightgbm"}
    assert with_task_defaults({"model": "xgboost"}, spec, SHIFTED) == {"model": "xgboost"}


@pytest.mark.parametrize("prof", [SIMILAR, profile("пропуски в a"), None])
def test_weights_kept_without_shift(prof):
    # нет сдвига или ml/ ещё не пишет AUC — поведение как до правила
    spec = spec_for()
    assert wants_proba_weights(spec, prof)
    assert baseline_hypothesis(spec, prof).params == {"model": "lightgbm", "proba_weights": True}
    assert with_task_defaults({"model": "xgboost"}, spec, prof)["proba_weights"] is True


def test_explicit_value_not_overridden():
    assert with_task_defaults({"proba_weights": True}, spec_for(), SHIFTED) == {"proba_weights": True}


@pytest.mark.parametrize("value, expected", [("0", True), ("0.9", True), ("0.85", False), ("мусор", False)])
def test_limit_from_env(monkeypatch, value, expected):
    # 0 — проверка выключена; порог выше AUC — множители остаются; мусор — порог по умолчанию 0.8
    monkeypatch.setenv(hypotheses.PROBA_SHIFT_ENV, value)
    assert wants_proba_weights(spec_for(), SHIFTED) is expected


def test_regression_untouched():
    assert not wants_proba_weights(spec_for(TaskType.REGRESSION, "rmse"), SIMILAR)


def test_playbook_hypotheses_follow_baseline():
    # гипотезы playbook сравниваются с бейзлайном без множителей — и сами без них
    spec = spec_for()
    for prof, expected in [(SHIFTED, False), (SIMILAR, True)]:
        hyps = propose_hypotheses(spec, prof, [], k=3, llm=None)
        assert hyps and all(bool(h.params.get("proba_weights")) is expected for h in hyps)
