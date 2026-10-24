"""Гипотезы от LLM: разбор ответа, фильтры, запасной playbook и живучесть цикла. Настоящая LLM не вызывается."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from agent import loop
from agent.hypotheses import canonical_params, load_playbook, params_key, propose_hypotheses
from agent.llm_client import LLMClient
from agent.llm_hypotheses import parse_hypotheses
from interfaces.schemas import (
    DataProfile,
    ExperimentResult,
    ExperimentStatus,
    Hypothesis,
    HypothesisKind,
    RunBudget,
    TaskType,
)
from interfaces.stubs import stub_task_spec

GOOD = {
    "id": "lgbm_l2",
    "kind": "hyperparams",
    "title": "Сильнее L2-регуляризация",
    "rationale": "Мало строк — модель переобучается",
    "params": {"model": "lightgbm", "model_params": {"reg_lambda": 5.0}},
}


class FakeLLM:
    """Подмена LLMClient: chat_json отдаёт заранее заданный ответ."""

    def __init__(self, answer, available=True):
        self.answer, self._available = answer, available
        self.requests, self.rub_spent, self.calls = 0, 0.0, 0
        self.kwargs = None

    def available(self):
        return self._available

    def log_summary(self):
        pass

    def chat_json(self, messages, **kwargs):
        self.requests += 1
        self.kwargs = kwargs
        if isinstance(self.answer, Exception):
            raise self.answer
        return self.answer


@pytest.fixture
def spec():
    return stub_task_spec(Path("tasks/demo"))


@pytest.fixture
def profile():
    return DataProfile(n_train=500, n_test=200, numeric_features=["a", "b"], categorical_features=[])


def baseline_history():
    hyp = Hypothesis(id="baseline", kind=HypothesisKind.BASELINE, title="база", params={"model": "lightgbm"})
    res = ExperimentResult(hypothesis_id="baseline", status=ExperimentStatus.OK, metric_name="rmse", mean_score=1.0)
    return [(hyp, res)]


def playbook_ids(spec, profile, k):
    return [h.id for h in propose_hypotheses(spec, profile, baseline_history(), k=k)]


def test_good_answer_goes_first(spec, profile, monkeypatch):
    monkeypatch.setenv("AGENT_LLM_FIRST", "1")  # механика LLM-части; порядок ADR 0014 — в test_playbook_first
    out = propose_hypotheses(spec, profile, baseline_history(), k=3, llm=FakeLLM([GOOD]))
    assert out[0].id == "lgbm_l2" and out[0].source == "llm"
    assert [h.source for h in out[1:]] == ["playbook", "playbook"]  # недостающие — из playbook


@pytest.mark.parametrize("answer", [None, "текст", 42, {"foo": "bar"}, [], [1, "x", None], RuntimeError("сбой")])
def test_garbage_falls_back_to_playbook(spec, profile, answer):
    out = propose_hypotheses(spec, profile, baseline_history(), k=3, llm=FakeLLM(answer))
    assert [h.id for h in out] == playbook_ids(spec, profile, 3)
    assert all(h.source == "playbook" for h in out)


def test_unavailable_llm_not_called(spec, profile):
    llm = FakeLLM([GOOD], available=False)
    assert [h.id for h in propose_hypotheses(spec, profile, baseline_history(), k=2, llm=llm)] == playbook_ids(
        spec, profile, 2
    )
    assert llm.requests == 0


def test_wrapped_in_object(spec, profile):
    out = parse_hypotheses({"hypotheses": [GOOD]}, baseline_history(), k=3)
    assert [h.id for h in out] == ["lgbm_l2"]


@pytest.mark.parametrize(
    "bad",
    [
        {**GOOD, "kind": "magic"},  # неизвестный kind
        {**GOOD, "kind": "features"},  # нужен код и песочница — пока нельзя
        {**GOOD, "params": {"model": "lightgbm", "use_test_target": True}},  # неизвестный ключ params
        {**GOOD, "params": {"model_params": "много деревьев"}},  # model_params не словарь
        {k: v for k, v in GOOD.items() if k != "title"},  # нет обязательного поля
        {**GOOD, "params": {"model": "lightgbm"}},  # те же params, что у бейзлайна
    ],
)
def test_invalid_items_dropped(bad):
    out = parse_hypotheses([bad, GOOD], baseline_history(), k=3)
    assert [h.params for h in out] == [GOOD["params"]]


def test_extra_fields_and_code_ignored():
    out = parse_hypotheses([{**GOOD, "code": "import os", "confidence": 0.9}], baseline_history(), k=3)
    assert out[0].code is None


def test_duplicate_id_renamed_and_ids_sanitized():
    history = baseline_history() + [(Hypothesis(id="lgbm_l2", kind="hyperparams", title="старая"), None)]
    out = parse_hypotheses([GOOD, {**GOOD, "id": "новая идея!", "params": {"n_folds": 10}}], history, k=3)
    assert out[0].id == "lgbm_l2_2"
    assert out[1].id == "llm"  # id без латиницы → запасное имя


def test_duplicates_inside_answer_and_k_limit():
    second = {**GOOD, "id": "other", "params": {"log_target": True}}
    out = parse_hypotheses([GOOD, {**GOOD, "id": "copy"}, second, {**second, "id": "x", "params": {}}], [], k=2)
    assert [h.id for h in out] == ["lgbm_l2", "other"]


def test_playbook_skips_params_already_proposed_by_llm(spec, profile):
    first = load_playbook()[0]
    same = {**GOOD, "id": "llm_copy", "kind": first["kind"], "params": first["params"]}
    out = propose_hypotheses(spec, profile, baseline_history(), k=5, llm=FakeLLM([same]))
    assert first["id"] not in [h.id for h in out]


def test_real_client_json_in_markdown(spec, profile, monkeypatch):
    monkeypatch.setenv("AGENT_LLM_FIRST", "1")  # механика LLM-части; порядок ADR 0014 — в test_playbook_first
    monkeypatch.setenv("LLM_MOCK", "1")
    client = LLMClient()
    monkeypatch.setattr(client, "available", lambda: True)
    monkeypatch.setattr(
        client,
        "chat",
        lambda messages, **kw: '```json\n[{"id": "a", "kind": "hyperparams", '
        '"title": "t", "params": {"n_folds": 7}}]\n```',
    )
    out = propose_hypotheses(spec, profile, baseline_history(), k=1, llm=client)
    assert [(h.id, h.source) for h in out] == [("a", "llm")]


@pytest.mark.parametrize("answer", [[GOOD], None, "мусор"])
def test_loop_survives_any_llm_answer(spec, profile, monkeypatch, answer):
    monkeypatch.setenv("AGENT_LLM_FIRST", "1")  # механика LLM-части; порядок ADR 0014 — в test_playbook_first
    monkeypatch.setenv("AGENT_FEATURE_CODE", "0")  # считаем только запросы гипотез и рефлексии

    def fake_experiment(spec, profile, hyp, train, budget):
        return ExperimentResult(
            hypothesis_id=hyp.id, status=ExperimentStatus.OK, metric_name="rmse", mean_score=1.0, std_score=0.1
        )

    monkeypatch.setattr(loop, "run_experiment", fake_experiment)
    history = baseline_history()
    llm = FakeLLM(answer)
    loop.run_autoresearch(
        spec,
        profile,
        None,
        RunBudget(),
        history,
        history[0],
        on_new_best=lambda h, r: None,
        max_iterations=4,
        llm=llm,
    )
    assert len(history) == 5
    assert len({h.id for h, _ in history}) == 5  # без повторов
    assert llm.requests == 2  # 4 итерации по 3 гипотезы за запрос


def test_budget_positional_as_in_protocol(spec, profile, monkeypatch):
    monkeypatch.setenv("AGENT_LLM_FIRST", "1")  # механика LLM-части; порядок ADR 0014 — в test_playbook_first
    # контракт 1.1 (D3): propose_hypotheses(spec, profile, history, k, budget)
    llm = FakeLLM([GOOD])
    budget = RunBudget(total_seconds=1000, safety_seconds=0)
    out = propose_hypotheses(spec, profile, baseline_history(), 3, budget, llm=llm)
    assert out[0].id == "lgbm_l2"
    assert 990 < llm.kwargs["seconds_left"] <= 1000  # остаток бюджета дошёл до LLM-клиента


def test_no_time_left_playbook_only(spec, profile, monkeypatch):
    def no_request(**kwargs):
        raise AssertionError("при нехватке времени запроса к LLM быть не должно")

    monkeypatch.setenv("LLM_MOCK", "1")
    monkeypatch.delenv("LLM_CACHE_DIR", raising=False)
    client = LLMClient()
    client._client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=no_request)))
    budget = RunBudget(total_seconds=60, safety_seconds=0)
    out = propose_hypotheses(spec, profile, baseline_history(), k=2, budget=budget, llm=client)
    assert [h.source for h in out] == ["playbook", "playbook"]


def test_loop_passes_budget(spec, profile, monkeypatch):
    seen = []

    def fake_propose(spec, profile, history, k=3, budget=None, llm=None, memory=None, llm_first=None):
        seen.append(budget)
        return []

    monkeypatch.setattr(loop, "propose_hypotheses", fake_propose)
    budget = RunBudget()
    history = baseline_history()
    loop.run_autoresearch(spec, profile, None, budget, history, history[0], lambda h, r: None, llm=FakeLLM(None))
    assert seen == [budget]


def make_hyp(hid):
    return Hypothesis(id=hid, kind=HypothesisKind.HYPERPARAMS, title=hid, params={"n_folds": 3}, source="playbook")


@pytest.mark.parametrize(
    "share, available, expected",
    [
        (
            0.0,
            True,
            [True, None],
        ),  # середина времени прошла, LLM не ходила — один раз вперёд playbook, дальше как обычно
        (0.5, True, [None, None]),  # время ещё есть — обычный порядок ADR 0014
        (0.0, False, [None, None]),  # LLM выключена — звать некого
    ],
)
def test_llm_gets_a_turn_by_half_time(spec, profile, monkeypatch, share, available, expected):
    """Ревью #257: решения должна принимать LLM (вебинар) — на больших данных playbook мог съесть весь бюджет."""
    seen = []
    batches = [[make_hyp("p1")], [make_hyp("p2")]]

    def fake_propose(spec, profile, history, k=3, budget=None, llm=None, memory=None, llm_first=None):
        seen.append(llm_first)
        return batches.pop(0) if batches else []

    def fake_experiment(spec, profile, h, train, budget):
        return ExperimentResult(hypothesis_id=h.id, status=ExperimentStatus.OK, metric_name="rmse", mean_score=50.0)

    monkeypatch.setattr(loop, "LLM_TURN_SHARE", share)
    monkeypatch.setattr(loop, "propose_hypotheses", fake_propose)
    monkeypatch.setattr(loop, "run_experiment", fake_experiment)
    monkeypatch.setattr(loop, "fallback_hypotheses", lambda *a, **kw: [])
    history = baseline_history()
    llm = FakeLLM(None, available=available)
    loop.run_autoresearch(spec, profile, None, RunBudget(), history, history[0], lambda h, r: None, llm=llm)
    assert seen[:2] == expected


@pytest.mark.parametrize(
    "params",
    [
        {"class_weight": "balanced"},
        {"model": "lightgbm", "class_weight": "balanced"},
        {"model": "LightGBM ", "class_weight": "Balanced", "n_folds": 5},
        {"model": "lightgbm", "class_weight": "balanced", "log_target": False, "feature_ops": [], "model_params": {}},
        {"model": "lightgbm", "class_weight": "balanced", "model_params": {"learning_rate": 0.05, "num_leaves": 63.0}},
    ],
)
def test_same_experiment_written_differently_is_a_repeat(params):
    # замер 22.10: LLM предложила balanced-веса, а playbook прогнал balanced_classes ещё раз с тем же скором
    assert params_key(params) == params_key({"model": "lightgbm", "class_weight": "balanced"})


@pytest.mark.parametrize(
    "params",
    [
        {"model": "lightgbm"},
        {"model": "catboost", "class_weight": "balanced"},
        {"class_weight": "balanced", "n_folds": 10},
        {"class_weight": "balanced", "model_params": {"learning_rate": 0.02}},
        {"class_weight": "balanced", "proba_weights": True},
        {"class_weight": "balanced", "seed": 0},  # число 0 — не «выключенный» ключ
        {"class_weight": "balanced", "min_child_samples": 0.0},
        {"class_weight": "balanced", "validation": "time:Year"},
    ],
)
def test_really_different_experiments_differ(params):
    assert params_key(params) != params_key({"model": "lightgbm", "class_weight": "balanced"})


def test_baseline_equivalent_is_a_repeat():
    # «LightGBM с параметрами по умолчанию» = бейзлайн: не тратим на него эксперимент
    assert canonical_params({"model": "lightgbm", "model_params": {"n_estimators": 3000}}) == {}


def test_playbook_skips_llm_repeat_written_differently(spec, profile, monkeypatch):
    monkeypatch.setenv("AGENT_LLM_FIRST", "1")  # механика LLM-части; порядок ADR 0014 — в test_playbook_first
    spec = spec.model_copy(update={"task_type": TaskType.MULTICLASS})
    profile = profile.model_copy(update={"target_summary": {"imbalance_ratio": 10}})
    same = {**GOOD, "id": "llm_balanced", "params": {"model": "lightgbm", "class_weight": "balanced", "n_folds": 5}}
    out = propose_hypotheses(spec, profile, baseline_history(), k=5, llm=FakeLLM([same]))
    assert "llm_balanced" in [h.id for h in out]
    assert "balanced_classes" not in [h.id for h in out]


def test_zero_is_not_switched_off():
    # ревью #164: 0 == False в Python — числовой ноль не должен пропадать как «выключенный ключ»
    assert params_key({"seed": 0}) != params_key({})
    assert canonical_params({"min_child_samples": 0.0, "x": 1}) == {"min_child_samples": 0, "x": 1}


def test_column_names_keep_case():
    # ревью #164: в validation и feature_ops — имена колонок, «Year» и «year» могут быть разными
    assert params_key({"validation": "time:Year"}) != params_key({"validation": "time:year"})
    assert params_key({"model": "CatBoost"}) == params_key({"model": "catboost"})


def test_zero_seed_survives_with_off_keys():
    # #169: 0 остаётся, а выключенный log_target=False пропадает
    assert canonical_params({"seed": 0, "log_target": False}) == {"seed": 0}


def test_validation_column_name_is_stripped_not_lowered():
    assert params_key({"validation": " time:Year "}) == params_key({"validation": "time:Year"})


# ---------------------------------------------------------------- ADR 0014: сначала playbook, LLM — когда он исчерпан


def test_playbook_first_llm_not_asked_while_playbook_has_hypotheses(spec, profile):
    llm = FakeLLM([GOOD])
    out = propose_hypotheses(spec, profile, baseline_history(), k=3, llm=llm)
    assert [h.id for h in out] == playbook_ids(spec, profile, 3) and llm.requests == 0


def test_llm_fills_when_playbook_exhausted(spec, profile, monkeypatch):
    from agent import hypotheses

    first = playbook_ids(spec, profile, 1)[0]
    items = [item for item in load_playbook() if item["id"] == first]
    monkeypatch.setattr(hypotheses, "load_playbook", lambda: items)  # в playbook осталась одна применимая гипотеза
    llm = FakeLLM([GOOD])
    out = propose_hypotheses(spec, profile, baseline_history(), k=3, llm=llm)
    assert [h.source for h in out] == ["playbook", "llm"] and llm.requests == 1


def test_llm_repeat_of_playbook_in_same_batch_dropped(spec, profile, monkeypatch):
    from agent import hypotheses

    item = {"id": "pb", "kind": "hyperparams", "title": "t", "params": GOOD["params"]}
    monkeypatch.setattr(hypotheses, "load_playbook", lambda: [item])
    out = propose_hypotheses(spec, profile, baseline_history(), k=3, llm=FakeLLM([GOOD]))
    assert [h.id for h in out] == ["pb"]  # LLM повторила гипотезу playbook той же пачки — не берём


def test_rejected_reasons_collected():
    """Отброшенные при разборе гипотезы LLM — с причиной: они попадают в итог решений LLM в agent.log."""
    bad_key = {**GOOD, "id": "bad_key", "params": {"model": "lightgbm", "секретный_ключ": 1}}
    no_schema = {"id": "no_schema", "kind": "hyperparams"}
    rejected: list[str] = []
    out = parse_hypotheses([GOOD, bad_key, no_schema, "строка"], [], k=3, rejected=rejected)
    assert [h.id for h in out] == ["lgbm_l2"]
    assert rejected[0].startswith("bad_key: kind=hyperparams, неизвестные params=['секретный_ключ']")
    assert rejected[1:] == ["no_schema: не по схеме", "?: не объект JSON"]
