"""Шаг 0 до LLM-аналитика (#239): тривиальный ответ пишется по разбору без LLM, аналитик — вторым разбором.

Настоящая LLM не вызывается. Главное, что проверяем: к моменту запроса к аналитику `submission.csv` уже на диске,
а любой сбой аналитика оставляет ответ и разбор по данным.
"""

import logging
import time

import numpy as np
import pandas as pd
import pytest

import solution
from agent import task_analyst
from agent.task_analyst import ANALYST_ENV
from agent.task_parser import parse_task
from ml.profiling import load_data
from runtime.budget import make_budget, plan_stages
from solution import AnswerKeeper, refine_with_analyst, run_pipeline, spec_changed, trivial_answer

log = logging.getLogger("test_step0")


class FakeLLM:
    """Подмена LLMClient: chat_json отдаёт заранее заданный ответ."""

    def __init__(self, answer):
        self.answer = answer
        self.requests = 0

    def available(self):
        return True

    def chat_json(self, messages, **kwargs):
        self.requests += 1
        return self.answer


def make_task(folder, test_cols=("id", "a", "b"), train_extra=(), n=40, n_test=10):
    """Маленькая задача без sample_submission и без метрики в тексте: id и метрику может уточнить аналитик."""
    rng = np.random.default_rng(0)
    train = pd.DataFrame({"id": range(n), "a": rng.normal(size=n), "b": rng.normal(size=n)})
    train["y"] = train["a"] * 2 + rng.normal(scale=0.1, size=n)
    if "id" not in test_cols:
        train = train.drop(columns="id")  # иначе id — вторая колонка «только в train» и target неоднозначен
    for name in train_extra:
        train[name] = rng.normal(size=n)
    test = pd.DataFrame({"id": range(n, n + n_test), "a": rng.normal(size=n_test), "b": rng.normal(size=n_test)})
    test["row_id"] = test["id"]
    train.to_csv(folder / "train.csv", index=False)
    test[list(test_cols)].to_csv(folder / "test.csv", index=False)
    (folder / "task.md").write_text("Предскажи y.", encoding="utf-8")
    return folder / "task.md"


@pytest.fixture(autouse=True)
def no_real_llm(monkeypatch):
    for var in ("OPENAI_API_KEY", "OPENAI_MODEL", "OPENAI_BASE_URL", "LLM_MOCK", ANALYST_ENV, "AGENT_ANALYST_APPLY"):
        monkeypatch.delenv(var, raising=False)


def ready_keeper(task_md):
    """Состояние `run_pipeline` после шага 0 по разбору без LLM."""
    spec = parse_task(task_md, use_analyst=False)
    train, test = load_data(spec)
    keeper = AnswerKeeper(task_md)
    keeper.spec, keeper.test = spec, test
    assert keeper.write(trivial_answer(spec, train, test)) == []
    return keeper, spec, train, test


def test_parse_task_without_analyst_never_touches_llm(tmp_path, monkeypatch):
    """use_analyst=False: клиент LLM не создаётся и аналитик не зовётся."""

    def boom(*args, **kwargs):
        raise AssertionError("LLM не должна вызываться при use_analyst=False")

    monkeypatch.setattr(task_analyst, "ask_llm", boom)
    monkeypatch.setattr("agent.llm_client.LLMClient", boom)
    spec = parse_task(make_task(tmp_path), use_analyst=False)
    assert spec.target == "y"
    assert spec.id_column == "id"


def test_step0_is_on_disk_before_analyst_is_asked(tmp_path, monkeypatch):
    """Главное: пока аналитик «думает» (медленная LLM), тривиальный ответ уже записан."""
    task_md = make_task(tmp_path)
    seen = {}

    def slow_analyst(text, train_cols, test_cols, sample, llm):
        seen["answer_on_disk"] = (tmp_path / "submission.csv").exists()
        raise TimeoutError("медленная LLM")

    class Stop(Exception):
        pass

    def stop_after_refine(*args, **kwargs):
        raise Stop

    monkeypatch.setattr(task_analyst, "ask_llm", slow_analyst)
    monkeypatch.setattr(solution, "profile_data", stop_after_refine)  # дальше шага 0 и аналитика не идём
    started = time.time()
    budget = make_budget(started)
    with pytest.raises(Stop):
        run_pipeline(task_md, budget, plan_stages(budget), AnswerKeeper(task_md), started)
    assert seen["answer_on_disk"] is True


def test_ambiguous_target_falls_back_to_analyst_parse(tmp_path, monkeypatch):
    """Разбор по данным не определил target (две колонки только в train): как раньше, решает аналитик, второго
    разбора нет — запрос к LLM один, ответ записан."""
    task_md = make_task(tmp_path, train_extra=["y_leak"])
    fake = FakeLLM({"target": "y", "id_column": "id"})
    monkeypatch.setattr("agent.llm_client.LLMClient", lambda *args, **kwargs: fake)

    class Stop(Exception):
        pass

    def stop_after_step0(*args, **kwargs):
        raise Stop

    monkeypatch.setattr(solution, "profile_data", stop_after_step0)
    started = time.time()
    budget = make_budget(started)
    with pytest.raises(Stop):
        run_pipeline(task_md, budget, plan_stages(budget), AnswerKeeper(task_md), started)
    assert fake.requests == 1
    assert (tmp_path / "submission.csv").exists()


def test_analyst_changes_metric_step0_is_rewritten(tmp_path, monkeypatch):
    """Аналитик уточнил метрику (в тексте её нет — эвристика взяла rmse) — шаг 0 перезаписан по новому разбору."""
    task_md = make_task(tmp_path)
    keeper, spec, train, test = ready_keeper(task_md)
    assert spec.metric.name == "rmse"  # без LLM метрика по умолчанию для регрессии
    fake = FakeLLM({"target": "y", "id_column": "id", "metric": "mae"})
    monkeypatch.setattr("agent.llm_client.LLMClient", lambda *args, **kwargs: fake)
    monkeypatch.setenv("AGENT_ANALYST_APPLY", "1")  # явно: с AGENT_ANALYST_APPLY=0 угаданное не меняется (#276)

    new_spec, new_train, new_test = refine_with_analyst(task_md, spec, train, test, keeper, log)

    assert fake.requests == 1
    assert new_spec.metric.name == "mae"
    assert keeper.spec is new_spec and keeper.test is new_test
    answer = pd.read_csv(keeper.path)
    assert list(answer.columns) == ["id", "y"]
    assert len(answer) == len(new_test)


def test_analyst_unchanged_spec_keeps_everything(tmp_path):
    """Аналитик ничего не меняет (LLM недоступна): тот же spec и данные, ответ на месте."""
    task_md = make_task(tmp_path)
    keeper, spec, train, test = ready_keeper(task_md)
    before = keeper.path.read_bytes()
    out = refine_with_analyst(task_md, spec, train, test, keeper, log)
    assert out == (spec, train, test) or (out[0] is spec and out[1] is train and out[2] is test)
    assert keeper.spec is spec
    assert keeper.path.read_bytes() == before


def test_second_parse_failure_keeps_step0(tmp_path, monkeypatch):
    """Разбор с аналитиком упал целиком — остаёмся на разборе по данным, ответ не тронут."""
    task_md = make_task(tmp_path)
    keeper, spec, train, test = ready_keeper(task_md)
    before = keeper.path.read_bytes()

    def broken(*args, **kwargs):
        raise ValueError("аналитик сломал разбор")

    monkeypatch.setattr(solution, "parse_task", broken)
    out_spec, out_train, out_test = refine_with_analyst(task_md, spec, train, test, keeper, log)
    assert out_spec is spec and out_train is train and out_test is test
    assert keeper.path.read_bytes() == before


def test_invalid_answer_by_analyst_spec_is_rolled_back(tmp_path, monkeypatch):
    """Ответ по разбору аналитика не прошёл проверку — возвращаем прежний spec и тесты keeper."""
    task_md = make_task(tmp_path)
    keeper, spec, train, test = ready_keeper(task_md)
    fake = FakeLLM({"target": "y", "id_column": "id", "metric": "mae"})
    monkeypatch.setattr("agent.llm_client.LLMClient", lambda *args, **kwargs: fake)
    monkeypatch.setattr(keeper, "write", lambda answer: ["проверка не прошла"])

    out_spec, out_train, out_test = refine_with_analyst(task_md, spec, train, test, keeper, log)

    assert out_spec is spec and out_train is train and out_test is test
    assert keeper.spec is spec and keeper.test is test


def test_spec_changed_sees_metric_and_columns(tmp_path):
    spec = parse_task(make_task(tmp_path), use_analyst=False)
    assert not spec_changed(spec, spec.model_copy())
    assert spec_changed(spec, spec.model_copy(update={"id_column": "row_id"}))
    other_metric = spec.metric.model_copy(update={"name": "mae"})
    assert spec_changed(spec, spec.model_copy(update={"metric": other_metric}))
