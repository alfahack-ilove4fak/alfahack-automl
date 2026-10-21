"""Бюджет по этапам: доли, резерв в конце, жёсткая отсечка и подстройка резерва под замер финала."""

import pytest

from runtime.budget import (
    BASELINE_SHARE,
    FINAL_SHARE,
    SAFETY_SHARE,
    Shares,
    ensure_final_reserve,
    get_shares,
    make_budget,
    plan_stages,
)

ENV_NAMES = (
    "AGENT_TIME_BUDGET_SEC",
    "AGENT_SAFETY_SEC",
    "AGENT_BASELINE_SHARE",
    "AGENT_FINAL_SHARE",
    "AGENT_SAFETY_SHARE",
)


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for name in ENV_NAMES:
        monkeypatch.delenv(name, raising=False)


def test_default_shares_sum_to_one():
    shares = get_shares()
    assert shares == Shares(BASELINE_SHARE, FINAL_SHARE, SAFETY_SHARE)
    assert shares.baseline + shares.research + shares.final + shares.safety == pytest.approx(1.0)
    assert shares.research > 0.6


@pytest.mark.parametrize("total", [1800, 3 * 3600])
def test_default_plan_keeps_final_and_safety(total, monkeypatch):
    monkeypatch.setenv("AGENT_TIME_BUDGET_SEC", str(total))
    budget = make_budget(started_at=1000.0)
    plan = plan_stages(budget)
    end = 1000.0 + total
    assert plan.platform_end == end
    assert plan.baseline_end - 1000.0 == pytest.approx(0.20 * total)
    assert end - plan.research_end >= 0.15 * total - 1e-6  # финал ≥ 10% + запас ≥ 5%
    assert end - plan.hard_deadline >= 0.05 * total - 1e-6  # к лимиту платформы остаётся ≥ 5%
    assert plan.baseline_end < plan.research_end < plan.hard_deadline < plan.platform_end
    assert budget.safety_seconds == pytest.approx(end - plan.research_end)


def test_shares_from_env(monkeypatch):
    monkeypatch.setenv("AGENT_BASELINE_SHARE", "0.1")
    monkeypatch.setenv("AGENT_FINAL_SHARE", "0.2")
    monkeypatch.setenv("AGENT_SAFETY_SHARE", "0.1")
    monkeypatch.setenv("AGENT_TIME_BUDGET_SEC", "1000")
    budget = make_budget(started_at=0.0)
    plan = plan_stages(budget)
    assert budget.safety_seconds == pytest.approx(300)
    assert plan.baseline_end == pytest.approx(100)
    assert plan.hard_deadline == pytest.approx(900)


@pytest.mark.parametrize(
    "env",
    [
        {"AGENT_BASELINE_SHARE": "0.9"},  # не оставляет места AutoResearch
        {"AGENT_FINAL_SHARE": "abc"},  # не число
        {"AGENT_SAFETY_SHARE": "-0.1"},  # вне [0, 1)
    ],
)
def test_bad_shares_fall_back_to_defaults(env, monkeypatch):
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    shares = get_shares()
    assert shares.research >= 0.1
    assert shares.safety >= 0 and shares.final >= 0


def test_manual_safety_keeps_hard_deadline_after_research(monkeypatch):
    # Так запускают дымовые тесты: бюджет 40 с и резерв 5 с
    monkeypatch.setenv("AGENT_TIME_BUDGET_SEC", "40")
    monkeypatch.setenv("AGENT_SAFETY_SEC", "5")
    budget = make_budget(started_at=0.0)
    plan = plan_stages(budget)
    assert plan.research_end == pytest.approx(35)
    assert plan.research_end < plan.hard_deadline < plan.platform_end


def test_final_reserve_grows_only_and_is_capped(monkeypatch):
    monkeypatch.setenv("AGENT_TIME_BUDGET_SEC", "1800")
    budget = make_budget(started_at=0.0)
    base = budget.safety_seconds  # 15% = 270 с
    assert ensure_final_reserve(budget, 5) == base  # быстрый финал — резерв не трогаем
    grown = ensure_final_reserve(budget, 200)  # финал 200 с → резерв ≥ 90 + 3·200
    assert grown == pytest.approx(0.05 * 1800 + 3 * 200)
    assert ensure_final_reserve(budget, 5) == grown  # резерв не уменьшается
    assert ensure_final_reserve(budget, 10_000) == pytest.approx(0.5 * 1800)  # и не съедает больше половины
