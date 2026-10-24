"""Правило принятия гипотезы (agent/acceptance.py): граничные случаи и направление метрики."""

import math

import pytest

from agent.acceptance import (
    LLM_GAIN_ENV,
    LLM_T_ENV,
    LLM_T_MIN,
    LLM_TIME_MIN_REL_GAIN,
    MIN_REL_GAIN,
    T_MIN,
    TIME_MIN_REL_GAIN,
    accept,
    corrected_t,
    fold_gains,
    thresholds,
)
from interfaces.schemas import ExperimentResult, ExperimentStatus


def res(folds, status=ExperimentStatus.OK):
    ok = status == ExperimentStatus.OK and folds
    mean = sum(folds) / len(folds) if ok else None
    std = (sum((f - mean) ** 2 for f in folds) / len(folds)) ** 0.5 if ok else None
    return ExperimentResult(
        hypothesis_id="h", status=status, metric_name="m", fold_scores=folds, mean_score=mean, std_score=std
    )


BEST = res([43.35, 41.86, 41.31, 42.85, 39.68])  # бейзлайн регрессии: RMSE 41.81 ± 1.28


def test_consistent_small_gain_accepted_despite_big_fold_spread():
    # Улучшение 0.19 при std по фолдам 1.28: старое правило (порог 0.32) его отсекало
    new = res([f - 0.19 + d for f, d in zip(BEST.fold_scores, [0.02, -0.03, 0.01, 0.0, -0.01], strict=False)])
    ok, reason = accept(new, BEST, greater_is_better=False)
    assert ok, reason


def test_noise_rejected():
    new = res(
        [f + d for f, d in zip(BEST.fold_scores, [-0.4, 0.35, -0.3, 0.45, -0.2], strict=False)]
    )  # среднее ~0, разброс
    assert not accept(new, BEST, greater_is_better=False)[0]


def test_gain_on_average_but_unstable_rejected():
    new = res(
        [f + d for f, d in zip(BEST.fold_scores, [-1.2, 0.3, -0.1, 0.4, 0.2], strict=False)]
    )  # один фолд тянет среднее
    assert not accept(new, BEST, greater_is_better=False)[0]


def test_direction_of_metric():
    worse_rmse = res([f + 0.2 for f in BEST.fold_scores])
    assert not accept(worse_rmse, BEST, greater_is_better=False)[0]
    assert accept(worse_rmse, BEST, greater_is_better=True)[0]  # для «больше — лучше» это рост


def test_tiny_gain_below_min_rel_rejected_even_if_consistent():
    gain = MIN_REL_GAIN * 41.8 / 2
    new = res([f - gain for f in BEST.fold_scores])
    ok, reason = accept(new, BEST, greater_is_better=False)
    assert not ok and "слишком мало" in reason


def test_identical_result_rejected():
    assert not accept(res(list(BEST.fold_scores)), BEST, greater_is_better=False)[0]


@pytest.mark.parametrize("status", [ExperimentStatus.FAILED, ExperimentStatus.TIMEOUT])
def test_failed_experiment_rejected(status):
    ok, reason = accept(res([1.0, 2.0], status=status), BEST, greater_is_better=False)
    assert not ok and status.value in reason


def test_no_previous_result_accepts_any_ok():
    empty = ExperimentResult(hypothesis_id="b", status=ExperimentStatus.FAILED, metric_name="m")
    assert accept(BEST, empty, greater_is_better=False)[0]


def test_different_n_folds_falls_back_to_mean_rule():
    ten = res([f - 1.0 for f in BEST.fold_scores] * 2)  # 10 фолдов против 5: попарно нельзя
    assert fold_gains(ten, BEST, greater_is_better=False) is None
    ok, reason = accept(ten, BEST, greater_is_better=False)
    assert ok and "несравнимы" in reason
    near = res([f - 0.1 for f in BEST.fold_scores] * 2)  # 0.1 < 0.25·std = 0.32
    assert not accept(near, BEST, greater_is_better=False)[0]


def test_corrected_t():
    assert corrected_t([0.1] * 5) == math.inf
    assert corrected_t([-0.1] * 5) == -math.inf
    assert corrected_t([0.0] * 5) == 0.0
    plain_t = 0.2 / math.sqrt(0.01 / 5)  # обычный парный t для gains со средним 0.2 и дисперсией 0.01
    gains = [0.2 + d for d in (-0.1, 0.1, 0.0, -0.1, 0.1)]
    assert corrected_t(gains) == pytest.approx(0.2 / math.sqrt((1 / 5 + 1 / 4) * 0.01))
    assert corrected_t(gains) < plain_t  # поправка делает тест осторожнее
    assert T_MIN > 0


def test_different_validation_scheme_rejected():
    # контракт 1.1 (D1): результаты разных схем CV несравнимы, даже если «лучше» по числам
    better = res([f - 1.0 for f in BEST.fold_scores]).model_copy(update={"validation": "group-5"})
    ok, reason = accept(better, BEST, greater_is_better=False)
    assert not ok and "схема валидации" in reason


# Схема по времени (#236): 3 фолда, последний — самый поздний период. Числа — как у ансамбля на public/regression:
# лучше на 3/3 фолдах, но t < T_MIN (у ансамбля там t = 1.91).
TIME_BEST = res([52.0, 48.0, 49.5])
TIME_GAINS_ALL = [0.2, 1.5, 3.0]  # лучше на всех периодах, сильнее всего на последнем
TIME_GAINS_NOT_LAST = [2.5, 1.5, -0.4]  # в среднем лучше, на последнем — хуже


def time_pair(gains, scheme="time:Year_Factor"):
    best = TIME_BEST.model_copy(update={"validation": scheme})
    new = res([b - g for b, g in zip(TIME_BEST.fold_scores, gains, strict=True)]).model_copy(
        update={"validation": scheme}
    )
    return new, best


def test_time_scheme_better_on_last_period_and_mean_accepted():
    new, best = time_pair(TIME_GAINS_ALL)
    assert corrected_t(fold_gains(new, best, greater_is_better=False)) < T_MIN  # по t не прошла бы
    ok, reason = accept(new, best, greater_is_better=False)
    assert ok and "последнем периоде" in reason


def test_time_scheme_worse_on_last_period_rejected():
    new, best = time_pair(TIME_GAINS_NOT_LAST)
    assert sum(TIME_GAINS_NOT_LAST) > 0
    assert not accept(new, best, greater_is_better=False)[0]


def test_time_scheme_direction_of_metric():
    # для «больше — лучше» те же числа означают ухудшение на всех периодах
    new, best = time_pair(TIME_GAINS_ALL)
    assert not accept(new, best, greater_is_better=True)[0]


def test_time_scheme_tiny_gain_still_rejected():
    new, best = time_pair([MIN_REL_GAIN * 50 / 4] * 3)
    ok, reason = accept(new, best, greater_is_better=False)
    assert not ok and "слишком мало" in reason


@pytest.mark.parametrize("scheme", [None, "kfold", "stratified", "group:duplicates"])
def test_non_time_schemes_keep_t_rule(scheme):
    new, best = time_pair(TIME_GAINS_ALL, scheme=scheme)
    ok, reason = accept(new, best, greater_is_better=False)
    assert not ok and "последнем периоде" not in reason


def test_time_scheme_incomparable_folds_use_mean_rule():
    best = TIME_BEST.model_copy(update={"validation": "time:Year_Factor"})
    new = res([45.0, 44.0]).model_copy(update={"validation": "time:Year_Factor"})  # упал фолд: 2 против 3
    assert fold_gains(new, best, greater_is_better=False) is None
    ok, reason = accept(new, best, greater_is_better=False)
    assert ok and "несравнимы" in reason


# Регрессия после #259: на public/regression Optuna (подбор на тех же time-фолдах) дала +0.6% на CV, t=1.16,
# лучше на последнем периоде — и была принята; holdout RMSE 44.5 → 53.1. Poisson (+3.5%) — честное улучшение.
def test_time_scheme_small_gain_below_time_threshold_rejected():
    gains = [0.7, 0.0, 0.2]  # +0.6% от среднего TIME_BEST, на последнем периоде лучше
    new, best = time_pair(gains)
    mean_gain = sum(gains) / len(gains)
    assert mean_gain < TIME_MIN_REL_GAIN * best.mean_score
    assert corrected_t(fold_gains(new, best, greater_is_better=False)) < T_MIN
    ok, reason = accept(new, best, greater_is_better=False)
    assert not ok and "последнем периоде" not in reason


def test_time_scheme_gain_above_time_threshold_accepted():
    gains = [0.2, 1.8, 3.2]  # +3.5%, как objective_poisson на public/regression
    new, best = time_pair(gains)
    assert sum(gains) / len(gains) >= TIME_MIN_REL_GAIN * best.mean_score
    assert corrected_t(fold_gains(new, best, greater_is_better=False)) < T_MIN
    ok, reason = accept(new, best, greater_is_better=False)
    assert ok and "последнем периоде" in reason


# ---------- гипотезы LLM: свой порог (#276) — по умолчанию как у playbook, строже через переменные ----------
def gains_result(gains):
    return res([b - g for b, g in zip(BEST.fold_scores, gains, strict=True)])


MID_T_GAINS = [0.3, 0.09, 0.25, 0.03, 0.2]  # t ≈ 2.3
LOW_T_GAINS = [0.3, -0.02, 0.25, 0.0, 0.2]  # t ≈ 1.5


def test_llm_default_threshold_like_playbook(monkeypatch):
    """Сабмит №6: штраф −8 не от LLM — по умолчанию гипотеза LLM с t = 2.3 проходит, с t = 1.5 нет."""
    monkeypatch.delenv(LLM_T_ENV, raising=False)
    assert (LLM_T_MIN, LLM_TIME_MIN_REL_GAIN) == (T_MIN, TIME_MIN_REL_GAIN)
    assert corrected_t(MID_T_GAINS) == pytest.approx(2.3, abs=0.05)
    assert corrected_t(LOW_T_GAINS) == pytest.approx(1.5, abs=0.05)
    ok, reason = accept(gains_result(MID_T_GAINS), BEST, greater_is_better=False, strict=True)
    assert ok and "порог 2, LLM" in reason
    assert not accept(gains_result(LOW_T_GAINS), BEST, greater_is_better=False, strict=True)[0]


def test_llm_strict_threshold_from_env(monkeypatch):
    """Выключатель остаётся: AGENT_LLM_T_MIN=3 отсекает t = 2.3 только у LLM, playbook — по-прежнему по 2."""
    monkeypatch.setenv(LLM_T_ENV, "3")
    ok, reason = accept(gains_result(MID_T_GAINS), BEST, greater_is_better=False, strict=True)
    assert not ok and "порог 3, LLM" in reason
    assert accept(gains_result(MID_T_GAINS), BEST, greater_is_better=False)[0]


@pytest.mark.parametrize("garbage", ["", "abc", "-1", "nan", "inf"])
def test_llm_thresholds_garbage_env_uses_defaults(monkeypatch, garbage):
    monkeypatch.setenv(LLM_T_ENV, garbage)
    monkeypatch.setenv(LLM_GAIN_ENV, garbage)
    assert thresholds(strict=True) == (LLM_T_MIN, LLM_TIME_MIN_REL_GAIN)
    assert thresholds(strict=False) == (T_MIN, TIME_MIN_REL_GAIN)


def test_llm_time_scheme_gain(monkeypatch):
    """time-схема: по умолчанию +1.5% проходит и у LLM (порог 1%); AGENT_LLM_TIME_GAIN=0.02 — только у playbook."""
    new, best = time_pair([0.2, 0.7, 1.4])  # +1.5%
    monkeypatch.delenv(LLM_GAIN_ENV, raising=False)
    assert accept(new, best, greater_is_better=False, strict=True)[0]
    monkeypatch.setenv(LLM_GAIN_ENV, "0.02")
    assert accept(new, best, greater_is_better=False)[0]
    assert not accept(new, best, greater_is_better=False, strict=True)[0]
    new, best = time_pair([0.2, 1.8, 3.2])  # +3.5% — проходит и при строгом пороге
    assert accept(new, best, greater_is_better=False, strict=True)[0]
