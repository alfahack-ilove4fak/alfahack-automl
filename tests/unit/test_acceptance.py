"""Правило принятия гипотезы (agent/acceptance.py): граничные случаи и направление метрики."""

import math

import pytest

from agent.acceptance import MIN_REL_GAIN, T_MIN, accept, corrected_t, fold_gains
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


TIME_BEST = res([52.0, 47.0, 50.0]).model_copy(update={"validation": "time:Year_Factor"})  # 3 периода по возрастанию


def time_res(gains, validation="time:Year_Factor"):
    """RMSE: улучшение g на фолде — это минус g к ошибке."""
    return res([b - g for b, g in zip(TIME_BEST.fold_scores, gains, strict=True)]).model_copy(
        update={"validation": validation}
    )


def test_time_scheme_better_on_last_period_and_mean_accepted():
    # #236: лучше на 3/3 фолдах, но t < 2 — раньше отклоняли (ансамбль на regression: t = 1.91)
    gains = [0.2, 2.5, 0.4]
    assert corrected_t(gains) < T_MIN
    ok, reason = accept(time_res(gains), TIME_BEST, greater_is_better=False)
    assert ok and "последнем периоде" in reason


def test_time_scheme_worse_on_last_period_rejected():
    gains = [1.5, 1.0, -0.2]  # в среднем лучше, но на самом позднем периоде хуже
    ok, reason = accept(time_res(gains), TIME_BEST, greater_is_better=False)
    assert not ok and "последнем периоде" not in reason


def test_time_scheme_better_on_last_but_not_on_mean_rejected():
    ok, reason = accept(time_res([-2.0, -1.0, 0.5]), TIME_BEST, greater_is_better=False)
    assert not ok and "слишком мало" in reason


def test_time_scheme_metric_direction():
    # «больше — лучше»: улучшение — рост метрики на последнем периоде
    best = res([0.70, 0.72, 0.71]).model_copy(update={"validation": "time:date"})
    new = res([0.69, 0.75, 0.72]).model_copy(update={"validation": "time:date"})
    assert accept(new, best, greater_is_better=True)[0]
    assert not accept(new, best, greater_is_better=False)[0]


@pytest.mark.parametrize("validation", [None, "kfold", "stratified", "group:duplicates"])
def test_other_schemes_keep_t_rule(validation):
    # те же числа, что принимает time-схема, в остальных схемах по-прежнему режет t ≥ 2
    best = TIME_BEST.model_copy(update={"validation": validation})
    ok, reason = accept(time_res([0.2, 2.5, 0.4], validation), best, greater_is_better=False)
    assert not ok and "последнем периоде" not in reason


def test_time_scheme_incomparable_folds_use_mean_rule():
    two = res([50.0, 46.0]).model_copy(update={"validation": "time:Year_Factor"})  # 2 фолда против 3
    ok, reason = accept(two, TIME_BEST, greater_is_better=False)
    assert "несравнимы" in reason
