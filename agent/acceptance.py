"""Правило принятия гипотезы: лучше ли новый результат текущего лучшего с учётом шума. Владелец — agent.

Решение и цифры — docs/decisions/0010-acceptance-rule.md.

Основное правило — попарное сравнение по фолдам. Разбиения у всех экспериментов одинаковые
(KFold/StratifiedKFold с фиксированным seed), поэтому разница на фолде i — это эффект гипотезы
на одних и тех же данных. Сложность фолда при этом вычитается, и порог становится гораздо точнее,
чем «разница средних против разброса по фолдам».

Значимость — t-статистика Nadeau–Bengio: обычный парный t-тест на CV переоценивает уверенность,
потому что обучающие части фолдов пересекаются; поправка увеличивает дисперсию на n_test/n_train.

Если фолды несравнимы (другое n_folds, упавшие фолды) — старое правило: среднее лучше на NOISE_FACTOR·std.
"""

from __future__ import annotations

import math

from interfaces.schemas import ExperimentResult, ExperimentStatus

T_MIN = 2.0  # порог скорректированной t-статистики (подобран на «нулевых» гипотезах, см. ADR 0010)
MIN_REL_GAIN = 1e-4  # улучшение меньше 0.01% от метрики не стоит переобучения на всём train
NOISE_FACTOR = 0.25  # запасное правило: какую долю std лучшего результата считаем шумом


def fold_gains(new: ExperimentResult, best: ExperimentResult, greater_is_better: bool) -> list[float] | None:
    """Улучшение на каждом фолде (> 0 — новая лучше). None — фолды сравнивать нельзя."""
    if not new.fold_scores or len(new.fold_scores) != len(best.fold_scores) or len(new.fold_scores) < 2:
        return None
    sign = 1.0 if greater_is_better else -1.0
    return [sign * (a - b) for a, b in zip(new.fold_scores, best.fold_scores, strict=True)]


def corrected_t(gains: list[float]) -> float:
    """t-статистика Nadeau–Bengio для k-fold: дисперсия умножается на (1/k + 1/(k-1)) вместо 1/k."""
    k = len(gains)
    mean = sum(gains) / k
    var = sum((g - mean) ** 2 for g in gains) / (k - 1)
    if var == 0:
        return math.inf if mean > 0 else (-math.inf if mean < 0 else 0.0)
    return mean / math.sqrt((1 / k + 1 / (k - 1)) * var)


def accept(new: ExperimentResult, best: ExperimentResult, greater_is_better: bool) -> tuple[bool, str]:
    """Принять ли new вместо best. Возвращает решение и короткую причину для лога и истории."""
    if new.status != ExperimentStatus.OK or new.mean_score is None:
        return False, f"эксперимент не удался ({new.status.value})"
    if best.mean_score is None:
        return True, "прежнего результата нет"
    if new.validation != best.validation:
        # разные схемы CV несравнимы (контракт 1.1, D1): схему выбирают один раз до цикла
        return False, f"другая схема валидации: {new.validation} против {best.validation}"
    gains = fold_gains(new, best, greater_is_better)
    if gains is None:
        delta = NOISE_FACTOR * (best.std_score or 0.0)
        ok = new.beats(best, greater_is_better, delta)
        return ok, f"фолды несравнимы — по среднему с порогом {delta:.4g}"
    mean_gain = sum(gains) / len(gains)
    if mean_gain <= MIN_REL_GAIN * abs(best.mean_score):
        return False, f"среднее улучшение {mean_gain:.4g} слишком мало"
    t = corrected_t(gains)
    wins = sum(g > 0 for g in gains)
    reason = f"улучшение {mean_gain:.4g}, лучше на {wins}/{len(gains)} фолдах, t={t:.2f} (порог {T_MIN})"
    return t >= T_MIN, reason
