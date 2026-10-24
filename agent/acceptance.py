"""Правило принятия гипотезы: лучше ли новый результат текущего лучшего с учётом шума. Владелец — agent.

Решение и цифры — docs/decisions/0011-acceptance-rule.md.

Основное правило — попарное сравнение по фолдам. Разбиения у всех экспериментов одинаковые
(KFold/StratifiedKFold с фиксированным seed), поэтому разница на фолде i — это эффект гипотезы
на одних и тех же данных. Сложность фолда при этом вычитается, и порог становится гораздо точнее,
чем «разница средних против разброса по фолдам».

Значимость — t-статистика Nadeau–Bengio: обычный парный t-тест на CV переоценивает уверенность,
потому что обучающие части фолдов пересекаются; поправка увеличивает дисперсию на n_test/n_train.

Схема по времени (time:…, #236): фолдов всего 3–4, и t ≥ 2 почти недостижим даже для честно лучших гипотез.
Поэтому там принимаем ещё и гипотезу, которая лучше в среднем и на последнем (самом позднем, ближайшем к test) периоде,
но только если среднее улучшение не меньше TIME_MIN_REL_GAIN от метрики: иначе проходит шум подбора на тех же
фолдах (Optuna на public/regression: +0.6% на CV, на holdout RMSE 44.5 → 53.1). Порог — страховка, подобран на одном
примере; надёжнее не тюнить на time-фолдах вовсе (#236).
Остальные схемы — только по t. Исследование — docs/research/accept-rule.md.

Гипотезы от LLM (source="llm", #276) проходят по строгому порогу: t ≥ LLM_T_MIN (3), во time-схеме выигрыш
≥ LLM_TIME_MIN_REL_GAIN (2%). Переменные AGENT_LLM_T_MIN и AGENT_LLM_TIME_GAIN меняют их; 2 и 0.01 — как у playbook.

Если фолды несравнимы (другое n_folds, упавшие фолды) — старое правило: среднее лучше на NOISE_FACTOR·std.
"""

from __future__ import annotations

import math
import os
from statistics import NormalDist

from interfaces.schemas import ExperimentResult, ExperimentStatus

T_MIN = 2.0  # порог скорректированной t-статистики (подобран на «нулевых» гипотезах, см. ADR 0011)
MIN_REL_GAIN = 1e-4  # улучшение меньше 0.01% от метрики не стоит переобучения на всём train
NOISE_FACTOR = 0.25  # запасное правило: какую долю std лучшего результата считаем шумом
TIME_PREFIX = "time:"  # схема по времени: фолды по возрастанию периода (ml/validation._time_folds)
TIME_MIN_REL_GAIN = 0.01  # time-правило без t: улучшение меньше 1% от метрики считаем шумом подбора
LLM_T_ENV = "AGENT_LLM_T_MIN"  # порог t для гипотез LLM (#276, штраф −8)
LLM_T_MIN = 3.0
LLM_GAIN_ENV = "AGENT_LLM_TIME_GAIN"  # порог выигрыша time-правила для гипотез LLM, доля метрики
LLM_TIME_MIN_REL_GAIN = 0.02
# Поправка на множественные сравнения (#276, гипотеза штрафа −7): за 3 часа цикл сравнивает с лучшим сотни гипотез
# на одних и тех же фолдах, и правило t ≥ 2 (≈ 2.3% ложных принятий на гипотезу) пропускает шум: на «нулевых»
# гипотезах ≈ 6 ложных принятий на 300 проверок при 5 фолдах, ≈ 13 при 3. С поправкой Бонферрони на тот же уровень
# на одну проверку порог растёт как квантиль нормального: m=20 → 3.05, 100 → 3.51, 300 → 3.79 (≈ 0.7 ложных на 300).
# Нормальный, а не t-квантиль: при 3 фолдах (2 степени свободы) t-квантиль уходит в десятки и цикл замирает.
# Во time-схеме и для несравнимых фолдов порог выигрыша растёт в том же отношении. По умолчанию выключено.
MULTI_ENV = "AGENT_CONFIRM"  # "1" — порог растёт с числом сравнений (вместе с agent/confirm.py)
T_MULTI_MAX = 4.0  # потолок порога: m ≈ 1000; дальше честные улучшения почти не проходят
_ALPHA = 1 - NormalDist().cdf(T_MIN)  # уровень одной проверки, который даёт T_MIN


def _env_float(name: str, default: float) -> float:
    """Число из переменной окружения; пусто или мусор — значение по умолчанию (агент не должен падать)."""
    try:
        value = float(os.getenv(name, ""))
    except ValueError:
        return default
    return value if math.isfinite(value) and value >= 0 else default


def multi_factor(n_tested: int) -> float:
    """Во сколько раз поднять пороги после n_tested сравнений с лучшим (≥ 1). 1 — поправка выключена."""
    if os.getenv(MULTI_ENV, "0") != "1" or n_tested <= 1:
        return 1.0
    z = NormalDist().inv_cdf(1 - _ALPHA / n_tested)
    return max(1.0, min(z, T_MULTI_MAX) / T_MIN)


def thresholds(strict: bool, n_tested: int = 1) -> tuple[float, float]:
    """(порог t, порог выигрыша time-правила): strict — для гипотез LLM, n_tested — поправка на число сравнений."""
    if strict:
        t_min, gain = _env_float(LLM_T_ENV, LLM_T_MIN), _env_float(LLM_GAIN_ENV, LLM_TIME_MIN_REL_GAIN)
    else:
        t_min, gain = T_MIN, TIME_MIN_REL_GAIN
    factor = multi_factor(n_tested)
    return max(t_min, T_MIN * factor), max(gain, TIME_MIN_REL_GAIN * factor)


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


def accept(
    new: ExperimentResult,
    best: ExperimentResult,
    greater_is_better: bool,
    strict: bool = False,
    n_tested: int = 1,
) -> tuple[bool, str]:
    """Принять ли new вместо best. Возвращает решение и короткую причину для лога и истории.

    strict=True — гипотеза от LLM: пороги t и time-правила строже (thresholds).
    n_tested — сколько гипотез уже сравнили с лучшим за прогон, включая эту (поправка при AGENT_CONFIRM=1).
    """
    if new.status != ExperimentStatus.OK or new.mean_score is None:
        return False, f"эксперимент не удался ({new.status.value})"
    if best.mean_score is None:
        return True, "прежнего результата нет"
    if new.validation != best.validation:
        # разные схемы CV несравнимы (контракт 1.1, D1): схему выбирают один раз до цикла
        return False, f"другая схема валидации: {new.validation} против {best.validation}"
    gains = fold_gains(new, best, greater_is_better)
    if gains is None:
        delta = NOISE_FACTOR * (best.std_score or 0.0) * multi_factor(n_tested)
        ok = new.beats(best, greater_is_better, delta)
        return ok, f"фолды несравнимы — по среднему с порогом {delta:.4g}"
    mean_gain = sum(gains) / len(gains)
    if mean_gain <= MIN_REL_GAIN * abs(best.mean_score):
        return False, f"среднее улучшение {mean_gain:.4g} слишком мало"
    t_min, time_gain = thresholds(strict, n_tested)
    t = corrected_t(gains)
    wins = sum(g > 0 for g in gains)
    rule = (
        f"порог {t_min:.3g}" + (", LLM" if strict else "") + (f", m={n_tested}" if multi_factor(n_tested) > 1 else "")
    )
    reason = f"улучшение {mean_gain:.4g}, лучше на {wins}/{len(gains)} фолдах, t={t:.2f} ({rule})"
    if (
        t < t_min
        and (new.validation or "").startswith(TIME_PREFIX)
        and gains[-1] > 0
        and mean_gain >= time_gain * abs(best.mean_score)
    ):
        return True, f"{reason}; лучше на последнем периоде ({gains[-1]:.4g}) и в среднем — time-схема"
    return t >= t_min, reason
