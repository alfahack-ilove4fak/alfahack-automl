"""Сдвиг train → test: AUC классификатора «строка из train или из test» и признаки, по которым они различаются.

Только измерение (#294, решение 25.10): результат идёт в agent.log и DataProfile.warnings, решения агента не меняет.
Строка в warnings начинается машиночитаемо — `adversarial_auc=0.88; …` — её читает правило про множители классов
(agent/hypotheses.py). Колонку времени или групп, которую выбрала choose_validation, из признаков убираем: по ней test
отделяется всегда (год 6, новые пациенты), хотя это устройство разбиения, а не сдвиг данных. Остальные признаки
могут выдать сдвиг сами (погода года 6 на public/regression), поэтому AUC без колонки времени там тоже может быть ≈ 1.

Идея и первая версия — evaluation/harness.py (#298, Егор): LightGBM, 4 фолда, подвыборка с каждой стороны.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

from interfaces.schemas import DataProfile
from runtime.log import get_logger

log = get_logger("shift")

SEED = 42
SHIFT_ENV = "AGENT_SHIFT"  # "0" — не считать сдвиг (экономит ≈ 5–15 с на старте)
SHIFT_MAX_ROWS = 20_000  # строк с каждой стороны: AUC по такой выборке стабилен до ±0.01, а время — секунды
SHIFT_FOLDS = 4
SHIFT_TOP = 3  # сколько признаков сдвига показывать
SHIFT_NONE_AUC = 0.6  # ниже — «сдвига нет» (случайное угадывание даёт 0.5 ± шум)
SHIFT_PREFIX = "adversarial_auc="  # начало строки в warnings: по нему её находит agent/


def shift_features(profile: DataProfile, train: pd.DataFrame, test: pd.DataFrame, exclude: tuple = ()) -> list[str]:
    """Признаки для классификатора «train против test»: признаки профиля без колонок exclude и идентификаторов.

    Идентификатор — целые числа или строки, все разные и в train, и в test: по нему test отделяется всегда (AUC 1.0),
    хотя модели он не нужен. Дробные числа с уникальными значениями (площадь, сумма) — обычные признаки.
    """
    features = []
    for col in profile.numeric_features + profile.categorical_features:
        if col in exclude or col not in train.columns or col not in test.columns:
            continue
        tr, te = train[col], test[col]
        id_like = not pd.api.types.is_float_dtype(tr) or not pd.api.types.is_float_dtype(te)
        if id_like and tr.nunique(dropna=False) == len(tr) and te.nunique(dropna=False) == len(te):
            continue
        features.append(col)
    return features


def adversarial_shift(
    profile: DataProfile, train: pd.DataFrame, test: pd.DataFrame, exclude: tuple = ()
) -> tuple[float, list[str]] | None:
    """(AUC по out-of-fold, признаки сдвига по убыванию вклада) или None, если считать не из чего."""
    import lightgbm as lgb
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold

    features = shift_features(profile, train, test, exclude)
    if not features or min(len(train), len(test)) < 4 * SHIFT_FOLDS:
        return None
    rng = np.random.default_rng(SEED)
    tr_rows = np.sort(rng.choice(len(train), min(len(train), SHIFT_MAX_ROWS), replace=False))
    te_rows = np.sort(rng.choice(len(test), min(len(test), SHIFT_MAX_ROWS), replace=False))
    x = pd.concat([train[features].iloc[tr_rows], test[features].iloc[te_rows]], ignore_index=True)
    for col in features:
        if not (pd.api.types.is_numeric_dtype(x[col]) or pd.api.types.is_bool_dtype(x[col])):
            x[col] = x[col].astype(str).astype("category")  # один словарь категорий для train и test
    is_test = np.r_[np.zeros(len(tr_rows)), np.ones(len(te_rows))]
    oof, gain = np.zeros(len(x)), np.zeros(len(features))
    for fit_rows, pred_rows in StratifiedKFold(SHIFT_FOLDS, shuffle=True, random_state=SEED).split(x, is_test):
        model = lgb.LGBMClassifier(
            n_estimators=100,
            learning_rate=0.1,
            random_state=SEED,
            n_jobs=min(os.cpu_count() or 1, 4),
            verbose=-1,
            deterministic=True,
            force_row_wise=True,
            importance_type="gain",
        )
        model.fit(x.iloc[fit_rows], is_test[fit_rows])
        oof[pred_rows] = model.predict_proba(x.iloc[pred_rows])[:, 1]
        gain += model.feature_importances_
    top = [features[i] for i in np.argsort(-gain)[:SHIFT_TOP] if gain[i] > 0]
    return float(roc_auc_score(is_test, oof)), top


def shift_warning(auc: float, top: list[str], excluded: str | None = None) -> str:
    """Строка для DataProfile.warnings; начало — `adversarial_auc=0.88` (формат согласован с agent/)."""
    text = f"{SHIFT_PREFIX}{auc:.2f}; "
    text += "сдвига train→test нет" if auc < SHIFT_NONE_AUC else "сдвиг train→test, сильнее всего: " + ", ".join(top)
    return text + (f" (без колонки {excluded}, по ней разбиение train/test)" if excluded else "")


def log_shift(profile: DataProfile, train: pd.DataFrame, test: pd.DataFrame, scheme: str) -> None:
    """Посчитать сдвиг, записать в agent.log и profile.warnings. Любая ошибка — только предупреждение в логе."""
    if os.getenv(SHIFT_ENV, "1") == "0" or any(w.startswith(SHIFT_PREFIX) for w in profile.warnings):
        return
    kind, _, column = scheme.partition(":")
    excluded = column if kind in ("time", "group") and column != "duplicates" else None
    try:
        result = adversarial_shift(profile, train, test, (excluded,) if excluded else ())
        if result is None:
            return
        auc, top = result
        if excluded:  # полный AUC — только в лог: показывает, насколько разбиение train/test видно по колонке
            full = adversarial_shift(profile, train, test)
            if full:
                log.info("Сдвиг train→test: AUC %.2f со всеми признаками, %.2f без «%s»", full[0], auc, excluded)
        profile.warnings.append(shift_warning(auc, top, excluded))
        log.info("Сдвиг train→test: %s", profile.warnings[-1])
    except Exception as exc:  # noqa: BLE001 — измерение вспомогательное: без него агент работает как раньше
        log.warning("Сдвиг train→test не посчитан: %s: %s", type(exc).__name__, exc)
