"""Паспорт данных (DataProfile) и подготовка признаков для моделей. Владелец — ML-pipeline инженер."""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from interfaces.schemas import ROW_ID_COLUMN, DataProfile, TaskSpec, TaskType
from runtime.log import get_logger

log = get_logger("profiling")


def load_data(spec: TaskSpec) -> tuple[pd.DataFrame, pd.DataFrame]:
    """train и test. id читается строкой («001300» не превращается в 1300), строки с пустым target выкидываются."""
    dtype = {spec.id_column: str}
    train = pd.read_csv(spec.train_path, low_memory=False, dtype=dtype)
    test = pd.read_csv(spec.test_path, low_memory=False, dtype=dtype)
    if spec.id_column not in test.columns:
        test[spec.id_column] = _missing_ids(spec, len(test))
    missing = train[spec.target].isna()
    if missing.any():
        log.warning("В train %d строк с пустым target — выкидываем, учиться на них нельзя", int(missing.sum()))
        train = train.loc[~missing].reset_index(drop=True)
    return train, test


def _missing_ids(spec: TaskSpec, n_rows: int) -> pd.Series:
    """id, которого нет в test (1.1.2): значения из sample_submission, если он есть и той же длины, иначе номер строки.

    Шаблон без id-колонки (ROW_ID_COLUMN) — ответ идёт по порядку строк test; колонка нужна внутри агента,
    в submission.csv её нет.
    """
    if spec.id_column != ROW_ID_COLUMN and spec.sample_submission_path is not None:
        try:
            sample = pd.read_csv(spec.sample_submission_path, usecols=[spec.id_column], dtype=str)[spec.id_column]
            if len(sample) == n_rows:
                return sample
        except Exception:  # noqa: BLE001 — нет такой колонки в образце: ниже номера строк
            pass
    return pd.Series(range(n_rows)).astype(str)


LOG_SCALE_METRICS = {"rmsle", "msle"}  # ошибка считается на log1p(y): обучение на log1p(target) совпадает с метрикой


def _skew_warning(skew: float, metric: str, non_negative: bool) -> str:
    """Факт о скошенности и что он значит для метрики задачи. Это читает LLM — совет должен быть верным.

    Для RMSLE/MSLE обучение на log1p(target) совпадает с метрикой. Для метрик в исходной шкале это смена цели:
    на public/regression CV показывает проигрыш, а честная отложенная часть — выигрыш (docs/research/playbook-v1.md),
    поэтому не советуем и не отговариваем — это гипотеза для проверки.
    """
    if metric in LOG_SCALE_METRICS and non_negative:
        return f"Скошенный target (skew={skew:.1f}); метрика {metric} считается в логарифмах — log1p(target) совпадает с ней"
    return (
        f"Скошенный target (skew={skew:.1f}); метрика {metric} — в исходной шкале: log1p(target) меняет "
        "оптимизируемую цель — это одна гипотеза для проверки, а не правило"
    )


# Утечка / признак, которого нет на момент прогноза (#240, чужая задача leak_column_auc: CV 0.988 против test 0.729)
TEST_EMPTY_SHARE = 0.95  # колонка пуста почти во всём test …
TRAIN_FILLED_SHARE = 0.5  # … а в train заполнена хотя бы наполовину — модель выучит то, чего в test не будет
LEAK_SIGNAL = 0.9  # одиночный сигнал: |ρ Спирмена| с target (регрессия) или |2·AUC−1| (бинарная)
LEAK_SHIFT_KS = 0.5  # и распределение в test другое (статистика Колмогорова — Смирнова)
LEAK_SAMPLE = 100_000  # строк для оценки сигнала и сдвига — больше не нужно


def _single_signal(x: pd.Series, y: pd.Series, task_type: TaskType) -> float:
    """Сила одиночного признака: |ρ Спирмена| (регрессия) или |2·AUC−1| (бинарная); мультикласс и мало данных — 0."""
    ok = x.notna()
    if task_type == TaskType.MULTICLASS or ok.sum() < 50:
        return 0.0
    xr = x[ok].rank().to_numpy()
    if task_type == TaskType.REGRESSION:
        yr = pd.to_numeric(y[ok], errors="coerce").rank().to_numpy()
        if np.isnan(yr).any() or xr.std() == 0 or yr.std() == 0:
            return 0.0
        return float(abs(np.corrcoef(xr, yr)[0, 1]))
    codes = y[ok].astype(str)
    positive = codes == codes.max()  # для |2·AUC−1| неважно, какой класс «положительный»
    n_pos, n_neg = int(positive.sum()), int((~positive).sum())
    if n_pos == 0 or n_neg == 0:
        return 0.0
    auc = (xr[positive.to_numpy()].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    return float(abs(2 * auc - 1))


def leak_check(spec: TaskSpec, train: pd.DataFrame, test: pd.DataFrame, features: list[str]) -> tuple[list, list]:
    """Признаки, которых не будет на момент прогноза: (что выбросить, предупреждения).

    1. Колонка почти пуста в test, а в train заполнена — на ней нельзя предсказывать: модель выучит зависимость, а в
       test увидит только пропуски (часто это поле, которое заполняют после события — утечка target).
    2. Числовой признак с почти идеальным одиночным сигналом и другим распределением в test — та же утечка, но
       test заполнен иначе (нулями, заглушкой). Сильный признак с тем же распределением в test — честный: только
       предупреждение.
    Пустой test (нечего сравнивать) — ничего не выбрасываем.
    """
    if len(test) == 0:
        return [], []
    drop, warnings, strong = [], [], []
    tr = train.sample(LEAK_SAMPLE, random_state=0) if len(train) > LEAK_SAMPLE else train
    te = test.sample(LEAK_SAMPLE, random_state=0) if len(test) > LEAK_SAMPLE else test
    for col in features:
        empty_test, empty_train = float(te[col].isna().mean()), float(tr[col].isna().mean())
        if empty_test >= TEST_EMPTY_SHARE and empty_train <= 1 - TRAIN_FILLED_SHARE:
            drop.append(col)
            warnings.append(
                f"Признак {col} пуст в {empty_test:.0%} test, а в train заполнен на {1 - empty_train:.0%} — "
                "на нём нельзя предсказывать (похоже на поле, которое заполняют после события), выброшен"
            )
            continue
        if not pd.api.types.is_numeric_dtype(tr[col]) or pd.api.types.is_bool_dtype(tr[col]):
            continue
        signal = _single_signal(tr[col], tr[spec.target], spec.task_type)
        if signal < LEAK_SIGNAL:
            continue
        a, b = tr[col].dropna(), te[col].dropna()
        shift = float(ks_2samp(a, b).statistic) if len(a) > 20 and len(b) > 20 else 0.0
        if shift >= LEAK_SHIFT_KS:
            drop.append(col)
            warnings.append(
                f"Признак {col} почти повторяет target в train (сила {signal:.2f}), а в test распределён иначе "
                f"(KS={shift:.2f}) — похоже на утечку, выброшен"
            )
        else:
            strong.append(f"{col} ({signal:.2f})")
    if strong:
        # Одна строка на все: её читает LLM, пять одинаковых предупреждений — только шум в промпте
        warnings.append(
            f"Очень сильные в одиночку признаки, в test распределены так же — оставлены: {', '.join(strong[:5])}"
            f"{' …' if len(strong) > 5 else ''}; если поле заполняют после события, это утечка"
        )
    return drop, warnings


def profile_data(spec: TaskSpec, train: pd.DataFrame, test: pd.DataFrame) -> DataProfile:
    features = [c for c in train.columns if c not in (spec.target, spec.id_column) and c in test.columns]
    numeric, categorical, dropped, warnings = [], [], [], []
    leaks, leak_warnings = leak_check(spec, train, test, features)
    for warning in leak_warnings:
        log.warning(warning)
    warnings.extend(leak_warnings)
    for col in features:
        series = train[col]
        if col in leaks or series.nunique(dropna=False) <= 1:
            dropped.append(col)
            continue
        if series.dtype == object or str(series.dtype) in ("category", "bool", "string"):
            categorical.append(col)
        else:
            numeric.append(col)

    missing = {c: round(float(train[c].isna().mean()), 4) for c in features if train[c].isna().any()}
    y = train[spec.target]
    if spec.task_type == TaskType.REGRESSION:
        target_summary = {k: float(v) for k, v in y.describe().items()}
        target_summary["skew"] = float(y.skew())
        if y.skew() > 2:
            warnings.append(_skew_warning(float(y.skew()), spec.metric.name, bool((y >= 0).all())))
    else:
        counts = y.astype(str).value_counts()
        target_summary = {"class_counts": counts.to_dict(), "imbalance_ratio": float(counts.max() / counts.min())}
        if counts.max() / counts.min() > 10:
            warnings.append("Сильный дисбаланс классов — стоит попробовать class_weight='balanced'")

    profile = DataProfile(
        n_train=len(train),
        n_test=len(test),
        numeric_features=numeric,
        categorical_features=categorical,
        dropped_features=dropped,
        missing_share=missing,
        target_summary=target_summary,
        memory_mb=float((train.memory_usage(deep=True).sum() + test.memory_usage(deep=True).sum()) / 2**20),
        warnings=warnings,
    )
    log.info(
        "Данные: train=%d, test=%d, числовых=%d, категориальных=%d, выброшено=%d",
        len(train),
        len(test),
        len(numeric),
        len(categorical),
        len(dropped),
    )
    return profile


def make_matrix(profile: DataProfile, train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Признаки для бустинга: числа как есть, категории → pandas category с общим словарём train+test."""
    cols = profile.numeric_features + profile.categorical_features
    x_train, x_test = train[cols].copy(), test[cols].copy()
    for col in profile.categorical_features:
        cats = pd.Index(pd.concat([x_train[col], x_test[col]]).astype(str).unique())
        x_train[col] = pd.Categorical(x_train[col].astype(str), categories=cats)
        x_test[col] = pd.Categorical(x_test[col].astype(str), categories=cats)
    for col in profile.numeric_features:
        x_train[col] = pd.to_numeric(x_train[col], errors="coerce").astype(np.float32)
        x_test[col] = pd.to_numeric(x_test[col], errors="coerce").astype(np.float32)
    names = _safe_names(cols)
    x_train.columns, x_test.columns = names, names
    return x_train, x_test


def _safe_names(cols: list[str]) -> list[str]:
    """Имена признаков, которые примут все модели: LightGBM падает на спецсимволах JSON, XGBoost — на [ ] <.

    Всё, кроме букв, цифр и _, заменяется на _; совпавшие после замены имена получают суффикс _1, _2…
    """
    names, seen = [], set()
    for col in cols:
        base = re.sub(r"\W+", "_", str(col)).strip("_") or "feature"
        name, k = base, 1
        while name in seen:
            name, k = f"{base}_{k}", k + 1
        seen.add(name)
        names.append(name)
    return names
