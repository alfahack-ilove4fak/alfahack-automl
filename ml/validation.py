"""Схемы кросс-валидации (params.validation) и выбор схемы по умолчанию. Владелец — ML-pipeline инженер.

ADR 0008, D1: схему выбирают один раз до цикла (choose_validation), все гипотезы считаются в ней, а
ExperimentResult.validation = "<схема>-<число фолдов>" — результаты разных схем не сравниваются.
Решение о схеме по умолчанию и цифры — docs/decisions/0013-validation-default.md.

Значения params.validation:
  kfold              — случайные фолды (по умолчанию для регрессии)
  stratified         — с сохранением долей классов (по умолчанию для классификации)
  group:<колонка>    — строки с одинаковым значением колонки — в одном фолде (здание, клиент…)
  group:duplicates   — полные дубликаты строк признаков — в одном фолде (иначе модель «видит ответ» на валидации)
  time:<колонка>     — по времени: обучение на прошлых периодах, проверка на следующем (как test после train)
"""

from __future__ import annotations

import re
import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold

from interfaces.schemas import DataProfile, TaskSpec, TaskType
from runtime.log import get_logger

log = get_logger("validation")

SEED = 42
DUPLICATES = "duplicates"
DUPLICATE_SHARE = 0.10  # от этой доли дубликатов в train случайные фолды заметно завышают оценку (EDA multiclass: 45%)
TIME_PARSE_SHARE = 0.9  # колонка — время, если столько значений читается как число или дата
TIME_MIN_PERIODS = 3
TIME_MAX_PERIODS = 50  # больше различных значений — режем на равные по числу строк периоды
# Имя «про время»: слово в начале имени, после не-буквы или с заглавной после строчной (Year_Factor, sale_date,
# SaleDate), но не update_count
# Группы по колонке (#288): test — другие объекты (пациенты, молекулы), и их значения колонки в train не встречаются.
# Пороги проверены на BeyondArena grouped: там новых значений в test 100%, строк на группу от ~2 до сотен
GROUP_NEW_SHARE = 0.9  # столько строк test должны нести значение, которого нет в train
GROUP_MIN_ROWS = 1.5  # строк train на значение в среднем: у идентификатора 1.0 — это не группы, а номера строк
GROUP_MIN_GROUPS = 10  # меньше значений — это свойство (тип, регион), а не объекты; и фолдов по группам не хватит
TIME_NAME_HINT = re.compile(
    r"(?:(?<![^\W\d_])|(?<=[a-zа-я])(?=[A-ZА-Я]))(?i:year|date|time|month|week|day|period|год|дата|время|месяц)"
)


def _is_classification(spec: TaskSpec) -> bool:
    return spec.task_type in (TaskType.BINARY, TaskType.MULTICLASS)


def default_scheme(spec: TaskSpec) -> str:
    return "stratified" if _is_classification(spec) else "kfold"


def parse_validation(value, spec: TaskSpec) -> tuple[str, str | None]:
    """params.validation → (схема, колонка). Пусто — схема по умолчанию для типа задачи."""
    if not value:
        return default_scheme(spec), None
    if not isinstance(value, str):
        raise ValueError(f"validation должна быть строкой вида 'time:колонка', а не {value!r}")
    scheme, _, column = value.partition(":")
    scheme = scheme.strip().lower()
    if scheme in ("kfold", "stratified"):
        return (scheme if _is_classification(spec) else "kfold"), None
    if scheme in ("group", "time") and column.strip():
        return scheme, column.strip()
    raise ValueError(f"неизвестная схема валидации {value!r}: kfold | stratified | group:<колонка> | time:<колонка>")


def time_values(series: pd.Series) -> np.ndarray | None:
    """Колонка → числа для упорядочивания по времени (даты из строк — тоже) или None, если это не время."""
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().mean() >= TIME_PARSE_SHARE:
        return numeric.to_numpy(dtype=float)
    if series.dtype != object and not isinstance(series.dtype, pd.CategoricalDtype):
        return None
    with warnings.catch_warnings():  # pandas предупреждает, что формат угадывается по строкам
        warnings.simplefilter("ignore")
        dates = pd.to_datetime(series.astype(str), errors="coerce")
    if dates.notna().mean() < TIME_PARSE_SHARE:
        return None
    return (dates - pd.Timestamp("1970-01-01")).dt.total_seconds().to_numpy(dtype=float)


def duplicate_groups(features: pd.DataFrame) -> np.ndarray:
    """Номер группы для каждой строки: строки с одинаковыми признаками — одна группа.

    Числа и категории pandas хеширует сам (NaN — одинаково); к строке приводим только object, где бывают смешанные
    типы. astype(str) для всей таблицы на 1M × 40 — 78 с и +2.6 ГБ на каждую гипотезу (ревью #150).
    """
    text = [c for c in features.columns if features[c].dtype == object]
    if text:
        features = features.astype(dict.fromkeys(text, str))
    return pd.util.hash_pandas_object(features, index=False).to_numpy()


def _rare_mask(spec: TaskSpec, y: np.ndarray, n_folds: int) -> np.ndarray:
    if not _is_classification(spec):
        return np.zeros(len(y), dtype=bool)
    counts = pd.Series(y).value_counts()
    return np.isin(y, counts.index[counts < n_folds].to_numpy())


def _group_folds(
    spec: TaskSpec, y: np.ndarray, groups: np.ndarray, n_folds: int
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Фолды по группам; редкие классы (меньше n_folds объектов) всегда в обучении, как в обычной схеме."""
    idx = np.arange(len(y))
    rare = _rare_mask(spec, y, n_folds)
    common = idx[~rare]
    n_folds = min(n_folds, len(np.unique(groups[common])))
    if n_folds < 2:
        raise ValueError("для схемы по группам нужно хотя бы 2 разные группы")
    if _is_classification(spec):
        splitter = StratifiedGroupKFold(n_folds, shuffle=True, random_state=SEED)
        parts = splitter.split(common, y[common], groups[common])
    else:
        parts = GroupKFold(n_folds).split(common, y[common], groups[common])
    return [(np.sort(np.concatenate([common[tr], idx[rare]])), common[va]) for tr, va in parts]


def _time_folds(t: np.ndarray, n_folds: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """Скользящее окно вперёд: периоды по возрастанию, фолд i — обучение на периодах до i, проверка на периоде i.

    Строки без времени (NaN) идут только в обучение. Периодов больше TIME_MAX_PERIODS — режем на равные по числу строк.
    """
    known = ~np.isnan(t)
    values = np.unique(t[known])
    if len(values) > TIME_MAX_PERIODS:
        edges = np.unique(np.quantile(t[known], np.linspace(0, 1, n_folds + 2)))
        period = np.searchsorted(edges[1:-1], t, side="right").astype(float)
    else:
        period = np.searchsorted(values, t).astype(float)
    period[~known] = -1
    periods = np.unique(period[known])
    if len(periods) < 2:
        raise ValueError("для схемы по времени нужно хотя бы 2 разных периода")
    chunks = np.array_split(periods, min(n_folds, len(periods) - 1) + 1)
    folds = []
    for i in range(1, len(chunks)):
        valid = np.isin(period, chunks[i])
        train = (period < chunks[i].min()) | ~known
        folds.append((np.flatnonzero(train), np.flatnonzero(valid)))
    return folds


def make_folds(
    spec: TaskSpec,
    scheme: str,
    column: str | None,
    train: pd.DataFrame,
    features: list[str],
    y: np.ndarray,
    n_folds: int,
    split_default,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Фолды для схемы. split_default(spec, y, n_folds) — обычные kfold/stratified (с правилом редких классов)."""
    if scheme in ("kfold", "stratified"):
        return split_default(spec, y, n_folds)
    if scheme == "group":
        if column == DUPLICATES:
            groups = duplicate_groups(train[features])
        elif column in train.columns:
            groups = train[column].astype(str).to_numpy()
        else:
            raise ValueError(f"нет колонки {column!r} для схемы group")
        return _group_folds(spec, y, groups, n_folds)
    if column not in train.columns:
        raise ValueError(f"нет колонки {column!r} для схемы time")
    t = time_values(train[column])
    if t is None:
        raise ValueError(f"колонка {column!r} не похожа на время")
    folds = _time_folds(t, n_folds)
    if _is_classification(spec):
        # Период с одним классом (редкое событие) ломает и обучение, и метрику вроде ROC-AUC — такие фолды выбрасываем
        y = np.asarray(y)
        folds = [(tr, va) for tr, va in folds if len(np.unique(y[tr])) > 1 and len(np.unique(y[va])) > 1]
        if not folds:
            raise ValueError(f"схема time:{column}: в каждом фолде обучение или проверка содержат один класс")
    return folds


def _time_folds_ok(spec: TaskSpec, train: pd.DataFrame, column: str, features: list[str]) -> bool:
    """Схема time:<колонка> даёт хотя бы 2 рабочих фолда (проверка «всухую», без обучения)."""
    if spec.target not in train.columns:
        return True
    try:
        folds = make_folds(spec, "time", column, train, features, train[spec.target].to_numpy(), 5, None)
    except ValueError as err:
        log.info("Схема time:%s не подходит: %s", column, err)
        return False
    if len(folds) < 2:
        log.info("Схема time:%s не подходит: рабочих фолдов %d", column, len(folds))
        return False
    return True


def _id_like(series: pd.Series) -> bool:
    """Значения — имена или номера объектов: строки/категории или целые числа. Непрерывное число (температура года)
    при сдвиге во времени тоже «новое» в test, но объектом не является (public/regression: погода года 6)."""
    if not pd.api.types.is_numeric_dtype(series) or pd.api.types.is_bool_dtype(series):
        return True
    values = series.dropna().to_numpy(dtype=float)
    return bool(np.all(values == np.round(values)))


def group_candidates(train: pd.DataFrame, test: pd.DataFrame, features: list[str]) -> list[str]:
    """Колонки, похожие на объекты, которые целиком ушли в test: в train значения повторяются, в test почти все новые.
    Значения — имена или целые номера (_id_like), не непрерывные числа.

    Порядок — от более крупных групп (меньше значений) к мелким: если test — новые больницы, то и пациенты в нём новые,
    а честные фолды дают только больницы. При случайном разбиении повторяющиеся значения есть и в test — не кандидат.
    """
    found = []
    for col in features:
        if col not in test.columns:
            continue
        a, b = train[col].dropna(), test[col].dropna()
        n_groups = a.nunique()
        if len(b) == 0 or n_groups < GROUP_MIN_GROUPS or len(a) < GROUP_MIN_ROWS * n_groups or not _id_like(a):
            continue
        seen = set(a.astype(str).unique())
        new_share = float((~b.astype(str).isin(seen)).mean())
        if new_share >= GROUP_NEW_SHARE:
            found.append((n_groups, col))
    return [col for _, col in sorted(found)]


def _group_folds_ok(spec: TaskSpec, train: pd.DataFrame, column: str, features: list[str]) -> bool:
    """Схема group:<колонка> даёт хотя бы 2 фолда, а у классификации в каждом фолде больше одного класса."""
    if spec.target not in train.columns:
        return True
    y = train[spec.target].to_numpy()
    try:
        folds = make_folds(spec, "group", column, train, features, y, 5, None)
    except ValueError as err:
        log.info("Схема group:%s не подходит: %s", column, err)
        return False
    if len(folds) < 2:
        log.info("Схема group:%s не подходит: фолдов %d", column, len(folds))
        return False
    if _is_classification(spec) and any(len(np.unique(y[tr])) < 2 or len(np.unique(y[va])) < 2 for tr, va in folds):
        log.info("Схема group:%s не подходит: в фолде один класс", column)
        return False
    return True


def choose_validation(spec: TaskSpec, profile: DataProfile, train: pd.DataFrame, test: pd.DataFrame) -> str:
    """Схема по умолчанию, похожая на то, как устроен скрытый test. Вызывать один раз до цикла (ADR 0008, D1).

    1. Есть колонка времени, и test лежит после train → time:<колонка> (обучение на прошлом, проверка на будущем).
       Колонка — время, только если имя «про время», это даты строкой или значений не больше TIME_MAX_PERIODS:
       номер строки (record_no) на случайном разбиении тоже «продолжается» в test, но временем не является.
       Кандидат, у которого после выброса фолдов с одним классом осталось меньше 2 фолдов, пропускается.
    2. Иначе есть колонка объектов, которые целиком ушли в test (в train значения повторяются, в test ≥ GROUP_NEW_SHARE
       строк с новыми значениями) → group:<колонка> (#288). Идентификатор строки не повторяется в train — не группы.
       Фолды не получаются (мало групп, один класс в фолде) — следующий кандидат, затем прежние правила.
    3. Иначе ≥ DUPLICATE_SHARE строк train — полные дубликаты признаков → group:duplicates.
    4. Иначе kfold / stratified.
    """
    features = profile.numeric_features + profile.categorical_features
    candidates = []
    for col in features:
        if col not in test.columns:
            continue
        t_train, t_test = time_values(train[col]), time_values(test[col])
        if t_train is None or t_test is None or np.isnan(t_test).all():
            continue
        t_train, t_test = t_train[~np.isnan(t_train)], t_test[~np.isnan(t_test)]
        n_values = len(np.unique(t_train))
        if n_values < TIME_MIN_PERIODS:
            continue
        is_date_text = pd.to_numeric(train[col], errors="coerce").notna().mean() < TIME_PARSE_SHARE
        if not (TIME_NAME_HINT.search(col) or is_date_text or n_values <= TIME_MAX_PERIODS):
            continue
        # test «после» train: почти весь test не раньше почти всего train, а середина test — строго позже.
        # Строгость отсекает почти постоянные колонки (флаг, который почти всегда 0): у них «0 ≥ 0» выполняется
        q95_train = np.quantile(t_train, 0.95)
        if np.quantile(t_test, 0.05) >= q95_train and np.median(t_test) > q95_train:
            candidates.append(col)
    candidates.sort(key=lambda c: (not TIME_NAME_HINT.search(c), len(np.unique(train[c].dropna()))))
    for col in candidates:
        if _time_folds_ok(spec, train, col, features):
            log.info("Схема валидации: по времени (%s) — test лежит после train; кандидаты: %s", col, candidates)
            return f"time:{col}"
    groups = group_candidates(train, test, features)
    for col in groups:
        if _group_folds_ok(spec, train, col, features):
            log.info("Схема валидации: по группам (%s) — в test другие объекты; кандидаты: %s", col, groups)
            return f"group:{col}"
    if features:
        share = float(pd.Series(duplicate_groups(train[features])).duplicated(keep=False).mean())
        if share >= DUPLICATE_SHARE:
            log.info("Схема валидации: группы дубликатов — %.0f%% строк train повторяются", share * 100)
            return f"group:{DUPLICATES}"
    scheme = default_scheme(spec)
    log.info("Схема валидации: %s — времени, групп и заметных дубликатов нет", scheme)
    return scheme
