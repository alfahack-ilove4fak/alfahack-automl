"""Библиотека преобразований признаков для params.feature_ops. Владелец — ML-pipeline инженер.

Операции добавляют новые колонки к матрице из make_matrix и ничего не удаляют. Всё, что «учится» (какие колонки
с пропусками, частоты категорий, средние target, какие числовые признаки брать в пары), считается только на
train-части фолда (fit_transform) и затем применяется к валидации фолда или к test (transform) — без утечки.

Операции (имя в params.feature_ops → что добавляет):
  missing_indicators — <колонка>__isna (0/1) для колонок, где в train-части есть пропуски, и n_missing на строку
  frequency_encoding — <категория>__freq: доля значения в train-части; новое значение → 0
  target_encoding    — <категория>__te: сглаженное среднее target по значению (multiclass — по колонке на класс).
                       Для строк train-части считается вне фолда (внутренние фолды), иначе строка «видит»
                       свой target и модель переобучается. Параметры: smoothing (20), folds (5)
  pair_diffs         — <a>__minus__<b> для пар из top_k (6) числовых признаков, сильнее связанных с target
  pair_ratios        — <a>__div__<b> для тех же пар; деление на ~0 → NaN, а не inf

Формат params.feature_ops: имя, список имён или список словарей с параметрами, например
  feature_ops: [missing_indicators, {op: target_encoding, smoothing: 50}]
Неизвестная операция или параметр → ValueError/TypeError → гипотеза получает FAILED.
"""

from __future__ import annotations

import warnings
from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.feature_selection import f_classif, f_regression
from sklearn.model_selection import KFold

from interfaces.schemas import TaskSpec, TaskType

SEED = 42
MAX_INDICATORS = 100  # больше колонок-индикаторов не добавляем: на широких данных они раздувают матрицу
MAX_TE_CLASSES = 10  # multiclass: колонки target encoding только для самых частых классов
EPS = 1e-12
MISSING_STRINGS = ["nan", "None", "<NA>"]  # так astype(str) в make_matrix записывает пропуски


def _categorical(x: pd.DataFrame) -> list[str]:
    return [c for c in x.columns if isinstance(x[c].dtype, pd.CategoricalDtype)]


def _numeric(x: pd.DataFrame) -> list[str]:
    return [c for c in x.columns if not isinstance(x[c].dtype, pd.CategoricalDtype)]


def _is_missing(series: pd.Series) -> np.ndarray:
    """Пропуск: NaN в числах; в категориях make_matrix делает из пропуска строку («nan», «None» или «<NA>»)."""
    missing = series.isna().to_numpy()
    if isinstance(series.dtype, pd.CategoricalDtype):
        missing |= series.isin(MISSING_STRINGS).to_numpy()
    return missing


class _Op:
    """Операция: fit_transform на train-части, transform на остальном. Возвращает только новые колонки."""

    def fit_transform(self, x: pd.DataFrame, y: np.ndarray) -> pd.DataFrame:
        self.fit(x, y)
        return self.transform(x)

    def fit(self, x: pd.DataFrame, y: np.ndarray) -> None:
        raise NotImplementedError

    def transform(self, x: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError


class MissingIndicators(_Op):
    def __init__(self, spec: TaskSpec):
        self.columns: list[str] = []

    def fit(self, x, y):
        share = {c: _is_missing(x[c]).mean() for c in x.columns}
        with_missing = [c for c in x.columns if share[c] > 0]
        # Самые «дырявые» — первыми, если колонок больше лимита
        self.columns = sorted(with_missing, key=lambda c: -share[c])[:MAX_INDICATORS]

    def transform(self, x):
        if not self.columns:
            return pd.DataFrame(index=x.index)
        flags = {f"{c}__isna": _is_missing(x[c]).astype(np.float32) for c in self.columns}
        out = pd.DataFrame(flags, index=x.index)
        out["n_missing"] = out.sum(axis=1).astype(np.float32)
        return out


class FrequencyEncoding(_Op):
    def __init__(self, spec: TaskSpec):
        self.freq: dict[str, np.ndarray] = {}

    def fit(self, x, y):
        # Коды категорий общие для train и test (словарь из make_matrix): частота — по коду, -1 (NaN) → ячейка 0
        self.freq = {}
        for c in _categorical(x):
            counts = np.bincount(x[c].cat.codes.to_numpy() + 1, minlength=len(x[c].cat.categories) + 1)
            self.freq[c] = counts / max(len(x), 1)

    def transform(self, x):
        cols = {f"{c}__freq": freq[x[c].cat.codes.to_numpy() + 1].astype(np.float32) for c, freq in self.freq.items()}
        return pd.DataFrame(cols, index=x.index)


class TargetEncoding(_Op):
    def __init__(self, spec: TaskSpec, smoothing: float = 20.0, folds: int = 5):
        if smoothing < 0 or folds < 2:
            raise ValueError("target_encoding: smoothing ≥ 0, folds ≥ 2")
        self.spec, self.smoothing, self.folds = spec, float(smoothing), int(folds)
        self.columns: list[str] = []
        self.classes: np.ndarray | None = None
        self.prior = np.zeros(1)
        self.tables: dict[str, np.ndarray] = {}

    def _targets(self, y: np.ndarray) -> np.ndarray:
        """Матрица n × m того, что усредняем: y для регрессии, 0/1 класса 1 для binary, 0/1 по классам для multiclass."""
        if self.spec.task_type == TaskType.REGRESSION:
            return np.asarray(y, dtype=float)[:, None]
        if self.spec.task_type == TaskType.BINARY:
            return (np.asarray(y) == 1).astype(float)[:, None]
        return (np.asarray(y)[:, None] == self.classes[None, :]).astype(float)

    def _table(self, codes: np.ndarray, t: np.ndarray, n_cats: int, prior: np.ndarray) -> np.ndarray:
        """Сглаженное среднее по каждому коду (строка 0 — пропуск): (сумма + prior·m) / (число + m).

        Значения, которых не было в train-части, получают prior (и при smoothing = 0).
        """
        idx = codes + 1
        count = np.bincount(idx, minlength=n_cats + 1).astype(float)[:, None]
        sums = np.stack([np.bincount(idx, weights=t[:, j], minlength=n_cats + 1) for j in range(t.shape[1])], axis=1)
        table = (sums + prior * self.smoothing) / np.maximum(count + self.smoothing, EPS)
        return np.where(count > 0, table, prior)

    def _names(self, c: str) -> list[str]:
        if self.classes is None:
            return [f"{c}__te"]
        return [f"{c}__te_{k}" for k in self.classes]

    def _frame(self, values: dict[str, np.ndarray], index: pd.Index) -> pd.DataFrame:
        cols = {}
        for c, v in values.items():
            for name, col in zip(self._names(c), v.T, strict=True):
                cols[name] = col.astype(np.float32)
        return pd.DataFrame(cols, index=index)

    def fit(self, x, y):
        self.columns = _categorical(x)
        if self.spec.task_type == TaskType.MULTICLASS:
            labels, counts = np.unique(y, return_counts=True)
            self.classes = np.sort(labels[np.argsort(-counts, kind="stable")[:MAX_TE_CLASSES]])
        t = self._targets(y)
        self.prior = t.mean(axis=0)
        self.tables = {
            c: self._table(x[c].cat.codes.to_numpy(), t, len(x[c].cat.categories), self.prior) for c in self.columns
        }

    def fit_transform(self, x, y):
        """Для train-части — значения вне фолда: строка кодируется по остальным внутренним фолдам."""
        self.fit(x, y)
        if not self.columns:
            return pd.DataFrame(index=x.index)
        t = self._targets(y)
        oof = {c: np.empty((len(x), t.shape[1])) for c in self.columns}
        inner = KFold(min(self.folds, len(x)), shuffle=True, random_state=SEED)
        for fit_rows, enc_rows in inner.split(np.arange(len(x))):
            prior = t[fit_rows].mean(axis=0)
            for c in self.columns:
                codes = x[c].cat.codes.to_numpy()
                table = self._table(codes[fit_rows], t[fit_rows], len(x[c].cat.categories), prior)
                oof[c][enc_rows] = table[codes[enc_rows] + 1]
        return self._frame(oof, x.index)

    def transform(self, x):
        values = {c: self.tables[c][x[c].cat.codes.to_numpy() + 1] for c in self.columns}
        return self._frame(values, x.index)


class _Pairs(_Op):
    """Общая часть pair_diffs и pair_ratios: выбор top_k числовых признаков по F-статистике с target."""

    def __init__(self, spec: TaskSpec, top_k: int = 6):
        if top_k < 2:
            raise ValueError(f"{type(self).__name__}: top_k ≥ 2")
        self.spec, self.top_k = spec, int(top_k)
        self.pairs: list[tuple[str, str]] = []

    def fit(self, x, y):
        numeric = _numeric(x)
        if len(numeric) < 2:
            self.pairs = []
            return
        values = x[numeric].astype(float)
        values = values.fillna(values.median()).fillna(0.0).to_numpy()
        score_fn = f_regression if self.spec.task_type == TaskType.REGRESSION else f_classif
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # постоянная колонка даёт NaN и предупреждение — считаем её слабой
            scores = np.nan_to_num(score_fn(values, y)[0], nan=0.0)
        # Порядок внутри пары — как в матрице, чтобы имена колонок не зависели от скоров
        top = [numeric[i] for i in sorted(np.argsort(-scores, kind="stable")[: self.top_k])]
        self.pairs = list(combinations(top, 2))

    def transform(self, x):
        cols = {}
        for a, b in self.pairs:
            name, value = self._combine(a, b, x[a].to_numpy(dtype=float), x[b].to_numpy(dtype=float))
            value = value.astype(np.float32)
            value[~np.isfinite(value)] = np.nan  # inf ломает линейную модель; переполнение float32 — тоже inf
            cols[name] = value
        return pd.DataFrame(cols, index=x.index)

    def _combine(self, a: str, b: str, va: np.ndarray, vb: np.ndarray) -> tuple[str, np.ndarray]:
        raise NotImplementedError


class PairDiffs(_Pairs):
    def _combine(self, a, b, va, vb):
        return f"{a}__minus__{b}", va - vb


class PairRatios(_Pairs):
    def _combine(self, a, b, va, vb):
        with np.errstate(divide="ignore", invalid="ignore"):
            return f"{a}__div__{b}", np.where(np.abs(vb) > EPS, va / vb, np.nan)


# Операции, которые понимает params.feature_ops. Новая = класс с fit/transform + строка здесь + строка в docstring
FEATURE_OPS = {
    "missing_indicators": MissingIndicators,
    "frequency_encoding": FrequencyEncoding,
    "target_encoding": TargetEncoding,
    "pair_diffs": PairDiffs,
    "pair_ratios": PairRatios,
}


def parse_feature_ops(raw) -> list[tuple[str, dict]]:
    """params.feature_ops → [(имя, параметры)]. None или [] — операций нет."""
    if not raw:
        return []
    items = [raw] if isinstance(raw, str | dict) else list(raw)
    ops = []
    for item in items:
        if isinstance(item, str):
            name, options = item, {}
        elif isinstance(item, dict) and "op" in item:
            options = dict(item)
            name = options.pop("op")
        else:
            raise ValueError(f"feature_ops: ожидается имя или {{op: имя, ...}}, получено {item!r}")
        name = str(name).strip().lower()
        if name not in FEATURE_OPS:
            raise ValueError(f"неизвестная операция признаков {name!r}, доступны: {sorted(FEATURE_OPS)}")
        ops.append((name, options))
    return ops


class FeatureOps:
    """Набор операций из params.feature_ops. fit_transform — на train-части, transform — на валидации/test.

    Без операций возвращает матрицу как есть. Колонки transform всегда те же и в том же порядке, что у fit_transform.
    """

    def __init__(self, spec: TaskSpec, raw):
        self.spec = spec
        self.ops = [(name, FEATURE_OPS[name], options) for name, options in parse_feature_ops(raw)]
        # Создаём сразу: неверный параметр операции — ошибка до обучения, а не на первом фолде
        for _, cls, options in self.ops:
            cls(spec, **options)
        self.fitted: list[_Op] = []
        self.names: list[str] = []

    def __bool__(self) -> bool:
        return bool(self.ops)

    def fit_transform(self, x: pd.DataFrame, y: np.ndarray) -> pd.DataFrame:
        if not self.ops:
            return x
        self.fitted = [cls(self.spec, **options) for _, cls, options in self.ops]
        parts = [op.fit_transform(x, y) for op in self.fitted]
        self.names = _unique_names(x.columns, [c for p in parts for c in p.columns])
        return self._join(x, parts)

    def transform(self, x: pd.DataFrame) -> pd.DataFrame:
        if not self.ops:
            return x
        return self._join(x, [op.transform(x) for op in self.fitted])

    def _join(self, x: pd.DataFrame, parts: list[pd.DataFrame]) -> pd.DataFrame:
        new = pd.concat(parts, axis=1) if parts else pd.DataFrame(index=x.index)
        new.columns = self.names
        return pd.concat([x, new], axis=1)


def _unique_names(taken, names: list[str]) -> list[str]:
    """Имена новых колонок без совпадений со старыми и между собой: дубль получает суффикс _1, _2…"""
    seen, out = set(taken), []
    for base in names:
        name, k = base, 1
        while name in seen:
            name, k = f"{base}_{k}", k + 1
        seen.add(name)
        out.append(name)
    return out
