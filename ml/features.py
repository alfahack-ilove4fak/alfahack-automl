"""Преобразования признаков для params.feature_ops. Владелец — ML-pipeline инженер.

Каждая операция — класс с fit(x, y) и transform(x) -> новые колонки (без исходных).
fit видит только train-часть фолда (в CV) или весь train (финальное обучение) — так нет утечки из валидации и test.

Операции (ключ в feature_ops → класс):
  missing_flags    — флаг «пропуск» для колонок, где в train были пропуски (и в категориях), + n_missing на строку;
                     не больше 100 флагов — самые «дырявые» колонки
  freq_encoding    — доля категории в train (редкое и новое значение → маленькое число, а не «неизвестно»)
  target_encoding  — среднее target по категории со сглаживанием; для строк самого train — out-of-fold,
                     иначе строка видит собственный target. Классы — по вероятности каждого класса (до 20 самых
                     частых). Параметры: smoothing (20), folds (5)
  pair_ops         — разности и отношения пар самых информативных числовых признаков. Параметр: top_k (5 признаков
                     → 20 колонок)

Формат params.feature_ops: имя, список имён или словари с параметрами, например
  feature_ops: [missing_flags, {op: target_encoding, smoothing: 50}, {op: pair_ops, top_k: 3}]
Неизвестная операция, параметр или его значение → ошибка до обучения → гипотеза получает FAILED.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.feature_selection import f_classif
from sklearn.model_selection import KFold

SEED = 42
TE_SMOOTHING = 20.0  # вес априорного среднего: категория из 20 объектов — наполовину своё среднее, наполовину общее
TE_INNER_FOLDS = 5
TE_MAX_CLASSES = 20
PAIR_TOP_K = 5
MAX_FLAGS = 100  # больше флагов пропусков не добавляем: на широких данных они раздувают матрицу
MISSING_STRINGS = ["nan", "None", "<NA>"]  # так astype(str) в make_matrix записывает пропуск в категории


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


class MissingFlags:
    def fit(self, x: pd.DataFrame, y: np.ndarray, classification: bool) -> MissingFlags:
        share = {c: _is_missing(x[c]).mean() for c in x.columns}
        # Если колонок с пропусками больше лимита — берём самые «дырявые»
        self.cols = sorted((c for c in x.columns if share[c] > 0), key=lambda c: -share[c])[:MAX_FLAGS]
        return self

    def transform(self, x: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame({f"{c}__isna": _is_missing(x[c]).astype(np.float32) for c in self.cols}, index=x.index)
        if self.cols:
            out["n_missing"] = out.sum(axis=1).astype(np.float32)  # сколько пропусков в строке
        return out


class FreqEncoding:
    def fit(self, x: pd.DataFrame, y: np.ndarray, classification: bool) -> FreqEncoding:
        self.freq = {c: x[c].astype(str).value_counts(normalize=True) for c in _categorical(x)}
        return self

    def transform(self, x: pd.DataFrame) -> pd.DataFrame:
        out = {f"{c}__freq": x[c].astype(str).map(f).fillna(0.0).astype(np.float32) for c, f in self.freq.items()}
        return pd.DataFrame(out, index=x.index)


class TargetEncoding:
    """Сглаженное среднее target по категории. transform того же объекта, что в fit, отдаёт out-of-fold значения."""

    def __init__(self, smoothing: float = TE_SMOOTHING, folds: int = TE_INNER_FOLDS):
        if smoothing < 0 or folds < 2:
            raise ValueError("target_encoding: smoothing ≥ 0, folds ≥ 2")
        self.smoothing, self.folds = float(smoothing), int(folds)

    def fit(self, x: pd.DataFrame, y: np.ndarray, classification: bool) -> TargetEncoding:
        self.cols = _categorical(x)
        self.targets = self._targets(y, classification)
        self.tables = {c: self._table(x[c].astype(str), self.targets) for c in self.cols}
        self._fit_x = x  # transform(тот же объект) → out-of-fold; по индексу нельзя: у test он тоже с 0
        self.oof = self._out_of_fold(x)
        return self

    @staticmethod
    def _targets(y: np.ndarray, classification: bool) -> pd.DataFrame:
        if not classification:
            return pd.DataFrame({"mean": y.astype(float)})
        classes, counts = np.unique(y, return_counts=True)
        if len(classes) == 2:
            return pd.DataFrame({"p": (y == classes[1]).astype(float)})
        # Классов больше лимита — колонки для самых частых, а не для первых по номеру
        top = np.sort(classes[np.argsort(-counts, kind="stable")[:TE_MAX_CLASSES]])
        return pd.DataFrame({f"p{k}": (y == k).astype(float) for k in top})

    def _table(self, cat: pd.Series, targets: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        stats = targets.groupby(cat.to_numpy()).agg(["sum", "count"])
        prior = targets.mean()
        table = pd.DataFrame(
            {
                t: (stats[(t, "sum")] + self.smoothing * prior[t]) / (stats[(t, "count")] + self.smoothing)
                for t in targets.columns
            }
        )
        return table, prior

    @staticmethod
    def _encode(cat: pd.Series, table: pd.DataFrame, prior: pd.Series, col: str) -> dict[str, np.ndarray]:
        joined = table.reindex(cat.to_numpy())
        return {f"{col}__te_{t}": joined[t].fillna(prior[t]).to_numpy(np.float32) for t in table.columns}

    def _out_of_fold(self, x: pd.DataFrame) -> pd.DataFrame:
        parts = []
        for tr, va in KFold(min(self.folds, len(x)), shuffle=True, random_state=SEED).split(x):
            cols: dict[str, np.ndarray] = {}
            for c in self.cols:
                table, prior = self._table(x[c].iloc[tr].astype(str), self.targets.iloc[tr].reset_index(drop=True))
                cols.update(self._encode(x[c].iloc[va].astype(str), table, prior, c))
            parts.append(pd.DataFrame(cols, index=x.index[va]))
        return pd.concat(parts).reindex(x.index) if parts else pd.DataFrame(index=x.index)

    def transform(self, x: pd.DataFrame) -> pd.DataFrame:
        if x is self._fit_x:
            return self.oof
        cols: dict[str, np.ndarray] = {}
        for c in self.cols:
            table, prior = self.tables[c]
            cols.update(self._encode(x[c].astype(str), table, prior, c))
        return pd.DataFrame(cols, index=x.index)


class PairOps:
    def __init__(self, top_k: int = PAIR_TOP_K):
        if top_k < 2:
            raise ValueError("pair_ops: top_k ≥ 2")
        self.top_k = int(top_k)

    def fit(self, x: pd.DataFrame, y: np.ndarray, classification: bool) -> PairOps:
        numeric = [c for c in _numeric(x) if x[c].nunique() > 2]
        if not numeric:
            self.top = []
            return self
        filled = x[numeric].fillna(x[numeric].median())
        if classification:
            score = np.nan_to_num(f_classif(filled, y)[0])
        else:
            score = np.nan_to_num(np.abs(filled.corrwith(pd.Series(y, index=x.index).astype(float)).to_numpy()))
        self.top = [numeric[i] for i in np.argsort(-score)[: self.top_k]]
        return self

    def transform(self, x: pd.DataFrame) -> pd.DataFrame:
        out = {}
        for a, b in combinations(self.top, 2):
            out[f"{a}__minus__{b}"] = (x[a] - x[b]).astype(np.float32)
            ratio = x[a] / x[b].where(x[b] != 0)
            out[f"{a}__div__{b}"] = ratio.replace([np.inf, -np.inf], np.nan).astype(np.float32)
        return pd.DataFrame(out, index=x.index)


FEATURE_OPS = {
    "missing_flags": MissingFlags,
    "freq_encoding": FreqEncoding,
    "target_encoding": TargetEncoding,
    "pair_ops": PairOps,
}


def parse_feature_ops(raw) -> list[tuple[str, dict]]:
    """params.feature_ops → [(имя, параметры)]. None или [] — операций нет.

    Принимает имя, список имён и словарей {op: имя, параметр: значение}. Каждую операцию сразу создаём:
    неверный параметр — ошибка здесь, до обучения, а не на первом фолде.
    """
    if not raw:
        return []
    if isinstance(raw, str | dict):
        raw = [raw]
    if not isinstance(raw, list | tuple):
        raise ValueError(f"feature_ops: ожидается имя или список, получено {raw!r}")
    ops, unknown = [], []
    for item in raw:
        if isinstance(item, str):
            name, options = item, {}
        elif isinstance(item, dict) and "op" in item:
            options = dict(item)
            name = options.pop("op")
        else:
            raise ValueError(f"feature_ops: ожидается имя или {{op: имя, ...}}, получено {item!r}")
        name = str(name).strip().lower()
        if name not in FEATURE_OPS:
            unknown.append(name)
            continue
        FEATURE_OPS[name](**options)  # TypeError на чужой параметр, ValueError на плохое значение
        ops.append((name, options))
    if unknown:
        raise ValueError(f"неизвестные feature_ops {unknown}, доступны: {sorted(FEATURE_OPS)}")
    return ops


def apply_feature_ops(
    ops, x_fit: pd.DataFrame, y_fit: np.ndarray, classification: bool, *others: pd.DataFrame
) -> list[pd.DataFrame]:
    """Обучить операции на (x_fit, y_fit) и добавить их колонки к x_fit и ко всем others (валидация, test).

    ops — params.feature_ops в любом формате parse_feature_ops. Имена новых колонок не совпадают со старыми.
    """
    frames = [x_fit, *others]
    for name, options in parse_feature_ops(ops):
        fitted = FEATURE_OPS[name](**options).fit(x_fit, y_fit, classification)
        new = [fitted.transform(src) for src in [x_fit, *others]]  # x_fit — тот же объект: так TE даёт out-of-fold
        names = _unique_names(frames[0].columns, list(new[0].columns))
        frames = [pd.concat([f, n.set_axis(names, axis=1)], axis=1) for f, n in zip(frames, new, strict=True)]
    return frames


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
