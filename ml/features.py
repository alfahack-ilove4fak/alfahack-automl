"""Преобразования признаков для params.feature_ops. Владелец — ML-pipeline инженер.

Каждая операция — класс с fit(x, y) и transform(x) -> новые колонки (без исходных).
fit видит только train-часть фолда (в CV) или весь train (финальное обучение) — так нет утечки из валидации и test.

Операции (ключ в feature_ops → класс):
  missing_flags    — флаг «пропуск» для числовых колонок, где в train были пропуски
  freq_encoding    — доля категории в train (редкое и новое значение → маленькое число, а не «неизвестно»)
  target_encoding  — среднее target по категории со сглаживанием; для строк самого train — out-of-fold,
                     иначе строка видит собственный target. Классы — по вероятности каждого класса (до 20 классов)
  pair_ops         — разности и отношения пар самых информативных числовых признаков (до 5 признаков → 20 колонок)
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


def _categorical(x: pd.DataFrame) -> list[str]:
    return [c for c in x.columns if isinstance(x[c].dtype, pd.CategoricalDtype)]


def _numeric(x: pd.DataFrame) -> list[str]:
    return [c for c in x.columns if not isinstance(x[c].dtype, pd.CategoricalDtype)]


class MissingFlags:
    def fit(self, x: pd.DataFrame, y: np.ndarray, classification: bool) -> MissingFlags:
        self.cols = [c for c in _numeric(x) if x[c].isna().any()]
        return self

    def transform(self, x: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame({f"{c}__isna": x[c].isna().astype(np.float32) for c in self.cols}, index=x.index)


class FreqEncoding:
    def fit(self, x: pd.DataFrame, y: np.ndarray, classification: bool) -> FreqEncoding:
        self.freq = {c: x[c].astype(str).value_counts(normalize=True) for c in _categorical(x)}
        return self

    def transform(self, x: pd.DataFrame) -> pd.DataFrame:
        out = {f"{c}__freq": x[c].astype(str).map(f).fillna(0.0).astype(np.float32) for c, f in self.freq.items()}
        return pd.DataFrame(out, index=x.index)


class TargetEncoding:
    """Сглаженное среднее target по категории. transform того же объекта, что в fit, отдаёт out-of-fold значения."""

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
        classes = np.unique(y)
        if len(classes) == 2:
            return pd.DataFrame({"p": (y == classes[1]).astype(float)})
        return pd.DataFrame({f"p{k}": (y == k).astype(float) for k in classes[:TE_MAX_CLASSES]})

    @staticmethod
    def _table(cat: pd.Series, targets: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        stats = targets.groupby(cat.to_numpy()).agg(["sum", "count"])
        prior = targets.mean()
        table = pd.DataFrame(
            {
                t: (stats[(t, "sum")] + TE_SMOOTHING * prior[t]) / (stats[(t, "count")] + TE_SMOOTHING)
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
        for tr, va in KFold(TE_INNER_FOLDS, shuffle=True, random_state=SEED).split(x):
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
        self.top = [numeric[i] for i in np.argsort(-score)[:PAIR_TOP_K]]
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


def apply_feature_ops(
    ops: list[str], x_fit: pd.DataFrame, y_fit: np.ndarray, classification: bool, *others: pd.DataFrame
) -> list[pd.DataFrame]:
    """Обучить операции на (x_fit, y_fit) и добавить их колонки к x_fit и ко всем others (валидация, test)."""
    unknown = [op for op in ops if op not in FEATURE_OPS]
    if unknown:
        raise ValueError(f"неизвестные feature_ops {unknown}, доступны: {sorted(FEATURE_OPS)}")
    frames = [x_fit, *others]
    for op in ops:
        fitted = FEATURE_OPS[op]().fit(x_fit, y_fit, classification)
        frames = [
            pd.concat([f, fitted.transform(src)], axis=1) for f, src in zip(frames, [x_fit, *others], strict=True)
        ]
    return frames
