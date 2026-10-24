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
  object_history   — история объекта: по строкам того же объекта строго из прошлых периодов — среднее и последнее
                     значение target, число прошлых строк и время с последнего периода. Ключ объекта ищется по данным
                     (колонки, которые вместе отличают объекты внутри периода и повторяются из периода в период), без
                     имён колонок в коде; колонка времени — по имени «про время» и тому же правилу. Параметры: key
                     (список колонок), time (колонка) — их может выбрать LLM из object_history_candidates; не нашли
                     ключ или время — операция ничего не добавляет

Формат params.feature_ops: имя, список имён или словари с параметрами, например
  feature_ops: [missing_flags, {op: target_encoding, smoothing: 50}, {op: pair_ops, top_k: 3}]
При схеме валидации time:<колонка> object_history без явного time берёт эту колонку (with_history_time).
Неизвестная операция, параметр или его значение → ошибка до обучения → гипотеза получает FAILED.
"""

from __future__ import annotations

import re
from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.feature_selection import f_classif
from sklearn.model_selection import KFold

from ml.validation import TIME_MIN_PERIODS, TIME_NAME_HINT, time_values

SEED = 42
TE_SMOOTHING = 20.0  # вес априорного среднего: категория из 20 объектов — наполовину своё среднее, наполовину общее
TE_INNER_FOLDS = 5
TE_MAX_CLASSES = 20
PAIR_TOP_K = 5
MAX_FLAGS = 100  # больше флагов пропусков не добавляем: на широких данных они раздувают матрицу
MISSING_STRINGS = ["nan", "None", "<NA>"]  # так astype(str) в make_matrix записывает пропуск в категории
HISTORY_MAX_KEY_COLS = 4  # ключ объекта — не больше стольких колонок
HISTORY_CANDIDATE_COLS = 30  # в ключ пробуем столько колонок с наибольшим числом значений
HISTORY_SEARCH_ROWS = 200_000  # поиск ключа — на первых строках: дальше он только дольше, а не точнее
HISTORY_MIN_UNIQUE = 0.8  # доля строк, где (объект, период) встречается один раз
HISTORY_MIN_REPEAT = 0.3  # доля строк, чей объект есть больше чем в одном периоде
HISTORY_ATTR_SHARE = 0.95  # атрибут объекта постоянен хотя бы в такой доле повторяющихся объектов
HISTORY_SCORE_TIE = 0.02  # кандидаты времени с такой разницей качества считаем равными


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


def _history_times(series: pd.Series) -> np.ndarray | None:
    """Колонка → числа для упорядочивания по времени; категории сначала в строки (пропуск → NaN)."""
    if isinstance(series.dtype, pd.CategoricalDtype):
        series = series.astype(object).where(~series.isin(MISSING_STRINGS))
    return time_values(series)


def _column_hashes(x: pd.DataFrame, cols: list[str]) -> dict[str, np.ndarray]:
    return {c: pd.util.hash_pandas_object(x[c].astype(object), index=False).to_numpy() for c in cols}


def _key_hash(hashes: dict[str, np.ndarray], cols: list[str]) -> np.ndarray:
    """Хеш набора колонок из хешей отдельных колонок (переполнение uint64 — нормально)."""
    out = np.zeros(len(next(iter(hashes.values()))), dtype=np.uint64)
    with np.errstate(over="ignore"):
        for c in cols:
            out = out * np.uint64(1_000_003) + hashes[c]
    return out


def _key_quality(key: np.ndarray, t_codes: np.ndarray, n_periods: int) -> tuple[float, float]:
    """(доля строк с единственной парой «объект, период», доля строк, чей объект есть в нескольких периодах)."""
    k_codes, k_uniques = pd.factorize(key)
    pair_codes, pairs = pd.factorize(k_codes.astype(np.int64) * n_periods + t_codes)
    unique = float((np.bincount(pair_codes)[pair_codes] == 1).mean())
    periods = np.bincount(pairs // n_periods, minlength=len(k_uniques))  # различные пары → периодов у объекта
    return unique, float((periods[k_codes] > 1).mean())


def _has_attribute(x: pd.DataFrame, key: list[str], time_col: str) -> bool:
    """Есть колонка вне ключа, постоянная внутри объекта (город магазина, штат здания) и разная у объектов.

    Без такого «атрибута» совпадения ключа — скорее разные объекты с одинаковым значением (площадь квартиры).
    """
    groups = pd.Series(_key_hash(_column_hashes(x, key), key), index=x.index)
    repeated = groups.duplicated(keep=False).to_numpy()
    if not repeated.any():
        return False
    rest = x.loc[repeated, [c for c in x.columns if c not in key and c != time_col]]
    by = groups[repeated].to_numpy()
    for c in rest.columns:
        values = rest[c].astype(object)
        if values.nunique() > 1 and (values.groupby(by).nunique(dropna=False) == 1).mean() >= HISTORY_ATTR_SHARE:
            return True
    return False


def _search_key(
    x: pd.DataFrame, t: np.ndarray, time_col: str, hashes: dict[str, np.ndarray] | None = None
) -> tuple[list[str], float]:
    """Жадно собрать ключ объекта: на каждом шаге колонка, лучше всего поднимающая unique·repeat.

    Постоянные внутри объекта колонки (площадь, год постройки, тип) вместе отличают объекты внутри периода
    и повторяются из периода в период. Меняющиеся (погода, показания) уникальны, но не повторяются — repeat
    низкий; общие для многих объектов (штат) повторяются, но не уникальны. Target не используется.
    """
    hashes = hashes if hashes is not None else _column_hashes(x, list(x.columns))
    ok = ~np.isnan(t)
    x, t = x[ok], t[ok]
    if len(x) == 0:
        return [], 0.0
    t_codes, periods = pd.factorize(t)
    hashes = {c: h[ok] for c, h in hashes.items() if c != time_col}
    cardinality = {c: len(pd.unique(h)) for c, h in hashes.items()}
    cols = sorted((c for c, n in cardinality.items() if n > 1), key=lambda c: -cardinality[c])
    cols = cols[:HISTORY_CANDIDATE_COLS]
    if not cols:
        return [], 0.0
    key, best, unique, repeat = [], 0.0, 0.0, 0.0
    while len(key) < HISTORY_MAX_KEY_COLS:
        scored = []
        for c in cols:
            if c not in key:
                u, r = _key_quality(_key_hash(hashes, [*key, c]), t_codes, len(periods))
                scored.append((u * r, u, r, c))
        if not scored:
            break
        score, u, r, col = max(scored, key=lambda s: s[0])
        if score <= best + 1e-3:
            break
        key, best, unique, repeat = [*key, col], score, u, r
        if unique >= 0.999:
            break
    if unique < HISTORY_MIN_UNIQUE or repeat < HISTORY_MIN_REPEAT or not _has_attribute(x, key, time_col):
        return [], 0.0
    return key, best


def object_history_candidates(x: pd.DataFrame) -> list[dict]:
    """Кандидаты для object_history, найденные правилами: [{time, key, score}], лучший первым.

    Время — колонки с именем «про время», которые читаются как числа или даты и имеют ≥ 3 периодов; из почти
    равных по качеству первым идёт самое крупное (меньше периодов).
    Отсюда LLM выбирает key и time для гипотезы; признаки считает наш код.
    """
    x = x.iloc[:HISTORY_SEARCH_ROWS]
    hashes = None  # хеши колонок — один раз на всех кандидатов времени
    out = []
    for col in x.columns:
        if not TIME_NAME_HINT.search(str(col)):
            continue
        t = _history_times(x[col])
        if t is None or len(np.unique(t[~np.isnan(t)])) < TIME_MIN_PERIODS:
            continue
        hashes = hashes if hashes is not None else _column_hashes(x, list(x.columns))
        key, score = _search_key(x, t, col, hashes)
        if key:
            out.append(
                {"time": col, "key": key, "score": round(score, 4), "periods": int(len(np.unique(t[~np.isnan(t)])))}
            )
    if not out:
        return []
    # Из почти равных по качеству берём самое крупное время: погода по году (градусо-дни) делит строки почти
    # как сам год, но «прошлое» по ней — не прошлое
    top = max(c["score"] for c in out)
    return sorted(out, key=lambda c: (c["score"] < top - HISTORY_SCORE_TIE, c["periods"], -c["score"]))


class ObjectHistory:
    """Признаки по прошлым периодам того же объекта. Для любой строки — только строки fit с тем же ключом и
    временем строго меньше: текущая строка, её период и будущее не видны ни в train, ни в валидации, ни в test.
    """

    def __init__(self, key: list[str] | str | None = None, time: str | None = None, time_hint: str | None = None):
        """key, time — явный выбор (LLM): нет колонки → ошибка. time_hint — колонка схемы time:, если она есть."""
        if isinstance(key, str):
            key = [key]
        if key is not None and (not isinstance(key, list | tuple) or not key):
            raise ValueError("object_history: key — непустой список колонок")
        if time is not None and not isinstance(time, str):
            raise ValueError("object_history: time — имя колонки")
        self.key_param, self.time_param, self.time_hint = (list(key) if key else None), time, time_hint

    @staticmethod
    def _column(x: pd.DataFrame, name: str) -> str | None:
        """Имя колонки в матрице: как есть или после замены спецсимволов (make_matrix переименовывает)."""
        if name in x.columns:
            return name
        safe = re.sub(r"\W+", "_", name).strip("_")
        return safe if safe in x.columns else None

    def fit(self, x: pd.DataFrame, y: np.ndarray, classification: bool) -> ObjectHistory:
        self.key, self.time = [], None
        time = self._column(x, self.time_param) if self.time_param else None
        key = [self._column(x, k) for k in self.key_param or []]
        if (self.time_param and time is None) or None in key:
            raise ValueError(f"object_history: нет колонок key={self.key_param} time={self.time_param}")
        if time is None and self.time_hint:
            time = self._column(x, self.time_hint)  # колонки схемы может не быть среди признаков — тогда ищем сами
        if time is None:
            found = [c for c in object_history_candidates(x) if not key or c["key"] == key]
            if not found:
                return self
            time = found[0]["time"]
            key = key or found[0]["key"]
        elif not key:
            head = x.iloc[:HISTORY_SEARCH_ROWS]
            t_head = _history_times(head[time])
            key = _search_key(head, t_head, time)[0] if t_head is not None else []
            if not key:
                return self
        t = _history_times(x[time])
        if t is None:
            return self
        self.key, self.time = key, time
        targets = TargetEncoding._targets(y, classification)
        rows = pd.DataFrame({"_k": _key_hash(_column_hashes(x, key), key), "_t": t})
        rows = pd.concat([rows, targets.set_axis(rows.index)], axis=1).dropna(subset=["_t"])
        self.target_cols = list(targets.columns)
        # Таблица периодов объекта: сумма и число строк за период, накопленные суммы до периода включительно
        per = rows.groupby(["_k", "_t"]).agg({**{c: "sum" for c in self.target_cols}, "_t": "size"})
        per = per.rename(columns={"_t": "_n"}).reset_index()
        cum = per.groupby("_k")[[*self.target_cols, "_n"]].cumsum()
        table = pd.DataFrame({"_k": per["_k"], "_t": per["_t"], "_last_t": per["_t"], "_cnt": cum["_n"]})
        for c in self.target_cols:
            table[f"_mean_{c}"] = cum[c] / cum["_n"]
            table[f"_last_{c}"] = per[c] / per["_n"]
        self.table = table.sort_values("_t", kind="stable").reset_index(drop=True)
        return self

    def transform(self, x: pd.DataFrame) -> pd.DataFrame:
        if not self.key:
            return pd.DataFrame(index=x.index)
        t = _history_times(x[self.time])
        query = pd.DataFrame({"_k": _key_hash(_column_hashes(x, self.key), self.key), "_t": np.nan if t is None else t})
        query["_row"] = np.arange(len(x))
        known = query.dropna(subset=["_t"]).sort_values("_t", kind="stable")
        # Последний период того же объекта строго раньше строки: allow_exact_matches=False не берёт её период
        joined = (
            pd.merge_asof(known, self.table, on="_t", by="_k", direction="backward", allow_exact_matches=False)
            .set_index("_row")
            .reindex(np.arange(len(x)))
        )
        out = {}
        for c in self.target_cols:
            out[f"hist__{c}__mean"] = joined[f"_mean_{c}"].to_numpy(np.float32)
            out[f"hist__{c}__last"] = joined[f"_last_{c}"].to_numpy(np.float32)
        out["hist__count"] = joined["_cnt"].fillna(0).to_numpy(np.float32)
        since = query["_t"].to_numpy() - joined["_last_t"].to_numpy()
        out["hist__since_last"] = since.astype(np.float32)
        return pd.DataFrame(out, index=x.index)


FEATURE_OPS = {
    "missing_flags": MissingFlags,
    "freq_encoding": FreqEncoding,
    "target_encoding": TargetEncoding,
    "pair_ops": PairOps,
    "object_history": ObjectHistory,
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
        if isinstance(item, tuple) and len(item) == 2:  # уже разобранная (имя, параметры) — из with_history_time
            item = {"op": item[0], **item[1]}
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


def with_history_time(raw, time_column: str | None) -> list[tuple[str, dict]]:
    """Операциям object_history без явного time — колонка времени из схемы валидации time:<колонка>.

    История по тому же времени, по которому режутся фолды: «прошлое» признака и «прошлое» валидации совпадают.
    """
    ops = parse_feature_ops(raw)
    if not time_column:
        return ops
    return [
        (name, {**options, "time_hint": time_column}) if name == "object_history" else (name, options)
        for name, options in ops
    ]


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
