"""Эксперимент = гипотеза → кросс-валидация → ExperimentResult. Владелец — ML-pipeline инженер.

Поддерживаемые params гипотезы (расширяйте — и документируйте в research/playbook.yaml):
  model          — "lightgbm" (по умолчанию) | "catboost" | "xgboost" | "linear". Список — MODEL_FITTERS.
                   linear — Ridge (регрессия) / LogisticRegression (классы) с масштабированием и one-hot;
                   её model_params: alpha (Ridge), C (LogisticRegression)
                   Неизвестное имя → ExperimentResult(status=FAILED), а не тихая подмена на LightGBM
  model_params   — словарь гиперпараметров именно выбранной модели (у LightGBM num_leaves, у CatBoost depth);
                   чужой параметр → FAILED с ошибкой библиотеки
  n_folds        — число фолдов (по умолчанию 5)
  log_target     — для регрессии: учиться на log1p(y), предсказывать expm1
  class_weight   — для классификации: "balanced" или None
"""

from __future__ import annotations

import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

from evaluation.metrics import METRICS, compute_metric
from interfaces.schemas import (
    DataProfile,
    ExperimentResult,
    ExperimentStatus,
    Hypothesis,
    RunBudget,
    TaskSpec,
    TaskType,
)
from ml.profiling import make_matrix
from runtime.budget import used_ram_mb
from runtime.log import get_logger

log = get_logger("experiment")

SEED = 42
BASE_PARAMS = {
    "n_estimators": 3000,
    "learning_rate": 0.05,
    "num_leaves": 63,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "random_state": SEED,
    "verbose": -1,
    "n_jobs": -1,
}
EARLY_STOPPING = 100
CATBOOST_PARAMS = {
    "iterations": 3000,
    "learning_rate": 0.05,
    "depth": 6,
    "random_seed": SEED,
    "thread_count": -1,
    "verbose": False,
    "allow_writing_files": False,  # иначе CatBoost создаёт папку catboost_info в каталоге задачи
}
XGBOOST_PARAMS = {
    "n_estimators": 3000,
    "learning_rate": 0.05,
    "max_depth": 6,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "tree_method": "hist",
    "enable_categorical": True,  # категории pandas из make_matrix — без ручного кодирования
    "max_cat_to_onehot": 1,
    "random_state": SEED,
    "n_jobs": -1,
    "verbosity": 0,
}


def _is_classification(spec: TaskSpec) -> bool:
    return spec.task_type in (TaskType.BINARY, TaskType.MULTICLASS)


def _encode_target(spec: TaskSpec, y: pd.Series) -> np.ndarray:
    if _is_classification(spec):
        index = {label: i for i, label in enumerate(spec.class_labels or [])}
        return y.astype(str).map(index).to_numpy()
    return y.to_numpy(dtype=float)


def _make_lightgbm(spec: TaskSpec, hypothesis: Hypothesis):
    params = {**BASE_PARAMS, **hypothesis.params.get("model_params", {})}
    if _is_classification(spec):
        return lgb.LGBMClassifier(class_weight=hypothesis.params.get("class_weight"), **params)
    return lgb.LGBMRegressor(**params)


# Старое имя: им пользуется research/bench_playbook.py. Убрать, когда стенд перейдёт на _fit_model
_make_model = _make_lightgbm


def _fit_lightgbm(
    spec: TaskSpec,
    hypothesis: Hypothesis,
    x: pd.DataFrame,
    y: np.ndarray,
    eval_set: tuple[pd.DataFrame, np.ndarray] | None = None,
    n_estimators: int | None = None,
) -> tuple[object, int]:
    """Обучить LightGBM. Возвращает (модель, лучшее число деревьев).

    С eval_set — ранняя остановка на валидации фолда (CV).
    Без eval_set — ровно n_estimators деревьев (финальное обучение на всём train).
    """
    model = _make_lightgbm(spec, hypothesis)
    if n_estimators is not None:
        model.set_params(n_estimators=n_estimators)
    if eval_set is None:
        model.fit(x, y)
        return model, model.n_estimators
    model.fit(x, y, eval_set=[eval_set], callbacks=[lgb.early_stopping(EARLY_STOPPING, verbose=False)])
    return model, model.best_iteration_ or model.n_estimators


def _fit_catboost(
    spec: TaskSpec,
    hypothesis: Hypothesis,
    x: pd.DataFrame,
    y: np.ndarray,
    eval_set: tuple[pd.DataFrame, np.ndarray] | None = None,
    n_estimators: int | None = None,
) -> tuple[object, int]:
    """Обучить CatBoost. Возвращает (модель, лучшее число деревьев). Режимы — как у _fit_lightgbm.

    Категории (pandas category из make_matrix) CatBoost кодирует сам — упорядоченными target-статистиками
    внутри обучения, без утечки на валидацию.
    """
    # Импорт здесь, а не в начале файла: catboost — из группы full, без него падает только эта гипотеза
    from catboost import CatBoostClassifier, CatBoostRegressor

    params = {**CATBOOST_PARAMS, **hypothesis.params.get("model_params", {})}
    if n_estimators is not None:
        params["iterations"] = n_estimators
    params["cat_features"] = [c for c in x.columns if isinstance(x[c].dtype, pd.CategoricalDtype)]
    if _is_classification(spec):
        if hypothesis.params.get("class_weight") == "balanced":
            params["auto_class_weights"] = "Balanced"
        model = CatBoostClassifier(**params)
    else:
        model = CatBoostRegressor(**params)
    if eval_set is None:
        model.fit(x, y)
    else:
        # use_best_model: после ранней остановки в модели остаются деревья до лучшей итерации
        model.fit(x, y, eval_set=eval_set, early_stopping_rounds=EARLY_STOPPING, use_best_model=True)
    return model, model.tree_count_


def _fit_xgboost(
    spec: TaskSpec,
    hypothesis: Hypothesis,
    x: pd.DataFrame,
    y: np.ndarray,
    eval_set: tuple[pd.DataFrame, np.ndarray] | None = None,
    n_estimators: int | None = None,
) -> tuple[object, int]:
    """Обучить XGBoost. Возвращает (модель, лучшее число деревьев). Режимы — как у _fit_lightgbm."""
    # Импорт здесь: xgboost — из группы full, без него падает только эта гипотеза
    from xgboost import XGBClassifier, XGBRegressor

    params = {**XGBOOST_PARAMS, **hypothesis.params.get("model_params", {})}
    if n_estimators is not None:
        params["n_estimators"] = n_estimators
    if eval_set is not None:
        params["early_stopping_rounds"] = EARLY_STOPPING  # без eval_set XGBoost с этим параметром падает
    model = (XGBClassifier if _is_classification(spec) else XGBRegressor)(**params)
    # У XGBoost нет class_weight — «balanced» передаём весами строк
    balanced = _is_classification(spec) and hypothesis.params.get("class_weight") == "balanced"
    weights = compute_sample_weight("balanced", y) if balanced else None
    if eval_set is None:
        model.fit(x, y, sample_weight=weights)
        return model, params["n_estimators"]
    model.fit(x, y, sample_weight=weights, eval_set=[eval_set], verbose=False)
    return model, model.best_iteration + 1


def _fit_linear(
    spec: TaskSpec,
    hypothesis: Hypothesis,
    x: pd.DataFrame,
    y: np.ndarray,
    eval_set: tuple[pd.DataFrame, np.ndarray] | None = None,
    n_estimators: int | None = None,
) -> tuple[object, int]:
    """Линейная модель: пропуски → медиана, масштабирование, one-hot категорий. Деревьев нет — возвращает 0.

    Импьютер, скейлер и one-hot — шаги одного Pipeline: учатся только на train-части фолда, без утечки.
    eval_set и n_estimators не нужны (ранней остановки нет), принимаются ради общей сигнатуры.
    """
    categorical = [c for c in x.columns if isinstance(x[c].dtype, pd.CategoricalDtype)]
    numeric = [c for c in x.columns if c not in categorical]
    prep = ColumnTransformer(
        [
            ("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), numeric),
            # min_frequency: редкие категории — в одну колонку «прочее», чтобы one-hot не раздувал память
            ("cat", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=10), categorical),
        ]
    )
    model_params = hypothesis.params.get("model_params", {})
    if _is_classification(spec):
        estimator = LogisticRegression(
            max_iter=1000, class_weight=hypothesis.params.get("class_weight"), random_state=SEED, **model_params
        )
    else:
        estimator = Ridge(random_state=SEED, **model_params)
    model = make_pipeline(prep, estimator)
    model.fit(x, y)
    return model, 0


# Модели, которые понимает params.model. Новая модель = функция _fit_<имя> с той же сигнатурой + строка здесь
MODEL_FITTERS = {
    "lightgbm": _fit_lightgbm,
    "catboost": _fit_catboost,
    "xgboost": _fit_xgboost,
    "linear": _fit_linear,
}
DEFAULT_MODEL = "lightgbm"


def _fit_model(
    spec: TaskSpec,
    hypothesis: Hypothesis,
    x: pd.DataFrame,
    y: np.ndarray,
    eval_set: tuple[pd.DataFrame, np.ndarray] | None = None,
    n_estimators: int | None = None,
) -> tuple[object, int]:
    """Выбрать модель по params.model и обучить её."""
    name = str(hypothesis.params.get("model") or DEFAULT_MODEL).strip().lower()
    if name not in MODEL_FITTERS:
        raise ValueError(f"неизвестная модель {name!r}, доступны: {sorted(MODEL_FITTERS)}")
    return MODEL_FITTERS[name](spec, hypothesis, x, y, eval_set, n_estimators)


def _predict(spec: TaskSpec, model, x: pd.DataFrame, log_target: bool):
    """Предсказание в том виде, который нужен метрике задачи."""
    if not _is_classification(spec):
        pred = model.predict(x)
        return np.expm1(pred) if log_target else pred
    proba = model.predict_proba(x)
    if METRICS[spec.metric.name].prediction == "proba":
        return proba
    return proba.argmax(axis=1)


def _split(spec: TaskSpec, y: np.ndarray, n_folds: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """Индексы (train, valid) фолдов: KFold для регрессии, StratifiedKFold для классов.

    Объекты редких классов (меньше n_folds штук) всегда в train-части каждого фолда и не попадают в валидацию:
    иначе в части фолдов модель не увидит класс и упадёт («unseen labels», XGBoost требует все метки 0..K-1).
    Без редких классов разбиение то же, что у StratifiedKFold на всех данных, — скоры сравнимы со старыми.
    """
    idx = np.arange(len(y))
    if not _is_classification(spec):
        return list(KFold(n_folds, shuffle=True, random_state=SEED).split(idx))
    counts = pd.Series(y).value_counts()
    rare = np.isin(y, counts.index[counts < n_folds].to_numpy())
    common = idx[~rare]
    if rare.any():
        log.info("Редкие классы (< %d объектов): %d строк всегда в обучении, в валидацию не идут", n_folds, rare.sum())
    splitter = StratifiedKFold(n_folds, shuffle=True, random_state=SEED)
    return [
        (np.sort(np.concatenate([common[tr], idx[rare]])), common[va]) for tr, va in splitter.split(common, y[common])
    ]


def run_experiment(
    spec: TaskSpec, profile: DataProfile, hypothesis: Hypothesis, train: pd.DataFrame, budget: RunBudget
) -> ExperimentResult:
    started = time.time()
    peak = used_ram_mb()
    params = hypothesis.params
    log_target = bool(params.get("log_target")) and spec.task_type == TaskType.REGRESSION
    n_folds = int(params.get("n_folds", 5))
    try:
        x, _ = make_matrix(profile, train, train.head(0))
        y = _encode_target(spec, train[spec.target])
        y_fit = np.log1p(np.clip(y, 0, None)) if log_target else y
        scores, best_iters = [], []
        for fold, (tr, va) in enumerate(_split(spec, y, n_folds)):
            if not budget.has_time():
                return ExperimentResult(
                    hypothesis_id=hypothesis.id,
                    status=ExperimentStatus.TIMEOUT,
                    metric_name=spec.metric.name,
                    fold_scores=scores,
                    fit_seconds=time.time() - started,
                    error=f"кончилось время на фолде {fold}",
                )
            model, best_iter = _fit_model(spec, hypothesis, x.iloc[tr], y_fit[tr], eval_set=(x.iloc[va], y_fit[va]))
            best_iters.append(best_iter)
            scores.append(compute_metric(spec.metric.name, y[va], _predict(spec, model, x.iloc[va], log_target)))
            peak = max(peak, used_ram_mb())
        hypothesis.params.setdefault("_best_iteration", int(np.mean(best_iters)))
        return ExperimentResult(
            hypothesis_id=hypothesis.id,
            status=ExperimentStatus.OK,
            metric_name=spec.metric.name,
            fold_scores=[float(s) for s in scores],
            mean_score=float(np.mean(scores)),
            std_score=float(np.std(scores)),
            fit_seconds=time.time() - started,
            peak_ram_mb=peak,
        )
    except Exception as exc:  # noqa: BLE001 — упавшая гипотеза не должна ронять агента
        log.exception("Гипотеза %s упала", hypothesis.id)
        return ExperimentResult(
            hypothesis_id=hypothesis.id,
            status=ExperimentStatus.FAILED,
            metric_name=spec.metric.name,
            fit_seconds=time.time() - started,
            error=f"{type(exc).__name__}: {exc}",
        )


def fit_predict_final(
    spec: TaskSpec, profile: DataProfile, hypothesis: Hypothesis, train: pd.DataFrame, test: pd.DataFrame
) -> pd.DataFrame:
    """Обучение на всём train с числом деревьев из CV и предсказание test в формате сабмита."""
    params = hypothesis.params
    log_target = bool(params.get("log_target")) and spec.task_type == TaskType.REGRESSION
    x, x_test = make_matrix(profile, train, test)
    y = _encode_target(spec, train[spec.target])
    y_fit = np.log1p(np.clip(y, 0, None)) if log_target else y
    # На всех данных деревьев нужно чуть больше, чем на 4/5 из CV
    n_estimators = max(50, int(params.get("_best_iteration", 500) * 1.1))
    model, _ = _fit_model(spec, hypothesis, x, y_fit, n_estimators=n_estimators)
    pred = _predict(spec, model, x_test, log_target)

    if _is_classification(spec):
        labels = np.asarray(spec.class_labels)
        if METRICS[spec.metric.name].prediction == "proba":
            # Бинарная задача с ROC-AUC/logloss: в ответ — вероятность «положительного» класса
            answer = pred[:, -1] if spec.task_type == TaskType.BINARY else labels[pred.argmax(axis=1)]
        else:
            answer = labels[pred]
    else:
        answer = pred
    return pd.DataFrame({spec.id_column: test[spec.id_column].to_numpy(), spec.target: answer})
