"""Эксперимент = гипотеза → кросс-валидация → ExperimentResult. Владелец — ML-pipeline инженер.

Защита по времени и памяти (без params, всегда): CV идёт на подвыборке, если строк больше MAX_CV_ROWS или матрица
не помещается в свободную память; после первого фолда — прогноз, и если остальные не успеют, эксперимент
останавливается с TIMEOUT. Подвыборка детерминированная — все гипотезы видят одни и те же строки.

Подбор optuna идёт на схеме гипотезы (params.validation), которую цикл ставит каждой гипотезе (#175, #176): на KFold он
оптимизировал утечку через повторы объектов (public/regression год 5: честно 53.99 против бейзлайна 51.25, ревью #115),
на time:Year_Factor — честно 45.92 (ADR 0013). Поэтому ключ снова открыт агенту (#200).

Финал (fit_predict_final) у бустингов — среднее AGENT_FINAL_SEEDS моделей с разными сидами (по умолчанию 3, #261 E3):
подвыборка строк и признаков у каждой своя, среднее снижает разброс ответа. AGENT_FINAL_SEEDS=1 — одна модель, как
раньше; первый сид всегда SEED. Время финала замеряет solution.py — резерв (ensure_final_reserve) растёт сам.

Поддерживаемые params гипотезы (расширяйте — и документируйте в research/playbook.yaml):
  model          — "lightgbm" (по умолчанию) | "catboost" | "xgboost" | "hist_gb" | "linear". Список — MODEL_FITTERS.
                   hist_gb — HistGradientBoosting из scikit-learn: самый быстрый бустинг, категории нативно
                   linear — Ridge (регрессия) / LogisticRegression (классы) с масштабированием и one-hot;
                   её model_params: alpha (Ridge), C (LogisticRegression)
                   Неизвестное имя → ExperimentResult(status=FAILED), а не тихая подмена на LightGBM
  model_params   — словарь гиперпараметров именно выбранной модели (у LightGBM num_leaves, у CatBoost depth);
                   чужой параметр → FAILED с ошибкой библиотеки
  n_folds        — число фолдов (по умолчанию 5)
  log_target     — для регрессии: учиться на log1p(y), предсказывать expm1. Не задан — включён для метрик на
                   лог-шкале (RMSLE, MSLE: это RMSE/MSE на log1p — обучение сразу под метрику, #206), иначе выключен
  class_weight   — для классификации: "balanced" или None
  feature_ops    — список преобразований признаков: missing_flags | freq_encoding | target_encoding | pair_ops;
                   элемент — имя или {op: имя, параметр: значение} (ml/features.py); обучаются только на train-части
                   фолда, без утечки
  validation     — схема CV: kfold | stratified | group:<колонка> | group:duplicates | time:<колонка> (ml/validation.py);
                   выбирается один раз до цикла (choose_validation) и одинакова у всех гипотез (ADR 0008, D1)
  proba_weights  — true: множители вероятностей классов под метрику по меткам (Macro F1…), подбор на OOF
                   (ml/class_weights.py), скор — кросс-подгонкой; для регрессии и метрик по вероятностям не действует
  optuna         — true или {trials, timeout, folds, rows} (любой ключ можно опустить — значение по умолчанию): подбор
                   model_params выбранной модели Optuna на подвыборке и схеме валидации гипотезы (ml/tuning.py), затем
                   обычная CV с найденными параметрами. Не дольше 30% оставшегося времени агента; не вместе с ensemble
  ensemble       — список моделей, прогноз — среднее их прогнозов (вероятностей у классификации): ["lightgbm", "xgboost"]
                   или [{model: xgboost, model_params: {...}}, ...]; у каждой своя ранняя остановка и число деревьев.
                   Остальные params (log_target, class_weight, feature_ops, proba_weights…) — общие для всех моделей
  objective      — "metric": цель обучения под метрику задачи для выбранной модели (регрессия; #186, #237): MAE/MedAE/
                   WAPE → L1, pinball@α → квантиль α, Huber → Huber δ=1, MAPE → MAPE, девиансы → Poisson/Gamma/Tweedie
                   (METRIC_OBJECTIVES). Нет своей цели у метрики или модели, обучение на log1p(y), цель задана в
                   model_params явно или таргет ей не подходит (Poisson на отрицательных) — L2, как без ключа

"""

from __future__ import annotations

import os
import time
from contextlib import contextmanager
from contextvars import ContextVar

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
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
from ml.class_weights import cross_fit_scores, fit_weights
from ml.features import apply_feature_ops, parse_feature_ops
from ml.profiling import make_matrix
from ml.validation import make_folds, parse_validation
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
    # n_jobs=-1 на многоядерных машинах и малых данных в разы медленнее из-за накладных расходов потоков
    "n_jobs": min(os.cpu_count() or 1, 4),
}
EARLY_STOPPING = 100
FINAL_TREES_MARGIN = 1.1  # на всём train деревьев нужно чуть больше, чем в фолде CV
CV_TRAIN_SHARE = 0.8  # доля train в фолде kfold-5 — под неё подобран FINAL_TREES_MARGIN
MAX_TREES_SCALE = 3.0  # поправка на долю train не больше ×3: деревьев с данными растёт медленнее, чем линейно
MAX_CV_ROWS = 500_000  # больше строк для сравнения гипотез не нужно: скор стабилен, а время растёт линейно
MIN_CV_ROWS = 1_000  # меньше — оценка бессмысленна, лучше честно упасть по памяти
RAM_FACTOR = 6  # пик памяти эксперимента ≈ RAM_FACTOR × размер матрицы (копии фолдов, бины и деревья модели)
# Скорость подобрана на public/multiclass (10 классов, 87 тыс. строк, 1 фолд): по умолчанию ~30 мин на фолд,
# с этими настройками 142 с и Macro F1 0.552. max_ctr_complexity=1 — без комбинаций категорий (×2 быстрее),
# rsm=0.3 — 30% признаков на выбор разбиения (×2.5), learning_rate 0.15 — меньше итераций до ранней остановки
CATBOOST_PARAMS = {
    "iterations": 3000,
    "learning_rate": 0.15,
    "depth": 6,
    "rsm": 0.3,
    "max_ctr_complexity": 1,
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
    # Категории до 64 значений — разбиения «одна против остальных», а не по подмножествам: подмножества на
    # десятках категорий запоминают train и ломаются при сдвиге (#145: public/regression честно RMSE 135.7 → 46.7)
    "max_cat_to_onehot": 64,
    "random_state": SEED,
    "n_jobs": -1,
    "verbosity": 0,
}


FINAL_SEEDS_ENV = "AGENT_FINAL_SEEDS"
DEFAULT_FINAL_SEEDS = 3
MAX_FINAL_SEEDS = 10
# Модели со случайностью в обучении (подвыборка строк/признаков): у linear и hist_gb без ранней остановки сид
# ничего не меняет — несколько копий только тратили бы время
SEEDED_MODELS = {"lightgbm", "xgboost", "catboost"}
_model_seed: ContextVar[int] = ContextVar("model_seed", default=SEED)


@contextmanager
def seeded(seed: int):
    """Сид моделей внутри блока (финал с несколькими сидами); model_params гипотезы с явным сидом важнее."""
    token = _model_seed.set(seed)
    try:
        yield
    finally:
        _model_seed.reset(token)


def final_seeds() -> int:
    """Сколько сидов у финальной модели: AGENT_FINAL_SEEDS (1…MAX_FINAL_SEEDS), не задан или мусор — по умолчанию."""
    try:
        value = int(os.environ.get(FINAL_SEEDS_ENV, DEFAULT_FINAL_SEEDS))
    except ValueError:
        return DEFAULT_FINAL_SEEDS
    return min(max(value, 1), MAX_FINAL_SEEDS)


def _seed_sensitive(hypothesis: Hypothesis) -> bool:
    """Есть ли в гипотезе модель, у которой сид меняет результат (у ансамбля — хотя бы один участник)."""
    if hypothesis.params.get("ensemble"):
        return any(m.params["model"] in SEEDED_MODELS for m in _ensemble_members(hypothesis))
    return str(hypothesis.params.get("model") or DEFAULT_MODEL).strip().lower() in SEEDED_MODELS


def _is_classification(spec: TaskSpec) -> bool:
    return spec.task_type in (TaskType.BINARY, TaskType.MULTICLASS)


LOG_SCALE_METRICS = ("rmsle", "msle")  # метрики на log1p(y): для них log_target по умолчанию включён


def _use_log_target(spec: TaskSpec, params: dict) -> bool:
    """Учиться ли на log1p(y): явный params.log_target, а без него — да для метрик на лог-шкале (только регрессия)."""
    if spec.task_type != TaskType.REGRESSION:
        return False
    if params.get("log_target") is not None:
        return bool(params["log_target"])
    return spec.metric.name in LOG_SCALE_METRICS


def _encode_target(spec: TaskSpec, y: pd.Series) -> np.ndarray:
    if _is_classification(spec):
        index = {label: i for i, label in enumerate(spec.class_labels or [])}
        return y.astype(str).map(index).to_numpy()
    return y.to_numpy(dtype=float)


# Цель обучения под метрику (#186, #237): метрика → вид цели. Метрик здесь нет — учимся на L2, как раньше.
# WAPE = Σ|e| / Σ|y| и MedAE оптимизируются медианой (L1), как MAE
METRIC_OBJECTIVES = {
    "mae": "l1",
    "medae": "l1",
    "wape": "l1",
    "pinball": "quantile",
    "huber": "huber",
    "mape": "mape",
    "poisson_deviance": "poisson",
    "gamma_deviance": "gamma",
    "tweedie_deviance": "tweedie",
}
HUBER_DELTA = 1.0  # как в evaluation.metrics._huber
TWEEDIE_POWER = 1.5  # как в метрике tweedie_deviance (evaluation/metrics.py)
OBJECTIVE_KEYS = ("objective", "loss_function", "loss")  # цель, заданная в model_params явно, важнее подстановки


def _objective_by_model(kind: str, alpha: float) -> dict[str, dict]:
    """Параметры цели вида kind для каждой модели; модели без подходящей цели в словаре нет (учится на L2)."""
    return {
        "l1": {
            "lightgbm": {"objective": "l1"},
            "xgboost": {"objective": "reg:absoluteerror"},
            "catboost": {"loss_function": "MAE"},
            "hist_gb": {"loss": "absolute_error"},
        },
        "quantile": {
            "lightgbm": {"objective": "quantile", "alpha": alpha},
            "xgboost": {"objective": "reg:quantileerror", "quantile_alpha": alpha},
            "catboost": {"loss_function": f"Quantile:alpha={alpha}"},
            "hist_gb": {"loss": "quantile", "quantile": alpha},
        },
        "huber": {
            "lightgbm": {"objective": "huber", "alpha": HUBER_DELTA},  # у LightGBM alpha — это δ Huber
            "xgboost": {"objective": "reg:pseudohubererror", "huber_slope": HUBER_DELTA},
            "catboost": {"loss_function": f"Huber:delta={HUBER_DELTA}"},
        },
        "mape": {"lightgbm": {"objective": "mape"}, "catboost": {"loss_function": "MAPE"}},
        "poisson": {
            "lightgbm": {"objective": "poisson"},
            "xgboost": {"objective": "count:poisson"},
            "catboost": {"loss_function": "Poisson"},
            "hist_gb": {"loss": "poisson"},
        },
        "gamma": {
            "lightgbm": {"objective": "gamma"},
            "xgboost": {"objective": "reg:gamma"},
            "hist_gb": {"loss": "gamma"},
        },
        "tweedie": {
            "lightgbm": {"objective": "tweedie", "tweedie_variance_power": TWEEDIE_POWER},
            "xgboost": {"objective": "reg:tweedie", "tweedie_variance_power": TWEEDIE_POWER},
            "catboost": {"loss_function": f"Tweedie:variance_power={TWEEDIE_POWER}"},
        },
    }[kind]


def metric_objective_kind(spec: TaskSpec, params: dict) -> tuple[str, float] | None:
    """(вид цели, α квантиля) под метрику задачи, если гипотеза просит params.objective="metric", иначе None.

    None и для классификации, и при обучении на log1p(y) (метрика тогда уже на другой шкале), и для метрик без
    своей цели (RMSE, R² — L2 по умолчанию и так под них).
    """
    objective = params.get("objective")
    if objective is None:
        return None
    if objective != "metric":
        raise ValueError(f"params.objective={objective!r}: поддерживается только 'metric'")
    if spec.task_type != TaskType.REGRESSION or _use_log_target(spec, params):
        return None
    base, _, level = spec.metric.name.partition("@")
    kind = METRIC_OBJECTIVES.get(base)
    if kind is None:
        return None
    return kind, float(level) if level else 0.5


def _objective_params(spec: TaskSpec, hypothesis: Hypothesis, model: str, y: np.ndarray | None) -> dict:
    """Параметры цели под метрику для модели model (или {}): см. params.objective в шапке модуля.

    Таргет не подходит цели (Poisson/Tweedie — отрицательные значения, Gamma — неположительные) — {} (L2), а не
    падение гипотезы: на другой задаче та же гипотеза полезна.
    """
    if any(key in hypothesis.params.get("model_params", {}) for key in OBJECTIVE_KEYS):
        return {}
    resolved = metric_objective_kind(spec, hypothesis.params)
    if resolved is None:
        return {}
    kind, alpha = resolved
    if y is not None and len(y):
        low = float(np.nanmin(y))
        if (kind in ("poisson", "tweedie") and low < 0) or (kind == "gamma" and low <= 0):
            return {}
    return dict(_objective_by_model(kind, alpha).get(model, {}))


def _make_lightgbm(spec: TaskSpec, hypothesis: Hypothesis, y: np.ndarray | None = None):
    params = {
        **BASE_PARAMS,
        "random_state": _model_seed.get(),
        **_objective_params(spec, hypothesis, "lightgbm", y),
        **hypothesis.params.get("model_params", {}),
    }
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
    model = _make_lightgbm(spec, hypothesis, y)
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

    params = {
        **CATBOOST_PARAMS,
        "random_seed": _model_seed.get(),
        **_objective_params(spec, hypothesis, "catboost", y),
        **hypothesis.params.get("model_params", {}),
    }
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

    params = {
        **XGBOOST_PARAMS,
        "random_state": _model_seed.get(),
        **_objective_params(spec, hypothesis, "xgboost", y),
        **hypothesis.params.get("model_params", {}),
    }
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


HIST_GB_MAX_BINS = 255  # больше категорий HistGradientBoosting не принимает — такие колонки идут как числа (коды)


def _category_codes(x: pd.DataFrame) -> pd.DataFrame:
    """Категории → их коды (общий словарь train+test из make_matrix), пропуск категории → NaN."""
    x = x.copy()
    for col in x.columns:
        if isinstance(x[col].dtype, pd.CategoricalDtype):
            x[col] = x[col].cat.codes.astype(np.float32).replace(-1, np.nan)
    return x


def _fit_hist_gb(
    spec: TaskSpec,
    hypothesis: Hypothesis,
    x: pd.DataFrame,
    y: np.ndarray,
    eval_set: tuple[pd.DataFrame, np.ndarray] | None = None,
    n_estimators: int | None = None,
) -> tuple[object, int]:
    """HistGradientBoosting (scikit-learn). Возвращает (модель, число итераций).

    Внешний eval_set scikit-learn 1.5 не принимает: в CV ранняя остановка идёт по своим 10% train-части фолда
    (валидацию фолда модель не видит). С n_estimators — ровно столько итераций, без остановки.
    """
    is_cat = [
        isinstance(x[c].dtype, pd.CategoricalDtype) and len(x[c].cat.categories) <= HIST_GB_MAX_BINS for c in x.columns
    ]
    params = {"max_iter": 3000, "learning_rate": 0.05, "random_state": SEED, "categorical_features": is_cat}
    if n_estimators is not None:
        params.update(max_iter=n_estimators, early_stopping=False)
    else:
        params.update(early_stopping=True, validation_fraction=0.1, n_iter_no_change=EARLY_STOPPING // 2)
        # Внутренняя валидация делится со стратификацией: класс из 1 объекта её ломает — тогда без остановки
        if _is_classification(spec) and np.unique(y, return_counts=True)[1].min() < 2:
            params.update(early_stopping=False, max_iter=300)
    params.update(_objective_params(spec, hypothesis, "hist_gb", y))
    params.update(hypothesis.params.get("model_params", {}))
    if _is_classification(spec):
        estimator = HistGradientBoostingClassifier(class_weight=hypothesis.params.get("class_weight"), **params)
    else:
        estimator = HistGradientBoostingRegressor(**params)
    model = make_pipeline(FunctionTransformer(_category_codes), estimator)
    model.fit(x, y)
    return model, int(estimator.n_iter_)


# Модели, которые понимает params.model. Новая модель = функция _fit_<имя> с той же сигнатурой + строка здесь
MODEL_FITTERS = {
    "lightgbm": _fit_lightgbm,
    "catboost": _fit_catboost,
    "xgboost": _fit_xgboost,
    "hist_gb": _fit_hist_gb,
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
    """Выбрать модель по params.model (или ансамбль по params.ensemble) и обучить её."""
    if hypothesis.params.get("ensemble"):
        return _fit_ensemble(spec, hypothesis, x, y, eval_set, n_estimators)
    name = str(hypothesis.params.get("model") or DEFAULT_MODEL).strip().lower()
    if name not in MODEL_FITTERS:
        raise ValueError(f"неизвестная модель {name!r}, доступны: {sorted(MODEL_FITTERS)}")
    return MODEL_FITTERS[name](spec, hypothesis, x, y, eval_set, n_estimators)


def _ensemble_members(hypothesis: Hypothesis) -> list[Hypothesis]:
    """params.ensemble → гипотезы-участники: общие params гипотезы + своя модель и её model_params."""
    members = hypothesis.params["ensemble"]
    if not isinstance(members, list) or len(members) < 2:
        raise ValueError("ensemble — список хотя бы из 2 моделей")
    common = {k: v for k, v in hypothesis.params.items() if k not in ("ensemble", "model", "model_params")}
    out = []
    for item in members:
        spec_item = {"model": item} if isinstance(item, str) else dict(item)
        if "ensemble" in spec_item:
            raise ValueError("вложенный ансамбль не поддерживается")
        model = str(spec_item.get("model") or DEFAULT_MODEL).strip().lower()
        if model not in MODEL_FITTERS:
            raise ValueError(f"неизвестная модель {model!r} в ensemble, доступны: {sorted(MODEL_FITTERS)}")
        params = {**common, "model": model, "model_params": dict(spec_item.get("model_params") or {})}
        out.append(hypothesis.model_copy(update={"params": params}))
    return out


def _fill_missing_categories(x: pd.DataFrame) -> pd.DataFrame:
    """Пропуск в категории → отдельная категория: CatBoost не принимает NaN в cat_features (пропуск даёт _hide_unseen)."""
    cols = [c for c in x.columns if isinstance(x[c].dtype, pd.CategoricalDtype) and x[c].isna().any()]
    if not cols:
        return x
    x = x.copy()
    for col in cols:
        if MISSING_CATEGORY not in x[col].cat.categories:
            x[col] = x[col].cat.add_categories(MISSING_CATEGORY)
        x[col] = x[col].fillna(MISSING_CATEGORY)
    return x


MISSING_CATEGORY = "__missing__"


class _Ensemble:
    """Среднее прогнозов моделей. У классификации — среднее вероятностей (все учились на одних закодированных y)."""

    def __init__(self, models: list, fill_categories: list[bool]):
        self.models = models
        self.fill_categories = fill_categories
        self.classes_ = getattr(models[0], "classes_", None)

    def _inputs(self, x: pd.DataFrame):
        filled = None
        for model, fill in zip(self.models, self.fill_categories, strict=True):
            if fill and filled is None:
                filled = _fill_missing_categories(x)
            yield model, filled if fill else x

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        return np.mean([np.asarray(m.predict(xi), dtype=float).reshape(len(x)) for m, xi in self._inputs(x)], axis=0)

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        return np.mean([m.predict_proba(xi) for m, xi in self._inputs(x)], axis=0)


def _fit_ensemble(
    spec: TaskSpec,
    hypothesis: Hypothesis,
    x: pd.DataFrame,
    y: np.ndarray,
    eval_set: tuple[pd.DataFrame, np.ndarray] | None = None,
    n_estimators: list[int] | int | None = None,
) -> tuple[_Ensemble, list[int]]:
    """Обучить участников ансамбля. Число деревьев — своё у каждого: список из CV (n_estimators) или ранняя остановка."""
    members = _ensemble_members(hypothesis)
    if n_estimators is not None and not isinstance(n_estimators, list):
        n_estimators = [n_estimators] * len(members)
    models, iters, fills = [], [], []
    for i, member in enumerate(members):
        fill = member.params["model"] == "catboost"
        x_m = _fill_missing_categories(x) if fill else x
        eval_m = eval_set
        if fill and eval_set is not None:
            eval_m = (_fill_missing_categories(eval_set[0]), eval_set[1])
        n_m = None if n_estimators is None else n_estimators[i]
        model, best_iter = MODEL_FITTERS[member.params["model"]](spec, member, x_m, y, eval_m, n_m)
        models.append(model)
        iters.append(int(best_iter))
        fills.append(fill)
    return _Ensemble(models, fills), iters


def _mean_best_iteration(best_iters: list) -> int | list[int]:
    """Среднее число деревьев по фолдам; у ансамбля — по каждому участнику отдельно."""
    if best_iters and isinstance(best_iters[0], list):
        return [int(v) for v in np.mean(best_iters, axis=0)]
    return int(np.mean(best_iters))


def _final_estimators(best: int | list[int] | None, scale: float = 1.0) -> int | list[int]:
    """Деревьев на всём train: чуть больше, чем в CV, × поправка на долю train (у ансамбля — для каждого участника)."""
    factor = FINAL_TREES_MARGIN * scale
    if isinstance(best, list):
        return [max(50, int(b * factor)) for b in best]
    return max(50, int((500 if best is None else best) * factor))


def _hide_unseen(hypothesis: Hypothesis, x_fit: pd.DataFrame, x_new: pd.DataFrame) -> pd.DataFrame:
    """Категории, которых нет в обучающей части, → пропуск в x_new (#145).

    Словарь категорий общий для train и test (make_matrix): у новой категории есть код, но ни одной строки в обучении.
    XGBoost отправляет её по произвольной ветке разбиения: на practice/sales (все даты отложенной части новые) честно
    RMSLE 1.63 против 0.49 с пропуском.
    Пропуск модели обрабатывают штатно (ветка, выученная для пропусков). CatBoost не трогаем: пропуск в категории
    он не принимает, а новую категорию и так кодирует априорной статистикой.
    """
    if str(hypothesis.params.get("model") or DEFAULT_MODEL).strip().lower() == "catboost":
        return x_new
    cols = [c for c in x_new.columns if isinstance(x_new[c].dtype, pd.CategoricalDtype)]
    unseen = {c: ~x_new[c].isin(x_fit[c].unique()) & x_new[c].notna() for c in cols}
    unseen = {c: mask for c, mask in unseen.items() if mask.any()}
    if not unseen:
        return x_new
    x_new = x_new.copy()
    for col, mask in unseen.items():
        x_new[col] = x_new[col].mask(mask)
    return x_new


def _predict(spec: TaskSpec, model, x: pd.DataFrame, log_target: bool):
    """Предсказание в том виде, который нужен метрике задачи."""
    if not _is_classification(spec):
        pred = model.predict(x)
        # expm1 от отрицательного прогноза < 0, а y ≥ 0 (иначе log1p не брали бы) — отсекаем снизу
        return np.clip(np.expm1(pred), 0, None) if log_target else pred
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
    if (counts < n_folds).all():
        # Все классы редкие — не на чем делить: фолдов столько, сколько объектов в самом малом классе (≥ 2)
        n_folds = max(2, int(counts.min()))
        log.info("Во всех классах меньше объектов, чем фолдов — CV на %d фолдах", n_folds)
    rare = np.isin(y, counts.index[counts < n_folds].to_numpy())
    common = idx[~rare]
    if len(np.unique(y[common])) < 2:
        raise ValueError("слишком мало объектов на класс для кросс-валидации")
    if rare.any():
        log.info("Редкие классы (< %d объектов): %d строк всегда в обучении, в валидацию не идут", n_folds, rare.sum())
    splitter = StratifiedKFold(n_folds, shuffle=True, random_state=SEED)
    return [
        (np.sort(np.concatenate([common[tr], idx[rare]])), common[va]) for tr, va in splitter.split(common, y[common])
    ]


def _sample_rows(spec: TaskSpec, y: np.ndarray, size: int) -> np.ndarray:
    """Детерминированная подвыборка size строк; для классов — пропорционально, редкие классы не теряются."""
    rng = np.random.default_rng(SEED)
    if not _is_classification(spec):
        return np.sort(rng.choice(len(y), size, replace=False))
    frac = size / len(y)
    parts = []
    for cls in np.unique(y):
        rows = np.flatnonzero(y == cls)
        keep = min(len(rows), max(round(len(rows) * frac), 10))
        parts.append(rng.choice(rows, keep, replace=False))
    return np.sort(np.concatenate(parts))


def _cv_rows(spec: TaskSpec, profile: DataProfile, y: np.ndarray, budget: RunBudget) -> np.ndarray | None:
    """Строки для CV или None — все. Ограничения: MAX_CV_ROWS и свободная память (оценка до построения матрицы)."""
    n = len(y)
    n_cols = max(len(profile.numeric_features) + len(profile.categorical_features), 1)
    need_mb = RAM_FACTOR * n * n_cols * 4 / 2**20  # признаки — float32
    free_mb = budget.ram_limit_mb - used_ram_mb()
    limit, reason = MAX_CV_ROWS, f"строк больше {MAX_CV_ROWS}"
    if need_mb > free_mb:
        by_ram = int(n * max(free_mb, 0) / need_mb)
        if by_ram < limit:
            limit, reason = by_ram, f"нужно ~{need_mb:.0f} МБ, свободно {free_mb:.0f} МБ"
    if n <= limit:
        return None
    if limit < MIN_CV_ROWS:
        raise MemoryError(f"не хватает памяти даже на {MIN_CV_ROWS} строк: {reason}")
    rows = _sample_rows(spec, y, limit)
    log.warning("CV на подвыборке %d из %d строк: %s", len(rows), n, reason)
    return rows


def _use_proba_weights(spec: TaskSpec, params: dict) -> bool:
    """proba_weights действует только для классификации с метрикой по меткам — иначе молча не применяется."""
    return (
        bool(params.get("proba_weights"))
        and _is_classification(spec)
        and METRICS[spec.metric.name].prediction == "label"
    )


def _fold_score(
    spec: TaskSpec,
    hypothesis: Hypothesis,
    x: pd.DataFrame,
    y: np.ndarray,
    y_fit: np.ndarray,
    tr: np.ndarray,
    va: np.ndarray,
    ops,  # params.feature_ops в любом формате parse_feature_ops
    log_target: bool,
) -> tuple[float, int, np.ndarray | None]:
    """Один фолд: признаки из feature_ops (учатся на train-части) → модель → метрика на валидации.

    Третий элемент — вероятности классов на валидации (для proba_weights), у регрессии None.
    """
    x_tr, x_va = x.iloc[tr], x.iloc[va]
    x_va = _hide_unseen(hypothesis, x_tr, x_va)
    if ops:
        x_tr, x_va = apply_feature_ops(ops, x_tr, y[tr], _is_classification(spec), x_va)
    known = np.ones(len(va), dtype=bool)
    if _is_classification(spec):
        # Схема time: в валидации бывает класс, которого ещё не было в обучении — ранняя остановка на нём падает
        # («unseen labels»), поэтому для неё берём только известные классы; метрика — по всей валидации
        known = np.isin(y_fit[va], np.unique(y_fit[tr]))
        if not known.any():
            raise ValueError("в валидации фолда нет ни одного класса из обучения")
    eval_set = (x_va[known], y_fit[va][known])
    model, best_iter = _fit_model(spec, hypothesis, x_tr, y_fit[tr], eval_set=eval_set)
    if not _is_classification(spec):
        return compute_metric(spec.metric.name, y[va], _predict(spec, model, x_va, log_target)), best_iter, None
    proba = _full_proba(spec, model, model.predict_proba(x_va))  # один раз: и для метрики, и для proba_weights
    pred = proba if METRICS[spec.metric.name].prediction == "proba" else proba.argmax(axis=1)
    return compute_metric(spec.metric.name, y[va], pred), best_iter, proba


def _full_proba(spec: TaskSpec, model, proba: np.ndarray) -> np.ndarray:
    """Вероятности по всем классам задачи. В схеме time ранний период может не знать поздний класс — его колонка 0."""
    n_classes = len(spec.class_labels or [])
    if proba.shape[1] == n_classes:
        return proba
    full = np.zeros((len(proba), n_classes))
    full[:, np.asarray(model.classes_, dtype=int)] = proba
    return full


def run_experiment(
    spec: TaskSpec, profile: DataProfile, hypothesis: Hypothesis, train: pd.DataFrame, budget: RunBudget
) -> ExperimentResult:
    started = time.time()
    peak = used_ram_mb()
    params = hypothesis.params
    log_target = _use_log_target(spec, params)
    try:
        # Разбор params — внутри try: кривой n_folds или feature_ops от LLM → FAILED, а не падение цикла агента
        n_folds = int(params.get("n_folds", 5))
        ops = params.get("feature_ops")
        parse_feature_ops(ops)
        y = _encode_target(spec, train[spec.target])
        rows = _cv_rows(spec, profile, y, budget)
        if rows is not None:
            train, y = train.iloc[rows], y[rows]
        profile, train, _ = with_feature_code(spec, profile, hypothesis, train)
        x, _ = make_matrix(profile, train, train.head(0))
        y_fit = np.log1p(np.clip(y, 0, None)) if log_target else y
        scheme, column = parse_validation(params.get("validation"), spec)
        features = profile.numeric_features + profile.categorical_features
        if params.get("optuna") and params.get("ensemble"):
            raise ValueError("optuna и ensemble вместе не поддерживаются: подбор идёт для одной модели")
        if params.get("optuna") and "_optuna_best" not in params:
            from ml.tuning import tune  # здесь: tuning сам импортирует этот модуль

            # Подбор — на той же схеме валидации, что и CV: на KFold он оптимизировал бы утечку (ревью #115)
            def split(rows: np.ndarray, k: int) -> list[tuple[np.ndarray, np.ndarray]]:
                return make_folds(spec, scheme, column, train.iloc[rows], features, y[rows], k, _split)

            best = tune(spec, hypothesis, x, y, y_fit, ops, log_target, budget, split=split)
            params["_optuna_best"] = best  # повторный запуск той же гипотезы не подбирает заново
            params["model_params"] = {**(params.get("model_params") or {}), **best}
        folds = make_folds(spec, scheme, column, train, features, y, n_folds, _split)
        n_folds = len(folds)  # group/time могут дать меньше фолдов, чем просили
        validation = f"{scheme}:{column}-{n_folds}" if column else f"{scheme}-{n_folds}"
        scores, best_iters, oof = [], [], []
        keep_oof = _use_proba_weights(spec, params)  # OOF-вероятности копим только когда они нужны (память)
        for fold, (tr, va) in enumerate(folds):
            fold_started = time.time()
            if not budget.has_time():
                return ExperimentResult(
                    hypothesis_id=hypothesis.id,
                    status=ExperimentStatus.TIMEOUT,
                    metric_name=spec.metric.name,
                    fold_scores=scores,
                    fit_seconds=time.time() - started,
                    error=f"кончилось время на фолде {fold}",
                    validation=validation,
                )
            score, best_iter, proba = _fold_score(spec, hypothesis, x, y, y_fit, tr, va, ops, log_target)
            best_iters.append(best_iter)
            scores.append(score)
            if keep_oof:
                oof.append((y[va], proba))
            peak = max(peak, used_ram_mb())
            if fold == 0 and n_folds > 1:
                # Прогноз по первому фолду: не успеем остальные — стоп сейчас, а не через час на последнем фолде
                forecast = (time.time() - fold_started) * (n_folds - 1)
                if forecast > budget.seconds_left():
                    return ExperimentResult(
                        hypothesis_id=hypothesis.id,
                        status=ExperimentStatus.TIMEOUT,
                        metric_name=spec.metric.name,
                        fold_scores=[float(s) for s in scores],
                        fit_seconds=time.time() - started,
                        peak_ram_mb=peak,
                        error=f"не успеет: прогноз ещё {forecast:.0f} с, осталось {budget.seconds_left():.0f} с",
                        validation=validation,
                    )
        hypothesis.params.setdefault("_best_iteration", _mean_best_iteration(best_iters))
        # Доля train в фолде: у time-схемы — расширяющееся окно (в среднем ≈ половина), финалу нужно больше деревьев
        hypothesis.params.setdefault("_train_fraction", round(float(np.mean([len(tr) for tr, _ in folds])) / len(y), 4))
        if _use_proba_weights(spec, params):
            gib = spec.metric.greater_is_better
            plain = float(np.mean(scores))
            scores = cross_fit_scores(spec.metric.name, gib, oof)
            weights = fit_weights(
                spec.metric.name, gib, np.concatenate([f[0] for f in oof]), np.vstack([f[1] for f in oof])
            )
            params["_proba_weights"] = weights.round(4).tolist()
            log.info(
                "proba_weights: %s %.5f → %.5f, множители %s",
                spec.metric.name,
                plain,
                np.mean(scores),
                weights.round(2),
            )
        return ExperimentResult(
            hypothesis_id=hypothesis.id,
            status=ExperimentStatus.OK,
            metric_name=spec.metric.name,
            fold_scores=[float(s) for s in scores],
            mean_score=float(np.mean(scores)),
            std_score=float(np.std(scores)),
            fit_seconds=time.time() - started,
            peak_ram_mb=peak,
            validation=validation,
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


PROBA_EPS = 1e-6  # вероятности 0 и 1 logloss штрафует бесконечно — слегка отодвигаем от краёв


def _label_key(value: str) -> str:
    """Имя класса без регистра и пробелов по краям; число — в одном виде («1», «1.0» → «1»)."""
    text = str(value).strip().lower()
    try:
        number = float(text)
    except ValueError:
        return text
    return str(int(number)) if number.is_integer() else repr(number)


def _name_variants(columns: list[str]):
    """Имена колонок как есть, затем без общей приставки и окончания — от самой длинной обрезки к короткой.

    «proba_low», «proba_long» → общая приставка «proba_lo», но нужна «proba_»: поэтому перебираем все длины,
    а какая верная — решает сопоставление с классами. Имя целиком не обрезаем («p1», «p10» → «1», «10»).
    """
    yield columns  # точное совпадение — раньше любой обрезки
    shortest = min(len(c) for c in columns)
    prefix = os.path.commonprefix(columns)
    suffix = os.path.commonprefix([c[::-1] for c in columns])
    for p in range(min(len(prefix), shortest - 1), -1, -1):
        for s in range(min(len(suffix), shortest - p - 1), -1, -1):
            yield [c[p : len(c) - s] for c in columns]


def match_class_columns(columns: list[str], class_labels: list[str]) -> list[str] | None:
    """Колонка ответа → класс по имени (#238): без регистра и без общей приставки/окончания («proba_<класс>»).

    None — по имени однозначно не сопоставить (тогда вызывающий сопоставляет по порядку). Сопоставление только
    взаимно однозначное: два класса с одним ключом, колонка без класса или две колонки на класс — None.
    """
    labels: dict[str, list[str]] = {}
    for label in class_labels:
        labels.setdefault(_label_key(label), []).append(label)
    if not columns or any(len(same) > 1 for same in labels.values()):
        return None
    for names in _name_variants(list(map(str, columns))):
        matched = [labels.get(_label_key(name), [None])[0] for name in names]
        if None not in matched and len(set(matched)) == len(matched) == len(class_labels):
            return matched
    return None


def _per_class_columns(columns: list[str], class_labels: list[str], proba: np.ndarray) -> dict[str, np.ndarray]:
    """Колонки ответа → вероятности классов: по имени (match_class_columns), иначе — по порядку (с записью в лог).

    Столбцы proba идут в порядке class_labels (так кодируется target). Строки нормируются к сумме 1.
    """
    if proba.shape[1] != len(columns):
        raise ValueError(f"колонок ответа {len(columns)}, а классов {proba.shape[1]}: {columns} / {class_labels}")
    proba = np.clip(proba, PROBA_EPS, 1 - PROBA_EPS)
    proba = proba / proba.sum(axis=1, keepdims=True)
    matched = match_class_columns(columns, class_labels)
    if matched is not None:
        index = {label: i for i, label in enumerate(class_labels)}
        if matched != list(columns):
            log.info("Колонки ответа ↔ классы по имени: %s", dict(zip(columns, matched, strict=True)))
        return {col: proba[:, index[label]] for col, label in zip(columns, matched, strict=True)}
    log.warning("Колонки ответа %s не совпадают с классами %s — сопоставляю по порядку", columns, class_labels)
    return {col: proba[:, i] for i, col in enumerate(columns)}


def with_feature_code(
    spec: TaskSpec,
    profile: DataProfile,
    hypothesis: Hypothesis,
    train: pd.DataFrame,
    test: pd.DataFrame | None = None,
) -> tuple[DataProfile, pd.DataFrame, pd.DataFrame | None]:
    """Код признаков от LLM (`Hypothesis.code`, ADR 0008 D4) → новые числовые колонки в train/test и профиле.

    Код исполняется в песочнице (agent/feature_code.apply_feature_code) на признаках train+test вместе,
    **без target и id**; новые колонки берутся только числовые. Ошибка кода → исключение → FAILED гипотезы
    (в run_experiment) или прежний ответ (в fit_predict_final). Без кода — всё как было.
    """
    if not hypothesis.code:
        return profile, train, test
    from agent.feature_code import apply_feature_code  # здесь: agent сам импортирует ml — без цикла импортов

    features = profile.numeric_features + profile.categorical_features
    parts = [train[features]] + ([test[features]] if test is not None else [])
    table = pd.concat(parts, ignore_index=True)
    out, note = apply_feature_code(hypothesis.code, table, target=spec.target)
    if out is None:
        raise ValueError(f"код признаков отклонён: {note}")
    new = [c for c in out.columns if c not in features]
    clash = [c for c in new if c in train.columns or (test is not None and c in test.columns)]
    if clash:
        raise ValueError(f"код признаков перезаписал бы колонки {clash}")
    values = out[new].to_numpy()
    train = train.copy()
    train[new] = values[: len(train)]
    if test is not None:
        test = test.copy()
        test[new] = values[len(train) :]
    log.info("Код признаков «%s»: +%d колонок %s%s", hypothesis.id, len(new), new, f" ({note})" if note else "")
    return profile.model_copy(update={"numeric_features": profile.numeric_features + new}), train, test


def _trees_scale(params: dict) -> float:
    """Во сколько раз финалу нужно больше деревьев, чем в CV, из-за доли train в фолдах (#218: схема time:).

    Ранняя остановка в фолде, где train — треть данных, наступает раньше, чем нужно модели на всём train.
    Для kfold-5 (доля 0.8) поправки нет — там хватает FINAL_TREES_MARGIN. Нет _train_fraction — тоже без поправки.
    """
    fraction = params.get("_train_fraction")
    if not fraction:
        return 1.0
    return min(MAX_TREES_SCALE, max(1.0, CV_TRAIN_SHARE / float(fraction)))


def _fit_final_models(spec: TaskSpec, hypothesis: Hypothesis, x: pd.DataFrame, y: np.ndarray, n_estimators):
    """Финальная модель: одна или среднее final_seeds() копий с сидами s, s+1, … от текущего s (SEED по умолчанию, #261 E3)."""
    n_seeds = final_seeds() if _seed_sensitive(hypothesis) else 1
    first = _model_seed.get()
    models = []
    for i in range(n_seeds):
        with seeded(first + i):
            models.append(_fit_model(spec, hypothesis, x, y, n_estimators=n_estimators)[0])
    if n_seeds == 1:
        return models[0]
    log.info("Финал «%s»: среднее %d моделей с разными сидами", hypothesis.id, n_seeds)
    return _Ensemble(models, [False] * n_seeds)


def fit_predict_final(
    spec: TaskSpec, profile: DataProfile, hypothesis: Hypothesis, train: pd.DataFrame, test: pd.DataFrame
) -> pd.DataFrame:
    """Обучение на всём train с числом деревьев из CV и предсказание test в формате сабмита."""
    params = hypothesis.params
    log_target = _use_log_target(spec, params)
    profile, train, test = with_feature_code(spec, profile, hypothesis, train, test)
    x, x_test = make_matrix(profile, train, test)
    x_test = _hide_unseen(hypothesis, x, x_test)
    y = _encode_target(spec, train[spec.target])
    y_fit = np.log1p(np.clip(y, 0, None)) if log_target else y
    ops = params.get("feature_ops")
    if ops:
        x, x_test = apply_feature_ops(ops, x, y, _is_classification(spec), x_test)
    n_estimators = _final_estimators(params.get("_best_iteration"), _trees_scale(params))
    model = _fit_final_models(spec, hypothesis, x, y_fit, n_estimators)
    pred = _predict(spec, model, x_test, log_target)
    if _use_proba_weights(spec, params) and params.get("_proba_weights"):
        pred = (model.predict_proba(x_test) * np.asarray(params["_proba_weights"])).argmax(axis=1)

    answer_columns = [c for c in spec.submission_columns if c != spec.id_column]
    if _is_classification(spec) and len(answer_columns) > 1:
        # Шаблон «колонка на класс» (#117): вероятности по классам в колонках шаблона
        proba = _per_class_columns(answer_columns, spec.class_labels or [], model.predict_proba(x_test))
        return pd.DataFrame({spec.id_column: test[spec.id_column].to_numpy(), **proba})
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
