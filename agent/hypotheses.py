"""Откуда берутся гипотезы. Владелец — LLM/agent-инженер (playbook — Research/DS).

Сначала — research/playbook.yaml по порядку. Когда применимые гипотезы playbook кончились и LLM доступна, она
смотрит на DataProfile и историю экспериментов и предлагает новые (agent/llm_hypotheses.py; ADR 0014).
Повторы (тот же id или те же params, что уже проверялись) отсекаются. params сравниваются в каноническом виде
(canonical_params): значения по умолчанию и «выключенные» ключи не делают гипотезу новой.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import yaml

from agent.llm_client import LLMClient
from agent.memory import ResearchMemory
from evaluation.metrics import METRICS
from interfaces.schemas import (
    DataProfile,
    ExperimentResult,
    Hypothesis,
    HypothesisKind,
    RunBudget,
    TaskSpec,
    TaskType,
)

PLAYBOOK_PATH = Path(__file__).resolve().parent.parent / "research" / "playbook.yaml"


# Метрики по меткам, которым помогает подбор множителей классов (proba_weights, #191): чувствительные к балансу
# классов. Accuracy — нет: её максимизирует argmax откалиброванных вероятностей, множители на OOF добавляют только шум
# (харнесс 23.10: digits accuracy 0.9873 → 0.9841)
PROBA_WEIGHTS_SKIP = {"accuracy"}


def wants_proba_weights(spec: TaskSpec | None) -> bool:
    """Подбирать ли множители классов в этой задаче: классификация + метрика по меткам, кроме accuracy."""
    if spec is None or spec.task_type == TaskType.REGRESSION or spec.metric.name in PROBA_WEIGHTS_SKIP:
        return False
    try:
        metric = METRICS[spec.metric.name]  # не .get(): параметрические (fbeta@2, pinball@0.9) собирает __missing__
    except KeyError:
        return False
    return metric.prediction == "label"


def baseline_hypothesis(spec: TaskSpec | None = None) -> Hypothesis:
    """Бейзлайн — LightGBM по умолчанию; для метрик по меткам — с подбором множителей классов (#191).

    Без подбора ответ — argmax, то есть порог 0.5: при редком классе это лотерея (payments: F1 0.09 против 0.34).
    """
    params = {"model": "lightgbm"}
    if wants_proba_weights(spec):
        params["proba_weights"] = True
    return Hypothesis(
        id="baseline",
        kind=HypothesisKind.BASELINE,
        title="LightGBM с параметрами по умолчанию",
        params=params,
        source="baseline",
    )


def with_task_defaults(params: dict, spec: TaskSpec) -> dict:
    """params гипотезы + то, что у бейзлайна включено для этой задачи (#191).

    Бейзлайн подбирает множители классов, если метрика по меткам (wants_proba_weights). Гипотезы playbook и LLM —
    отдельные наборы params, не наследуют их от лучшего; без этого каждая сравнивалась бы с бейзлайном без множителей
    и проигрывала бы заведомо. Явное значение в гипотезе (proba_weights: false) не трогаем.
    """
    if "proba_weights" in params or not wants_proba_weights(spec):
        return params
    return {**params, "proba_weights": True}


def load_playbook(path: Path = PLAYBOOK_PATH) -> list[dict]:
    if not path.exists():
        return []
    return yaml.safe_load(path.read_text(encoding="utf-8")).get("hypotheses", [])


def _applies(item: dict, spec: TaskSpec, profile: DataProfile) -> bool:
    task_types = item.get("task_types")
    if task_types and spec.task_type.value not in task_types:
        return False
    if item.get("requires") == "categorical" and not profile.categorical_features:
        return False
    if item.get("requires") == "imbalance" and spec.task_type != TaskType.REGRESSION:
        return profile.target_summary.get("imbalance_ratio", 1) > 3
    return item.get("requires") != "imbalance"


DEFAULT_MODEL = "lightgbm"  # как ml/experiment.py: без model — LightGBM
DEFAULT_N_FOLDS = 5


def _model_defaults() -> dict[str, dict]:
    """Гиперпараметры моделей по умолчанию из ml/experiment.py: такой же model_params — это не новая гипотеза."""
    try:
        from ml.experiment import BASE_PARAMS, CATBOOST_PARAMS, XGBOOST_PARAMS
    except Exception:  # noqa: BLE001 — без ml сравниваем без учёта умолчаний модели
        return {}
    return {"lightgbm": BASE_PARAMS, "catboost": CATBOOST_PARAMS, "xgboost": XGBOOST_PARAMS}


CASE_INSENSITIVE_KEYS = ("model", "class_weight")  # значения-названия; в остальных строках бывают имена колонок


def _norm(value, key: str = ""):
    """5.0 → 5, « Balanced » → «balanced»: одно значение, записанное по-разному.

    Регистр приводим только у CASE_INSENSITIVE_KEYS: в validation, feature_ops и т. п. — имена колонок,
    а «Year» и «year» могут быть разными колонками.
    """
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str):
        return value.strip().lower() if key in CASE_INSENSITIVE_KEYS else value.strip()
    if isinstance(value, dict):
        return {k: _norm(v, k) for k, v in value.items()}
    if isinstance(value, list):
        return [_norm(v) for v in value]
    return value


def _is_off(value) -> bool:
    """Ключ «выключен»: None, False, пустая строка/словарь/список. Число 0 — не выключено (0 == False в Python)."""
    return value is None or value is False or (isinstance(value, dict | list | str) and not value)


def canonical_params(params: dict) -> dict:
    """params без того, что не меняет эксперимент: пустые и выключенные ключи, значения по умолчанию.

    {model: lightgbm, class_weight: balanced, n_folds: 5, log_target: false, model_params: {learning_rate: 0.05}}
    и {class_weight: balanced} — один и тот же эксперимент.
    Служебные ключи исполнителя (_best_iteration, _proba_weights, _optuna_best…) — результат прогона, а не идея.
    """
    out = {k: _norm(v, k) for k, v in params.items() if not _is_off(v) and not str(k).startswith("_")}
    model = out.get("model", DEFAULT_MODEL)
    if model == DEFAULT_MODEL:
        out.pop("model", None)
    if out.get("n_folds") == DEFAULT_N_FOLDS:
        out.pop("n_folds")
    defaults = _model_defaults().get(model, {})
    if isinstance(out.get("model_params"), dict):
        changed = {k: v for k, v in out["model_params"].items() if k not in defaults or _norm(defaults[k]) != v}
        if changed:
            out["model_params"] = changed
        else:
            out.pop("model_params")
    return out


def params_key(params: dict, run_scheme: str | None = None) -> str:
    """Ключ для поиска повторов: одинаковые по сути гипотезы с разными id и записью params совпадают.

    run_scheme — схема валидации прогона (её задаёт бейзлайн, ADR 0008, D1). validation, равная ей, из ключа
    убирается: у гипотез из истории схема записана, а у новых кандидатов её ещё нет — иначе LightGBM по
    умолчанию не опознавался бы как повтор бейзлайна. Другая схема — другая гипотеза (ревью #164).
    """
    if run_scheme and _norm(params.get("validation"), "validation") == _norm(run_scheme, "validation"):
        params = {k: v for k, v in params.items() if k != "validation"}
    return json.dumps(canonical_params(params), sort_keys=True, ensure_ascii=False, default=str)


def run_scheme(history: list[tuple[Hypothesis, ExperimentResult]]) -> str | None:
    """Схема валидации прогона — из params бейзлайна (первая запись истории); None — обычная CV."""
    return history[0][0].params.get("validation") if history else None


LLM_FIRST_ENV = "AGENT_LLM_FIRST"  # "1" — прежний порядок: сначала LLM, потом playbook (для сравнения, ADR 0014)


def propose_hypotheses(
    spec: TaskSpec,
    profile: DataProfile,
    history: list[tuple[Hypothesis, ExperimentResult]],
    k: int = 3,
    budget: RunBudget | None = None,
    llm: LLMClient | None = None,
    memory: ResearchMemory | None = None,
) -> list[Hypothesis]:
    """Сначала playbook, недостающие — от LLM (ADR 0014). Повторы истории отсекаются.

    Playbook бесплатен и мгновенен, а LLM на замере (#179) выигрыша не дала, но стоила времени и токенов
    (тай-брейки платформы). Поэтому LLM зовём, когда применимые гипотезы playbook кончились: она видит всю их
    историю и предлагает то, чего в playbook нет. AGENT_LLM_FIRST=1 — прежний порядок.
    budget (контракт 1.1, D3): по остатку времени LLM-клиент решает, звать ли модель и сколько ждать.
    """
    llm_first = os.getenv(LLM_FIRST_ENV, "0") == "1"
    out: list[Hypothesis] = _llm_part(spec, profile, history, k, llm, memory, budget) if llm_first else []
    out += _playbook_part(spec, profile, history, k - len(out), out)
    if not llm_first and len(out) < k:
        out += _dedup(_llm_part(spec, profile, history, k - len(out), llm, memory, budget), history, out)
    return out


def _llm_part(spec, profile, history, k, llm, memory, budget) -> list[Hypothesis]:
    """Гипотезы LLM — с тем, что у бейзлайна включено для задачи (with_task_defaults, #191)."""
    if llm is None or k <= 0:
        return []
    from agent.llm_hypotheses import llm_hypotheses  # здесь: llm_hypotheses сам импортирует этот модуль

    out = llm_hypotheses(spec, profile, history, k, llm, memory, budget)
    return [h.model_copy(update={"params": with_task_defaults(h.params, spec)}) for h in out]


def _dedup(hyps: list[Hypothesis], history, out: list[Hypothesis]) -> list[Hypothesis]:
    """Гипотезы без повторов истории и уже выбранных (по id и по сути params)."""
    scheme = run_scheme(history)
    seen_ids = {h.id for h, _ in history} | {h.id for h in out}
    seen = {params_key(h.params, scheme) for h, _ in history} | {params_key(h.params, scheme) for h in out}
    fresh = []
    for hyp in hyps:
        key = params_key(hyp.params, scheme)
        if hyp.id not in seen_ids and key not in seen:
            fresh.append(hyp)
            seen_ids.add(hyp.id)
            seen.add(key)
    return fresh


def _playbook_part(spec, profile, history, k, out: list[Hypothesis]) -> list[Hypothesis]:
    """До k применимых гипотез playbook, которых ещё не было в истории и среди out."""
    picked: list[Hypothesis] = []
    tried = {h.id for h, _ in history} | {h.id for h in out}
    scheme = run_scheme(history)
    seen_params = {params_key(h.params, scheme) for h, _ in history} | {params_key(h.params, scheme) for h in out}
    for item in load_playbook():
        if len(picked) >= k:
            break
        params = with_task_defaults(item.get("params", {}), spec)
        if item["id"] in tried or params_key(params, scheme) in seen_params or not _applies(item, spec, profile):
            continue
        picked.append(
            Hypothesis(
                id=item["id"],
                kind=HypothesisKind(item["kind"]),
                title=item["title"],
                rationale=item.get("rationale", ""),
                params=params,
                source="playbook",
            )
        )
        seen_params.add(params_key(params, scheme))
    return picked
