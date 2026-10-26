"""Qwen — инженер данных (главная роль LLM, 26.10). Владелец — agent.

Сразу после бейзлайна Qwen получает условие задачи, профиль и карточки колонок и отвечает планом (JSON):
  plan         — что и зачем делает с данными;
  code         — функция prepare(df) → DataFrame: те же строки, только колонки для модели (чистка, отбор, новые
                 признаки). Исполняется в песочнице (runtime/sandbox.py), как код признаков (agent/feature_code.py);
  model        — модель из реестра ml/experiment.MODEL_FITTERS и её model_params;
  log_target / objective_metric / class_weight — готовые ключи params ml/.
Агент считает CV на подготовленных данных в той же схеме и на тех же фолдах, что бейзлайн, и показывает Qwen
результат или ошибку — до MAX_ROUNDS раундов. Затем Qwen выбирает вариант для ответа и пишет вывод
(qwen_report.md в папке прогона и agent.log). Страховка: вариант, который по CV хуже бейзлайна больше чем на разброс
бейзлайна по фолдам, не берётся — остаётся бейзлайн. Без LLM, при мусорном ответе или сбое — None: агент работает
как раньше. AGENT_DATA_ENGINEER=0 выключает.
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from string import Template

import numpy as np
import pandas as pd

from agent.feature_code import ALLOWED_IMPORTS, SANDBOX_MEM_MB, check_feature_code
from agent.prompting import describe_profile, describe_task, load_prompt
from interfaces.schemas import (
    DataProfile,
    ExperimentResult,
    ExperimentStatus,
    Hypothesis,
    HypothesisKind,
    RunBudget,
    TaskSpec,
    TaskType,
)
from ml.experiment import MODEL_FITTERS, run_experiment
from ml.profiling import profile_data
from ml.validation import parse_validation
from runtime.log import get_logger
from runtime.sandbox import run_code

log = get_logger("data_engineer")

DE_ENV = "AGENT_DATA_ENGINEER"  # "0" — выключить роль инженера данных
FUNCTION = "prepare"
MAX_ROUNDS = 3  # планов от Qwen: первый + до двух исправлений по ошибке или результату CV
MAX_CODE_CHARS = 6000
MAX_COLUMNS_SHOWN = 80
MAX_OUT_COLUMNS = 500
EXAMPLES = 3
PLAN_MAX_TOKENS = 3000
VERDICT_MAX_TOKENS = 800
TEMPERATURE = 0.2
SANDBOX_TIMEOUT = 300.0
DE_TIME_SHARE = 0.5  # не больше этой доли остатка бюджета на всю роль: остальное — циклу AutoResearch
ROUND_MIN_SECONDS = 45.0  # раунд = запрос к LLM + песочница + CV; меньше — не начинаем
PLAN_KEYS = ("plan", "code", "model", "model_params", "log_target", "objective_metric", "class_weight")


def enabled() -> bool:
    return os.getenv(DE_ENV, "1") != "0"


@dataclass
class Attempt:
    """Один план Qwen: что предложил, что вышло."""

    round: int
    plan: str
    hyp: Hypothesis | None = None
    result: ExperimentResult | None = None
    error: str = ""
    train: pd.DataFrame | None = None
    test: pd.DataFrame | None = None
    profile: DataProfile | None = None
    columns: list[str] = field(default_factory=list)


@dataclass
class Outcome:
    """Выбор Qwen: данные и гипотеза для ответа (chosen=None — остаётся бейзлайн) и вывод для отчёта."""

    chosen: Attempt | None
    attempts: list[Attempt]
    conclusion: str


# --- 1. Промпт


def _short(value, limit: int = 30) -> str:
    text = str(value).replace("\n", " ")
    return text if len(text) <= limit else text[: limit - 1] + "…"


def describe_columns(frame: pd.DataFrame) -> str:
    """Карточки колонок: тип, доля пропусков, число разных значений и несколько примеров (не строки целиком)."""
    lines = []
    for col in list(frame.columns)[:MAX_COLUMNS_SHOWN]:
        s = frame[col]
        kind = "число" if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s) else "текст/категория"
        values = s.dropna()
        examples = ", ".join(_short(v) for v in values.drop_duplicates().head(EXAMPLES))
        lines.append(
            f"- {col} — {kind}; пропуски {s.isna().mean():.2f}; разных {values.nunique()}; примеры: {examples or '—'}"
        )
    if frame.shape[1] > MAX_COLUMNS_SHOWN:
        lines.append(
            f"- … и ещё {frame.shape[1] - MAX_COLUMNS_SHOWN} колонок: {', '.join(map(str, frame.columns[MAX_COLUMNS_SHOWN:]))[:1500]}"
        )
    return "\n".join(lines)


def _score(res: ExperimentResult | None) -> str:
    if res is None or res.mean_score is None:
        return "не посчитано"
    return f"{res.mean_score:.5g} ± {res.std_score or 0:.3g} (фолды: {', '.join(f'{v:.4g}' for v in res.fold_scores)})"


def describe_attempts(attempts: list[Attempt], base_result: ExperimentResult) -> str:
    if not attempts:
        return "Пока ничего — это первый план."
    lines = []
    for a in attempts:
        model = a.hyp.params.get("model") if a.hyp else "—"
        outcome = f"ошибка: {a.error}" if a.error else f"CV {_score(a.result)} (бейзлайн {_score(base_result)})"
        lines.append(f"- Раунд {a.round}: {_short(a.plan, 300)} | модель {model} | {outcome}")
    lines.append(
        "Улучши план: исправь ошибку или попробуй другую подготовку/модель, если выигрыша над бейзлайном нет. "
        'Если лучший вариант уже устраивает и улучшать нечего — ответь {"stop": true}.'
    )
    return "\n".join(lines)


def build_plan_messages(
    spec: TaskSpec, profile: DataProfile, raw: pd.DataFrame, base_result: ExperimentResult, attempts: list[Attempt]
) -> list[dict] | None:
    system, template = load_prompt("system"), load_prompt("data_engineer")
    if system is None or template is None:
        return None
    user = Template(template).safe_substitute(
        task=describe_task(spec),
        profile=describe_profile(profile),
        columns=describe_columns(raw),
        baseline=f"LightGBM на всех колонках как есть: CV {spec.metric.name} = {_score(base_result)}",
        feedback=describe_attempts(attempts, base_result),
        function=FUNCTION,
        models=", ".join(MODEL_FITTERS),
        imports=", ".join(sorted(ALLOWED_IMPORTS)),
    )
    return [{"role": "system", "content": system.strip()}, {"role": "user", "content": user.strip()}]


# --- 2. Разбор плана


def parse_plan(raw, spec: TaskSpec) -> tuple[dict | None, str]:
    """Ответ Qwen → (проверенный план, "") или (None, причина для следующего раунда)."""
    if isinstance(raw, list) and raw:
        raw = raw[0]
    if not isinstance(raw, dict):
        return None, "ответ не JSON-объект"
    plan = {k: raw.get(k) for k in PLAN_KEYS}
    reason = check_feature_code(str(plan["code"] or ""), spec.target, function=FUNCTION, max_chars=MAX_CODE_CHARS)
    if reason:
        return None, f"код не принят: {reason}"
    model = str(plan["model"] or "lightgbm").strip().lower()
    if model not in MODEL_FITTERS:
        return None, f"неизвестная модель {model!r}, доступны: {', '.join(MODEL_FITTERS)}"
    plan["model"] = model
    if not isinstance(plan["model_params"], dict):
        plan["model_params"] = {}
    return plan, ""


def plan_params(plan: dict, spec: TaskSpec, validation: str | None) -> dict:
    """План → params гипотезы ml/ (лишнее для типа задачи отбрасываем: иначе гипотеза совпадёт или упадёт)."""
    params: dict = {"model": plan["model"]}
    if plan["model_params"]:
        params["model_params"] = dict(plan["model_params"])
    regression = spec.task_type == TaskType.REGRESSION
    if regression and plan.get("log_target") is True:
        params["log_target"] = True
    if regression and plan.get("objective_metric") is True and not params.get("log_target"):
        params["objective"] = "metric"
    if not regression and plan.get("class_weight") == "balanced":
        params["class_weight"] = "balanced"
    if validation:
        params["validation"] = validation
    return params


# --- 3. Код в песочнице

# Обвязка: таблицу читает песочница (pickle пишем мы), назад — CSV и JSON (их чтение не исполняет код)
_RUNNER = """

import json as _json

import pandas as _pd

_df = _pd.read_pickle("prepare_in.pkl")
_out = {function}(_df.copy())
if not isinstance(_out, _pd.DataFrame):
    raise TypeError(f"{function} вернула {{type(_out).__name__}}, а не DataFrame")
_report = {{"rows": len(_out), "same_index": bool(_out.index.equals(_df.index)), "columns": [str(c) for c in _out.columns]}}
if _report["rows"] == len(_df) and _report["same_index"]:
    _out.columns = _report["columns"]
    _out.to_csv("prepare_out.csv", index=False)
    _report["dtypes"] = {{c: ("num" if _pd.api.types.is_numeric_dtype(_out[c]) else "cat") for c in _out.columns}}
with open("prepare_report.json", "w", encoding="utf-8") as _f:
    _json.dump(_report, _f, ensure_ascii=False)
"""


def apply_prepare_code(
    code: str, df: pd.DataFrame, target: str | None = None, timeout: float = SANDBOX_TIMEOUT
) -> tuple[pd.DataFrame | None, str]:
    """Выполнить prepare(df) в песочнице. → (подготовленная таблица с индексом df, "") или (None, причина)."""
    reason = check_feature_code(code, target, function=FUNCTION, max_chars=MAX_CODE_CHARS)
    if reason:
        return None, reason
    try:
        with tempfile.TemporaryDirectory(prefix="prepare-") as tmp:
            folder = Path(tmp)
            df.to_pickle(folder / "prepare_in.pkl")
            res = run_code(code + _RUNNER.format(function=FUNCTION), folder, timeout=timeout, mem_mb=SANDBOX_MEM_MB)
            if res.timed_out:
                return None, f"код не уложился в {timeout:.0f} с"
            if res.exit_code != 0:
                last = (res.stderr.strip().splitlines() or ["без сообщения"])[-1]
                return None, f"код упал: {last[:300]}"
            report = json.loads((folder / "prepare_report.json").read_text(encoding="utf-8"))
            if report.get("rows") != len(df) or not report.get("same_index"):
                return (
                    None,
                    f"код изменил строки: было {len(df)}, стало {report.get('rows')} (или другой порядок/индекс)",
                )
            columns = report.get("columns") or []
            if not columns:
                return None, "prepare вернула таблицу без колонок"
            if len(columns) > MAX_OUT_COLUMNS or len(set(columns)) != len(columns):
                return None, f"колонок {len(columns)} (больше {MAX_OUT_COLUMNS}) или повторяются имена"
            dtypes = report.get("dtypes") or {}
            text = [c for c, kind in dtypes.items() if kind == "cat"]
            out = pd.read_csv(folder / "prepare_out.csv", dtype=dict.fromkeys(text, str), low_memory=False)
    except Exception as exc:  # noqa: BLE001 — сбой обмена = план не удался, агент работает дальше
        return None, f"обмен с песочницей не удался: {type(exc).__name__}: {exc}"[:300]
    out.index = df.index
    numeric = out.select_dtypes(include="number").columns
    out[numeric] = out[numeric].replace([np.inf, -np.inf], np.nan)
    useful = [c for c in out.columns if out[c].nunique(dropna=False) > 1]
    if not useful:
        return None, "все колонки после prepare константные"
    return out[useful], ""


# --- 4. Раунды и вывод


def _targets(spec: TaskSpec) -> list[str]:
    return [spec.target, *spec.extra_targets]


def raw_features(spec: TaskSpec, train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    """train и test вместе без target и id — то, что видит prepare(df)."""
    cols = [c for c in train.columns if c in test.columns and c not in (*_targets(spec), spec.id_column)]
    return pd.concat([train[cols], test[cols]], ignore_index=True)


def split_prepared(
    spec: TaskSpec, prepared: pd.DataFrame, train: pd.DataFrame, test: pd.DataFrame, keep: str | None
) -> tuple[pd.DataFrame, pd.DataFrame, str | None]:
    """Подготовленная таблица → train и test с id и target. keep — колонка схемы CV (время/группы): если Qwen её
    убрал, она возвращается служебной (по ней строятся фолды), в признаки не идёт. → (train, test, служебная)."""
    n = len(train)
    clash = [c for c in prepared.columns if c in (spec.id_column, *_targets(spec))]  # имя id/target — не признак
    prepared = prepared.drop(columns=clash)
    tr = prepared.iloc[:n].reset_index(drop=True)
    te = prepared.iloc[n:].reset_index(drop=True)
    helper = None
    if keep and keep not in tr.columns and keep in train.columns and keep in test.columns:
        helper = keep
        tr[keep], te[keep] = train[keep].to_numpy(), test[keep].to_numpy()
    tr.insert(0, spec.id_column, train[spec.id_column].to_numpy())
    te.insert(0, spec.id_column, test[spec.id_column].to_numpy())
    for target in _targets(spec):
        tr[target] = train[target].to_numpy()
    return tr, te, helper


def _without(profile: DataProfile, column: str | None) -> DataProfile:
    if column is None:
        return profile
    return profile.model_copy(
        update={
            "numeric_features": [c for c in profile.numeric_features if c != column],
            "categorical_features": [c for c in profile.categorical_features if c != column],
        }
    )


def _round(
    k: int,
    plan: dict,
    spec: TaskSpec,
    train: pd.DataFrame,
    test: pd.DataFrame,
    validation: str | None,
    budget: RunBudget,
) -> Attempt:
    attempt = Attempt(round=k, plan=str(plan.get("plan") or "без описания"))
    prepared, reason = apply_prepare_code(str(plan["code"]), raw_features(spec, train, test), spec.target)
    if prepared is None:
        attempt.error = reason
        return attempt
    scheme, column = parse_validation(validation, spec)
    keep = column if scheme in ("time", "group") else None
    tr, te, helper = split_prepared(spec, prepared, train, test, keep)
    profile = _without(profile_data(spec, tr, te), helper)
    hyp = Hypothesis(
        id=f"qwen_data_{k}",
        kind=HypothesisKind.PREPROCESSING,
        title=f"Qwen — подготовка данных, раунд {k}",
        rationale=attempt.plan[:1000],
        params=plan_params(plan, spec, validation),
        source="llm",
    )
    result = run_experiment(spec, profile, hyp, tr, budget)
    attempt.hyp, attempt.result, attempt.columns = hyp, result, list(prepared.columns)
    attempt.train, attempt.test, attempt.profile = tr, te, profile
    if result.status != ExperimentStatus.OK:
        attempt.error = f"CV не посчиталась: {result.status.value} {result.error or ''}"[:300]
    return attempt


def too_worse(res: ExperimentResult, base: ExperimentResult, greater_is_better: bool) -> bool:
    """Страховка вердикта: вариант хуже бейзлайна больше, чем на разброс бейзлайна по фолдам."""
    if res.mean_score is None or base.mean_score is None or res.validation != base.validation:
        return True
    gap = (base.mean_score - res.mean_score) if greater_is_better else (res.mean_score - base.mean_score)
    return gap > (base.std_score or 0.0)


def _best_by_cv(attempts: list[Attempt], base: ExperimentResult, gib: bool) -> Attempt | None:
    ok = [a for a in attempts if not a.error and a.result is not None and a.result.mean_score is not None]
    if not ok:
        return None
    best = (max if gib else min)(ok, key=lambda a: a.result.mean_score)
    better = best.result.mean_score > base.mean_score if gib else best.result.mean_score < base.mean_score
    return best if better and best.result.validation == base.validation else None


def ask_verdict(
    spec: TaskSpec, base: Hypothesis, base_result: ExperimentResult, attempts: list[Attempt], llm, budget: RunBudget
) -> tuple[str | None, str]:
    """Qwen выбирает вариант и пишет вывод. → (id варианта или None — LLM не ответила, вывод)."""
    template, system = load_prompt("data_engineer_verdict"), load_prompt("system")
    if template is None or system is None:
        return None, ""
    rows = [f"- {base.id}: бейзлайн (LightGBM на сырых колонках) — CV {_score(base_result)}"]
    for a in attempts:
        if a.hyp is not None and not a.error:
            rows.append(
                f"- {a.hyp.id}: {_short(a.plan, 300)} | модель {a.hyp.params.get('model')} — CV {_score(a.result)}"
            )
        else:
            rows.append(f"- раунд {a.round} (не участвует): {_short(a.plan, 200)} — {a.error}")
    user = Template(template).safe_substitute(
        task=describe_task(spec),
        metric=spec.metric.name,
        direction="больше — лучше" if spec.metric.greater_is_better else "меньше — лучше",
        results="\n".join(rows),
    )
    messages = [{"role": "system", "content": system.strip()}, {"role": "user", "content": user.strip()}]
    try:
        raw = llm.chat_json(
            messages, temperature=TEMPERATURE, max_tokens=VERDICT_MAX_TOKENS, seconds_left=budget.seconds_left()
        )
    except Exception:  # noqa: BLE001
        log.exception("Вывод Qwen не получен")
        return None, ""
    if isinstance(raw, list) and raw:
        raw = raw[0]
    if not isinstance(raw, dict):
        return None, ""
    return (str(raw.get("choice")) if raw.get("choice") else None), str(raw.get("conclusion") or "")


def write_report(outcome: Outcome, spec: TaskSpec, base_result: ExperimentResult) -> None:
    """Отчёт инженера данных: в agent.log и qwen_report.md в папке прогона (AGENT_RUN_DIR), если она задана."""
    lines = [
        f"# Qwen — инженер данных: {spec.target}, {spec.metric.name}",
        "",
        f"Бейзлайн: CV {_score(base_result)}",
        "",
    ]
    for a in outcome.attempts:
        lines += [f"## Раунд {a.round}", a.plan, ""]
        if a.columns:
            lines.append(f"Колонки в модель ({len(a.columns)}): {', '.join(a.columns[:60])}")
        if a.hyp is not None:
            lines.append(f"Модель: {a.hyp.params}")
        lines += [f"Итог: {a.error or 'CV ' + _score(a.result)}", ""]
    chosen = outcome.chosen.hyp.id if outcome.chosen and outcome.chosen.hyp else "бейзлайн"
    lines += ["## Вывод", f"В ответ: {chosen}", outcome.conclusion or "—"]
    text = "\n".join(lines)
    log.info("Qwen — инженер данных, вывод: в ответ %s. %s", chosen, outcome.conclusion)
    run_dir = os.getenv("AGENT_RUN_DIR")
    if run_dir:
        try:
            Path(run_dir).mkdir(parents=True, exist_ok=True)
            (Path(run_dir) / "qwen_report.md").write_text(text, encoding="utf-8")
        except OSError:
            log.warning("qwen_report.md не записан")


def run_data_engineer(
    spec: TaskSpec,
    profile: DataProfile,
    train: pd.DataFrame,
    test: pd.DataFrame,
    base: Hypothesis,
    base_result: ExperimentResult,
    budget: RunBudget,
    llm,
) -> Outcome | None:
    """Роль инженера данных целиком. None — роль не работала (выключена, нет LLM, нет времени, все планы мимо)."""
    if not enabled() or llm is None or not llm.available() or base_result.mean_score is None:
        return None
    if load_prompt("data_engineer") is None:
        return None
    deadline = time.time() + DE_TIME_SHARE * budget.seconds_left()
    validation = base.params.get("validation")
    raw = raw_features(spec, train, test).iloc[: len(train)]
    attempts: list[Attempt] = []
    gib = spec.metric.greater_is_better
    for k in range(1, MAX_ROUNDS + 1):
        left = min(deadline - time.time(), budget.seconds_left())
        if left < ROUND_MIN_SECONDS + 2 * (base_result.fit_seconds or 0):
            log.info("Инженер данных: на раунд %d нет времени (осталось %.0f с)", k, left)
            break
        messages = build_plan_messages(spec, profile, raw, base_result, attempts)
        if messages is None:
            break
        try:
            answer = llm.chat_json(
                messages, temperature=TEMPERATURE, max_tokens=PLAN_MAX_TOKENS, seconds_left=budget.seconds_left()
            )
        except Exception:  # noqa: BLE001 — сбой LLM не роняет агента
            log.exception("Инженер данных: план не получен")
            answer = None
        if answer is None:
            log.info("Инженер данных: LLM не ответила в раунде %d", k)
            if not llm.available():
                break
            attempts.append(
                Attempt(round=k, plan="", error="ответ не разобран как JSON — нужен ровно один JSON-объект")
            )
            continue
        if isinstance(answer, dict) and answer.get("stop") is True and attempts:
            log.info("Инженер данных: Qwen остановился после раунда %d — улучшать нечего", k - 1)
            break
        plan, reason = parse_plan(answer, spec)
        if plan is None:
            log.info("Инженер данных, раунд %d: план отклонён — %s", k, reason)
            attempts.append(
                Attempt(
                    round=k, plan=str((answer or {}).get("plan", "")) if isinstance(answer, dict) else "", error=reason
                )
            )
            continue
        log.info("Инженер данных, раунд %d: %s | модель %s", k, _short(plan["plan"], 400), plan["model"])
        try:
            attempt = _round(k, plan, spec, train, test, validation, budget)
        except Exception as exc:  # noqa: BLE001 — раунд не удался, а не агент
            log.exception("Инженер данных, раунд %d упал", k)
            attempt = Attempt(round=k, plan=str(plan.get("plan") or ""), error=f"{type(exc).__name__}: {exc}"[:300])
        attempts.append(attempt)
        log.info(
            "Инженер данных, раунд %d: %s",
            k,
            attempt.error or f"CV {_score(attempt.result)}, бейзлайн {_score(base_result)}",
        )
    if not any(a.hyp is not None and not a.error for a in attempts):
        if attempts:
            write_report(
                Outcome(None, attempts, "ни один план не дал рабочих данных — остаётся бейзлайн"), spec, base_result
            )
        return None if not attempts else Outcome(None, attempts, "ни один план не дал рабочих данных")
    choice, conclusion = ask_verdict(spec, base, base_result, attempts, llm, budget)
    by_id = {a.hyp.id: a for a in attempts if a.hyp is not None and not a.error}
    chosen = by_id.get(choice) if choice else None
    if choice is None:  # LLM не дала вердикт — по CV, только если лучше бейзлайна
        chosen = _best_by_cv(attempts, base_result, gib)
        conclusion = conclusion or "вывод LLM не получен — выбран лучший по CV вариант, если он лучше бейзлайна"
    if chosen is not None and too_worse(chosen.result, base_result, gib):
        conclusion += (
            f" [страховка агента: {chosen.hyp.id} хуже бейзлайна больше разброса по фолдам — остаётся бейзлайн]"
        )
        chosen = None
    outcome = Outcome(chosen, attempts, conclusion)
    write_report(outcome, spec, base_result)
    return outcome
