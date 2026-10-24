"""LLM-аналитик задачи: Qwen читает task.md и заголовки файлов, агент сверяет ответ с данными. Владелец — agent.

Вебинар организаторов (docs/resources/webinar-qa.md): формат задач не гарантирован, «парсеры перестанут
работать» — агент должен понимать текст task.md. Поэтому разбор двухслойный:
  1. детерминированный (task_parser) — работает всегда, без LLM;
  2. этот модуль — один запрос к LLM до шага 0. Каждое поле ответа проверяется по данным (колонка есть
     в train/test, метрика есть в реестре). LLM заполняет то, что эвристика лишь угадала; при споре
     с разбором, подтверждённым данными, побеждают данные, а расхождение пишется в лог и в notes.

Без LLM, при мусоре в ответе или ошибке — разбор остаётся детерминированным.
"""

from __future__ import annotations

import os
from pathlib import Path
from string import Template

import pandas as pd

from agent.llm_client import LLMClient
from agent.prompting import load_prompt
from evaluation.metrics import METRICS, find_metric_in_text
from runtime.log import get_logger

log = get_logger("task_analyst")

ANALYST_ENV = "AGENT_LLM_TASK"  # "0" — не спрашивать LLM
APPLY_ENV = "AGENT_ANALYST_APPLY"  # "1" — LLM заменяет угаданные поля; по умолчанию только заметки (#276, штраф −8)
MAX_TOKENS = 600
MAX_WAIT = 90.0  # сек: разбор идёт до шага 0 (тривиального ответа) — дольше ждать нельзя
MAX_TEXT_CHARS = 8000
MAX_COLUMNS = 80
SAMPLE_ROWS = 3
MAX_NOTE_CHARS = 160
PREFIX = "LLM-аналитик"
SYSTEM = (
    "Ты — опытный ML-инженер внутри AutoML-агента. Тебе дают описание табличной задачи и колонки файлов; "
    "нужно точно понять, что предсказывать и как это оценят. Отвечай только JSON, без пояснений."
)


def enabled() -> bool:
    return os.getenv(ANALYST_ENV, "1") != "0"


def apply_enabled() -> bool:
    """Может ли аналитик заменить угаданные target/id/метрику. По умолчанию нет: только заполнить пустое (#276)."""
    return os.getenv(APPLY_ENV, "0") == "1"


def _only_fill_gaps(accepted: dict[str, str], heuristic: dict[str, str | None]) -> tuple[dict[str, str], list[str]]:
    """Без AGENT_ANALYST_APPLY=1 оставить только поля, где разбор по данным ничего не нашёл; прочее — в заметки."""
    kept = {k: v for k, v in accepted.items() if heuristic.get(k) is None}
    notes = [
        f"предлагает {key}={value} вместо {heuristic[key]} — не применено ({APPLY_ENV}=1 включает)"
        for key, value in accepted.items()
        if key not in kept
    ]
    return kept, notes


def _columns(cols: list[str]) -> str:
    shown = ", ".join(cols[:MAX_COLUMNS])
    return shown + (f" … и ещё {len(cols) - MAX_COLUMNS}" if len(cols) > MAX_COLUMNS else "")


def build_messages(text: str, train_cols: list[str], test_cols: list[str], sample: Path | None) -> list[dict] | None:
    template = load_prompt("task")
    if template is None:
        return None
    sample_cols, sample_rows = "нет файла", "нет"
    if sample is not None:
        try:
            head = pd.read_csv(sample, nrows=SAMPLE_ROWS)
            sample_cols, sample_rows = _columns(list(head.columns)), head.to_csv(index=False).strip()
        except Exception:  # noqa: BLE001 — шаблон ответа необязателен
            pass
    if len(text) > MAX_TEXT_CHARS:
        text = text[:MAX_TEXT_CHARS] + " …"
    try:
        user = Template(template).substitute(
            text=text.strip(),
            n_train=len(train_cols),
            train=_columns(train_cols),
            n_test=len(test_cols),
            test=_columns(test_cols),
            sample=sample_cols,
            sample_rows=sample_rows,
            metrics=", ".join(METRICS),
        )
    except (KeyError, ValueError) as exc:
        log.warning("Промпт task.md не собран: %s", exc)
        return None
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user.strip()}]


def ask_llm(
    text: str, train_cols: list[str], test_cols: list[str], sample: Path | None, llm: LLMClient | None
) -> dict | None:
    """Ответ LLM о задаче (dict) или None — выключено, LLM недоступна, мусор или ошибка."""
    if llm is None or not enabled() or not llm.available():
        return None
    messages = build_messages(text, train_cols, test_cols, sample)
    if messages is None:
        return None
    try:
        raw = llm.chat_json(messages, temperature=0.0, max_tokens=MAX_TOKENS, max_wait=MAX_WAIT, purpose="task")
    except Exception:  # noqa: BLE001 — разбор без LLM всё равно есть
        log.exception("%s: запрос не удался — остаёмся на разборе по данным", PREFIX)
        return None
    if isinstance(raw, list):
        raw = next((item for item in raw if isinstance(item, dict)), None)
    return raw if isinstance(raw, dict) else None


def metric_key(value) -> str | None:
    """Название метрики от LLM → ключ реестра evaluation/metrics.py или None."""
    if not isinstance(value, str) or not value.strip():
        return None
    name = value.strip().lower()
    return name if name in METRICS else find_metric_in_text(name)


def reconcile(
    view: dict | None,
    heuristic: dict[str, str | None],
    guessed: set[str],
    train_cols: list[str],
    test_cols: list[str],
) -> tuple[dict[str, str], list[str]]:
    """Сверить ответ LLM с разбором по данным.

    heuristic — {"target", "id_column", "metric"} из детерминированного разбора; guessed — какие из них только
    угаданы (нет sample_submission, метрика не найдена в тексте…). Возвращает (что заменить, заметки для notes).
    """
    if not view:
        return {}, []
    accepted: dict[str, str] = {}
    notes: list[str] = []
    checks: list[str] = []

    targets = view.get("target")
    targets = [str(t) for t in targets] if isinstance(targets, list) else [str(targets)] if targets else []
    if len(targets) > 1:
        notes.append(f"несколько таргетов {targets}: агент пока предсказывает один ({heuristic['target']})")
    if targets:
        target = targets[0]
        valid = target in train_cols and target not in test_cols
        if target == heuristic["target"]:
            checks.append(f"target={target} ✓")
        elif valid and "target" in guessed:
            accepted["target"] = target
            checks.append(f"target={target} (вместо угаданного {heuristic['target']})")
        elif not valid:
            notes.append(f"target по тексту — {target}, но такой колонки нет только в train — не принят")
        else:
            notes.append(f"target по тексту — {target}, оставлен {heuristic['target']} (подтверждён данными)")

    id_column = view.get("id_column")
    if isinstance(id_column, str) and id_column:
        if id_column == heuristic["id_column"]:
            checks.append(f"id={id_column} ✓")
        elif id_column in test_cols and "id_column" in guessed:
            accepted["id_column"] = id_column
            checks.append(f"id={id_column} (вместо угаданного {heuristic['id_column']})")
        else:
            notes.append(f"id по тексту — {id_column}, оставлен {heuristic['id_column']}")

    raw_metric = view.get("metric")
    metric = metric_key(raw_metric)
    if raw_metric and metric is None:
        # метрики нет в реестре (#186): оптимизируем ближайшую из реестра, которую назвала LLM, если поиск
        # по тексту ничего не нашёл; иначе — найденную поиском. Настоящая метрика остаётся в notes
        closest = metric_key(view.get("closest_metric"))
        if closest and "metric" in guessed:
            accepted["metric"] = closest
            checks.append(f"метрика={closest} (ближайшая к «{str(raw_metric)[:40]}»)")
        chosen = accepted.get("metric", heuristic["metric"])
        notes.append(f"метрика по тексту — «{str(raw_metric)[:40]}», её нет в реестре: оптимизируем {chosen}")
    elif metric == heuristic["metric"]:
        checks.append(f"метрика={metric} ✓")
    elif metric and "metric" in guessed:
        accepted["metric"] = metric
        checks.append(f"метрика={metric} (в тексте не нашли по словарю)")
    elif metric:
        notes.append(f"метрика по тексту — {metric}, поиск по тексту нашёл {heuristic['metric']}: оставлен поиск")

    for key, label in (("time_column", "колонка времени"), ("group_column", "колонка групп")):
        column = view.get(key)
        if isinstance(column, str) and column in train_cols:
            notes.append(f"{label} для валидации — {column}")

    answer_format = view.get("answer_format")
    if answer_format in ("probability", "class_probabilities"):
        notes.append(f"формат ответа — {answer_format}")

    constraints = view.get("constraints")
    if isinstance(constraints, list):
        notes += [" ".join(str(c).split())[:MAX_NOTE_CHARS] for c in constraints[:3] if str(c).strip()]

    if not apply_enabled():
        accepted, kept_back = _only_fill_gaps(accepted, heuristic)
        notes += kept_back
    log.info("%s: %s", PREFIX, ", ".join(checks) or "поля не распознаны")
    for note in notes:
        log.info("%s: %s", PREFIX, note)
    return accepted, [f"{PREFIX}: {n}" for n in notes]
