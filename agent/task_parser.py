"""Разбор задачи: task.md + файлы рядом → TaskSpec. Владелец — LLM/agent-инженер.

Сейчас разбор детерминированный (без LLM) — он должен работать ВСЕГДА, даже если LLM недоступна.
LLM-уточнение (agent/llm_client.py) добавляется поверх и может только дополнить notes/метрику.

Важно: пути в тексте task.md не обязаны совпадать с реальной раскладкой
(в публичных задачах так и есть), поэтому файлы ищем сами.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from evaluation.metrics import METRICS, find_metric_in_text
from interfaces.schemas import MetricSpec, TaskSpec, TaskType
from runtime.log import get_logger

log = get_logger("task_parser")

FILE_PATTERNS = {
    "train": re.compile(r"^train.*\.csv$", re.I),
    "test": re.compile(r"^test.*\.csv$", re.I),
    "sample_submission": re.compile(r"^sample.*submission.*\.csv$", re.I),
    "data_dictionary": re.compile(r"^(data_)?dictionary.*\.csv$", re.I),
}
DEFAULT_METRIC = {TaskType.REGRESSION: "rmse", TaskType.BINARY: "roc_auc", TaskType.MULTICLASS: "f1_macro"}
MAX_CLASSES_FOR_NUMERIC_TARGET = 20
PROBA_HINTS = ("вероятност", "probabilit")
# Названия вероятностных метрик, которых пока нет в реестре evaluation/metrics.py (зона qa) —
# убрать, когда реестр начнёт их узнавать
EXTRA_PROBA_ALIASES = {
    "logloss": ("binary cross-entropy", "binary crossentropy", "crossentropy", "cross entropy", "bce", "log-loss")
}


def find_task_files(task_md: Path) -> dict[str, Path]:
    """Ищет файлы задачи: сначала по путям из текста, потом рекурсивно рядом с task.md."""
    task_dir = task_md.parent
    text = task_md.read_text(encoding="utf-8")
    found: dict[str, Path] = {}
    # 1) пути, упомянутые в тексте (`folder/train.csv`)
    for mention in re.findall(r"`([^`\s]+\.csv)`", text):
        candidate = task_dir / mention
        if candidate.exists():
            for key, pattern in FILE_PATTERNS.items():
                if key not in found and pattern.match(candidate.name):
                    found[key] = candidate
    # 2) рекурсивный поиск, ближайшие файлы — первыми
    for csv in sorted(task_dir.rglob("*.csv"), key=lambda p: (len(p.parts), p.name)):
        if csv.name.lower() in ("submission.csv",) or csv.name.endswith(".tmp.csv"):
            continue
        for key, pattern in FILE_PATTERNS.items():
            if key not in found and pattern.match(csv.name):
                found[key] = csv
    return found


def parse_task(task_md: Path) -> TaskSpec:
    task_md = task_md.resolve()
    text = task_md.read_text(encoding="utf-8")
    files = find_task_files(task_md)
    notes: list[str] = []
    for required in ("train", "test"):
        if required not in files:
            raise FileNotFoundError(f"Не найден {required}*.csv рядом с {task_md}")

    train_cols = list(pd.read_csv(files["train"], nrows=0).columns)
    test_cols = list(pd.read_csv(files["test"], nrows=0).columns)
    dictionary = _read_dictionary(files.get("data_dictionary"))

    # id и target: надёжнее всего — заголовок sample_submission
    if "sample_submission" in files:
        submission_columns = list(pd.read_csv(files["sample_submission"], nrows=0).columns)
        id_column, target = submission_columns[0], submission_columns[-1]
    else:
        target = _guess_target(train_cols, test_cols, dictionary)
        id_column = _guess_id(test_cols, dictionary)
        submission_columns = [id_column, target]
        notes.append("Нет sample_submission: id и target определены по train/test/словарю")

    y = pd.read_csv(files["train"], usecols=[target])[target]
    metric_name = find_metric_in_text(text)
    task_type = _guess_task_type(y, metric_name, dictionary.get(target))
    metric_name = _check_needs_proba(metric_name, task_type, text, files.get("sample_submission"), target, notes)
    if metric_name is None:
        metric_name = DEFAULT_METRIC.get(task_type, "rmse")
        notes.append(f"Метрика в тексте не найдена — взяли {metric_name} по типу задачи")
    metric = METRICS[metric_name]

    class_labels = None
    if task_type in (TaskType.BINARY, TaskType.MULTICLASS):
        class_labels = sorted(map(str, y.dropna().unique()))

    spec = TaskSpec(
        task_md=task_md,
        task_dir=task_md.parent,
        train_path=files["train"],
        test_path=files["test"],
        sample_submission_path=files.get("sample_submission"),
        data_dictionary_path=files.get("data_dictionary"),
        submission_path=task_md.parent / "submission.csv",
        task_type=task_type,
        target=target,
        id_column=id_column,
        metric=MetricSpec(name=metric.name, greater_is_better=metric.greater_is_better),
        submission_columns=submission_columns,
        class_labels=class_labels,
        description=text,
        notes=notes,
    )
    log.info(
        "Задача: %s, target=%s, метрика=%s, классов=%s",
        task_type.value,
        target,
        metric.name,
        len(class_labels) if class_labels else "-",
    )
    return spec


def _check_needs_proba(
    metric_name: str | None, task_type: TaskType, text: str, sample: Path | None, target: str, notes: list[str]
) -> str | None:
    """Страховка от неверной метрики: задача ждёт вероятности, а найденная метрика оценивает метки.

    Ловушка: «accuracy не подходит, оцениваем binary cross-entropy» — поиск метрики берёт accuracy,
    агент пишет 0/1 и получает огромный logloss. Признаки, что нужны вероятности:
    дробные значения в sample_submission или вероятностная метрика + слово «вероятность» в тексте.
    """
    if task_type != TaskType.BINARY:
        # multiclass: fit_predict_final пишет метки, а ответ «колонка на класс» парсер пока не разбирает —
        # переключение на logloss там только ухудшит балл (ревью #83, задача ml + agent)
        return metric_name
    low = text.lower()
    proba_metric = _mentioned_proba_metric(low)
    if metric_name is None:
        return proba_metric  # «BCE» и т.п. реестр не знает — но это точно вероятности
    if METRICS[metric_name].prediction != "label":
        return metric_name
    from_sample = _sample_has_probabilities(sample, target)
    from_text = proba_metric is not None and any(hint in low for hint in PROBA_HINTS)
    if not (from_sample or from_text):
        return metric_name
    chosen = proba_metric or "logloss"
    reason = "в sample_submission дробные значения" if from_sample else f"в тексте есть {chosen} и «вероятность»"
    notes.append(f"Метрика {metric_name} оценивает метки, но задача ждёт вероятности ({reason}) — взяли {chosen}")
    log.warning("Метрика %s → %s: %s", metric_name, chosen, reason)
    return chosen


def _mentioned_proba_metric(low: str) -> str | None:
    """Самая ранняя по тексту вероятностная метрика (logloss, roc_auc, …) или None."""
    best: tuple[int, str] | None = None
    for name, metric in METRICS.items():
        if metric.prediction != "proba":
            continue
        for alias in metric.aliases + EXTRA_PROBA_ALIASES.get(name, ()):
            alias = alias.rstrip("*")
            # слева — граница слова; справа — не латиница/цифра: «кросс-энтропией» подходит, «auction» — нет
            match = re.search(rf"(?<!\w){re.escape(alias)}(?![a-z0-9])", low)
            if match and (best is None or match.start() < best[0]):
                best = (match.start(), name)
    return best[1] if best else None


def _sample_has_probabilities(sample: Path | None, target: str) -> bool:
    if sample is None:
        return False
    try:
        values = pd.read_csv(sample, usecols=[target], nrows=2000)[target].dropna()
    except Exception:  # noqa: BLE001 — шаблон необязателен
        return False
    if values.empty or not pd.api.types.is_numeric_dtype(values):
        return False
    return bool(values.between(0, 1).all() and (values % 1 != 0).any())


def _read_dictionary(path: Path | None) -> dict[str, dict]:
    if path is None:
        return {}
    try:
        df = pd.read_csv(path)
        key = "feature" if "feature" in df.columns else df.columns[0]
        return {str(row[key]): row.to_dict() for _, row in df.iterrows()}
    except Exception:  # noqa: BLE001 — словарь не обязателен
        return {}


def _guess_target(train_cols, test_cols, dictionary) -> str:
    for name, row in dictionary.items():
        if str(row.get("role", "")).lower() == "target" and name in train_cols:
            return name
    extra = [c for c in train_cols if c not in test_cols]
    if len(extra) == 1:
        return extra[0]
    raise ValueError(f"Не удалось определить target: колонки только в train = {extra}")


def _guess_id(test_cols, dictionary) -> str:
    for name, row in dictionary.items():
        if str(row.get("role", "")).lower() == "id" and name in test_cols:
            return name
    return "id" if "id" in test_cols else test_cols[0]


def _guess_task_type(y: pd.Series, metric_name: str | None, dict_row: dict | None) -> TaskType:
    if metric_name is not None and METRICS[metric_name].prediction == "value":
        return TaskType.REGRESSION
    n_unique = y.nunique(dropna=True)
    is_categorical = (
        y.dtype == object
        or str((dict_row or {}).get("dtype", "")).lower() in ("category", "categorical", "binary")
        or metric_name is not None  # метрика классификации найдена в тексте
        or (pd.api.types.is_integer_dtype(y) and n_unique <= MAX_CLASSES_FOR_NUMERIC_TARGET)
    )
    if not is_categorical:
        return TaskType.REGRESSION
    return TaskType.BINARY if n_unique == 2 else TaskType.MULTICLASS
