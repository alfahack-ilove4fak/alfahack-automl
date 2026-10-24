"""Разбор задачи: task.md + файлы рядом → TaskSpec. Владелец — LLM/agent-инженер.

Детерминированный разбор работает ВСЕГДА, даже если LLM недоступна. Поверх него — LLM-аналитик
(agent/task_analyst.py): Qwen читает task.md, агент сверяет её ответ с данными. LLM заменяет только то,
что эвристика угадала (нет sample_submission, метрика не найдена), остальное попадает в notes.

Важно: пути в тексте task.md не обязаны совпадать с реальной раскладкой
(в публичных задачах так и есть), поэтому файлы ищем сами.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from agent.csv_dialect import read_text_any, standard_csv
from evaluation.metrics import METRICS, find_metric_in_text
from interfaces.schemas import ROW_ID_COLUMN, MetricSpec, TaskSpec, TaskType
from runtime.log import get_logger

log = get_logger("task_parser")

FILE_PATTERNS = {
    "train": re.compile(r"^train.*\.csv$", re.I),
    "test": re.compile(r"^test.*\.csv$", re.I),
    "sample_submission": re.compile(r"^sample.*submission.*\.csv$", re.I),
    "data_dictionary": re.compile(r"^(data_)?dictionary.*\.csv$", re.I),
}
DEFAULT_METRIC = {TaskType.REGRESSION: "rmse", TaskType.BINARY: "roc_auc", TaskType.MULTICLASS: "f1_macro"}
LABEL_METRIC = {TaskType.BINARY: "accuracy", TaskType.MULTICLASS: "f1_macro"}  # когда в шаблоне метки
MAX_CLASSES_FOR_NUMERIC_TARGET = 20


def find_task_files(task_md: Path) -> dict[str, Path]:
    """Ищет файлы задачи: сначала по путям из текста, потом рекурсивно рядом с task.md.

    Файлы не в обычном виде (cp1251, «;», десятичная запятая) подменяются обычной копией (agent/csv_dialect.py):
    дальше их читают как есть все модули агента (#238).
    """
    task_dir = task_md.parent
    text = read_text_any(task_md)
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
    return {key: standard_csv(path) for key, path in found.items()}


def parse_task(task_md: Path, llm=None, use_analyst: bool = True) -> TaskSpec:
    """task.md → TaskSpec. llm — LLMClient для аналитика; None — создать свой (без ключа он сразу недоступен).

    use_analyst=False — только разбор по данным, без LLM (клиент не создаётся): так `solution.py` пишет шаг 0 до
    любого запроса к LLM, а аналитика зовёт вторым разбором (#239).
    """
    task_md = task_md.resolve()
    text = read_text_any(task_md)
    files = find_task_files(task_md)
    notes: list[str] = []
    for required in ("train", "test"):
        if required not in files:
            raise FileNotFoundError(f"Не найден {required}*.csv рядом с {task_md}")

    train_cols = list(pd.read_csv(files["train"], nrows=0).columns)
    test_cols = list(pd.read_csv(files["test"], nrows=0).columns)
    dictionary = _read_dictionary(files.get("data_dictionary"))

    # id и target: надёжнее всего — заголовок sample_submission
    guessed: set[str] = set()  # что эвристика лишь угадала — это может уточнить LLM-аналитик
    if "sample_submission" in files:
        submission_columns = list(pd.read_csv(files["sample_submission"], nrows=0).columns)
        id_column, target = submission_columns[0], submission_columns[-1]
        if id_column not in test_cols and (
            len(submission_columns) == 1 or (id_column in train_cols and not _is_train_id(files["train"], id_column))
        ):
            # Шаблон без id (первая колонка — таргет из train или единственная, строки в порядке test): ответ — по номеру строки, id в ответ не входит
            id_column = ROW_ID_COLUMN
            notes.append("В шаблоне ответа нет id: строки ответа идут в порядке test.csv")
        if target not in train_cols:
            # Формат «колонка на класс» (id, класс_1, …, класс_k): в шаблоне не target, а вероятности классов
            target = _guess_target(train_cols, test_cols, dictionary, files["test"], strict=False)
            guessed.add("target")
            notes.append(
                f"Ответ — колонка на класс ({len(submission_columns) - 1} шт.): нужны вероятности, target={target}"
            )
    else:
        target = _guess_target(train_cols, test_cols, dictionary, files["test"], strict=False)
        id_column = _guess_id(test_cols, dictionary, files["test"])
        guessed |= {"target", "id_column"}
        notes.append("Нет sample_submission: id и target определены по train/test/словарю")
    metric_name = find_metric_in_text(text)
    if metric_name is None:
        guessed.add("metric")

    if use_analyst:
        target, id_column, metric_name = _ask_analyst(
            text, files, train_cols, test_cols, target, id_column, metric_name, guessed, notes, llm
        )
    if target is None:
        raise ValueError(f"Не удалось определить target: колонки только в train = {_train_only(train_cols, test_cols)}")
    if "sample_submission" not in files:
        submission_columns = [id_column, target]

    # Несколько таргетов (#174): другие колонки ответа, которые есть только в train, — тоже таргеты.
    # Основной (по нему CV и цикл) — тот, к которому подходит метрика из текста
    answer_columns = [c for c in submission_columns if c != id_column]
    targets = [target] + [c for c in answer_columns if c != target and c in train_cols and c not in test_cols]
    if len(targets) > 1:
        target = _primary_target(files["train"], targets, metric_name)
        notes.append(f"Несколько таргетов: {targets} — модель на каждый, цикл — по {target}")
    extra_targets = [t for t in targets if t != target]

    y = pd.read_csv(files["train"], usecols=[target])[target]
    task_type = _guess_task_type(y, metric_name, dictionary.get(target))
    metric_name = _check_needs_proba(metric_name, task_type, text, files.get("sample_submission"), target, notes)
    if metric_name is None:
        metric_name = DEFAULT_METRIC.get(task_type, "rmse")
        notes.append(f"Метрика в тексте не найдена — взяли {metric_name} по типу задачи")
    metric = METRICS[metric_name]

    class_labels = None
    if task_type in (TaskType.BINARY, TaskType.MULTICLASS):
        class_labels = sorted(map(str, y.dropna().unique()))
        if len(answer_columns) > 1 and not extra_targets and set(answer_columns) != set(class_labels):
            notes.append(f"Колонки ответа {answer_columns} не совпадают с классами train {class_labels}")

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
        extra_targets=extra_targets,
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


def _ask_analyst(text, files, train_cols, test_cols, target, id_column, metric_name, guessed, notes, llm):
    """LLM-аналитик (agent/task_analyst.py): уточняет угаданные поля, пишет расхождения в notes. Никогда не падает."""
    try:
        from agent import task_analyst  # здесь: разбор без LLM не должен зависеть от промптов и клиента

        if not task_analyst.enabled():
            return target, id_column, metric_name
        if llm is None:
            from agent.llm_client import LLMClient

            llm = LLMClient()
        view = task_analyst.ask_llm(text, train_cols, test_cols, files.get("sample_submission"), llm)
        heuristic = {"target": target, "id_column": id_column, "metric": metric_name}
        accepted, extra = task_analyst.reconcile(view, heuristic, guessed, train_cols, test_cols)
    except Exception:  # noqa: BLE001 — разбор по данным важнее
        log.exception("LLM-аналитик задачи упал — остаёмся на разборе по данным")
        return target, id_column, metric_name
    notes += extra
    return accepted.get("target", target), accepted.get("id_column", id_column), accepted.get("metric", metric_name)


def _train_only(train_cols, test_cols) -> list[str]:
    return [c for c in train_cols if c not in test_cols]


def _check_needs_proba(
    metric_name: str | None, task_type: TaskType, text: str, sample: Path | None, target: str, notes: list[str]
) -> str | None:
    """Страховка от неверной метрики: шаблон ответа требует вероятности, а найденная метрика оценивает метки.

    Ловушка (#85): «accuracy не подходит, оцениваем binary cross-entropy» — если поиск метрики ошибётся
    и возьмёт accuracy, агент запишет 0/1 и получит огромный logloss. Отрицания теперь понимает сам поиск
    (evaluation/metrics.py), а здесь — независимая проверка по данным: дробные значения в sample_submission.
    Текст как признак не используем: «для справки считаем log loss по вероятностям» дал бы ложное переключение.
    """
    if task_type != TaskType.BINARY or metric_name is None:
        # multiclass: ответ «колонка на класс» пока не поддержан целиком (#110) — переключение только навредит
        return metric_name
    if METRICS[metric_name].prediction != "label" or not _sample_has_probabilities(sample, target):
        return metric_name
    chosen = _mentioned_proba_metric(text.lower()) or "logloss"
    notes.append(
        f"Метрика {metric_name} оценивает метки, но в sample_submission дробные значения — нужны вероятности, "
        f"взяли {chosen}"
    )
    log.warning("Метрика %s → %s: в sample_submission дробные значения", metric_name, chosen)
    return chosen


def _mentioned_proba_metric(low: str) -> str | None:
    """Самая ранняя по тексту вероятностная метрика (logloss, roc_auc, …) или None."""
    best: tuple[int, str] | None = None
    for name, metric in METRICS.items():
        if metric.prediction != "proba":
            continue
        for alias in metric.aliases:
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
        # формат словаря не гарантирован (вебинар): «Feature, Role, Type» и «feature, role, dtype» — одно и то же
        df.columns = [str(c).strip().lower() for c in df.columns]
        key = "feature" if "feature" in df.columns else df.columns[0]
        return {str(row[key]): row.to_dict() for _, row in df.iterrows()}
    except Exception:  # noqa: BLE001 — словарь не обязателен
        return {}


def _guess_target(train_cols, test_cols, dictionary, test_path: Path | None = None, strict: bool = True) -> str | None:
    """target по словарю, единственной колонке «только в train» или пустой колонке test. strict=False — None вместо ошибки (решит LLM)."""
    for name, row in dictionary.items():
        if str(row.get("role", "")).strip().lower() == "target" and name in train_cols:
            return name
    extra = _train_only(train_cols, test_cols)
    if len(extra) == 1:
        return extra[0]
    if not extra and test_path is not None:
        # target есть и в test, но пустой (заготовка под ответ): ищем колонку train, которая в test вся пустая
        empty = _empty_test_columns(test_path, [c for c in train_cols if c in test_cols])
        if len(empty) == 1:
            return empty[0]
    if not strict:
        return None
    raise ValueError(f"Не удалось определить target: колонки только в train = {extra}")


def _empty_test_columns(test_path: Path, columns: list[str]) -> list[str]:
    try:
        test = pd.read_csv(test_path, usecols=columns)
    except Exception:  # noqa: BLE001 — не прочитали — значит, не угадали
        return []
    return [c for c in columns if test[c].isna().all()]


ID_TOKENS = {"id", "uid", "key", "ид", "номер", "идентификатор"}


def _looks_like_id(name: str) -> bool:
    """«txn_id», «clientID», «Номер заявки», «row-key» — по словам имени, а не по подстроке («paid» — не id)."""
    words = re.sub(r"([a-z])([A-Z])", r"\1 \2", str(name)).lower()
    return bool(ID_TOKENS & set(re.split(r"[^0-9a-zа-яё]+", words)))


def _is_train_id(train_path: Path, column: str) -> bool:
    """Первая колонка шаблона лежит в train, но не в test: это id (значения возьмём из образца) или таргет?

    id — если имя похоже на id или колонка — уникальные целые («index», «row», номер строки): имена надёжно не
    угадать, а таргет в train почти никогда не состоит из одних уникальных целых (#268, ревью artem и dzigoku-ku).
    """
    if _looks_like_id(column):
        return True
    try:
        values = pd.read_csv(train_path, usecols=[column])[column]
    except (ValueError, OSError):
        return False
    return bool(pd.api.types.is_integer_dtype(values) and len(values) > 1 and values.is_unique)


def _guess_id(test_cols, dictionary, test_path: Path | None = None) -> str:
    """id без шаблона ответа: роль в словаре → «id» → колонка с id-именем и уникальными значениями в test.

    Раньше без «id» бралась первая колонка test — на чужой задаче это был признак «tenure» (ночной аудит #238).
    """
    for name, row in dictionary.items():
        if str(row.get("role", "")).lower() == "id" and name in test_cols:
            return name
    if "id" in test_cols:
        return "id"
    candidates = [c for c in test_cols if _looks_like_id(c)]
    if candidates and test_path is not None:
        try:
            test = pd.read_csv(test_path, usecols=candidates)
            unique = [c for c in candidates if test[c].notna().all() and test[c].is_unique]
            if unique:
                return unique[0]
        except Exception:  # noqa: BLE001 — проверка уникальности не обязательна
            pass
    return test_cols[0]  # неуникальный «store_id» не лучше первой колонки (ревью #246)


def _primary_target(train_path: Path, targets: list[str], metric_name: str | None) -> str:
    """Основной таргет из нескольких: первый, тип которого подходит метрике (число — для регрессионной)."""
    if metric_name is None:
        return targets[0]
    ys = pd.read_csv(train_path, usecols=targets)
    wants_value = METRICS[metric_name].prediction == "value"
    for t in targets:
        if (_guess_task_type(ys[t], None, None) == TaskType.REGRESSION) == wants_value:
            return t
    return targets[0]


def _guess_task_type(y: pd.Series, metric_name: str | None, dict_row: dict | None) -> TaskType:
    numeric = pd.api.types.is_numeric_dtype(y)
    if metric_name is not None and METRICS[metric_name].prediction == "value" and numeric:
        return TaskType.REGRESSION  # текстовый таргет регрессией не бывает, даже при метрике вроде MAE
    n_unique = y.nunique(dropna=True)
    # 0.0/1.0 (так читаются целые метки с пропусками или записанные как float) — те же целые метки
    integer_values = pd.api.types.is_integer_dtype(y) or (
        pd.api.types.is_float_dtype(y) and bool((y.dropna() % 1 == 0).all())
    )
    is_categorical = (
        y.dtype == object
        or str((dict_row or {}).get("dtype", "")).lower() in ("category", "categorical", "binary")
        or metric_name is not None  # метрика классификации найдена в тексте
        or (integer_values and n_unique <= MAX_CLASSES_FOR_NUMERIC_TARGET)
    )
    if not is_categorical:
        return TaskType.REGRESSION
    return TaskType.BINARY if n_unique == 2 else TaskType.MULTICLASS


def spec_for_target(spec: TaskSpec, target: str, y: pd.Series) -> TaskSpec:
    """TaskSpec одного таргета задачи с несколькими (#174): тип и классы — по его значениям; метрика — общая,
    если подходит к типу, иначе метрика по умолчанию для типа."""
    task_type = _guess_task_type(y, spec.metric.name, None)
    labels = sorted(map(str, y.dropna().unique())) if task_type != TaskType.REGRESSION else None
    metric = spec.metric
    if (METRICS[metric.name].prediction == "value") != (task_type == TaskType.REGRESSION):
        # метрика задачи — для другого таргета (MAE для класса): берём метрику по типу этого таргета,
        # а для классов — по шаблону ответа: дробные значения → вероятности, иначе метки
        name = DEFAULT_METRIC[task_type]
        if task_type != TaskType.REGRESSION and not _sample_has_probabilities(spec.sample_submission_path, target):
            name = LABEL_METRIC[task_type]
        metric = MetricSpec(name=name, greater_is_better=METRICS[name].greater_is_better)
    return spec.model_copy(
        update={
            "metric": metric,
            "target": target,
            "extra_targets": [],
            "task_type": task_type,
            "class_labels": labels,
            "submission_columns": [c for c in (spec.id_column, target) if c != ROW_ID_COLUMN],
        }
    )
