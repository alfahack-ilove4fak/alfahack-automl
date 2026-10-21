"""Устойчивость: на «кривой» задаче агент всё равно сдаёт валидный submission.csv.

Скрытые задачи могут прийти в неожиданном виде — без sample_submission, с лишними файлами, CRLF,
кириллицей в именах колонок, пропусками в target. Каждый тест — маленькая задача, запуск solution.py
как на платформе (отдельный процесс, рабочая папка = папка задачи) и проверка ответа
по ИЗВЕСТНОЙ правде: колонки, id в порядке test, нет пропусков, метки из списка классов.
Сверяем не через parse_task: если агент ошибся с target или id, проверка его же разбором это пропустит.
"""

from __future__ import annotations

import os
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
BUDGET_SEC = 30


@dataclass
class Task:
    """Задача в памяти: правим её в варианте, потом пишем на диск."""

    train: pd.DataFrame
    test: pd.DataFrame
    id_col: str = "id"
    target: str = "target"
    kind: str = "multiclass"  # regression | binary | multiclass
    proba: bool = False  # метрика по вероятностям → в ответе число 0..1
    text: str = "# Задача\n\nПредсказать `target` для строк test.csv. Метрика — Macro F1.\n"
    sample: bool = True
    extra_files: dict[str, str] = field(default_factory=dict)
    newline: str = "\n"
    subdir: str = ""


def base_task(kind: str = "multiclass", n: int = 300, seed: int = 0) -> Task:
    rng = np.random.default_rng(seed)
    x1, x2 = rng.normal(size=n + 100), rng.normal(size=n + 100)
    city = rng.choice(["msk", "spb", "kzn"], n + 100)
    if kind == "regression":
        y = 3 * x1 - x2 + rng.normal(scale=0.3, size=n + 100)
        text = "# Задача\n\nПредсказать `target`. Метрика — RMSE.\n"
    elif kind == "binary":
        y = np.where(x1 + x2 > 0, "yes", "no")
        text = "# Задача\n\nПредсказать `target`. Метрика — ROC-AUC, в ответе вероятность класса yes.\n"
    else:
        y = np.select([x1 > 0.5, x1 < -0.5], ["a", "c"], "b")
        text = "# Задача\n\nПредсказать `target` для строк test.csv. Метрика — Macro F1.\n"
    data = pd.DataFrame({"id": np.arange(1000, 1000 + n + 100), "x1": x1, "x2": x2, "city": city, "target": y})
    return Task(
        train=data.iloc[:n].reset_index(drop=True),
        test=data.iloc[n:].drop(columns="target").reset_index(drop=True),
        kind=kind,
        proba=kind == "binary",
        text=text,
    )


def write_task(task: Task, root: Path) -> Path:
    data_dir = root / task.subdir
    data_dir.mkdir(parents=True, exist_ok=True)
    nl = task.newline
    task.train.to_csv(data_dir / "train.csv", index=False, lineterminator=nl)
    task.test.to_csv(data_dir / "test.csv", index=False, lineterminator=nl)
    if task.sample:
        filler = 0.5 if task.proba else (0.0 if task.kind == "regression" else task.train[task.target].dropna().iloc[0])
        sample = task.test[[task.id_col]].assign(**{task.target: filler})
        sample.to_csv(data_dir / "sample_submission.csv", index=False, lineterminator=nl)
    for name, content in task.extra_files.items():
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_bytes(content.replace("\n", nl).encode("utf-8"))
    (root / "task.md").write_bytes(task.text.replace("\n", nl).encode("utf-8"))
    return root / "task.md"


def run_agent(task_md: Path, tmp_path: Path) -> subprocess.CompletedProcess:
    env = {
        **os.environ,
        "AGENT_TIME_BUDGET_SEC": str(BUDGET_SEC),
        "AGENT_SAFETY_SEC": "5",
        "LLM_MOCK": "1",
        "AGENT_RUN_DIR": str(tmp_path / "run"),
    }
    return subprocess.run(
        [sys.executable, str(ROOT / "solution.py"), "task.md"],
        cwd=task_md.parent,
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=BUDGET_SEC + 240,
    )


def check_answer(task: Task, task_md: Path, proc: subprocess.CompletedProcess) -> None:
    log = proc.stderr[-3000:]
    submission = task_md.parent / "submission.csv"
    assert submission.exists(), f"нет submission.csv рядом с task.md\n{log}"
    sub = pd.read_csv(submission, dtype={task.id_col: str})
    assert list(sub.columns) == [task.id_col, task.target], f"колонки {list(sub.columns)}\n{log}"
    assert sub[task.id_col].astype(str).tolist() == task.test[task.id_col].astype(str).tolist(), "id не как в test"
    pred = sub[task.target]
    assert pred.notna().all(), f"пропуски в ответе: {int(pred.isna().sum())}"
    if task.kind == "regression" or task.proba:
        values = pd.to_numeric(pred, errors="coerce")
        assert np.isfinite(values).all(), "нечисловые ответы"
        if task.proba:
            assert values.between(0, 1).all(), "вероятности вне [0, 1]"
    else:
        classes = set(task.train[task.target].dropna().astype(str))
        assert set(pred.astype(str)) <= classes, f"метки вне классов: {set(pred.astype(str)) - classes}"
    assert proc.returncode == 0, f"solution.py завершился с кодом {proc.returncode}\n{log}"


# --- Варианты «кривых» задач. Каждый получает базовую задачу и портит её.


def no_sample_submission(t: Task) -> Task:
    """Нет sample_submission — id и target агент определяет сам."""
    t.sample = False
    return t


def extra_files(t: Task) -> Task:
    """Лишние файлы рядом: старые версии данных глубже, README, json, внешний CSV."""
    t.extra_files = {
        "archive/old/train.csv": "a,b\n1,2\n",
        "archive/old/test.csv": "a\n1\n",
        "README.md": "Описание данных\n",
        "meta.json": '{"source": "x"}\n',
        "external_weather.csv": "date,temp\n2020-01-01,3\n",
    }
    return t


def crlf_everywhere(t: Task) -> Task:
    """Windows-переводы строк в task.md и во всех CSV."""
    t.newline = "\r\n"
    return t


def empty_and_constant_columns(t: Task) -> Task:
    """Колонка из одних пропусков, константа и пустая строка-категория."""
    for df in (t.train, t.test):
        df["all_nan"] = np.nan
        df["const"] = 7
        df["blank"] = ""
    return t


def unicode_column_names(t: Task) -> Task:
    """Кириллица, пробелы и спецсимволы в именах колонок (LightGBM не любит JSON-символы)."""
    names = {"x1": "Площадь, м²", "x2": 'доход "до налогов" [руб]', "city": "город:{регион}"}
    t.train, t.test = t.train.rename(columns=names), t.test.rename(columns=names)
    return t


def unicode_target_and_id(t: Task) -> Task:
    """Кириллица в имени target и id."""
    t.train = t.train.rename(columns={"target": "класс", "id": "номер"})
    t.test = t.test.rename(columns={"id": "номер"})
    t.target, t.id_col = "класс", "номер"
    t.text = "# Задача\n\nПредсказать `класс`. Метрика — Macro F1.\n"
    return t


def target_with_missing(t: Task) -> Task:
    """В train у части строк target пустой."""
    t.train.loc[t.train.index[::7], "target"] = np.nan
    return t


def class_with_one_object(t: Task) -> Task:
    """Класс, у которого в train ровно один объект (StratifiedKFold на 5 фолдов)."""
    t.train.loc[0, "target"] = "rare"
    return t


def string_ids(t: Task) -> Task:
    """Строковые id с ведущими нулями — нельзя превращать в числа."""
    for df in (t.train, t.test):
        df["id"] = [f"00{i}" for i in df["id"]]
    return t


def numeric_binary_target(t: Task) -> Task:
    """Бинарный target 0/1 числами и метрика по вероятностям."""
    t.train["target"] = (t.train["target"] == "yes").astype(int)
    return t


def no_metric_in_text(t: Task) -> Task:
    """В task.md нет ни слова о метрике."""
    t.text = "# Задача\n\nНужно предсказать колонку `target`.\n"
    return t


def files_in_subfolder(t: Task) -> Task:
    """Данные во вложенной папке, а в тексте путь указан неверно."""
    t.subdir = "data/v2"
    t.text += "\nФайлы: `input/train.csv`, `input/test.csv`.\n"
    return t


def few_rows(t: Task) -> Task:
    """Очень маленький train — 30 строк."""
    t.train = t.train.head(30)
    return t


def mixed_type_column(t: Task) -> Task:
    """Числовая колонка с мусорными строками («н/д», «—»)."""
    for df in (t.train, t.test):
        df["x2"] = df["x2"].round(3).astype(object)
        df.loc[df.index[::10], "x2"] = "н/д"
        df.loc[df.index[5::10], "x2"] = "—"
    return t


CASES: list[tuple[str, Callable[[Task], Task]]] = [
    ("multiclass", no_sample_submission),
    ("regression", no_sample_submission),
    ("multiclass", extra_files),
    ("binary", crlf_everywhere),
    ("regression", empty_and_constant_columns),
    ("multiclass", unicode_column_names),
    ("multiclass", unicode_target_and_id),
    ("multiclass", target_with_missing),
    ("regression", target_with_missing),
    ("multiclass", class_with_one_object),
    ("multiclass", string_ids),
    ("binary", numeric_binary_target),
    ("binary", no_metric_in_text),
    ("multiclass", files_in_subfolder),
    ("regression", few_rows),
    ("regression", mixed_type_column),
]


# Известные баги: тест помечен xfail(strict=True) — когда владелец починит, тест покраснеет «XPASS»,
# и отметку надо снять. Ключ — id случая, значение — что сломано и где.
KNOWN_INVALID = {
    "multiclass-string_ids": "ml/profiling.py load_data: id «001300» читается как число, в ответе «1300»",
}
KNOWN_NO_MODEL = {
    "multiclass-unicode_column_names": "ml: LightGBM не принимает JSON-символы в именах признаков",
    "multiclass-target_with_missing": "ml: строки с пустым target не выкидываются перед обучением",
    "regression-target_with_missing": "ml: строки с пустым target не выкидываются перед обучением",
    "multiclass-class_with_one_object": "ml: класса с 1 объектом нет в части фолдов — «unseen labels»",
}
IDS = [f"{k}-{f.__name__}" for k, f in CASES]


def _cases(known: dict[str, str]):
    return [
        pytest.param(case, id=case, marks=pytest.mark.xfail(strict=True, reason=known[case]) if case in known else ())
        for case in IDS
    ]


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    """Агент запускается один раз на случай, обе проверки читают результат из кэша."""
    cache: dict[str, tuple[Task, Path, subprocess.CompletedProcess]] = {}

    def get(case: str):
        if case not in cache:
            kind, spoil = CASES[IDS.index(case)]
            root = tmp_path_factory.mktemp(case)
            task = spoil(base_task(kind))
            task_md = write_task(task, root / "task")
            cache[case] = (task, task_md, run_agent(task_md, root))
        return cache[case]

    return get


@pytest.mark.parametrize("case", _cases(KNOWN_INVALID))
def test_valid_answer(case, runs):
    """Главное: на любой задаче есть валидный ответ — иначе 0 баллов."""
    check_answer(*runs(case))


@pytest.mark.parametrize("case", _cases(KNOWN_NO_MODEL))
def test_model_trained(case, runs):
    """Ответ дала модель, а не тривиальная страховка (среднее / частый класс) — иначе балл ≈ 0."""
    _, _, proc = runs(case)
    assert "Сохранён ответ «" in proc.stderr, "все гипотезы упали — сдан тривиальный ответ\n" + proc.stderr[-3000:]
