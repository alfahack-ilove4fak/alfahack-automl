"""Внешние задачи: настоящие табличные данные OpenML в формате платформы. Владелец — research.

    uv run python research/external/make_external_tasks.py                  # все из manifest.json → runs/night/external/
    uv run python research/external/make_external_tasks.py --only bank-marketing credit
    uv run python research/night/run_alien.py --set external --minutes 5    # прогнать агента с независимой проверкой

Зачем: «чужие» задачи research/night синтетические, из одного генератора; здесь данные живые, составленные другими
людьми. Это защита от нулей на реальных таблицах (грязь, шум, перекосы), а не повод подгонять playbook:
внешний набор — для гейта и сравнения «было/стало», параметры под него не подбираем.

Что делаем с каждым датасетом (manifest.json — список из select_tasks.py):
  1. скачиваем через sklearn.datasets.fetch_openml (кэш ~/scikit_learn_data; данные в git не кладём — у наборов разные
     лицензии, см. research/external/README.md);
  2. делим 75/25 случайно (классификация — стратифицированно), seed = data_id;
  3. метрика и вид задачи вариативны (по хэшу имени): регрессия rmse/mae/r2, бинарная roc_auc/logloss/pr_auc,
     многоклассовая f1_macro/accuracy/balanced_accuracy; метрику пишем в task.md словами и в разных стилях
     (английский, русский «как в банке», коротко), чтобы заодно проверять разбор текста;
  4. формат ответа тоже меняется: id в test с разными именами, без sample_submission, и ответ без id вовсе
     (контракт 1.1.2). Ответы и описание проверок — в runs/night/external/_answers (агент их не видит).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "runs" / "night" / "external"
MANIFEST = Path(__file__).with_name("manifest.json")
TEST_SHARE = 0.25

METRICS_BY_TYPE = {
    "regression": ["rmse", "mae", "r2"],
    "binary": ["roc_auc", "logloss", "pr_auc"],
    "multiclass": ["f1_macro", "accuracy", "balanced_accuracy"],
}
GREATER = {"rmse": False, "mae": False, "r2": True, "roc_auc": True, "logloss": False, "pr_auc": True,
           "f1_macro": True, "accuracy": True, "balanced_accuracy": True}  # fmt: skip
PROBA = {"roc_auc", "logloss", "pr_auc"}
WORDS = {  # метрика словами: en / ru
    "rmse": ("RMSE (root mean squared error), lower is better", "корень из среднеквадратичной ошибки (RMSE), чем меньше, тем лучше"),
    "mae": ("MAE (mean absolute error), lower is better", "средняя абсолютная ошибка (MAE), чем меньше, тем лучше"),
    "r2": ("R2 (R-squared), higher is better", "коэффициент детерминации R², чем больше, тем лучше"),
    "roc_auc": ("ROC-AUC, higher is better", "ROC-AUC, чем больше, тем лучше"),
    "logloss": ("logloss (binary cross-entropy), lower is better", "logloss, чем меньше, тем лучше"),
    "pr_auc": ("PR-AUC (average precision), higher is better", "PR-AUC (average precision), чем больше, тем лучше"),
    "f1_macro": ("Macro F1, higher is better", "Macro F1 (усреднение по классам), чем больше, тем лучше"),
    "accuracy": ("accuracy, higher is better", "доля правильных ответов (accuracy), чем больше, тем лучше"),
    "balanced_accuracy": ("balanced accuracy, higher is better", "balanced accuracy, чем больше, тем лучше"),
}  # fmt: skip
ID_NAMES = ["id", "ID", "row_id", "client_id", "application_id", "record_no"]
FORMATS = ["id_sample", "id_sample", "id_no_sample", "no_id_sample", "id_sample_shifted"]


def h(name: str, salt: str = "") -> int:
    return int(hashlib.sha1(f"{name}{salt}".encode()).hexdigest(), 16)


def slug(name: str) -> str:
    return re.sub(r"[^0-9a-z]+", "_", name.lower()).strip("_")[:28]


def load(data_id: int) -> tuple[pd.DataFrame, str] | None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        data = fetch_openml(data_id=data_id, as_frame=True, parser="pandas")
    if len(data.target_names) != 1:
        return None
    target = data.target_names[0]
    frame = data.frame.copy()
    for col in frame.columns:
        if isinstance(frame[col].dtype, pd.CategoricalDtype):
            frame[col] = frame[col].astype(object)
    frame.columns = [str(c) for c in frame.columns]
    return frame, str(target)


def describe(frame: pd.DataFrame, target: str, lang: str) -> str:
    lines = []
    for col in frame.columns:
        if col == target:
            continue
        kind = ("number" if lang == "en" else "число") if pd.api.types.is_numeric_dtype(frame[col]) else (
            "category/text" if lang == "en" else "категория/текст")  # fmt: skip
        lines.append(f"- `{col}` — {kind}")
    return "\n".join(lines[:60]) + ("\n- ..." if len(lines) > 60 else "")


def task_text(style: int, name: str, kind: str, metric: str, target: str, positive: str | None, fmt: str, id_col: str | None,
              frame: pd.DataFrame) -> str:  # fmt: skip
    en, ru = WORDS[metric]
    answer = {
        "regression": f"the predicted value of `{target}`",
        "binary": f"the probability that `{target}` = `{positive}`" if metric in PROBA else f"the predicted class of `{target}`",
        "multiclass": f"the predicted class of `{target}`",
    }[kind]  # fmt: skip
    answer_ru = {
        "regression": f"предсказанное значение `{target}`",
        "binary": f"вероятность того, что `{target}` = `{positive}`" if metric in PROBA else f"предсказанный класс `{target}`",
        "multiclass": f"предсказанный класс `{target}`",
    }[kind]  # fmt: skip
    files = "train.csv, test.csv" + (
        ", sample_submission.csv" if "no_sample" not in fmt and fmt != "id_no_sample" else ""
    )
    id_en = (
        f"Each row has an identifier `{id_col}`. "
        if id_col
        else "Rows have no identifier: answer in the same order as test.csv. "
    )
    id_ru = (
        f"У каждой строки есть идентификатор `{id_col}`. "
        if id_col
        else "Идентификатора нет: строки ответа идут в том же порядке, что и в test.csv. "
    )
    cols = [id_col, target] if id_col else [target]
    if style == 0:
        return (
            f"# {name}\n\nTabular prediction task. Files: {files}.\n\nTarget: `{target}` in train.csv. {id_en}"
            f"For every row of test.csv predict {answer}.\n\nSubmission: CSV with columns {', '.join(f'`{c}`' for c in cols)}.\n"
            f"Metric: {en}.\n\n## Features\n{describe(frame, target, 'en')}\n"
        )
    if style == 1:
        return (
            f"# Задача: {name}\n\n## Описание\nБанк-партнёр передал выгрузку данных. Нужно построить модель по train.csv "
            f"и сделать прогноз для test.csv.\n\nЦелевая переменная — `{target}`. {id_ru}Прогноз: {answer_ru}.\n\n"
            f"## Файлы\n{files}\n\n## Формат ответа\nsubmission.csv, колонки: {', '.join(f'`{c}`' for c in cols)}.\n\n"
            f"## Метрика качества\n{ru}.\n\n## Признаки\n{describe(frame, target, 'ru')}\n"
        )
    return (
        f"{name}. Предскажи `{target}` для test.csv ({answer_ru}). {id_ru}Ответ — submission.csv с колонками "
        f"{', '.join(cols)}. Оценка: {ru}.\n"
    )


def make(entry: dict) -> str | None:
    loaded = load(entry["data_id"])
    if loaded is None:
        return None
    frame, target = loaded
    frame = frame.dropna(subset=[target]).reset_index(drop=True)
    kind = entry["task_type"]
    name = f"{slug(entry['name'])}"
    if kind == "regression":
        y = pd.to_numeric(frame[target], errors="coerce")
        frame = frame.loc[y.notna()].reset_index(drop=True)
        frame[target] = y[y.notna()].to_numpy()
        if frame[target].nunique() < 5:
            return None
    else:
        frame[target] = frame[target].astype(str)
        counts = frame[target].value_counts()
        frame = frame[frame[target].isin(counts[counts >= 10].index)].reset_index(drop=True)
        if frame[target].nunique() < 2:
            return None
    metrics = METRICS_BY_TYPE[kind]
    metric = metrics[h(entry["name"], "metric") % len(metrics)]
    positive = None
    if kind == "binary":
        classes = sorted(frame[target].unique())
        positive = classes[-1]
        shares = frame[target].value_counts(normalize=True)
        if metric == "pr_auc" and shares.min() < 0.3:  # PR-AUC осмысленна на редком классе: положительный — меньший
            positive = shares.idxmin()
    train, test = train_test_split(
        frame, test_size=TEST_SHARE, random_state=entry["data_id"],
        stratify=frame[target] if kind != "regression" else None,
    )  # fmt: skip
    train, test = train.reset_index(drop=True), test.reset_index(drop=True)
    fmt = FORMATS[h(entry["name"], "fmt") % len(FORMATS)]
    style = h(entry["name"], "style") % 3
    id_col = None if fmt == "no_id_sample" else ID_NAMES[h(entry["name"], "id") % len(ID_NAMES)]
    clash = id_col in frame.columns
    id_col = None if clash else id_col
    if id_col is None and fmt != "no_id_sample":
        fmt = "no_id_sample"
    shift = 100000 + h(entry["name"], "shift") % 900000 if fmt == "id_sample_shifted" else 0
    answers = test[[target]].copy()
    test_x = test.drop(columns=[target])
    if id_col:
        ids = np.arange(len(test)) + shift + (7 if shift else 1)
        train.insert(0, id_col, np.arange(len(train)) + shift + len(test) + 100)
        test_x.insert(0, id_col, ids)
        answers.insert(0, id_col, ids)
    filler = {
        "regression": 0.0,
        "binary": 0.5 if metric in PROBA else positive,
        "multiclass": sorted(train[target].unique())[0],
    }[kind]
    sample = answers.assign(**{target: filler})
    task_dir = OUT / name
    shutil.rmtree(task_dir, ignore_errors=True)
    task_dir.mkdir(parents=True)
    train.to_csv(task_dir / "train.csv", index=False)
    test_x.to_csv(task_dir / "test.csv", index=False)
    if fmt != "id_no_sample":
        sample.to_csv(task_dir / "sample_submission.csv", index=False)
    (task_dir / "task.md").write_text(
        task_text(style, entry["name"], kind, metric, target, positive, fmt, id_col, frame), encoding="utf-8"
    )
    ans = OUT / "_answers"
    ans.mkdir(parents=True, exist_ok=True)
    answers.to_csv(ans / f"{name}.csv", index=False)
    meta = {
        "metric": metric,
        "greater_is_better": GREATER[metric],
        "task_type": "regression" if kind == "regression" else kind,
        "id": id_col,
        "targets": [target],
        "submission_columns": ([id_col] if id_col else []) + [target],
        "positive": positive,
        "checks": f"OpenML {entry['data_id']} ({', '.join(entry['suites'])}); формат {fmt}, стиль текста {style}",
        "source": f"https://www.openml.org/d/{entry['data_id']}",
    }
    (ans / f"{name}.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"✅ {name}: {kind}, {metric}, {train.shape[0]}+{test.shape[0]} строк, формат {fmt}, стиль {style}", flush=True
    )
    return name


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", nargs="*", help="части имён датасетов")
    parser.add_argument(
        "--core", action="store_true", help="только быстрый срез (core в manifest.json, ≈ 12 мелких задач)"
    )
    args = parser.parse_args()
    entries = json.loads(MANIFEST.read_text(encoding="utf-8"))["tasks"]
    if args.core:
        entries = [e for e in entries if e.get("core")]
    for entry in entries:
        if args.only and not any(o.lower() in entry["name"].lower() for o in args.only):
            continue
        try:
            if make(entry) is None:
                print(f"⏭ {entry['name']}: пропущен (несколько таргетов или мало разных значений)")
        except Exception as exc:  # noqa: BLE001 — один датасет не должен ронять остальные
            print(f"💥 {entry['name']}: {type(exc).__name__}: {exc}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
