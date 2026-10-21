"""Тренировочные «скрытые» задачи (R-L1-3): tasks/practice/<имя>/ в формате платформы + скрытые ответы.

Зачем: на private будут две задачи с неизвестными данными и метрикой. Эти задачи проверяют, что агент
не подогнан под две публичные: другие метрики, маленькие и средние данные, числовой таргет-класс,
даты строкой, сдвиг во времени, «кривой» task.md.

    uv run python research/make_practice_tasks.py            # пересоздать все задачи
    uv run python research/make_practice_tasks.py --only cancer_binary_auc

Ответы лежат ОТДЕЛЬНО от задач — tasks/practice/_answers/<имя>.csv (id, target), а метрики —
в tasks/practice/_answers/manifest.json: run_local.py копирует агенту только папку задачи, как платформа.
Всё детерминировано (SEED), без интернета: sklearn.datasets + переделка публичной задачи + синтетика.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent
PRACTICE = ROOT / "tasks" / "practice"
ANSWERS = PRACTICE / "_answers"
SEED = 2026

# имя задачи → (функция-генератор, метрика из evaluation.metrics, больше — лучше)
TASKS: dict[str, tuple[Callable[[Path], dict], str, bool]] = {}


def task(name: str, metric: str, greater_is_better: bool):
    def register(fn):
        TASKS[name] = (fn, metric, greater_is_better)
        return fn

    return register


def _split(df: pd.DataFrame, target: str, test_size: float, stratify: bool, id_col: str = "id"):
    """Перемешанные id, train/test, ответы и шаблон сабмита (заглушки — не правильные ответы)."""
    rng = np.random.default_rng(SEED)
    df = df.copy()
    df.insert(0, id_col, rng.permutation(len(df)) + 1000)
    train, test = train_test_split(
        df, test_size=test_size, random_state=SEED, stratify=df[target] if stratify else None
    )
    test = test.sort_values(id_col)
    answers = test[[id_col, target]].reset_index(drop=True)
    stub = train[target].mode()[0] if stratify else 0
    sample = answers.assign(**{target: stub})
    return train.reset_index(drop=True), test.drop(columns=[target]).reset_index(drop=True), answers, sample


def _write(task_dir: Path, files: dict[str, pd.DataFrame], task_md: str) -> None:
    for rel, frame in files.items():
        path = task_dir / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path, index=False)
    (task_dir / "task.md").write_text(task_md.strip() + "\n", encoding="utf-8")


def _dictionary(rows: list[tuple[str, str, str, str]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["feature", "role", "dtype", "description_ru"])


def _standard_md(
    title: str, about: str, n_train: int, n_test: int, fields: str, target_text: str, metric_text: str, header: str
) -> str:
    return f"""
# {title}

## Задача

{about}

## Данные

- `train.csv` — {n_train} строк с признаками и таргетом;
- `test.csv` — {n_test} строк без таргета;
- `data_dictionary.csv` — описание полей;
- `sample_submission.csv` — шаблон ответа; значения в нём не являются правильными ответами.

CSV в UTF-8, разделитель — запятая, десятичный разделитель — точка. Пустая ячейка — пропуск.

## Описание полей

{fields}

## Что предсказать

{target_text}

Метрика — **{metric_text}**.

## Формат результата

Файл `submission.csv` в каталоге этого файла, ровно два столбца:

```csv
{header}
```

Для каждого `id` из `test.csv` — одно предсказание, порядок строк как в `test.csv`.
"""


@task("cancer_binary_auc", "roc_auc", True)
def cancer(task_dir: Path) -> dict:
    """Маленькая бинарная задача (569 строк, 30 числовых признаков), ответ — вероятность."""
    data = datasets.load_breast_cancer(as_frame=True)
    df = data.data.copy()
    df.columns = [c.replace(" ", "_") for c in df.columns]
    df["is_malignant"] = 1 - data.target  # в sklearn 0 — злокачественная
    train, test, answers, sample = _split(df, "is_malignant", 0.3, stratify=True)
    sample["is_malignant"] = 0.5
    dictionary = _dictionary(
        [("id", "id", "int", "Идентификатор строки")]
        + [(c, "feature", "numeric", "Характеристика ядер клеток по снимку") for c in df.columns[:-1]]
        + [("is_malignant", "target", "binary", "1 — злокачественная опухоль, 0 — доброкачественная")]
    )
    md = _standard_md(
        "Диагностика опухоли по снимку",
        "По 30 характеристикам ядер клеток со снимка биопсии оцените, злокачественная ли опухоль.",
        len(train),
        len(test),
        "Все признаки числовые: среднее (`mean_*`), стандартная ошибка (`*_error`) и худшее значение (`worst_*`) "
        "для радиуса, текстуры, периметра, площади, гладкости и других характеристик ядер.",
        "Для каждой строки `test.csv` — **вероятность** того, что опухоль злокачественная (`is_malignant` = 1), "
        "число от 0 до 1.",
        "ROC-AUC (больше — лучше)",
        "id,is_malignant",
    )
    _write(
        task_dir,
        {"train.csv": train, "test.csv": test, "data_dictionary.csv": dictionary, "sample_submission.csv": sample},
        md,
    )
    return {"answers": answers}


@task("diabetes_regression_mae", "mae", False)
def diabetes(task_dir: Path) -> dict:
    """Очень маленькая регрессия (442 строки): проверка, что агент не переобучается и не тратит 3 часа зря."""
    data = datasets.load_diabetes(as_frame=True, scaled=False)
    df = data.data.rename(columns={"s1": "tc", "s2": "ldl", "s3": "hdl", "s4": "tch", "s5": "ltg", "s6": "glu"})
    df["sex"] = df["sex"].map({1.0: "F", 2.0: "M"})
    df["progression"] = data.target
    train, test, answers, sample = _split(df, "progression", 0.3, stratify=False)
    dictionary = _dictionary(
        [
            ("id", "id", "int", "Идентификатор пациента"),
            ("age", "feature", "numeric", "Возраст, лет"),
            ("sex", "feature", "category", "Пол"),
            ("bmi", "feature", "numeric", "Индекс массы тела"),
            ("bp", "feature", "numeric", "Среднее артериальное давление"),
        ]
        + [(c, "feature", "numeric", "Показатель анализа крови") for c in ["tc", "ldl", "hdl", "tch", "ltg", "glu"]]
        + [("progression", "target", "numeric", "Прогрессирование болезни через год (условная шкала)")]
    )
    md = _standard_md(
        "Прогрессирование диабета",
        "По возрасту, полу, индексу массы тела, давлению и шести анализам крови оцените, насколько "
        "прогрессирует болезнь через год после обследования.",
        len(train),
        len(test),
        "`age`, `bmi`, `bp` и шесть анализов крови — числа; `sex` — категория (`F`/`M`).",
        "Для каждой строки `test.csv` — число `progression`.",
        "MAE, средняя абсолютная ошибка (меньше — лучше)",
        "id,progression",
    )
    _write(
        task_dir,
        {"train.csv": train, "test.csv": test, "data_dictionary.csv": dictionary, "sample_submission.csv": sample},
        md,
    )
    return {"answers": answers}


@task("digits_multiclass_accuracy", "accuracy", True)
def digits(task_dir: Path) -> dict:
    """10 классов, записанных ЧИСЛАМИ 0–9: агент не должен принять задачу за регрессию."""
    data = datasets.load_digits(as_frame=True)
    df = data.data.astype(int)
    df["digit"] = data.target
    train, test, answers, sample = _split(df, "digit", 0.3, stratify=True)
    dictionary = _dictionary(
        [("id", "id", "int", "Идентификатор изображения")]
        + [(c, "feature", "numeric", "Яркость пикселя 0–16") for c in data.data.columns]
        + [("digit", "target", "category", "Цифра на изображении, класс 0–9")]
    )
    md = _standard_md(
        "Распознавание рукописных цифр",
        "Каждая строка — изображение 8×8 рукописной цифры. Определите, какая цифра изображена. "
        "Это **классификация**: таргет — номер класса, а не число для регрессии.",
        len(train),
        len(test),
        "64 признака `pixel_<строка>_<столбец>` — яркость пикселя от 0 до 16.",
        "Для каждой строки `test.csv` — класс `digit` (целое от 0 до 9).",
        "accuracy, доля правильных ответов (больше — лучше)",
        "id,digit",
    )
    _write(
        task_dir,
        {"train.csv": train, "test.csv": test, "data_dictionary.csv": dictionary, "sample_submission.csv": sample},
        md,
    )
    return {"answers": answers}


@task("sales_regression_dates_rmsle", "rmsle", False)
def sales(task_dir: Path) -> dict:
    """Синтетические продажи: дата строкой, магазины (высокая кардинальность), пропуски, test — позже train."""
    rng = np.random.default_rng(SEED)
    n_stores, n_cats = 150, 30
    stores = pd.DataFrame(
        {
            "store_id": [f"S{i:03d}" for i in range(n_stores)],
            "city": rng.choice([f"city_{i:02d}" for i in range(12)], n_stores),
            "store_type": rng.choice(["A", "B", "C"], n_stores, p=[0.5, 0.3, 0.2]),
            "store_level": rng.normal(0, 0.6, n_stores),
        }
    )
    cats = pd.DataFrame(
        {
            "product_category": [f"cat_{i:02d}" for i in range(n_cats)],
            "cat_level": rng.normal(1.0, 0.7, n_cats),
            "ref_price": rng.lognormal(5, 0.6, n_cats),
        }
    )
    holidays = pd.to_datetime(
        [
            "2023-01-01",
            "2023-03-08",
            "2023-05-09",
            "2023-12-31",
            "2024-01-01",
            "2024-03-08",
            "2024-05-09",
            "2024-12-31",
            "2025-01-01",
            "2025-03-08",
        ]
    )

    def sample_rows(n: int, start: str, end: str) -> pd.DataFrame:
        dates = pd.to_datetime(
            rng.integers(pd.Timestamp(start).value // 10**9, pd.Timestamp(end).value // 10**9, n), unit="s"
        ).normalize()
        df = pd.DataFrame({"date": dates})
        df = pd.concat(
            [
                df,
                stores.sample(n, replace=True, random_state=int(rng.integers(10**9))).reset_index(drop=True),
                cats.sample(n, replace=True, random_state=int(rng.integers(10**9))).reset_index(drop=True),
            ],
            axis=1,
        )
        df["promo"] = (rng.random(n) < 0.15).astype(int)
        df["price"] = (df["ref_price"] * rng.lognormal(0, 0.15, n) * np.where(df["promo"] == 1, 0.8, 1.0)).round(2)
        doy = df["date"].dt.dayofyear
        df["temperature"] = (8 - 14 * np.cos(2 * np.pi * doy / 365) + rng.normal(0, 3, n)).round(1)
        df["is_holiday"] = df["date"].isin(holidays).astype(int)
        years = (df["date"] - pd.Timestamp("2023-01-01")).dt.days / 365
        log_rate = (
            df["store_level"]
            + df["cat_level"]
            + 0.12 * years
            + 0.5 * df["promo"]
            - 0.9 * np.log(df["price"] / df["ref_price"])
            + 0.35 * (df["date"].dt.dayofweek >= 5)
            + 0.25 * np.sin(2 * np.pi * doy / 365)
            + 0.6 * df["is_holiday"]
            + 0.4 * (df["store_type"] == "A")
        )
        df["units_sold"] = rng.poisson(np.exp(log_rate))
        df.loc[rng.random(n) < 0.08, "price"] = np.nan
        df.loc[rng.random(n) < 0.05, "temperature"] = np.nan
        df["date"] = df["date"].dt.strftime("%Y-%m-%d")
        return df.drop(columns=["store_level", "cat_level", "ref_price"])

    train = sample_rows(30000, "2023-01-01", "2025-01-01")
    test_full = sample_rows(6000, "2025-01-01", "2025-04-01")
    ids = rng.permutation(len(train) + len(test_full)) + 1
    train.insert(0, "id", ids[: len(train)])
    test_full.insert(0, "id", ids[len(train) :])
    test_full = test_full.sort_values("id").reset_index(drop=True)
    answers = test_full[["id", "units_sold"]]
    test = test_full.drop(columns=["units_sold"])
    sample = answers.assign(units_sold=0)
    dictionary = _dictionary(
        [
            ("id", "id", "int", "Идентификатор строки"),
            ("date", "feature", "date", "Дата продаж, ГГГГ-ММ-ДД"),
            ("store_id", "feature", "category", "Магазин (150 значений)"),
            ("city", "feature", "category", "Город магазина"),
            ("store_type", "feature", "category", "Формат магазина"),
            ("product_category", "feature", "category", "Категория товара"),
            ("promo", "feature", "binary", "1 — товар участвует в акции"),
            ("price", "feature", "numeric", "Цена, ₽; бывают пропуски"),
            ("temperature", "feature", "numeric", "Температура воздуха в день продаж, °C; бывают пропуски"),
            ("is_holiday", "feature", "binary", "1 — праздничный день"),
            ("units_sold", "target", "numeric", "Продано штук за день"),
        ]
    )
    md = _standard_md(
        "Дневные продажи в магазинах",
        "Оцените, сколько штук товара категории продаст магазин за день. В train — продажи за 2023–2024 годы, "
        "в test — за первый квартал 2025 года.",
        len(train),
        len(test),
        "`date` — дата строкой; `store_id`, `city`, `store_type`, `product_category` — категории; "
        "`promo`, `is_holiday` — флаги; `price`, `temperature` — числа с пропусками.",
        "Для каждой строки `test.csv` — неотрицательное число `units_sold`.",
        "RMSLE, root mean squared log error (меньше — лучше)",
        "id,units_sold",
    )
    _write(
        task_dir,
        {"train.csv": train, "test.csv": test, "data_dictionary.csv": dictionary, "sample_submission.csv": sample},
        md,
    )
    return {"answers": answers}


@task("network_binary_logloss_crooked", "logloss", False)
def network(task_dir: Path) -> dict:
    """«Кривой» task.md: по-английски, неверные пути, нет sample_submission и словаря, ловушка с accuracy в тексте,
    файлы во вложенной папке, id называется не `id`. Данные — переделка публичной multiclass в бинарную."""
    src = ROOT / "tasks" / "public" / "multiclass"
    df = pd.read_csv(src / "train.csv")
    df = df.sample(n=28000, random_state=SEED).drop(columns=["id"])
    df["is_attack"] = (df.pop("attack_cat") != "Normal").astype(int)
    df = df.rename(columns={"proto": "Protocol", "service": "Service", "state": "State"})
    train, test, answers, _ = _split(df, "is_attack", 0.3, stratify=True, id_col="flow_id")
    md = f"""
Network flows — attack or not?
==============================

Predict whether each network flow is malicious. Files are in `/data/input/flows_train.csv` and
`/data/input/flows_test.csv` (paths may differ on your machine).

Train has {len(train)} rows, test has {len(test)}. The target column is the one that is missing in the test file.
Each row is identified by `flow_id`.

**Evaluation.** Accuracy is misleading here, so we do NOT use it. Submissions are scored with
binary cross-entropy on the predicted probability that the flow is an attack (target = 1). Lower is better.

Submission: a CSV with columns `flow_id` and `is_attack`, where `is_attack` is a probability in [0, 1].
"""
    _write(task_dir, {"data/train_flows.csv": train, "data/test_flows.csv": test}, md)
    (task_dir / "data" / "README.txt").write_text("Raw export, columns as is.\n", encoding="utf-8")
    return {"answers": answers}


def main() -> int:
    parser = argparse.ArgumentParser(description="Пересоздать тренировочные задачи tasks/practice/")
    parser.add_argument("--only", nargs="*", choices=sorted(TASKS), help="какие задачи (по умолчанию все)")
    args = parser.parse_args()
    ANSWERS.mkdir(parents=True, exist_ok=True)
    manifest_path = ANSWERS / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    for name in args.only or sorted(TASKS):
        fn, metric, gib = TASKS[name]
        task_dir = PRACTICE / name
        shutil.rmtree(task_dir, ignore_errors=True)
        answers = fn(task_dir)["answers"]
        answers.to_csv(ANSWERS / f"{name}.csv", index=False)
        manifest[name] = {
            "metric": metric,
            "greater_is_better": gib,
            "id_column": answers.columns[0],
            "target": answers.columns[1],
        }
        print(f"✅ {name}: {metric}, test {len(answers)} строк → {task_dir.relative_to(ROOT)}")
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    for _stream in (sys.stdout, sys.stderr):
        _stream.reconfigure(encoding="utf-8", errors="replace")
    sys.exit(main())
