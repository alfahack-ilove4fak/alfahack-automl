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


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-z))


def _intercept_for_rate(z: np.ndarray, rate: float) -> float:
    """Сдвиг логита, при котором средняя вероятность события равна rate (бисекция — детерминированно)."""
    lo, hi = -30.0, 30.0
    for _ in range(100):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if _sigmoid(z + mid).mean() < rate else (lo, mid)
    return (lo + hi) / 2


@task("payments_binary_f1_rare", "f1", True)
def payments(task_dir: Path) -> dict:
    """Платежи: редкое мошенничество (~2.5%), F1 по положительному классу — нужен порог, а не argmax 0.5.

    Дата строкой, test — следующий месяц после train (схема time + редкий класс, ревью #150). Продавцов 3 800,
    в test четверть операций — у продавцов, которых нет в train (новые категории, #145).
    """
    rng = np.random.default_rng(SEED + 1)

    def merchants(n: int, prefix: str) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "merchant_id": [f"{prefix}{i:05d}" for i in range(n)],
                "merchant_category": rng.choice([f"mcc_{i:02d}" for i in range(20)], n),
                "merchant_risk": rng.normal(0, 1, n),
            }
        )

    old, new = merchants(3000, "M"), merchants(800, "N")
    countries = np.array([f"C{i:02d}" for i in range(40)])
    country_p = np.r_[0.7, np.full(39, 0.3 / 39)]
    country_risk = dict(zip(countries, np.r_[0.0, rng.normal(0.8, 0.6, 39)], strict=True))

    def sample_rows(n: int, start: str, end: str, new_share: float) -> tuple[pd.DataFrame, np.ndarray]:
        stamps = pd.to_datetime(
            rng.integers(pd.Timestamp(start).value // 10**9, pd.Timestamp(end).value // 10**9, n), unit="s"
        )
        from_new = rng.random(n) < new_share
        merch = old.iloc[rng.integers(0, len(old), n)].reset_index(drop=True)
        merch.loc[from_new] = new.iloc[rng.integers(0, len(new), int(from_new.sum()))].to_numpy()
        df = pd.DataFrame({"date": stamps.strftime("%Y-%m-%d"), "hour": stamps.hour})
        df["merchant_id"] = merch["merchant_id"]
        df["merchant_category"] = merch["merchant_category"]
        df["amount"] = rng.lognormal(7, 1.1, n).round(2)
        df["device"] = rng.choice(["mobile", "web", "pos"], n, p=[0.5, 0.3, 0.2])
        df["country"] = rng.choice(countries, n, p=country_p)
        df["is_foreign"] = (df["country"] != "C00").astype(int)
        df["customer_age"] = np.clip(rng.normal(40, 13, n), 18, 90).astype(int)
        df["tx_last_24h"] = rng.poisson(2, n)
        z = 2.0 * (  # множитель — сила сигнала: иначе F1 даже с порогом ~0.13 и тонет в шуме
            1.1 * merch["merchant_risk"].to_numpy(dtype=float)
            + df["country"].map(country_risk).to_numpy()
            + 0.5 * np.log(df["amount"] / 1100)
            + 0.9 * (df["hour"] < 6)
            + 0.45 * np.sqrt(df["tx_last_24h"])
            + 0.5 * (df["device"] == "web")
            - 0.015 * (df["customer_age"] - 40)
        ).to_numpy(dtype=float)
        df.loc[rng.random(n) < 0.03, "amount"] = np.nan
        return df, z

    train, z_train = sample_rows(60000, "2024-01-01", "2024-10-01", new_share=0.0)
    test_full, z_test = sample_rows(15000, "2024-10-01", "2024-11-01", new_share=0.25)
    shift = _intercept_for_rate(z_train, 0.025)
    train["is_fraud"] = (rng.random(len(train)) < _sigmoid(z_train + shift)).astype(int)
    test_full["is_fraud"] = (rng.random(len(test_full)) < _sigmoid(z_test + shift)).astype(int)
    ids = rng.permutation(len(train) + len(test_full)) + 1
    train.insert(0, "id", ids[: len(train)])
    test_full.insert(0, "id", ids[len(train) :])
    test_full = test_full.sort_values("id").reset_index(drop=True)
    answers = test_full[["id", "is_fraud"]]
    test = test_full.drop(columns=["is_fraud"])
    dictionary = _dictionary(
        [
            ("id", "id", "int", "Идентификатор операции"),
            ("date", "feature", "date", "Дата операции, ГГГГ-ММ-ДД"),
            ("hour", "feature", "int", "Час операции, 0–23"),
            ("merchant_id", "feature", "category", "Продавец (тысячи значений; в test бывают новые)"),
            ("merchant_category", "feature", "category", "Категория продавца (MCC)"),
            ("amount", "feature", "numeric", "Сумма операции, ₽; бывают пропуски"),
            ("device", "feature", "category", "Канал: mobile, web, pos"),
            ("country", "feature", "category", "Страна операции"),
            ("is_foreign", "feature", "binary", "1 — операция не в стране карты"),
            ("customer_age", "feature", "int", "Возраст клиента"),
            ("tx_last_24h", "feature", "int", "Операций по карте за прошлые сутки"),
            ("is_fraud", "target", "binary", "1 — мошенническая операция"),
        ]
    )
    md = _standard_md(
        "Мошеннические платежи",
        "Найдите мошеннические операции по картам. Мошенничество редкое — около 2–3% операций. В train — "
        "операции за январь–сентябрь 2024 года, в test — за октябрь 2024 года.",
        len(train),
        len(test),
        "`date` — дата строкой; `merchant_id`, `merchant_category`, `device`, `country` — категории; "
        "`hour`, `amount`, `customer_age`, `tx_last_24h` — числа; `is_foreign` — флаг.",
        "Для каждой строки `test.csv` — метка `is_fraud`: 1 (мошенничество) или 0.",
        "F1-score по классу 1 (больше — лучше)",
        "id,is_fraud",
    )
    _write(
        task_dir,
        {
            "train.csv": train,
            "test.csv": test,
            "data_dictionary.csv": dictionary,
            "sample_submission.csv": answers.assign(is_fraud=0),
        },
        md,
    )
    return {"answers": answers}


@task("batches_ordinal_qwk", "qwk", True)
def batches(task_dir: Path) -> dict:
    """Оценка партии 1–5: порядковые классы числами, метрика QWK; в task.md список полей с ошибкой.

    Как на public по вебинару: в описании одного поля нет, другое названо неверно — верный набор только в
    data_dictionary.csv и в самих CSV.
    """
    rng = np.random.default_rng(SEED + 2)
    n = 10000
    suppliers = np.array([f"supplier_{i:02d}" for i in range(12)])
    supplier_effect = dict(zip(suppliers, rng.normal(0, 0.5, 12), strict=True))
    df = pd.DataFrame(
        {
            "temperature_c": rng.normal(72, 6, n).round(1),
            "ph": rng.normal(6.5, 0.4, n).round(2),
            "moisture": rng.beta(4, 12, n).round(3),
            "density": rng.normal(1.04, 0.02, n).round(4),
            "aging_days": rng.integers(1, 120, n),
            "viscosity": rng.lognormal(3, 0.3, n).round(1),
            "additive_mg": rng.gamma(2, 15, n).round(1),
            "supplier": rng.choice(suppliers, n),
            "line": rng.choice(["L1", "L2", "L3", "L4"], n, p=[0.4, 0.3, 0.2, 0.1]),
        }
    )
    latent = (
        -0.08 * (df["temperature_c"] - 72) ** 2 / 6
        - 1.2 * (df["ph"] - 6.6).abs()
        - 6 * (df["moisture"] - 0.25)
        + 0.9 * np.log(df["aging_days"])
        - 0.02 * df["additive_mg"]
        + df["supplier"].map(supplier_effect)
        + 0.4 * (df["line"] == "L1")
        + rng.normal(0, 0.8, n)
    )
    cuts = np.quantile(latent, [0.05, 0.25, 0.70, 0.95])
    df["grade"] = np.searchsorted(cuts, latent) + 1
    df.loc[rng.random(n) < 0.04, "viscosity"] = np.nan
    df.loc[rng.random(n) < 0.04, "ph"] = np.nan
    train, test, answers, sample = _split(df, "grade", test_size=0.2, stratify=True)
    dictionary = _dictionary(
        [
            ("id", "id", "int", "Идентификатор партии"),
            ("temperature_c", "feature", "numeric", "Температура процесса, °C"),
            ("ph", "feature", "numeric", "Кислотность; бывают пропуски"),
            ("moisture", "feature", "numeric", "Влажность, доля 0–1"),
            ("density", "feature", "numeric", "Плотность, г/см³"),
            ("aging_days", "feature", "int", "Выдержка, дней"),
            ("viscosity", "feature", "numeric", "Вязкость; бывают пропуски"),
            ("additive_mg", "feature", "numeric", "Добавка, мг на кг"),
            ("supplier", "feature", "category", "Поставщик сырья"),
            ("line", "feature", "category", "Производственная линия"),
            ("grade", "target", "int", "Оценка качества партии: 1 (худшая) … 5 (лучшая)"),
        ]
    )
    md = _standard_md(
        "Оценка качества партий",
        "Эксперты оценивают каждую партию продукции по шкале от 1 до 5. Предскажите оценку по параметрам "
        "производства. Ошибка на 2 балла хуже, чем на 1.",
        len(train),
        len(test),
        # Ошибка намеренная: нет additive_mg, а viscosity названа «vsc» — верный набор в словаре и CSV
        "`temperature_c`, `ph`, `moisture`, `density`, `aging_days`, `vsc` — числа; " "`supplier`, `line` — категории.",
        "Для каждой строки `test.csv` — целая оценка `grade` от 1 до 5.",
        "Quadratic Weighted Kappa, QWK (больше — лучше)",
        "id,grade",
    )
    _write(
        task_dir,
        {"train.csv": train, "test.csv": test, "data_dictionary.csv": dictionary, "sample_submission.csv": sample},
        md,
    )
    return {"answers": answers}


@task("claims_regression_mape_rowno", "mape", False)
def claims(task_dir: Path) -> dict:
    """Страховые выплаты: MAPE (завышать дорого), скошенная сумма > 0 и номер записи, похожий на время.

    record_no — порядок ввода записей; test — последние 20% записей, но данные от порядка не зависят. Схема time
    по record_no здесь лишняя (ревью #150: номер строки не время) — агент должен остаться на kfold.
    """
    rng = np.random.default_rng(SEED + 3)
    n = 25000
    regions = np.array([f"R{i:02d}" for i in range(15)])
    region_effect = dict(zip(regions, rng.normal(0, 0.25, 15), strict=True))
    policy_effect = {"basic": -0.3, "standard": 0.0, "plus": 0.2, "premium": 0.45, "fleet": 0.1}
    df = pd.DataFrame(
        {
            "record_no": np.arange(1, n + 1),
            "region": rng.choice(regions, n),
            "policy_type": rng.choice(list(policy_effect), n, p=[0.3, 0.35, 0.2, 0.1, 0.05]),
            "channel": rng.choice(["office", "online", "agent"], n, p=[0.3, 0.5, 0.2]),
            "vehicle_age": rng.integers(0, 25, n),
            "driver_age": np.clip(rng.normal(42, 12, n), 18, 85).astype(int),
            "sum_insured": rng.lognormal(13.5, 0.6, n).round(-3),
            "prior_claims": rng.poisson(0.4, n),
            "deductible": rng.choice([0, 5000, 15000, 30000], n, p=[0.4, 0.3, 0.2, 0.1]),
        }
    )
    log_amount = (
        2.5
        + 0.55 * np.log(df["sum_insured"])
        + df["region"].map(region_effect)
        + df["policy_type"].map(policy_effect)
        - 0.02 * df["vehicle_age"]
        + 0.15 * df["prior_claims"]
        - 0.000015 * df["deductible"]
        + 0.004 * (df["driver_age"] - 42) ** 2 / 10
        + rng.normal(0, 0.55, n)
    )
    df["claim_amount"] = np.exp(log_amount).round(0).clip(lower=500)
    df.loc[rng.random(n) < 0.05, "vehicle_age"] = np.nan
    df.insert(0, "id", rng.permutation(n) + 50000)
    cut = int(n * 0.8)  # test — последние записи по record_no, но без сдвига в данных
    train, test_full = df.iloc[:cut].reset_index(drop=True), df.iloc[cut:].sort_values("id").reset_index(drop=True)
    answers = test_full[["id", "claim_amount"]]
    test = test_full.drop(columns=["claim_amount"])
    dictionary = _dictionary(
        [
            ("id", "id", "int", "Идентификатор убытка"),
            ("record_no", "feature", "int", "Порядковый номер записи в журнале убытков"),
            ("region", "feature", "category", "Регион"),
            ("policy_type", "feature", "category", "Тип полиса"),
            ("channel", "feature", "category", "Канал продажи полиса"),
            ("vehicle_age", "feature", "int", "Возраст автомобиля, лет; бывают пропуски"),
            ("driver_age", "feature", "int", "Возраст водителя"),
            ("sum_insured", "feature", "numeric", "Страховая сумма, ₽"),
            ("prior_claims", "feature", "int", "Убытков по полису раньше"),
            ("deductible", "feature", "int", "Франшиза, ₽"),
            ("claim_amount", "target", "numeric", "Сумма выплаты, ₽"),
        ]
    )
    md = _standard_md(
        "Сумма страховой выплаты",
        "Оцените сумму выплаты по страховому убытку. Важна относительная точность: ошибиться на 10 000 ₽ "
        "в выплате на 20 000 ₽ хуже, чем в выплате на 500 000 ₽.",
        len(train),
        len(test),
        "`record_no`, `vehicle_age`, `driver_age`, `sum_insured`, `prior_claims`, `deductible` — числа; "
        "`region`, `policy_type`, `channel` — категории.",
        "Для каждой строки `test.csv` — положительное число `claim_amount`.",
        "MAPE, mean absolute percentage error (меньше — лучше)",
        "id,claim_amount",
    )
    _write(
        task_dir,
        {
            "train.csv": train,
            "test.csv": test,
            "data_dictionary.csv": dictionary,
            "sample_submission.csv": answers.assign(claim_amount=0),
        },
        md,
    )
    return {"answers": answers}


@task("search_ranking_ndcg", "ndcg", True)
def search(task_dir: Path) -> dict:
    """Ранжирование (вебинар: «может быть ранжирование»; #186): документы внутри запроса, метрика NDCG@10.

    Строка — пара «запрос — документ», query_id — группа; релевантность 0–3. В test — **новые** запросы
    (группы не пересекаются с train), поэтому query_id — не признак, а группа: схема валидации по query_id
    честнее kfold. Ответ — число на документ, важен только порядок внутри запроса.
    """
    rng = np.random.default_rng(SEED + 5)
    n_queries = 900
    sizes = rng.integers(8, 31, n_queries)
    qid = np.repeat(np.arange(n_queries) + 500, sizes)
    n = len(qid)
    query_len = np.repeat(rng.integers(1, 7, n_queries), sizes)
    query_hard = np.repeat(rng.normal(0, 0.6, n_queries), sizes)  # трудность запроса: сдвиг для всех его документов
    sources = np.array(["news", "wiki", "forum", "shop", "blog"])
    source_effect = {"news": 0.1, "wiki": 0.35, "forum": -0.2, "shop": 0.0, "blog": -0.1}
    df = pd.DataFrame(
        {
            "query_id": qid,
            "query_len": query_len,
            "bm25": np.clip(rng.gamma(2.0, 4.0, n) - query_hard * 2, 0, None).round(3),
            "title_match": rng.binomial(query_len, 0.4) / query_len,
            "click_rate": np.clip(rng.beta(1.2, 12, n), 0, 1).round(4),
            "doc_len": rng.lognormal(6.5, 0.8, n).round(0),
            "age_days": rng.integers(0, 2000, n),
            "source": rng.choice(sources, n, p=[0.25, 0.15, 0.25, 0.2, 0.15]),
        }
    )
    latent = (
        0.12 * df["bm25"]
        + 1.6 * df["title_match"]
        + 6.0 * df["click_rate"]
        - 0.25 * np.log1p(df["age_days"] / 30)
        + df["source"].map(source_effect)
        - 0.15 * np.abs(np.log(df["doc_len"]) - 6.5)
        + query_hard
        + rng.normal(0, 0.45, n)
    )
    df["relevance"] = np.digitize(latent, np.quantile(latent, [0.55, 0.8, 0.94]))  # 0 — нерелевантно, 3 — точно
    df.loc[rng.random(n) < 0.05, "click_rate"] = np.nan  # новые документы без истории кликов
    df.insert(0, "id", rng.permutation(n) + 300000)
    test_queries = rng.choice(np.unique(qid), size=int(n_queries * 0.25), replace=False)
    in_test = df["query_id"].isin(test_queries)
    train = df[~in_test].reset_index(drop=True)
    test_full = df[in_test].sort_values("id").reset_index(drop=True)
    answers = test_full[["id", "relevance"]]
    test = test_full.drop(columns=["relevance"])
    dictionary = _dictionary(
        [
            ("id", "id", "int", "Идентификатор пары «запрос — документ»"),
            ("query_id", "group", "int", "Номер поискового запроса; документы одного запроса ранжируются вместе"),
            ("query_len", "feature", "int", "Число слов в запросе"),
            ("bm25", "feature", "numeric", "Текстовая близость документа к запросу (BM25)"),
            ("title_match", "feature", "numeric", "Доля слов запроса в заголовке"),
            ("click_rate", "feature", "numeric", "Доля кликов по документу в прошлом; у новых документов пусто"),
            ("doc_len", "feature", "numeric", "Длина документа, слов"),
            ("age_days", "feature", "int", "Возраст документа, дней"),
            ("source", "feature", "category", "Тип источника"),
            ("relevance", "target", "int", "Оценка асессора: 0 — нерелевантно … 3 — точный ответ"),
        ]
    )
    md = f"""
# Ранжирование поисковой выдачи

## Задача

Для каждого поискового запроса нужно упорядочить найденные документы: самые полезные — наверх.
Асессоры оценили документы по шкале 0–3; по этим оценкам и проверяется порядок.

## Данные

- `train.csv` — {len(train)} пар «запрос — документ» ({train["query_id"].nunique()} запросов) с оценкой `relevance`;
- `test.csv` — {len(test)} пар для **других** запросов ({test["query_id"].nunique()}), без оценки;
- `data_dictionary.csv` — описание полей;
- `sample_submission.csv` — шаблон ответа; значения в нём не являются правильными ответами.

Документы одного запроса связаны через `query_id`.

## Что предсказать

Для каждой строки `test.csv` — число `relevance`: чем больше, тем выше документ в выдаче своего запроса.
Важен только порядок документов внутри запроса, шкала числа не важна.

Метрика — **NDCG@10**, считается отдельно для каждого `query_id` и усредняется по запросам (больше — лучше).

## Формат результата

Файл `submission.csv` в каталоге этого файла, два столбца:

```csv
id,relevance
```

Для каждого `id` из `test.csv` — одна строка, порядок как в `test.csv`.
"""
    _write(
        task_dir,
        {
            "train.csv": train,
            "test.csv": test,
            "data_dictionary.csv": dictionary,
            "sample_submission.csv": answers.assign(relevance=0.0),
        },
        md,
    )
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
