"""«Чужие» задачи для ночного аудита: всё, чем скрытая задача может отличаться от public. Владелец — research.

    uv run python research/night/make_alien_tasks.py                 # все задачи → runs/night/alien/
    uv run python research/night/make_alien_tasks.py --only wide_1500_logloss

Seed = сумма кодов букв ника (Richi-Marqez → 1164): у разных людей разные данные.
Каждая задача — папка с task.md и CSV (как на платформе); ответы и описание того, что проверяем, —
отдельно, в runs/night/alien/_answers/<задача>.json и .csv (агент их не видит: раннер копирует только папку задачи).
Данные — синтетика без интернета; в каждой задаче есть сигнал, чтобы LightGBM был заметно лучше тривиального ответа.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "runs" / "night" / "alien"
SEED = sum(map(ord, "Richi-Marqez"))  # 1164


def _rng(i: int) -> np.random.Generator:
    return np.random.default_rng(SEED + i)


def _save(name: str, files: dict[str, tuple[pd.DataFrame, dict]], task_md: str, answers: pd.DataFrame, meta: dict):
    """files: относительный путь → (таблица, kwargs для to_csv). answers — колонки ответа в порядке test."""
    task_dir = OUT / name
    task_dir.mkdir(parents=True, exist_ok=True)
    for rel, (df, kwargs) in files.items():
        path = task_dir / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False, **kwargs)
    (task_dir / "task.md").write_text(task_md, encoding=meta.pop("md_encoding", "utf-8"))
    ans = OUT / "_answers"
    ans.mkdir(parents=True, exist_ok=True)
    answers.to_csv(ans / f"{name}.csv", index=False)
    (ans / f"{name}.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"✅ {name}: {', '.join(f'{k} {v[0].shape}' for k, v in files.items())}")


def _split(df: pd.DataFrame, n_test: int, rng) -> tuple[pd.DataFrame, pd.DataFrame]:
    idx = rng.permutation(len(df))
    return df.iloc[idx[n_test:]].reset_index(drop=True), df.iloc[idx[:n_test]].reset_index(drop=True)


def _sigmoid(z):
    return 1 / (1 + np.exp(-z))


# ------------------------------------------------------------------ 1. BOM, русские заголовки, дубли имён, кавычки


def ru_headers_bom_auc():
    rng = _rng(1)
    n = 8000
    cities = [f"г. {c}, обл." for c in ["Ростов-на-Дону", "Казань", "Тула", "Омск", "Пермь", "Сочи", "Уфа", "Тверь"]]
    city = rng.choice(cities, n)
    age = rng.integers(18, 75, n)
    income = rng.lognormal(11, 0.6, n).round(0)
    income2 = (income * rng.uniform(0.5, 2.5, n)).round(0)  # «Доход, руб.» семьи — второе поле с тем же именем
    stage = rng.integers(0, 40, n).astype(float)
    stage[rng.random(n) < 0.1] = np.nan
    words = ["нужен кредит", "ремонт, кухня", 'покупка "авто"', "лечение", "учёба; курсы", "отпуск"]
    comment = rng.choice(words, n)
    city_eff = {c: e for c, e in zip(cities, rng.normal(0, 0.6, len(cities)), strict=True)}
    z = (
        -0.5
        + 0.9 * np.log(income / 60000)
        + 0.03 * (age - 40)
        - 0.0006 * (age - 40) ** 2
        + 0.4 * np.log1p(np.nan_to_num(stage, nan=2))
        + np.vectorize(city_eff.get)(city)
        + 0.3 * np.log(income2 / income)
    )
    y = (rng.random(n) < _sigmoid(z)).astype(int)
    ids = [f"Z-{i:06d}" for i in rng.permutation(n) + 100]
    cols = [
        "Возраст клиента",
        "Номер заявки",
        "Доход, руб.",
        "Город",
        "Стаж (лет)",
        "Одобрено",
        "Доход, руб.",
        "Комментарий",
    ]
    df = pd.DataFrame(list(zip(age, ids, income, city, stage, y, income2, comment, strict=True)), columns=cols)
    train, test = _split(df, 2500, rng)
    answers = test[["Номер заявки", "Одобрено"]].copy()
    test = test.drop(columns=["Одобрено"])
    sample = answers.assign(Одобрено=0.5)
    md = """# Одобрение заявок

Банк хочет заранее понимать, одобрит ли кредитный комитет заявку.

## Данные
- `data/train.csv` — заявки с решением (поле «Одобрено»: 1 — одобрена, 0 — нет);
- `data/test.csv` — новые заявки;
- `data/sample_submission.csv` — пример ответа.

Поле «Доход, руб.» встречается дважды: сначала личный доход, затем доход семьи.

## Метрика
Нужна вероятность одобрения. Качество оцениваем площадью под ROC-кривой — чем больше, тем лучше.
"""
    enc = {"encoding": "utf-8-sig"}
    _save(
        "ru_headers_bom_auc",
        {"train.csv": (train, enc), "test.csv": (test, enc), "sample_submission.csv": (sample, enc)},
        md,
        answers,
        {
            "metric": "roc_auc",
            "greater_is_better": True,
            "task_type": "binary",
            "id": "Номер заявки",
            "targets": ["Одобрено"],
            "submission_columns": ["Номер заявки", "Одобрено"],
            "positive": "1",
            "checks": "BOM utf-8-sig, русские заголовки с пробелами и запятыми, дубль имени колонки, "
            "кавычки и ; внутри полей, id не первая, target не последний, неверные пути в task.md",
        },
    )


# ------------------------------------------------------------------ 2. разделитель ; и запятая-десятичная (Excel RU)


def semicolon_decimal_mae():
    rng = _rng(2)
    n = 6000
    area = rng.uniform(20, 150, n).round(1)
    rooms = np.clip((area / 25 + rng.normal(0, 0.7, n)).round(), 1, 6).astype(int)
    floor = rng.integers(1, 25, n)
    district = rng.choice(["Центр", "Север", "Юг", "Запад", "Восток"], n)
    eff = {"Центр": 1.6, "Север": 1.0, "Юг": 0.9, "Запад": 1.1, "Восток": 0.85}
    price = area * 95_000 * np.vectorize(eff.get)(district) * (1 + 0.004 * floor) * rng.lognormal(0, 0.15, n)
    price = price.round(-3) / 1e6  # млн руб., дробное — с запятой
    df = pd.DataFrame(
        {"id": np.arange(1, n + 1), "площадь": area, "комнат": rooms, "этаж": floor, "район": district, "цена": price}
    )
    train, test = _split(df, 1500, rng)
    answers = test[["id", "цена"]].copy()
    test = test.drop(columns=["цена"])
    sample = answers.assign(цена=0.0)
    md = """# Цена квартиры

Файлы выгружены из Excel: train.csv, test.csv, sample_submission.csv. Цена — в млн руб.

Метрика — MAE (средняя абсолютная ошибка), меньше — лучше.
"""
    kw = {"sep": ";", "decimal": ","}
    _save(
        "semicolon_decimal_mae",
        {"train.csv": (train, kw), "test.csv": (test, kw), "sample_submission.csv": (sample, kw)},
        md,
        answers,
        {
            "metric": "mae",
            "greater_is_better": False,
            "task_type": "regression",
            "id": "id",
            "targets": ["цена"],
            "submission_columns": ["id", "цена"],
            "sep": ";",
            "decimal": ",",
            "checks": "разделитель ; и запятая-десятичная во всех CSV (выгрузка Excel RU)",
        },
    )


# ------------------------------------------------------------------ 3. кодировка cp1251


def cp1251_r2():
    rng = _rng(3)
    n = 5000
    x1, x2, x3 = rng.normal(0, 1, (3, n))
    region = rng.choice(["Москва", "Санкт-Петербург", "Новосибирск", "Екатеринбург"], n)
    eff = {"Москва": 3.0, "Санкт-Петербург": 2.0, "Новосибирск": 0.5, "Екатеринбург": 1.0}
    y = 10 + 3 * x1 + 2 * np.sin(2 * x2) + x1 * x3 + np.vectorize(eff.get)(region) + rng.normal(0, 1.5, n)
    df = pd.DataFrame(
        {"ид": np.arange(n), "признак_1": x1, "признак_2": x2, "признак_3": x3, "регион": region, "выручка": y.round(3)}
    )
    train, test = _split(df, 1200, rng)
    answers = test[["ид", "выручка"]].copy()
    test = test.drop(columns=["выручка"])
    sample = answers.assign(выручка=0.0)
    md = """# Выручка филиала

Файлы: train.csv, test.csv, sample_submission.csv (кодировка Windows-1251, как выгружает 1С).

Метрика — коэффициент детерминации R²: чем ближе к 1, тем лучше.
"""
    kw = {"encoding": "cp1251"}
    _save(
        "cp1251_r2",
        {"train.csv": (train, kw), "test.csv": (test, kw), "sample_submission.csv": (sample, kw)},
        md,
        answers,
        {
            "metric": "r2",
            "greater_is_better": True,
            "task_type": "regression",
            "id": "ид",
            "targets": ["выручка"],
            "submission_columns": ["ид", "выручка"],
            "encoding": "cp1251",
            "checks": "CSV в cp1251 (1С/Excel), R²",
        },
    )


# ------------------------------------------------------------------ 4. нет колонки id


def no_id_rmse():
    rng = _rng(4)
    n = 5000
    x = rng.normal(0, 1, (n, 6))
    y = 50 + 10 * x[:, 0] + 5 * x[:, 1] ** 2 + 4 * x[:, 2] * x[:, 3] + rng.normal(0, 3, n)
    df = pd.DataFrame(x.round(4), columns=[f"x{i}" for i in range(6)]).assign(price=y.round(2))
    train, test = _split(df, 1200, rng)
    answers = test[["price"]].copy()
    test = test.drop(columns=["price"])
    sample = answers.assign(price=0.0)
    md = """# Price

train.csv, test.csv. Submission: one column `price`, rows in the same order as test.csv (see sample_submission.csv).
Metric: RMSE (lower is better).
"""
    _save(
        "no_id_rmse",
        {"train.csv": (train, {}), "test.csv": (test, {}), "sample_submission.csv": (sample, {})},
        md,
        answers,
        {
            "metric": "rmse",
            "greater_is_better": False,
            "task_type": "regression",
            "id": None,
            "targets": ["price"],
            "submission_columns": ["price"],
            "checks": "нет колонки id ни в test, ни в шаблоне: ответ — одна колонка в порядке строк test",
        },
    )


# ------------------------------------------------------------------ 5. >1000 колонок, константы, пустые колонки


def wide_1500_logloss():
    rng = _rng(5)
    n, k = 5000, 1500
    x = rng.normal(0, 1, (n, k)).astype(np.float32)
    w = np.zeros(k)
    informative = rng.choice(k, 20, replace=False)
    w[informative] = rng.normal(0, 1, 20)
    z = x @ w * 0.6 + 0.8 * x[:, informative[0]] * x[:, informative[1]]
    y = (rng.random(n) < _sigmoid(z - 0.5)).astype(int)
    df = pd.DataFrame(x.round(3), columns=[f"f{i:04d}" for i in range(k)])
    for i in range(10):
        df[f"const_{i}"] = 7.0
    for i in range(5):
        df[f"empty_{i}"] = np.nan
    df.insert(0, "row_id", np.arange(100000, 100000 + n))
    df["label"] = y
    train, test = _split(df, 1500, rng)
    answers = test[["row_id", "label"]].copy()
    test = test.drop(columns=["label"])
    sample = answers.assign(label=0.5)
    md = """# Wide signals

1500 anonymised sensor features (`f0000`…`f1499`), plus a few technical columns.
Predict the probability that `label` = 1. Evaluation metric: binary log loss (lower is better).
"""
    _save(
        "wide_1500_logloss",
        {"train.csv": (train, {}), "test.csv": (test, {}), "sample_submission.csv": (sample, {})},
        md,
        answers,
        {
            "metric": "logloss",
            "greater_is_better": False,
            "task_type": "binary",
            "id": "row_id",
            "targets": ["label"],
            "submission_columns": ["row_id", "label"],
            "positive": "1",
            "checks": "1515 колонок (20 информативных), 10 константных, 5 полностью пустых; logloss",
        },
    )


# ------------------------------------------------------------------ 6. > 1 млн строк


def million_rows_rmsle():
    rng = _rng(6)
    n, n_test = 1_300_000, 300_000
    store = rng.integers(0, 1000, n)
    item = rng.integers(0, 40, n)
    price = rng.lognormal(3, 0.7, n).astype(np.float32)
    promo = (rng.random(n) < 0.2).astype(np.int8)
    week = rng.integers(1, 53, n)
    temp = rng.normal(15, 10, n).astype(np.float32)
    store_eff = rng.normal(0, 0.5, 1000)[store]
    item_eff = rng.normal(0, 0.7, 40)[item]
    mu = 1.5 + store_eff + item_eff + 0.6 * promo - 0.3 * np.log(price) + 0.3 * np.sin(week / 52 * 2 * np.pi)
    y = np.round(rng.poisson(np.exp(mu)) * 1.0, 0)
    df = pd.DataFrame(
        {
            "id": np.arange(n),
            "store": np.char.add("s", store.astype(str)),
            "item": item,
            "price": price.round(2),
            "promo": promo,
            "week": week,
            "temp": temp.round(1),
            "x_noise": rng.normal(0, 1, n).astype(np.float32).round(3),
            "sales": y,
        }
    )
    train, test = _split(df, n_test, rng)
    answers = test[["id", "sales"]].copy()
    test = test.drop(columns=["sales"])
    sample = answers.assign(sales=0.0)
    md = """# Продажи (большая выгрузка)

train.csv (1 млн строк), test.csv, sample_submission.csv. Прогноз продаж товара в магазине за неделю.
Метрика: RMSLE.
"""
    _save(
        "million_rows_rmsle",
        {"train.csv": (train, {}), "test.csv": (test, {}), "sample_submission.csv": (sample, {})},
        md,
        answers,
        {
            "metric": "rmsle",
            "greater_is_better": False,
            "task_type": "regression",
            "id": "id",
            "targets": ["sales"],
            "submission_columns": ["id", "sales"],
            "checks": "1 млн строк train, 300 тыс. test, 1000 магазинов строкой; RMSLE",
        },
    )


# ------------------------------------------------------------------ 7. даты разных форматов, текст, огромные числа, NaN в target, 50 классов


def dates_text_50cls_balacc():
    rng = _rng(7)
    n = 24000
    days = rng.integers(0, 730, n)
    dates = pd.Timestamp("2023-01-01") + pd.to_timedelta(days, unit="D")
    fmt = rng.choice(["%Y-%m-%d", "%d.%m.%Y", "%m/%d/%Y", "%d %b %Y"], n)
    date_str = [d.strftime(f) for d, f in zip(dates, fmt, strict=True)]
    cls = rng.integers(0, 50, n)
    vocab = [f"слово{i}" for i in range(300)]
    cls_words = rng.integers(0, 300, (50, 6))
    text = [" ".join(rng.choice(np.concatenate([np.array(vocab)[cls_words[c]], rng.choice(vocab, 4)]), 5)) for c in cls]
    amount = (10 ** rng.uniform(3, 12, n)) * (1 + cls / 50)
    balance = rng.normal(-5e6, 3e7, n) + cls * 1e5
    x1 = rng.normal(cls % 7, 2.0, n)
    x2 = rng.normal(cls // 7, 2.0, n)
    label = np.array([f"cat_{c:02d}" for c in cls], dtype=object)
    df = pd.DataFrame(
        {
            "order_id": [f"o{i}" for i in range(n)],
            "дата_заказа": date_str,
            "описание": text,
            "сумма": amount,
            "баланс": balance.round(2),
            "x1": x1.round(3),
            "x2": x2.round(3),
            "категория": label,
        }
    )
    train, test = _split(df, 6000, rng)
    answers = test[["order_id", "категория"]].copy()
    test = test.drop(columns=["категория"])
    train.loc[rng.random(len(train)) < 0.02, "категория"] = np.nan  # пропуски в target train
    sample = answers.assign(категория="cat_00")
    md = """# Категория заказа

Определите категорию заказа (50 категорий). Файлы train.csv, test.csv, sample_submission.csv.
В «дата_заказа» даты записаны по-разному (выгрузки из разных систем), «описание» — свободный текст.
Около 2% заказов в train без категории.

Метрика — сбалансированная точность (balanced accuracy): средняя полнота по категориям, больше — лучше.
"""
    _save(
        "dates_text_50cls_balacc",
        {"train.csv": (train, {}), "test.csv": (test, {}), "sample_submission.csv": (sample, {})},
        md,
        answers,
        {
            "metric": "balanced_accuracy",
            "greater_is_better": True,
            "task_type": "multiclass",
            "id": "order_id",
            "targets": ["категория"],
            "submission_columns": ["order_id", "категория"],
            "checks": "даты в 4 форматах строкой, свободный текст, суммы до 1e12 и отрицательные, 2% NaN в target, "
            "50 классов, balanced accuracy",
        },
    )


# ------------------------------------------------------------------ 8. 99.9% одного класса, PR-AUC


def rare_999_prauc():
    rng = _rng(8)
    n = 250_000
    x = rng.normal(0, 1, (n, 10)).astype(np.float32)
    merchant = rng.integers(0, 500, n)
    risky = rng.random(500) < 0.05
    z = -8.2 + 1.6 * x[:, 0] + 1.2 * x[:, 1] - 0.8 * x[:, 2] + 2.0 * risky[merchant] + 0.8 * (x[:, 3] > 1.5)
    y = (rng.random(n) < _sigmoid(z)).astype(int)
    df = pd.DataFrame(x.round(4), columns=[f"v{i}" for i in range(10)])
    df.insert(0, "tx_id", np.arange(n) + 5_000_000)
    df["merchant"] = np.char.add("m", merchant.astype(str))
    df["is_fraud"] = y
    train, test = _split(df, 50_000, rng)
    answers = test[["tx_id", "is_fraud"]].copy()
    test = test.drop(columns=["is_fraud"])
    sample = answers.assign(is_fraud=0.001)
    md = f"""# Fraud detection

Card transactions; fraud is very rare (about {y.mean():.2%} of rows).
Files: train.csv, test.csv, sample_submission.csv. Predict a fraud score for every `tx_id`.

Evaluation: area under the precision-recall curve (average precision). Higher is better.
"""
    _save(
        "rare_999_prauc",
        {"train.csv": (train, {}), "test.csv": (test, {}), "sample_submission.csv": (sample, {})},
        md,
        answers,
        {
            "metric": "pr_auc",
            "greater_is_better": True,
            "task_type": "binary",
            "id": "tx_id",
            "targets": ["is_fraud"],
            "submission_columns": ["tx_id", "is_fraud"],
            "positive": "1",
            "checks": f"доля положительного класса {y.mean():.3%}, 200 тыс. строк, PR-AUC",
        },
    )


# ------------------------------------------------------------------ 9. утечка: колонка = target в train, пустая в test


def leak_column_auc():
    rng = _rng(9)
    n = 12000
    x = rng.normal(0, 1, (n, 8))
    seg = rng.choice(["A", "B", "C", "D"], n)
    z = (
        0.9 * x[:, 0]
        - 0.7 * x[:, 1]
        + 0.5 * x[:, 2] * x[:, 3]
        + np.vectorize({"A": 0.5, "B": 0, "C": -0.4, "D": 0.2}.get)(seg)
    )
    y = (rng.random(n) < _sigmoid(z)).astype(int)
    df = pd.DataFrame(x.round(4), columns=[f"feat_{i}" for i in range(8)])
    df.insert(0, "client_id", np.arange(n))
    df["segment"] = seg
    # «Код итогового статуса» — заполняется ПОСЛЕ события: в train почти равен target, в test пуст
    leak = np.where(rng.random(n) < 0.98, y, 1 - y).astype(float)
    df["final_status_code"] = leak
    df["churn"] = y
    train, test = _split(df, 3000, rng)
    answers = test[["client_id", "churn"]].copy()
    test = test.drop(columns=["churn"])
    test["final_status_code"] = np.nan
    sample = answers.assign(churn=0.5)
    md = """# Отток клиентов

Предскажите вероятность ухода клиента (`churn`). Файлы train.csv, test.csv, sample_submission.csv.
Поле `final_status_code` заполняется системой после закрытия договора, поэтому в test оно пустое.

Метрика — ROC-AUC.
"""
    _save(
        "leak_column_auc",
        {"train.csv": (train, {}), "test.csv": (test, {}), "sample_submission.csv": (sample, {})},
        md,
        answers,
        {
            "metric": "roc_auc",
            "greater_is_better": True,
            "task_type": "binary",
            "id": "client_id",
            "targets": ["churn"],
            "submission_columns": ["client_id", "churn"],
            "positive": "1",
            "checks": "утечка: final_status_code = target на 98% train, в test 100% пусто (о чём сказано в task.md)",
        },
    )


# ------------------------------------------------------------------ 10. мультитаргет + квантильная потеря 0.9


def multitarget_pinball90():
    rng = _rng(10)
    n = 10000
    x = rng.normal(0, 1, (n, 5))
    region = rng.choice(["north", "south", "east", "west"], n)
    base = (
        100
        + 15 * x[:, 0]
        + 10 * np.abs(x[:, 1])
        + np.vectorize({"north": 10, "south": -5, "east": 0, "west": 5}.get)(region)
    )
    # шум несимметричный и зависит от x2: квантиль 0.9 далеко от среднего
    y1 = base + rng.exponential(20 * (1 + np.abs(x[:, 2])), n)
    y2 = 0.5 * base + rng.gamma(1.5, 10 * (1 + (x[:, 3] > 0)), n)
    df = pd.DataFrame(x.round(4), columns=[f"x{i}" for i in range(5)])
    df.insert(0, "sku", [f"SKU{i:05d}" for i in range(n)])
    df["region"] = region
    df["demand_store"] = y1.round(1)
    df["demand_online"] = y2.round(1)
    train, test = _split(df, 2500, rng)
    answers = test[["sku", "demand_store", "demand_online"]].copy()
    test = test.drop(columns=["demand_store", "demand_online"])
    sample = answers.assign(demand_store=0.0, demand_online=0.0)
    md = """# Запас на складе

Нужно спрогнозировать спрос в магазине (`demand_store`) и онлайн (`demand_online`) так, чтобы товара хватило
в 9 случаях из 10. Поэтому метрика — квантильная потеря (pinball loss) для квантиля 0.9, усреднённая по двум
колонкам ответа; меньше — лучше.

Файлы: train.csv, test.csv, sample_submission.csv.
"""
    _save(
        "multitarget_pinball90",
        {"train.csv": (train, {}), "test.csv": (test, {}), "sample_submission.csv": (sample, {})},
        md,
        answers,
        {
            "metric": "pinball@0.9",
            "greater_is_better": False,
            "task_type": "regression",
            "id": "sku",
            "targets": ["demand_store", "demand_online"],
            "submission_columns": ["sku", "demand_store", "demand_online"],
            "checks": "два таргета, квантиль 0.9 (несимметричный шум): нужна квантильная цель, а не среднее",
        },
    )


# ------------------------------------------------------------------ 11. порядковые классы, метрика словами


def ordinal_kappa_words():
    rng = _rng(11)
    n = 9000
    x = rng.normal(0, 1, (n, 6))
    latent = 1.2 * x[:, 0] + 0.8 * x[:, 1] - 0.5 * x[:, 2] + 0.4 * x[:, 3] * x[:, 4] + rng.normal(0, 0.8, n)
    grade = np.digitize(latent, [-1.8, -0.6, 0.6, 1.8])  # 0..4, крайние редкие
    df = pd.DataFrame(x.round(4), columns=["вязкость", "плотность", "влажность", "зола", "сера", "кислотность"])
    df.insert(0, "партия", np.arange(n) + 1)
    df["сорт"] = grade + 1  # 1..5
    train, test = _split(df, 2000, rng)
    answers = test[["партия", "сорт"]].copy()
    test = test.drop(columns=["сорт"])
    sample = answers.assign(сорт=3)
    md = """# Сорт партии

Эксперты присваивают партии сорт от 1 до 5 (1 — худший). Предскажите сорт.
Файлы: train.csv, test.csv, sample_submission.csv.

Согласованность прогноза с экспертами измеряют каппой Коэна с квадратичными весами: ошибка на два сорта
наказывается вчетверо сильнее ошибки на один. Чем выше значение, тем лучше.
"""
    _save(
        "ordinal_kappa_words",
        {"train.csv": (train, {}), "test.csv": (test, {}), "sample_submission.csv": (sample, {})},
        md,
        answers,
        {
            "metric": "qwk",
            "greater_is_better": True,
            "task_type": "multiclass",
            "id": "партия",
            "targets": ["сорт"],
            "submission_columns": ["партия", "сорт"],
            "checks": "порядковые классы 1–5 числами; QWK описана только словами по-русски («каппа Коэна с квадратичными весами»)",
        },
    )


# ------------------------------------------------------------------ 12. колонка на класс, имена не совпадают с метками, новые категории


def per_class_proba_newcats_mlogloss():
    rng = _rng(12)
    n = 15000
    x = rng.normal(0, 1, (n, 5))
    shop = rng.integers(0, 800, n)
    shop_eff = rng.normal(0, 1, 800)
    classes = np.array(["low", "mid", "high", "vip", "churn"])
    logits = np.stack(
        [
            1.0 * x[:, 0],
            0.6 * x[:, 1] + 0.3,
            -0.8 * x[:, 0] + 0.5 * x[:, 2],
            0.9 * shop_eff[shop] + x[:, 3] - 1.0,
            -0.7 * x[:, 4] - 0.3,
        ],
        axis=1,
    )
    proba = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
    cls = np.array([rng.choice(5, p=p) for p in proba])
    df = pd.DataFrame(x.round(4), columns=[f"a{i}" for i in range(5)])
    df.insert(0, "uid", np.arange(n) + 1)
    df["shop"] = np.char.add("shop_", shop.astype(str))
    df["segment"] = classes[cls]
    train, test = _split(df, 4000, rng)
    # в test четверть строк — магазины, которых нет в train
    new_rows = rng.random(len(test)) < 0.25
    test.loc[new_rows, "shop"] = "shop_new_" + pd.Series(rng.integers(0, 300, new_rows.sum())).astype(str).to_numpy()
    answers = test[["uid", "segment"]].copy()
    test = test.drop(columns=["segment"])
    proba_cols = [f"proba_{c}" for c in classes]  # порядок «по смыслу», а не по алфавиту
    sample = test[["uid"]].assign(**{c: 0.2 for c in proba_cols})
    md = """# Сегмент клиента

Для каждого клиента нужна вероятность каждого из пяти сегментов: low, mid, high, vip, churn.
Колонки ответа — `proba_low`, `proba_mid`, `proba_high`, `proba_vip`, `proba_churn` (см. sample_submission.csv).

Метрика — multi-class log loss (меньше — лучше).
"""
    _save(
        "per_class_proba_newcats_mlogloss",
        {"train.csv": (train, {}), "test.csv": (test, {}), "sample_submission.csv": (sample, {})},
        md,
        answers,
        {
            "metric": "logloss",
            "greater_is_better": False,
            "task_type": "multiclass",
            "id": "uid",
            "targets": ["segment"],
            "submission_columns": ["uid", *proba_cols],
            "per_class": {c: f"proba_{c}" for c in classes},
            "checks": "ответ — колонка на класс, имена proba_<класс> в смысловом (не алфавитном) порядке; "
            "25% строк test — новые магазины (800 магазинов в train)",
        },
    )


# ------------------------------------------------------------------ 13. английский, шум, неверные пути, F-beta, метки yes/no


def english_noisy_fbeta():
    rng = _rng(13)
    n = 20000
    x = rng.normal(0, 1, (n, 7))
    channel = rng.choice(["web", "app", "branch", "partner"], n, p=[0.5, 0.3, 0.15, 0.05])
    z = -2.6 + 1.1 * x[:, 0] + 0.8 * x[:, 1] * (x[:, 2] > 0) + 0.6 * (channel == "partner") + 0.4 * x[:, 3]
    y = np.where(rng.random(n) < _sigmoid(z), "yes", "no")
    df = pd.DataFrame(x.round(4), columns=["tenure", "spend", "visits", "age_z", "score_a", "score_b", "score_c"])
    df["channel"] = channel
    df["txn_id"] = [f"T{i:07d}" for i in rng.permutation(n)]
    df["responded"] = y
    train, test = _split(df, 5000, rng)
    answers = test[["txn_id", "responded"]].copy()
    test = test.drop(columns=["responded"])
    md = """# Marketing response — season 3

Background. Last season the team used accuracy as the headline number, but since only ~10% of customers respond,
that turned out to be useless. We also looked at precision and recall separately in the weekly report.
The data was exported by the CRM team on 2025-02-11 (export job #4471, see ticket MKT-2207).

Files: `/mnt/data/season3/train_v2.csv`, `/mnt/data/season3/test_v2.csv` (paths on the old server — may differ).

Target: `responded` (yes/no). Submit `txn_id,responded` with labels yes/no.

Scoring. Missing a responder costs us much more than contacting a non-responder, so submissions are scored with
the F2 score (F-beta with beta = 2) for the positive class "yes". Higher is better.
"""
    _save(
        "english_noisy_fbeta",
        {"input/train.csv": (train, {}), "input/test.csv": (test, {})},
        md,
        answers,
        {
            "metric": "f2",
            "greater_is_better": True,
            "task_type": "binary",
            "id": "txn_id",
            "targets": ["responded"],
            "submission_columns": ["txn_id", "responded"],
            "positive": "yes",
            "checks": "английский, шум про accuracy/precision/recall, неверные пути, файлы в input/, нет sample_submission, "
            "id и target не первые, метки yes/no, метрики F2 (F-beta) нет в реестре",
        },
    )


# ------------------------------------------------------------------ 14. id из train в test, SMAPE, ID заглавными


def smape_train_ids_in_test():
    rng = _rng(14)
    n = 8000
    x = rng.normal(0, 1, (n, 5))
    cat = rng.choice(list("ABCDEFGH"), n)
    y = np.exp(2 + 0.6 * x[:, 0] + 0.3 * x[:, 1] + 0.2 * (cat == "A")) * rng.lognormal(0, 0.2, n)
    df = pd.DataFrame(x.round(4), columns=[f"m{i}" for i in range(5)])
    df.insert(0, "ID", np.arange(n) + 1)
    df["cat"] = cat
    df["y"] = y.round(3)
    train, test = _split(df, 2000, rng)
    # 30% test — точные копии строк train (тот же ID и признаки): «старые» объекты в новом файле
    dup = train.sample(600, random_state=1)
    test = pd.concat([test, dup], ignore_index=True).sample(frac=1, random_state=2).reset_index(drop=True)
    answers = test[["ID", "y"]].copy()
    test = test.drop(columns=["y"])
    sample = answers.assign(y=1.0)
    md = """# Объём заказа

train.csv, test.csv, sample_submission.csv. Часть объектов test уже встречалась в train (тот же ID).

Метрика — SMAPE (симметричная средняя абсолютная процентная ошибка), меньше — лучше.
"""
    _save(
        "smape_train_ids_in_test",
        {"train.csv": (train, {}), "test.csv": (test, {}), "sample_submission.csv": (sample, {})},
        md,
        answers,
        {
            "metric": "smape",
            "greater_is_better": False,
            "task_type": "regression",
            "id": "ID",
            "targets": ["y"],
            "submission_columns": ["ID", "y"],
            "checks": "SMAPE; 23% строк test — копии строк train с тем же ID (можно ответить точно)",
        },
    )


# ------------------------------------------------------------------ 15. MAPE словами (легко спутать с MAE)


def mape_in_words():
    rng = _rng(15)
    n = 10000
    x = rng.normal(0, 1, (n, 4))
    kind = rng.choice(["малый", "средний", "крупный"], n, p=[0.6, 0.3, 0.1])
    scale = np.vectorize({"малый": 5, "средний": 80, "крупный": 1500}.get)(kind)
    y = scale * np.exp(0.4 * x[:, 0] + 0.2 * x[:, 1]) * rng.lognormal(0, 0.35, n)
    df = pd.DataFrame(x.round(4), columns=["k1", "k2", "k3", "k4"])
    df.insert(0, "номер", np.arange(n))
    df["тип_клиента"] = kind
    df["объём"] = y.round(2)
    train, test = _split(df, 2500, rng)
    answers = test[["номер", "объём"]].copy()
    test = test.drop(columns=["объём"])
    sample = answers.assign(объём=0.0)
    md = """# Объём закупки

Файлы: train.csv, test.csv, sample_submission.csv.

Качество оцениваем как среднее абсолютное отклонение прогноза от факта в процентах от факта: ошибка на 10 единиц
у малого клиента так же плоха, как ошибка на 3000 у крупного. Чем меньше, тем лучше.
"""
    _save(
        "mape_in_words",
        {"train.csv": (train, {}), "test.csv": (test, {}), "sample_submission.csv": (sample, {})},
        md,
        answers,
        {
            "metric": "mape",
            "greater_is_better": False,
            "task_type": "regression",
            "id": "номер",
            "targets": ["объём"],
            "submission_columns": ["номер", "объём"],
            "checks": "MAPE описана словами («среднее абсолютное отклонение … в процентах») — легко принять за MAE; "
            "масштаб таргета различается в 300 раз",
        },
    )


TASKS = {
    f.__name__: f
    for f in [
        ru_headers_bom_auc,
        semicolon_decimal_mae,
        cp1251_r2,
        no_id_rmse,
        wide_1500_logloss,
        million_rows_rmsle,
        dates_text_50cls_balacc,
        rare_999_prauc,
        leak_column_auc,
        multitarget_pinball90,
        ordinal_kappa_words,
        per_class_proba_newcats_mlogloss,
        english_noisy_fbeta,
        smape_train_ids_in_test,
        mape_in_words,
    ]
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", nargs="*", help="имена задач")
    args = parser.parse_args()
    for name, make in TASKS.items():
        if not args.only or name in args.only:
            make()


if __name__ == "__main__":
    main()
