"""«Чужие» задачи для ночного аудита: всё, чем скрытая задача может отличаться от public. Владелец — research.

    uv run python research/night/make_alien_tasks.py                 # все задачи → runs/night/alien/
    uv run python research/night/make_alien_tasks.py --only wide_1500_logloss
    uv run python research/night/make_alien_tasks.py --set leak         # 3 утечки разной природы → runs/night/leak/
    uv run python research/night/make_alien_tasks.py --set uncovered    # мультиметка, ранжирование, 600k строк

Seed = сумма кодов букв ника (Richi-Marqez → 1164): у разных людей разные данные.
Каждая задача — папка с task.md и CSV (как на платформе); ответы и описание того, что проверяем, —
отдельно, в runs/night/<набор>/_answers/<задача>.json и .csv (агент их не видит: раннер копирует только папку задачи).
Наборы leak и uncovered в гейт не входят (он считает нули на 15 задачах набора alien); прогон — run_alien.py --set.
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


# ================================================================== набор «leak»: три утечки разной природы
# Для проверки утечек в агенте (ml/, @marenkov17). У существующей leak_column_auc признак пуст в test и об этом сказано
# в task.md; здесь три другие природы, и в task.md об утечке ни слова — правило не должно ловить одну форму.
# meta["leak_columns"] — что надо отбросить; ориентир LightGBM считается без них (DS, заметивший утечку), а
# lgbm_with_leak — с ними: насколько ловушка бьёт по ответу.


def leak_idlike_regno_auc():
    """Id-подобный признак: номер регистрации растёт вместе с target в train, в test — новый диапазон без связи."""
    rng = _rng(16)
    n = 10000
    x = rng.normal(0, 1, (n, 6))
    channel = rng.choice(["офис", "сайт", "приложение", "партнёр"], n)
    z = (
        -1.0
        + 0.9 * x[:, 0]
        - 0.7 * x[:, 1]
        + 0.5 * x[:, 2] * x[:, 3]
        + np.vectorize({"офис": 0.4, "сайт": 0.0, "приложение": -0.3, "партнёр": 0.6}.get)(channel)
    )
    y = (rng.random(n) < _sigmoid(z)).astype(int)
    df = pd.DataFrame(x.round(4), columns=[f"f{i}" for i in range(6)])
    df.insert(0, "request_id", [f"Q-{i:06d}" for i in rng.permutation(n)])
    df["channel"] = channel
    df["complaint"] = y
    train, test = _split(df, 3000, rng)
    # Жалобы регистрировали вторым потоком: в train номер почти упорядочен по target (как выгрузка, отсортированная
    # по статусу). Test зарегистрирован позже — номера больше любого из train и с target не связаны.
    order = np.argsort(train["complaint"].to_numpy() + rng.normal(0, 0.35, len(train)))
    regno = np.empty(len(train), dtype=np.int64)
    regno[order] = 500_000 + np.arange(len(train)) * 3 + rng.integers(0, 3, len(train))
    train.insert(1, "reg_number", regno)
    test.insert(1, "reg_number", regno.max() + 1 + rng.permutation(len(test)) * 3 + rng.integers(0, 3, len(test)))
    answers = test[["request_id", "complaint"]].copy()
    test = test.drop(columns=["complaint"])
    sample = answers.assign(complaint=0.5)
    md = """# Жалоба или обычное обращение

По обращению клиента нужно оценить вероятность того, что это жалоба (`complaint` = 1).

Файлы: `train.csv` (обращения с разметкой), `test.csv` (новые обращения), `sample_submission.csv`.
Колонки: `request_id` — идентификатор обращения, `reg_number` — регистрационный номер, `channel` — канал,
`f0`…`f5` — числовые характеристики обращения.

Метрика — ROC-AUC, чем больше, тем лучше. Ответ: `request_id`, `complaint` (вероятность).
"""
    _save(
        "leak_idlike_regno_auc",
        {"train.csv": (train, {}), "test.csv": (test, {}), "sample_submission.csv": (sample, {})},
        md,
        answers,
        {
            "metric": "roc_auc",
            "greater_is_better": True,
            "task_type": "binary",
            "id": "request_id",
            "targets": ["complaint"],
            "submission_columns": ["request_id", "complaint"],
            "positive": "1",
            "leak_columns": ["reg_number"],
            "leak_kind": "id-подобный: монотонный номер, в train упорядочен по target, в test — новый диапазон",
            "checks": "утечка через id-подобный номер (не назван id, заполнен в test): CV с ним завышена, "
            "в test все номера правее train — модель на нём проваливается",
        },
    )


def leak_target_copy_mae():
    """Копия target: «сумма по договору» = цена сделки ±1% в train; в test поле заполнено шаблоном без связи."""
    rng = _rng(17)
    n = 9000
    area = rng.uniform(25, 140, n).round(1)
    rooms = np.clip((area / 30 + rng.normal(0, 0.7, n)).round(), 1, 5).astype(int)
    district = rng.choice([f"район_{i}" for i in range(12)], n)
    age = rng.integers(0, 60, n)
    floor = rng.integers(1, 25, n)
    d_eff = dict(zip(sorted(set(district)), rng.normal(0, 0.25, 12), strict=True))
    log_price = (
        np.log(140_000)
        + np.log(area)
        + np.vectorize(d_eff.get)(district)
        - 0.004 * age
        + 0.01 * np.minimum(floor, 10)
        + rng.normal(0, 0.12, n)
    )
    price = np.exp(log_price).round(-3)
    df = pd.DataFrame({"deal_id": np.arange(n) + 70_000, "area": area, "rooms": rooms, "district": district})
    df["building_age"] = age
    df["floor"] = floor
    df["contract_sum"] = (price * (1 + rng.normal(0, 0.01, n))).round(-3)
    df["price"] = price
    train, test = _split(df, 2500, rng)
    # В test договор ещё не подписан: поле заполнено значениями из шаблона выгрузки — то же распределение, без связи
    test["contract_sum"] = rng.permutation(train["contract_sum"].to_numpy())[: len(test)]
    answers = test[["deal_id", "price"]].copy()
    test = test.drop(columns=["price"])
    sample = answers.assign(price=0.0)
    md = """# Цена сделки

Оцените итоговую цену сделки с квартирой (`price`, руб.).

Данные: `train.csv` — прошлые сделки, `test.csv` — новые, `sample_submission.csv` — пример ответа.
Колонки: площадь, число комнат, район, возраст дома, этаж и `contract_sum` из системы договоров.

Метрика — MAE (средняя абсолютная ошибка), меньше — лучше. Ответ: `deal_id`, `price`.
"""
    _save(
        "leak_target_copy_mae",
        {"train.csv": (train, {}), "test.csv": (test, {}), "sample_submission.csv": (sample, {})},
        md,
        answers,
        {
            "metric": "mae",
            "greater_is_better": False,
            "task_type": "regression",
            "id": "deal_id",
            "targets": ["price"],
            "submission_columns": ["deal_id", "price"],
            "leak_columns": ["contract_sum"],
            "leak_kind": "копия target: corr ≈ 0.999 в train; в test то же распределение, но без связи с target",
            "checks": "признак-копия target, в test заполнен и по распределению не отличается: состязательная "
            "проверка train/test его не видит, видно только «слишком хорошо» в train",
        },
    )


def leak_future_window_rmse():
    """Будущее время: среднее продаж по окну [d−1, d+1] (включает день прогноза и завтра) — в test заморожено."""
    rng = _rng(18)
    stores = [f"S{i:02d}" for i in range(40)]
    days = pd.date_range("2024-01-01", periods=730 + 61, freq="D")
    st = np.repeat(stores, len(days))
    dt = np.tile(days, len(stores))
    base = np.repeat(rng.lognormal(4.5, 0.4, len(stores)), len(days))
    dow = pd.DatetimeIndex(dt).dayofweek.to_numpy()
    doy = pd.DatetimeIndex(dt).dayofyear.to_numpy()
    promo = (rng.random(len(st)) < 0.15).astype(int)
    temp = (10 - 12 * np.cos(2 * np.pi * doy / 365) + rng.normal(0, 3, len(st))).round(1)
    mean = base * (1 + 0.25 * (dow >= 5)) * (1 + 0.15 * np.sin(2 * np.pi * doy / 365)) * (1 + 0.35 * promo)
    sales = rng.poisson(mean).astype(float)
    df = pd.DataFrame({"store": st, "date": pd.DatetimeIndex(dt).strftime("%Y-%m-%d"), "promo": promo, "temp": temp})
    df["weekday"] = dow
    df["month"] = pd.DatetimeIndex(dt).month
    df["sales"] = sales
    # Витрина считала «среднее за 3 дня» центрированным окном: в него входят день прогноза и следующий день
    df["sales_3d_avg"] = (
        df.groupby("store")["sales"].transform(lambda s: s.rolling(3, center=True, min_periods=1).mean()).round(2)
    )
    df.insert(0, "row_id", np.arange(len(df)) + 1)
    cut = days[730]
    is_test = pd.to_datetime(df["date"]) >= cut
    train, test = df[~is_test].reset_index(drop=True), df[is_test].reset_index(drop=True)
    # В test будущих продаж нет: витрина повторяет последнее известное (честное, только прошлое) среднее магазина
    last = train.groupby("store")["sales"].apply(lambda s: s.iloc[-7:].mean()).round(2)
    test["sales_3d_avg"] = test["store"].map(last)
    answers = test[["row_id", "sales"]].copy()
    test = test.drop(columns=["sales"])
    sample = answers.assign(sales=0.0)
    md = """# Продажи магазинов по дням

Нужно предсказать продажи (`sales`, штук) каждого магазина на каждый день следующих двух месяцев.

Файлы: `train.csv` — история за два года, `test.csv` — дни, на которые нужен прогноз, `sample_submission.csv`.
Колонки: магазин, дата, день недели, месяц, флаг промо-акции, температура и `sales_3d_avg` — среднее продаж за 3 дня из витрины.

Метрика — RMSE, меньше — лучше. Ответ: `row_id`, `sales`.
"""
    _save(
        "leak_future_window_rmse",
        {"train.csv": (train, {}), "test.csv": (test, {}), "sample_submission.csv": (sample, {})},
        md,
        answers,
        {
            "metric": "rmse",
            "greater_is_better": False,
            "task_type": "regression",
            "id": "row_id",
            "targets": ["sales"],
            "submission_columns": ["row_id", "sales"],
            "leak_columns": ["sales_3d_avg"],
            "leak_kind": "будущее время: окно признака заходит за дату строки; в test заморожено на дате отсечки",
            "checks": "утечка из будущего: валидация по времени её НЕ ловит (окно строки внутри фолда), в test "
            "признак постоянен по магазину — модель на нём проваливается",
        },
    )


# ================================================================== набор «uncovered»: чего нет ни в public, ни выше


def multilabel_3targets_mean_auc():
    """Три бинарных таргета (мультиметка) с разной частотой, метрика — средний ROC-AUC по колонкам."""
    rng = _rng(19)
    n = 14000
    x = rng.normal(0, 1, (n, 8))
    segment = rng.choice(["mass", "affluent", "private", "youth"], n, p=[0.6, 0.2, 0.05, 0.15])
    seg = {"mass": 0.0, "affluent": 0.6, "private": 1.2, "youth": -0.5}
    s = np.vectorize(seg.get)(segment)
    shared = 0.6 * x[:, 0] + 0.4 * s  # общий фактор: таргеты коррелированы
    targets = {
        "credit_card": -0.8 + shared + 0.7 * x[:, 1] - 0.4 * x[:, 2] ** 2,
        "mortgage": -2.4 + shared + 0.9 * x[:, 3] + 0.5 * (x[:, 4] > 0.5),
        "deposit": -3.6 + 0.8 * shared + 0.8 * x[:, 5] * x[:, 6],
    }
    df = pd.DataFrame(x.round(4), columns=[f"x{i}" for i in range(8)])
    df.insert(0, "client", [f"C{i:06d}" for i in rng.permutation(n)])
    df["segment"] = segment
    for name, z in targets.items():
        df[name] = (rng.random(n) < _sigmoid(z)).astype(int)
    train, test = _split(df, 4000, rng)
    names = list(targets)
    answers = test[["client", *names]].copy()
    test = test.drop(columns=names)
    sample = answers.assign(**{t: 0.5 for t in names})
    md = """# Какие продукты подключит клиент

Для каждого клиента нужно оценить вероятность подключения трёх продуктов: кредитной карты (`credit_card`),
ипотеки (`mortgage`) и вклада (`deposit`). Клиент может подключить несколько продуктов сразу.

Файлы: `train.csv`, `test.csv`, `sample_submission.csv`.

Метрика — ROC-AUC, посчитанный отдельно по каждой из трёх колонок и усреднённый (больше — лучше).
Ответ: `client` и три колонки вероятностей.
"""
    _save(
        "multilabel_3targets_mean_auc",
        {"train.csv": (train, {}), "test.csv": (test, {}), "sample_submission.csv": (sample, {})},
        md,
        answers,
        {
            "metric": "mean_roc_auc",
            "greater_is_better": True,
            "task_type": "binary",
            "id": "client",
            "targets": names,
            "submission_columns": ["client", *names],
            "positive": "1",
            "checks": "три бинарных таргета в одном ответе (≈30% / 10% / 3% положительных), вероятности, "
            "средний AUC: агенту нужна модель на каждый таргет, а не один таргет + копии",
        },
    )


def ranking_query_map5():
    """Ранжирование с query_id: релевантность зависит от документа относительно других документов запроса."""
    rng = _rng(20)
    n_q = 2000
    sizes = rng.integers(5, 41, n_q)
    q = np.repeat(np.arange(n_q) + 1, sizes)
    n = len(q)
    q_shift = np.repeat(rng.normal(0, 2.0, n_q), sizes)  # у запросов свой уровень bm25 — сырой порог не работает
    q_len = np.repeat(rng.integers(1, 8, n_q), sizes)
    bm25 = (q_shift + rng.gamma(2.0, 1.5, n)).round(3)
    ctr = rng.beta(1, 12, n).round(4)
    fresh = rng.integers(0, 2000, n)
    doc_len = rng.lognormal(6, 0.7, n).round()
    cat_match = (rng.random(n) < 0.3).astype(int)
    score = (bm25 - q_shift) + 25 * ctr + 0.8 * cat_match - 0.0006 * fresh + rng.normal(0, 1.0, n)
    df = pd.DataFrame({"query_id": q, "bm25": bm25, "ctr": ctr, "freshness_days": fresh, "doc_len": doc_len})
    df["query_len"] = q_len
    df["category_match"] = cat_match
    df["relevant"] = 0
    order = df.assign(s=score).groupby("query_id")["s"].rank(ascending=False, method="first")
    df.loc[(order <= np.ceil(df.groupby("query_id")["query_id"].transform("size") * 0.15)).to_numpy(), "relevant"] = 1
    df.insert(0, "id", np.arange(n) + 1)
    test_q = rng.choice(np.arange(n_q) + 1, n_q // 4, replace=False)  # test — другие запросы
    is_test = df["query_id"].isin(test_q)
    train, test = df[~is_test].reset_index(drop=True), df[is_test].reset_index(drop=True)
    answers = test[["id", "relevant", "query_id"]].copy()
    test = test.drop(columns=["relevant"])
    sample = test[["id"]].assign(score=0.0)
    md = """# Ранжирование документов в поиске

Для каждого запроса (`query_id`) нужно упорядочить найденные документы так, чтобы релевантные (`relevant` = 1)
оказались наверху. В test — новые запросы, которых нет в train.

Файлы: `train.csv`, `test.csv`, `sample_submission.csv`.

Ответ — `id` и `score`: чем больше score, тем выше документ в выдаче своего запроса; шкала не важна.
Метрика — MAP@5: средняя точность по первым пяти документам, считается по каждому запросу и усредняется
(больше — лучше).
"""
    _save(
        "ranking_query_map5",
        {"train.csv": (train, {}), "test.csv": (test, {}), "sample_submission.csv": (sample, {})},
        md,
        answers,
        {
            "metric": "map@5",
            "greater_is_better": True,
            "task_type": "regression",
            "id": "id",
            "group": "query_id",
            "targets": ["relevant"],
            "answer_column": "score",
            "submission_columns": ["id", "score"],
            "checks": "ранжирование по запросам: колонка ответа ≠ имени таргета, test — новые query_id (нужны "
            "группы в CV), уровень bm25 свой у каждого запроса, метрика по запросам",
        },
    )


def big_600k_time_dups_auc():
    """Большая таблица: 600k строк, дата строкой «дд.мм.гггг», 15% полных дубликатов, редкий класс, test — позже."""
    rng = _rng(21)
    n_train, n_test = 600_000, 150_000
    n = n_train + n_test
    day = np.sort(rng.integers(0, 820, n))  # 730 дней train + 90 дней test, строки по времени
    day[n_train:] = np.maximum(day[n_train:], 731)
    day[:n_train] = np.minimum(day[:n_train], 730)
    x = rng.normal(0, 1, (n, 12)).astype(np.float32).round(3)
    merchant = rng.zipf(1.6, n) % 40_000
    mcc = rng.choice([f"mcc_{i}" for i in range(60)], n)
    amount = rng.lognormal(7, 1.1, n).round(2)
    m_risk = rng.normal(0, 1, 40_000)
    drift = day / 820  # доля положительных растёт со временем
    z = -4.3 + 0.8 * x[:, 0] + 0.5 * x[:, 1] * x[:, 2] + 0.35 * m_risk[merchant] + 0.3 * np.log(amount / 1000) + drift
    y = (rng.random(n) < _sigmoid(z)).astype(np.int8)
    df = pd.DataFrame(x, columns=[f"v{i}" for i in range(12)])
    df.insert(0, "op_date", (pd.Timestamp("2023-01-01") + pd.to_timedelta(day, "D")).strftime("%d.%m.%Y"))
    df["merchant"] = merchant
    df["mcc"] = mcc
    df["amount"] = amount
    df["fraud"] = y
    # 15% строк train — полные повторы других строк train (повторные списания): в случайной CV дубль в train и valid
    dup_from = rng.choice(n_train, int(0.15 * n_train), replace=False)
    dup_to = rng.choice(n_train, len(dup_from), replace=False)
    src = np.arange(n)
    src[dup_to] = dup_from
    df = df.iloc[src].reset_index(drop=True)  # строками целиком — типы колонок сохраняются
    df.insert(0, "op_id", np.arange(n) + 10_000_000)
    train, test = df.iloc[:n_train].reset_index(drop=True), df.iloc[n_train:].reset_index(drop=True)
    answers = test[["op_id", "fraud"]].copy()
    test = test.drop(columns=["fraud"])
    sample = answers.assign(fraud=0.0)
    md = """# Мошеннические операции

Оцените вероятность того, что операция по карте мошенническая (`fraud` = 1).

Файлы: `train.csv` — операции за два года, `test.csv` — операции следующего квартала, `sample_submission.csv`.
Дата операции — `op_date` (дд.мм.гггг), `merchant` — код торговой точки, `mcc` — категория, `amount` — сумма.

Метрика — ROC-AUC (больше — лучше). Ответ: `op_id`, `fraud`.
"""
    _save(
        "big_600k_time_dups_auc",
        {"train.csv": (train, {}), "test.csv": (test, {}), "sample_submission.csv": (sample, {})},
        md,
        answers,
        {
            "metric": "roc_auc",
            "greater_is_better": True,
            "task_type": "binary",
            "id": "op_id",
            "targets": ["fraud"],
            "submission_columns": ["op_id", "fraud"],
            "positive": "1",
            "checks": "≥ 500k строк: схема по времени (дата строкой дд.мм.гггг, test — следующий квартал) и группы "
            "дубликатов на большой таблице, подвыборка CV (MAX_CV_ROWS), id-подобный merchant (40k значений), "
            "≈2–3% положительных, дрейф доли во времени; время и RAM за бюджет",
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


# Отдельные наборы, а не TASKS: гейт (scripts/gate.py) считает нули ровно на 15 задачах runs/night/alien
LEAK = {f.__name__: f for f in [leak_idlike_regno_auc, leak_target_copy_mae, leak_future_window_rmse]}
UNCOVERED = {f.__name__: f for f in [multilabel_3targets_mean_auc, ranking_query_map5, big_600k_time_dups_auc]}
SETS = {"alien": TASKS, "leak": LEAK, "uncovered": UNCOVERED}


def main() -> None:
    global OUT
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", nargs="*", help="имена задач")
    parser.add_argument("--set", choices=sorted(SETS), default="alien", help="набор задач (папка runs/night/<набор>)")
    args = parser.parse_args()
    OUT = ROOT / "runs" / "night" / args.set
    for name, make in SETS[args.set].items():
        if not args.only or name in args.only:
            make()


if __name__ == "__main__":
    main()
