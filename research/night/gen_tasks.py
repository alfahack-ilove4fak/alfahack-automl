"""Ночной аудит: генератор «чужих» синтетических задач (не похожих на public). Автор — аудит @marenkov17.

    uv run python research/night/gen_tasks.py [--out runs/night/tasks] [--big]

Seed = сумма кодов букв ника «marenkov17» = 971. Каждая задача — папка с task.md и CSV, как на платформе.
Ответы и «паспорт» задачи (настоящие target, id, метрика, формат) — в <out>/_answers/<задача>/, агент их не видит.
--big добавляет задачу на 1.2 млн строк (долго генерируется и считается).
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

SEED = sum(map(ord, "marenkov17"))  # 971


def signal(rng: np.random.Generator, n: int, k: int = 8) -> tuple[pd.DataFrame, np.ndarray]:
    """k числовых признаков и линейно-нелинейный сигнал по ним."""
    x = rng.normal(size=(n, k))
    s = 1.5 * x[:, 0] - x[:, 1] + 0.8 * x[:, 2] * x[:, 3] + np.sin(2 * x[:, 4]) + 0.5 * (x[:, 5] > 0.5)
    return pd.DataFrame(x, columns=[f"f{i}" for i in range(k)]), s


def split(df: pd.DataFrame, frac: float = 0.25, seed: int = SEED) -> tuple[pd.DataFrame, pd.DataFrame]:
    test = df.sample(frac=frac, random_state=seed)
    return df.drop(test.index), test


def save_task(
    out: Path,
    name: str,
    md: str,
    train: pd.DataFrame,
    test_full: pd.DataFrame,
    meta: dict,
    sample: pd.DataFrame | None,
    csv_kw: dict | None = None,
    sub_dir: str = "",
) -> None:
    """test_full — test с ответами: колонки ответов (meta['targets']) уходят в _answers, из test их убираем."""
    csv_kw = csv_kw or {}
    task_dir = out / name
    data_dir = task_dir / sub_dir if sub_dir else task_dir
    data_dir.mkdir(parents=True, exist_ok=True)
    (task_dir / "task.md").write_text(md, encoding="utf-8")
    test = test_full.drop(
        columns=[c for c in meta["targets"] if c in test_full.columns and not meta.get("keep_in_test")]
    )
    train.to_csv(data_dir / "train.csv", index=False, **csv_kw)
    test.to_csv(data_dir / "test.csv", index=False, **csv_kw)
    if sample is not None:
        sample.to_csv(
            data_dir / "sample_submission.csv", index=False, **{k: v for k, v in csv_kw.items() if k == "sep"}
        )
    ans = out / "_answers" / name
    ans.mkdir(parents=True, exist_ok=True)
    test_full.to_pickle(ans / "test_full.pkl")
    train.to_pickle(ans / "train.pkl")
    (ans / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")


def t01_semicolon(out: Path, rng):
    """; + запятая-десятичная + BOM + русские заголовки с пробелами; MAE словами."""
    x, s = signal(rng, 6000)
    df = x.rename(
        columns={
            "f0": "Площадь, м2",
            "f1": "Этаж квартиры",
            "f2": "Расстояние до метро (км)",
            "f3": "Год постройки",
            "f4": "Индекс района",
            "f5": "Доля окон на юг",
            "f6": "Шум",
            "f7": "Шум 2",
        }
    )
    df.insert(0, "Номер объявления", np.arange(100000, 106000))
    df["Цена, млн руб"] = (10 + 3 * s + rng.normal(0, 1, len(df))).round(3)
    train, test = split(df)
    md = """# Оценка стоимости квартир

Данные выгружены из Excel: разделитель столбцов — точка с запятой, десятичный разделитель — запятая,
кодировка UTF-8 с BOM. Заголовки на русском.

Нужно предсказать столбец «Цена, млн руб» для каждого объявления из test.csv.
Качество считаем как среднюю абсолютную ошибку прогноза цены — чем она меньше, тем лучше.

Ответ — submission.csv с колонками «Номер объявления» и «Цена, млн руб» (как в sample_submission.csv).
"""
    sample = test[["Номер объявления"]].assign(**{"Цена, млн руб": 0.0})
    meta = {
        "targets": ["Цена, млн руб"],
        "id": "Номер объявления",
        "metric": "mae",
        "greater": False,
        "kind": "value",
        "submission": ["Номер объявления", "Цена, млн руб"],
        "csv": {"sep": ";", "decimal": ","},
    }
    save_task(
        out,
        "t01_semicolon_bom_ru",
        md,
        train,
        test,
        meta,
        sample,
        csv_kw={"sep": ";", "decimal": ",", "encoding": "utf-8-sig"},
    )


def t02_id_last_quotes(out: Path, rng):
    """id — последний столбец, target посередине, строки в кавычках с запятыми, английский текст, ROC-AUC."""
    x, s = signal(rng, 8000)
    city = rng.choice(["Moscow, RU", "Kazan, RU", "Paris, FR", 'New "York", US'], len(x))
    df = x.copy()
    df["city"] = city
    p = 1 / (1 + np.exp(-(s - 0.5 + (city == "Paris, FR"))))
    df.insert(4, "churned", (rng.random(len(x)) < p).astype(int))
    df["customer_ref"] = [f"C{i:06d}" for i in rng.permutation(len(x))]
    train, test = split(df)
    md = """# Customer churn

Predict whether a customer churns. Files: `train.csv`, `test.csv`, `sample_submission.csv`.
The target column is `churned` (1 = churned). Customer identifier is `customer_ref`.

Submissions are evaluated on area under the ROC curve between the predicted probability and the observed target.
"""
    sample = test[["customer_ref"]].assign(churned=0.5)
    meta = {
        "targets": ["churned"],
        "id": "customer_ref",
        "metric": "roc_auc",
        "greater": True,
        "kind": "proba",
        "submission": ["customer_ref", "churned"],
    }
    save_task(out, "t02_id_last_quotes_en_auc", md, train, test, meta, sample)


def t03_no_id(out: Path, rng):
    """В train/test нет id; sample_submission — индекс строки; R² словами; target не последний."""
    x, s = signal(rng, 5000)
    df = x.copy()
    df.insert(2, "y", 5 * s + rng.normal(0, 2, len(x)))
    train, test = split(df)
    test = test.reset_index(drop=True)
    md = """# Прогноз выхода продукции

В train.csv и test.csv нет идентификатора: строки test нумеруются с нуля по порядку (колонка `index` в ответе).
Целевая переменная — `y`. Метрика — коэффициент детерминации (R²), больше — лучше.
"""
    sample = pd.DataFrame({"index": np.arange(len(test)), "y": 0.0})
    meta = {
        "targets": ["y"],
        "id": "index",
        "id_is_rowno": True,
        "metric": "r2",
        "greater": True,
        "kind": "value",
        "submission": ["index", "y"],
    }
    save_task(out, "t03_no_id_r2", md, train, test, meta, sample)


def t04_big(out: Path, rng):
    """1.2 млн строк × 20 признаков, logloss."""
    n = 1_200_000
    x, s = signal(rng, n, k=16)
    df = x.astype(np.float32)
    df["cat_a"] = rng.integers(0, 500, n)
    df["cat_b"] = rng.choice(list("ABCDEFGH"), n)
    df["num_noise1"] = rng.normal(size=n).astype(np.float32)
    df["num_noise2"] = rng.normal(size=n).astype(np.float32)
    p = 1 / (1 + np.exp(-(s - 1)))
    df["click"] = (rng.random(n) < p).astype(int)
    df.insert(0, "id", np.arange(n))
    train, test = split(df, frac=0.2)
    md = """# Клики по баннерам

Большие данные: ~1 млн строк. Предскажите вероятность клика `click` для каждого `id` из test.csv.
Метрика — log loss (бинарная кросс-энтропия), меньше — лучше.
"""
    sample = test[["id"]].assign(click=0.5)
    meta = {
        "targets": ["click"],
        "id": "id",
        "metric": "logloss",
        "greater": False,
        "kind": "proba",
        "submission": ["id", "click"],
    }
    save_task(out, "t04_big_1m_logloss", md, train, test, meta, sample)


def t05_wide(out: Path, rng):
    """3000 строк × 1500 признаков (почти все шум), RMSE."""
    n, k = 3000, 1500
    x = rng.normal(size=(n, k)).astype(np.float32)
    s = x[:, 0] * 2 - x[:, 10] + x[:, 100] * x[:, 200] + 0.5 * x[:, 1000]
    df = pd.DataFrame(x, columns=[f"g{i}" for i in range(k)])
    df.insert(0, "id", np.arange(n))
    df["target"] = s + rng.normal(0, 0.5, n)
    train, test = split(df)
    md = """# Экспрессия генов

1500 признаков-генов, 3000 образцов. Предскажите `target`. Метрика — RMSE (меньше — лучше).
"""
    sample = test[["id"]].assign(target=0.0)
    meta = {
        "targets": ["target"],
        "id": "id",
        "metric": "rmse",
        "greater": False,
        "kind": "value",
        "submission": ["id", "target"],
    }
    save_task(out, "t05_wide_1500cols_rmse", md, train, test, meta, sample)


def t06_dirty(out: Path, rng):
    """100% NaN, константы, текст, даты разных форматов, огромные/отрицательные значения, NaN в target,
    дубль имени колонки, утечка (колонка = target только в train). MAPE."""
    n = 8000
    x, s = signal(rng, n, k=6)
    df = x.copy()
    df["all_nan"] = np.nan
    df["const"] = 7
    df["const_str"] = "same"
    words = np.array(["red car fast", "blue truck slow", "green bike", "old red truck", "new blue car"])
    df["comment"] = rng.choice(words, n)
    dates = pd.to_datetime("2020-01-01") + pd.to_timedelta(rng.integers(0, 1000, n), unit="D")
    fmt = rng.integers(0, 3, n)
    df["date"] = [d.strftime(["%Y-%m-%d", "%d.%m.%Y", "%m/%d/%Y"][f]) for d, f in zip(dates, fmt, strict=True)]
    df["huge"] = rng.normal(size=n) * 1e15
    df["neg"] = -np.abs(rng.normal(size=n)) * 1e6
    y = 100 + 20 * s + 10 * (df["comment"].str.contains("red")) + rng.normal(0, 5, n)
    y = np.clip(y, 5, None)
    df["amount"] = y.round(2)
    df.insert(0, "id", np.arange(n))
    train, test = split(df)
    train = train.copy()
    train.loc[train.sample(frac=0.03, random_state=1).index, "amount"] = np.nan
    train["amount_check"] = train["amount"]  # утечка: копия target, в test её смысл другой
    test = test.copy()
    test["amount_check"] = np.nan
    # дубль имени колонки: в CSV будут два столбца «f0»
    train = train.rename(columns={"f5": "f0_dup"})
    test = test.rename(columns={"f5": "f0_dup"})
    md = """# Сумма страховой выплаты

Предскажите `amount` (сумма выплаты) по полису. Метрика — MAPE (средняя абсолютная процентная ошибка), меньше — лучше.
Внимание: `amount_check` — сумма, сверенная бухгалтерией ПОСЛЕ выплаты; в test её нет.
"""
    sample = test[["id"]].assign(amount=0.0)
    meta = {
        "targets": ["amount"],
        "id": "id",
        "metric": "mape",
        "greater": False,
        "kind": "value",
        "submission": ["id", "amount"],
    }
    save_task(out, "t06_dirty_leak_mape", md, train, test, meta, sample)
    # дубль имени: переписываем заголовок CSV вручную
    for f in ("train.csv", "test.csv"):
        p = out / "t06_dirty_leak_mape" / f
        lines = p.read_text(encoding="utf-8").split("\n", 1)
        p.write_text(lines[0].replace("f0_dup", "f0") + "\n" + lines[1], encoding="utf-8")


def t07_50cls(out: Path, rng):
    """50 строковых классов, accuracy."""
    n = 15000
    x, s = signal(rng, n, k=10)
    cls = np.digitize(s + 0.5 * x["f6"].to_numpy(), np.quantile(s + 0.5 * x["f6"].to_numpy(), np.linspace(0, 1, 51)[1:-1]))
    df = x.copy()
    df.insert(0, "id", np.arange(n))
    df["species"] = [f"sp_{c:02d}" for c in cls]
    train, test = split(df)
    md = """# Определение вида

50 видов. Предскажите `species`. Метрика — accuracy (доля верных ответов).
"""
    sample = test[["id"]].assign(species="sp_00")
    meta = {
        "targets": ["species"],
        "id": "id",
        "metric": "accuracy",
        "greater": True,
        "kind": "label",
        "submission": ["id", "species"],
    }
    save_task(out, "t07_50classes_accuracy", md, train, test, meta, sample)


def t08_rare(out: Path, rng):
    """0.1% положительного класса, PR-AUC."""
    n = 40000
    x, s = signal(rng, n)
    logit = 3 * s - 13
    df = x.copy()
    df.insert(0, "id", np.arange(n))
    df["fraud"] = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    train, test = split(df)
    md = f"""# Мошенничество

Положительных примеров очень мало ({df['fraud'].mean():.2%}). Предскажите вероятность `fraud`.
Метрика — PR-AUC (average precision), больше — лучше.
"""
    sample = test[["id"]].assign(fraud=0.0)
    meta = {
        "targets": ["fraud"],
        "id": "id",
        "metric": "pr_auc",
        "greater": True,
        "kind": "proba",
        "submission": ["id", "fraud"],
    }
    save_task(out, "t08_rare_0p1_prauc", md, train, test, meta, sample)


def t09_kappa_ru(out: Path, rng):
    """Порядковые классы 0..5, «каппа Коэна» по-русски (невзвешенная)."""
    n = 8000
    x, s = signal(rng, n)
    grade = np.digitize(s + rng.normal(0, 0.7, n), [-3, -1.5, 0, 1.5, 3])
    df = x.copy()
    df.insert(0, "id", np.arange(n))
    df["grade"] = grade
    train, test = split(df)
    md = """# Сорт зерна

Предскажите сорт `grade` (целое от 0 до 5). Метрика — каппа Коэна между ответом и истинным сортом.
"""
    sample = test[["id"]].assign(grade=0)
    meta = {
        "targets": ["grade"],
        "id": "id",
        "metric": "cohen_kappa",
        "greater": True,
        "kind": "label",
        "submission": ["id", "grade"],
    }
    save_task(out, "t09_ordinal_kappa_ru", md, train, test, meta, sample)


def t10_f2(out: Path, rng):
    """F2-мера, метки yes/no, дисбаланс 8%."""
    n = 12000
    x, s = signal(rng, n)
    df = x.copy()
    df.insert(0, "id", np.arange(n))
    df["default"] = np.where(rng.random(n) < 1 / (1 + np.exp(-(1.5 * s - 4.5))), "yes", "no")
    train, test = split(df)
    md = """# Дефолт по кредиту

Предскажите `default` (yes/no). Пропустить дефолт дороже, чем ложная тревога, поэтому метрика — F2-мера
(F-beta с beta=2) по классу yes. Ответ — метка yes или no.
"""
    sample = test[["id"]].assign(default="no")
    meta = {
        "targets": ["default"],
        "id": "id",
        "metric": "f2",
        "greater": True,
        "kind": "label",
        "pos": "yes",
        "submission": ["id", "default"],
    }
    save_task(out, "t10_f2_labels_yesno", md, train, test, meta, sample)


def t11_multitarget(out: Path, rng):
    """Два регрессионных таргета, метрика — среднее RMSE."""
    n = 6000
    x, s = signal(rng, n)
    df = x.copy()
    df.insert(0, "id", np.arange(n))
    df["temp_out"] = 3 * s + rng.normal(0, 1, n)
    df["pressure_out"] = 100 + 10 * x["f6"] - 5 * s + rng.normal(0, 1, n)
    train, test = split(df)
    md = """# Режим реактора

Предскажите ДВЕ величины: `temp_out` и `pressure_out`. Метрика — среднее RMSE по двум колонкам (меньше — лучше).
"""
    sample = test[["id"]].assign(temp_out=0.0, pressure_out=0.0)
    meta = {
        "targets": ["temp_out", "pressure_out"],
        "id": "id",
        "metric": "mean_rmse",
        "greater": False,
        "kind": "value",
        "submission": ["id", "temp_out", "pressure_out"],
    }
    save_task(out, "t11_multitarget_rmse", md, train, test, meta, sample)


def t12_proba_cols(out: Path, rng):
    """4 класса, ответ — колонка на класс (вероятности), multi logloss; в test новые категории."""
    n = 10000
    x, s = signal(rng, n)
    shop = rng.integers(0, 300, n)
    cls = np.digitize(s + (shop % 7 == 0), [-1, 0.5, 2])
    df = x.copy()
    df.insert(0, "id", np.arange(n))
    df["shop"] = [f"shop{v}" for v in shop]
    df["segment"] = np.array(["low", "mid", "high", "vip"])[cls]
    train, test = split(df)
    test = test.copy()
    test.loc[test.sample(frac=0.3, random_state=2).index, "shop"] = "shop_new"
    md = """# Сегмент клиента

Для каждого id предскажите вероятности четырёх сегментов. Ответ: `id,high,low,mid,vip`.
Метрика — multi-class log loss.
"""
    sample = test[["id"]].assign(high=0.25, low=0.25, mid=0.25, vip=0.25)
    meta = {
        "targets": ["segment"],
        "id": "id",
        "metric": "logloss",
        "greater": False,
        "kind": "proba_cols",
        "submission": ["id", "high", "low", "mid", "vip"],
    }
    save_task(out, "t12_proba_columns_newcats", md, train, test, meta, sample)


def t13_ids_overlap(out: Path, rng):
    """В test повторяются объекты train (те же id и признаки), SMAPE."""
    n = 6000
    x, s = signal(rng, n)
    df = x.copy()
    df.insert(0, "id", np.arange(n))
    df["sales"] = np.exp(1 + 0.5 * s + rng.normal(0, 0.2, n)).round(3)
    train, test = split(df)
    dup = train.sample(400, random_state=3)
    test = pd.concat([test, dup]).sample(frac=1, random_state=4)
    md = """# Продажи

Предскажите `sales`. Часть объектов test встречалась в train. Метрика — SMAPE (меньше — лучше).
"""
    sample = test[["id"]].assign(sales=1.0)
    meta = {
        "targets": ["sales"],
        "id": "id",
        "metric": "smape",
        "greater": False,
        "kind": "value",
        "submission": ["id", "sales"],
        "dup_ids": True,
    }
    save_task(out, "t13_test_ids_from_train_smape", md, train, test, meta, sample)


def t14_noise_paths(out: Path, rng):
    """Неверные пути, файлы во вложенной папке, нет sample_submission, шум в тексте, RMSLE в конце."""
    n = 7000
    x, s = signal(rng, n)
    df = x.copy()
    df["trip_id"] = rng.permutation(n) + 50000
    df["duration_sec"] = np.exp(5 + 0.6 * s + rng.normal(0, 0.3, n)).round(1)
    train, test = split(df)
    md = """# Taxi trip duration (v2)

Data lives in `data/v2/train_final.csv` and `data/v2/test_final.csv` (paths may differ in your copy).

Some background: last year we used accuracy and then MAE, both were misleading. Teams reported R2 too.
Column `trip_id` identifies a trip. Predict `duration_sec`.

Lots of unrelated notes: the dataset was collected by the city; F1 is irrelevant here; AUC is not applicable.

Final ranking: submissions are scored by RMSLE.
"""
    meta = {
        "targets": ["duration_sec"],
        "id": "trip_id",
        "metric": "rmsle",
        "greater": False,
        "kind": "value",
        "submission": ["trip_id", "duration_sec"],
    }
    save_task(out, "t14_noise_paths_rmsle_nosample", md, train, test, meta, None, sub_dir="input/raw")


def t15_brier(out: Path, rng):
    """Brier score — нет в реестре."""
    n = 8000
    x, s = signal(rng, n)
    df = x.copy()
    df.insert(0, "id", np.arange(n))
    df["rain"] = (rng.random(n) < 1 / (1 + np.exp(-s))).astype(int)
    train, test = split(df)
    md = """# Дождь завтра

Предскажите вероятность дождя `rain`. Метрика — Brier score (средний квадрат ошибки вероятности), меньше — лучше.
"""
    sample = test[["id"]].assign(rain=0.5)
    meta = {
        "targets": ["rain"],
        "id": "id",
        "metric": "brier",
        "greater": False,
        "kind": "proba",
        "submission": ["id", "rain"],
    }
    save_task(out, "t15_brier_proba", md, train, test, meta, sample)


def t16_quantile(out: Path, rng):
    """Квантильная потеря 0.9 словами."""
    n = 6000
    x, s = signal(rng, n)
    df = x.copy()
    df.insert(0, "id", np.arange(n))
    df["demand"] = (50 + 10 * s + rng.gamma(2, 8, n)).round(1)
    train, test = split(df)
    md = """# Спрос на складе

Нам важно не остаться без товара, поэтому прогнозируем 90-й перцентиль спроса `demand`.
Оценка — quantile loss (pinball) для уровня 0.9; меньше — лучше.
"""
    sample = test[["id"]].assign(demand=0.0)
    meta = {
        "targets": ["demand"],
        "id": "id",
        "metric": "pinball@0.9",
        "greater": False,
        "kind": "value",
        "submission": ["id", "demand"],
    }
    save_task(out, "t16_quantile90", md, train, test, meta, sample)


def t17_balacc_words(out: Path, rng):
    """Числовые классы 1..3, balanced accuracy, направление словами, нет sample и словаря, test без id-колонки 'id'."""
    n = 6000
    x, s = signal(rng, n)
    df = x.copy()
    df.insert(3, "row_key", rng.permutation(n) + 1)
    df["quality"] = np.digitize(s, [-2.5, 2.0]) + 1
    train, test = split(df)
    md = """# Качество партии

Класс качества `quality` (1, 2 или 3). Идентификатор строки — `row_key`.
Оценивается сбалансированная точность (balanced accuracy): чем выше, тем лучше. Классы несбалансированы.
"""
    meta = {
        "targets": ["quality"],
        "id": "row_key",
        "metric": "balanced_accuracy",
        "greater": True,
        "kind": "label",
        "submission": ["row_key", "quality"],
    }
    save_task(out, "t17_balanced_acc_numeric_classes", md, train, test, meta, None)


GENERATORS = [
    t01_semicolon,
    t02_id_last_quotes,
    t03_no_id,
    t05_wide,
    t06_dirty,
    t07_50cls,
    t08_rare,
    t09_kappa_ru,
    t10_f2,
    t11_multitarget,
    t12_proba_cols,
    t13_ids_overlap,
    t14_noise_paths,
    t15_brier,
    t16_quantile,
    t17_balacc_words,
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/night/tasks")
    ap.add_argument("--big", action="store_true")
    ap.add_argument("--only", default="")
    args = ap.parse_args()
    out = Path(args.out)
    gens = GENERATORS + ([t04_big] if args.big else [])
    for i, gen in enumerate(gens):
        if args.only and args.only not in gen.__name__:
            continue
        rng = np.random.default_rng(SEED + i)
        target = [p for p in out.glob(gen.__name__ + "*")]
        for p in target:
            shutil.rmtree(p)
        gen(out, rng)
        print("ok", gen.__name__)


if __name__ == "__main__":
    main()
