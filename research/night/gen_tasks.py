"""Ночной аудит: генератор «чужих» синтетических задач (непохожих на public и practice).

    uv run python research/night/gen_tasks.py --out /tmp/night_tasks [--nick thenlaooo] [--only имя,имя]

seed = сумма кодов букв ника (thenlaooo → 969): у разных людей — разные данные, у одного — воспроизводимые.
Для каждой задачи: <out>/<имя>/{task.md, train.csv, test.csv, sample_submission.csv}
и <out>/_truth/<имя>.json (правильный разбор) + <имя>.csv (ответы на test) — вне каталога задачи, агент их не видит.
Данные и тексты — выдуманные, цель — сломать разбор формата/метрики/объёма, а не качество модели.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

TASKS: dict = {}


def task(fn):
    TASKS[fn.__name__.removeprefix("t_")] = fn
    return fn


def _split(df: pd.DataFrame, n_train: int):
    return df.iloc[:n_train].reset_index(drop=True), df.iloc[n_train:].reset_index(drop=True)


def _signal(rng, n, k=6):
    x = rng.normal(size=(n, k))
    w = rng.normal(size=k)
    return x, x @ w + 0.5 * x[:, 0] * x[:, 1]


def _spec(
    train,
    test,
    answers,
    sample,
    md,
    truth,
    sep=",",
    decimal=".",
    encoding="utf-8",
    answers_name="answer",
    task_md_encoding="utf-8",
):
    return dict(
        train=train,
        test=test,
        answers=answers,
        sample=sample,
        md=md,
        truth=truth,
        sep=sep,
        decimal=decimal,
        encoding=encoding,
        answers_name=answers_name,
        task_md_encoding=task_md_encoding,
    )


# ───────────────────────── формат ─────────────────────────


@task
def t_semicolon_ru(rng):
    """`;`, десятичная запятая, BOM, русские заголовки с пробелами, id не первая, target в середине. MAE ↓."""
    n, nt = 4000, 1500
    x, y = _signal(rng, n + nt, 4)
    price = 5e6 + 1e6 * y + rng.normal(0, 3e5, n + nt)
    df = pd.DataFrame(
        {
            "Площадь, м2": np.round(50 + 15 * x[:, 0], 1),
            "Цена": np.round(price, 0),
            "ID заказа": np.arange(100000, 100000 + n + nt),
            "Этаж": np.clip((6 + 3 * x[:, 1]).round(), 1, 25),
            "Район города": rng.choice(["Центральный", "Северный", "Южный", "Западный"], n + nt),
            "Год постройки": np.clip((1990 + 15 * x[:, 2]).round(), 1900, 2024),
            "Расстояние до метро (км)": np.abs(np.round(1.5 + x[:, 3], 2)),
        }
    )
    df["Цена"] += df["Район города"].map({"Центральный": 8e5, "Северный": 0, "Южный": -2e5, "Западный": 3e5})
    tr, te = _split(df, n)
    ans = te["Цена"].copy()
    te = te.drop(columns=["Цена"])
    sample = pd.DataFrame({"ID заказа": te["ID заказа"], "Цена": 0})
    md = (
        "# Оценка стоимости квартир\n\nПредскажите цену квартиры (`Цена`).\n\n"
        "Файлы: `train.csv`, `test.csv`, `sample_submission.csv`. CSV в UTF-8 с BOM, разделитель `;`, "
        "десятичный разделитель — запятая.\n\nМетрика качества: **MAE** (средняя абсолютная ошибка), чем меньше, тем лучше.\n\n"
        "Результат — `submission.csv` с колонками `ID заказа;Цена` (тот же формат, что в образце).\n"
    )
    truth = dict(metric="mae", target="Цена", id="ID заказа", greater_is_better=False)
    return _spec(tr, te, ans, sample, md, truth, sep=";", decimal=",", encoding="utf-8-sig", answers_name="Цена")


@task
def t_cp1251_yesno(rng):
    """CSV в cp1251 (Windows), метки «да/нет» строками, accuracy."""
    n, nt = 3000, 1000
    x, z = _signal(rng, n + nt, 4)
    p = 1 / (1 + np.exp(-z))
    y = np.where(rng.random(n + nt) < p, "да", "нет")
    df = pd.DataFrame(
        {
            "клиент": np.arange(n + nt),
            "возраст": np.clip((40 + 10 * x[:, 0]).round(), 18, 90),
            "доход": np.round(np.exp(10 + 0.4 * x[:, 1]), 0),
            "город": rng.choice(["Москва", "Казань", "Ярославль", "Тюмень"], n + nt),
            "кредитов": rng.poisson(2, n + nt),
            "отток": y,
        }
    )
    tr, te = _split(df, n)
    ans = te["отток"].copy()
    te = te.drop(columns=["отток"])
    sample = pd.DataFrame({"клиент": te["клиент"], "отток": "нет"})
    md = (
        "# Отток клиентов\n\nПредскажите, уйдёт ли клиент (`отток`: да/нет).\n\nФайлы — `train.csv`, `test.csv`, "
        "`sample_submission.csv`.\nКодировка файлов — Windows-1251 (так выгрузила старая система).\n\n"
        "Метрика: доля верных ответов (accuracy), больше — лучше.\n"
    )
    truth = dict(metric="accuracy", target="отток", id="клиент", greater_is_better=True)
    return _spec(tr, te, ans, sample, md, truth, encoding="cp1251", answers_name="отток")


@task
def t_no_id_column(rng):
    """В test нет колонки id; в образце ответа — номер строки `row` и цена. RMSE ↓."""
    n, nt = 3000, 1200
    x, y = _signal(rng, n + nt, 5)
    df = pd.DataFrame({f"f{i}": x[:, i] for i in range(5)})
    df["price"] = np.round(100 + 20 * y + rng.normal(0, 5, n + nt), 2)
    tr, te = _split(df, n)
    ans = te["price"].copy()
    te = te.drop(columns=["price"])
    sample = pd.DataFrame({"row": np.arange(nt), "price": 0.0})
    md = (
        "# Predict price\n\nTest rows have no identifier. Write `submission.csv` with columns `row,price`, "
        "where `row` is the zero-based position of the row in `test.csv`.\n\nMetric: RMSE (lower is better).\n"
    )
    truth = dict(metric="rmse", target="price", id="row", greater_is_better=False)
    return _spec(tr, te, ans, sample, md, truth)


@task
def t_dup_columns(rng):
    """Дублирующиеся имена колонок; id в середине; target не последний. ROC-AUC."""
    n, nt = 3000, 1000
    x, z = _signal(rng, n + nt, 4)
    y = (rng.random(n + nt) < 1 / (1 + np.exp(-z))).astype(int)
    cols = {
        "value": x[:, 0],
        "label": y,
        "value ": x[:, 1],  # то же имя с пробелом
        "uid": np.arange(n + nt),
        "Value": x[:, 2],
        "value_dup": x[:, 3],
    }
    df = pd.DataFrame(cols)
    tr, te = _split(df, n)
    ans = te["label"].copy()
    te = te.drop(columns=["label"])
    sample = pd.DataFrame({"uid": te["uid"], "label": 0.5})
    md = (
        "# Binary classification\n\nPredict `label` (probability of class 1). In the files some columns have "
        "similar names (`value`, `value `, `Value`) — they are different features.\n\n"
        "Evaluation: ROC AUC, higher is better. Output: `uid,label`.\n"
    )
    truth = dict(metric="roc_auc", target="label", id="uid", greater_is_better=True)
    s = _spec(tr, te, ans, sample, md, truth)
    s["raw_header_dups"] = True
    return s


# ───────────────────────── объём ─────────────────────────


@task
def t_wide_1500(rng):
    """1500 числовых колонок, 2500 строк; сигнал в 10. ROC-AUC."""
    n, nt, k = 2500, 1000, 1500
    x = rng.normal(size=(n + nt, k)).astype(np.float32)
    z = x[:, :10] @ rng.normal(size=10)
    y = (rng.random(n + nt) < 1 / (1 + np.exp(-z))).astype(int)
    df = pd.DataFrame(x, columns=[f"x{i}" for i in range(k)])
    df.insert(0, "id", np.arange(n + nt))
    df["y"] = y
    tr, te = _split(df, n)
    ans = te["y"].copy()
    te = te.drop(columns=["y"])
    sample = pd.DataFrame({"id": te["id"], "y": 0.5})
    md = "# Wide table\n\nPredict `y` (probability). 1500 anonymous features.\n\nMetric: ROC-AUC (higher is better).\n"
    truth = dict(metric="roc_auc", target="y", id="id", greater_is_better=True)
    return _spec(tr, te, ans, sample, md, truth)


@task
def t_big_1m_rows(rng):
    """1 000 000 строк train, 12 признаков. MAE ↓."""
    n, nt = 1_000_000, 200_000
    x = rng.normal(size=(n + nt, 10)).astype(np.float32)
    y = x[:, :5] @ np.array([3, -2, 1, 0.5, 4]) + np.sin(x[:, 5]) * 5 + rng.normal(0, 1, n + nt)
    df = pd.DataFrame(x, columns=[f"num{i}" for i in range(10)])
    df.insert(0, "rid", np.arange(n + nt))
    df["cat_a"] = rng.choice(list("ABCDEFGH"), n + nt)
    df["cat_b"] = rng.choice([f"k{i}" for i in range(300)], n + nt)
    df["target"] = np.round(y + df["cat_a"].map(dict(zip("ABCDEFGH", range(8)))), 4)
    tr, te = _split(df, n)
    ans = te["target"].copy()
    te = te.drop(columns=["target"])
    sample = pd.DataFrame({"rid": te["rid"], "target": 0.0})
    md = "# Big regression\n\nPredict `target`. About a million rows.\n\nMetric: MAE (lower is better).\n"
    truth = dict(metric="mae", target="target", id="rid", greater_is_better=False)
    return _spec(tr, te, ans, sample, md, truth)


# ───────────────────────── данные ─────────────────────────


@task
def t_degenerate_columns(rng):
    """Пустая, константные, текстовая, id-подобная, смесь форматов дат, огромные значения, bool. PR-AUC."""
    n, nt = 4000, 1500
    x, z = _signal(rng, n + nt, 4)
    y = (rng.random(n + nt) < 1 / (1 + np.exp(-(z - 1.5)))).astype(int)
    dates = pd.to_datetime("2023-01-01") + pd.to_timedelta(rng.integers(0, 700, n + nt), unit="D")
    fmt = rng.integers(0, 3, n + nt)
    date_mixed = [d.strftime(["%Y-%m-%d", "%d.%m.%Y", "%m/%d/%y"][f]) for d, f in zip(dates, fmt, strict=True)]
    df = pd.DataFrame(
        {
            "id": np.arange(n + nt),
            "good1": x[:, 0],
            "good2": x[:, 1],
            "all_nan": np.nan,
            "const_num": 7,
            "const_str": "same",
            "free_text": [
                f"Заказ №{i} от клиента {rng.integers(0, 10**6)}: {'срочно' if y_ else 'обычный'}"
                for i, y_ in enumerate(y)
            ],
            "hash_like": [f"{rng.integers(0, 2**62):x}" for _ in range(n + nt)],
            "date_mixed": date_mixed,
            "huge": x[:, 2] * 1e15,
            "negative": -np.abs(x[:, 3]) * 1e3,
            "flag": rng.choice([True, False], n + nt),
            "target": y,
        }
    )
    tr, te = _split(df, n)
    ans = te["target"].copy()
    te = te.drop(columns=["target"])
    sample = pd.DataFrame({"id": te["id"], "target": 0.5})
    md = (
        "# Rare event\n\nPredict the probability of `target`=1.\n\nThe metric is the area under the precision-recall "
        "curve (PR AUC); higher is better.\n"
    )
    truth = dict(metric="pr_auc", target="target", id="id", greater_is_better=True)
    return _spec(tr, te, ans, sample, md, truth)


@task
def t_target_nan(rng):
    """В train у 3% строк target пустой. R² ↑."""
    n, nt = 3000, 1000
    x, y = _signal(rng, n + nt, 5)
    df = pd.DataFrame({f"f{i}": x[:, i] for i in range(5)})
    df.insert(0, "id", np.arange(n + nt))
    df["y"] = np.round(10 + y + rng.normal(0, 0.5, n + nt), 3)
    tr, te = _split(df, n)
    ans = te["y"].copy()
    te = te.drop(columns=["y"])
    tr.loc[rng.choice(n, int(0.03 * n), replace=False), "y"] = np.nan
    sample = pd.DataFrame({"id": te["id"], "y": 0.0})
    md = "# Измерения\n\nПредскажите `y`. Часть значений `y` в train потеряна.\n\nМетрика: R² (коэффициент детерминации), чем больше, тем лучше.\n"
    truth = dict(metric="r2", target="y", id="id", greater_is_better=True)
    return _spec(tr, te, ans, sample, md, truth)


@task
def t_fifty_classes(rng):
    """50 классов, строковые метки. Balanced accuracy."""
    n, nt, k = 8000, 3000, 50
    centers = rng.normal(size=(k, 6)) * 2
    cls = rng.integers(0, k, n + nt)
    x = centers[cls] + rng.normal(size=(n + nt, 6))
    df = pd.DataFrame(x, columns=[f"f{i}" for i in range(6)])
    df.insert(0, "id", np.arange(n + nt))
    df["species"] = [f"cls_{c:02d}" for c in cls]
    tr, te = _split(df, n)
    ans = te["species"].copy()
    te = te.drop(columns=["species"])
    sample = pd.DataFrame({"id": te["id"], "species": "cls_00"})
    md = "# 50 видов\n\nПредскажите `species` (50 классов). Метрика — сбалансированная точность (balanced accuracy), больше — лучше.\n"
    truth = dict(metric="balanced_accuracy", target="species", id="id", greater_is_better=True)
    return _spec(tr, te, ans, sample, md, truth)


@task
def t_imbalance_999(rng):
    """0.1% положительных, 60k строк. Average precision (PR-AUC), вероятности."""
    n, nt = 60000, 20000
    x, z = _signal(rng, n + nt, 5)
    y = (rng.random(n + nt) < 1 / (1 + np.exp(-(z * 1.2 - 7.5)))).astype(int)
    df = pd.DataFrame({f"f{i}": x[:, i] for i in range(5)})
    df.insert(0, "tx_id", np.arange(n + nt))
    df["fraud"] = y
    tr, te = _split(df, n)
    ans = te["fraud"].copy()
    te = te.drop(columns=["fraud"])
    sample = pd.DataFrame({"tx_id": te["tx_id"], "fraud": 0.0123})
    md = "# Fraud\n\nThe event is extremely rare. Submit probabilities in `fraud`.\n\nMetric: average precision (area under the PR curve), higher is better.\n"
    truth = dict(metric="pr_auc", target="fraud", id="tx_id", greater_is_better=True, positives=int(y.sum()))
    return _spec(tr, te, ans, sample, md, truth)


@task
def t_train_only_leak(rng):
    """В train есть колонки, которых нет в test: утечка `post_outcome_flag` и `internal_score`. AUC."""
    n, nt = 3000, 1000
    x, z = _signal(rng, n + nt, 4)
    y = (rng.random(n + nt) < 1 / (1 + np.exp(-z))).astype(int)
    df = pd.DataFrame({f"f{i}": x[:, i] for i in range(4)})
    df.insert(0, "id", np.arange(n + nt))
    df["post_outcome_flag"] = y
    df["internal_score"] = y + rng.normal(0, 0.1, n + nt)
    df["target"] = y
    tr, te = _split(df, n)
    ans = te["target"].copy()
    te = te.drop(columns=["target", "post_outcome_flag", "internal_score"])
    sample = pd.DataFrame({"id": te["id"], "target": 0.5})
    md = (
        "# Churn\n\nPredict the probability of `target`. Train contains some columns that are computed after the "
        "event and are not available at prediction time.\n\nMetric: ROC AUC (higher is better).\n"
    )
    truth = dict(metric="roc_auc", target="target", id="id", greater_is_better=True)
    return _spec(tr, te, ans, sample, md, truth)


@task
def t_proba_columns(rng):
    """Ответ — вероятности по классам (id, a, b, c, d). LogLoss ↓."""
    n, nt, k = 4000, 1500, 4
    x = rng.normal(size=(n + nt, 5))
    logits = x @ rng.normal(size=(5, k)) * 1.2
    p = np.exp(logits) / np.exp(logits).sum(1, keepdims=True)
    cls = np.array([rng.choice(k, p=pi) for pi in p])
    names = np.array(["alpha", "beta", "gamma", "delta"])
    df = pd.DataFrame(x, columns=[f"f{i}" for i in range(5)])
    df.insert(0, "id", np.arange(n + nt))
    df["grade"] = names[cls]
    tr, te = _split(df, n)
    ans = te["grade"].copy()
    te = te.drop(columns=["grade"])
    sample = pd.DataFrame({"id": te["id"], **{c: 0.25 for c in sorted(names)}})
    md = (
        "# Grade\n\nFor each test row predict probabilities of all four grades. `submission.csv`: `id,alpha,beta,delta,gamma` "
        "(one column per class).\n\nMetric: multi-class log loss (lower is better).\n"
    )
    truth = dict(metric="logloss", target="grade", id="id", greater_is_better=False)
    return _spec(tr, te, ans, sample, md, truth)


@task
def t_mape_zeros_words(rng):
    """Нули в target; метрика словами «средняя абсолютная процентная ошибка», без аббревиатуры; направление словами."""
    n, nt = 3000, 1000
    x, y = _signal(rng, n + nt, 4)
    v = np.round(np.maximum(0, 30 + 8 * y + rng.normal(0, 3, n + nt)))
    v[rng.random(n + nt) < 0.05] = 0
    df = pd.DataFrame({f"f{i}": x[:, i] for i in range(4)})
    df.insert(0, "id", np.arange(n + nt))
    df["demand"] = v
    tr, te = _split(df, n)
    ans = te["demand"].copy()
    te = te.drop(columns=["demand"])
    sample = pd.DataFrame({"id": te["id"], "demand": 0})
    md = (
        "# Спрос на товар\n\nПредскажите `demand`.\n\nКачество решений оценивается средней абсолютной процентной "
        "ошибкой. Чем меньше значение, тем лучше решение.\n"
    )
    truth = dict(metric="mape", target="demand", id="id", greater_is_better=False)
    return _spec(tr, te, ans, sample, md, truth)


@task
def t_ids_overlap_newcats(rng):
    """id теста пересекаются с id train; в test новые категории; даты dd.mm.yyyy; метки yes/no. Accuracy."""
    n, nt = 3000, 1000
    x, z = _signal(rng, n + nt, 3)
    y = np.where(rng.random(n + nt) < 1 / (1 + np.exp(-z)), "yes", "no")
    cats = np.array(rng.choice(["a", "b", "c", "d"], n + nt), dtype=object)
    cats[n:][rng.random(nt) < 0.3] = "NEW_" + str(rng.integers(0, 5))
    dates = pd.to_datetime("2022-01-01") + pd.to_timedelta(np.arange(n + nt) // 4, unit="D")
    df = pd.DataFrame(
        {
            "id": np.concatenate([np.arange(n), np.arange(nt)]),  # test id 0..nt-1 — те же числа, что в train
            "f0": x[:, 0],
            "f1": x[:, 1],
            "cat": cats,
            "when": dates.strftime("%d.%m.%Y"),
            "bought": y,
        }
    )
    tr, te = _split(df, n)
    ans = te["bought"].copy()
    te = te.drop(columns=["bought"])
    sample = pd.DataFrame({"id": te["id"], "bought": "no"})
    md = "# Purchase\n\nPredict `bought` (yes/no). Identifiers are only unique inside a file.\n\nAccuracy, higher is better.\n"
    truth = dict(metric="accuracy", target="bought", id="id", greater_is_better=True)
    return _spec(tr, te, ans, sample, md, truth)


@task
def t_quantile_090(rng):
    """Квантильная ошибка, α=0.9: нужно предсказывать 90-й процентиль, а не среднее."""
    n, nt = 4000, 1500
    x = rng.normal(size=(n + nt, 3))
    scale = np.exp(0.6 * x[:, 0])
    y = 50 + 10 * x[:, 1] + scale * rng.gamma(2.0, 3.0, n + nt)
    df = pd.DataFrame(x, columns=["a", "b", "c"])
    df.insert(0, "id", np.arange(n + nt))
    df["wait_minutes"] = np.round(y, 3)
    tr, te = _split(df, n)
    ans = te["wait_minutes"].copy()
    te = te.drop(columns=["wait_minutes"])
    sample = pd.DataFrame({"id": te["id"], "wait_minutes": 0.0})
    md = (
        "# Время ожидания\n\nПредскажите 90-й перцентиль времени ожидания `wait_minutes`.\n\n"
        "Метрика: quantile loss (pinball) для квантиля 0.9, чем меньше, тем лучше.\n"
    )
    truth = dict(metric="pinball@0.9", target="wait_minutes", id="id", greater_is_better=False)
    return _spec(tr, te, ans, sample, md, truth)


@task
def t_fbeta2(rng):
    """F-beta (β=2) — метрики нет в реестре; нужны метки 0/1. Ближайшая метрика реестра — F1."""
    n, nt = 5000, 2000
    x, z = _signal(rng, n + nt, 4)
    y = (rng.random(n + nt) < 1 / (1 + np.exp(-(z * 1.3 - 2.2)))).astype(int)
    df = pd.DataFrame({f"f{i}": x[:, i] for i in range(4)})
    df.insert(0, "id", np.arange(n + nt))
    df["is_default"] = y
    tr, te = _split(df, n)
    ans = te["is_default"].copy()
    te = te.drop(columns=["is_default"])
    sample = pd.DataFrame({"id": te["id"], "is_default": 0})
    md = (
        "# Дефолты\n\nПредскажите метку `is_default` (0/1). Пропущенный дефолт дороже ложной тревоги, поэтому "
        "используется F-beta мера с beta = 2 (recall вдвое важнее precision). Больше — лучше.\n"
    )
    truth = dict(metric=None, target="is_default", id="id", greater_is_better=True, fbeta=2.0)
    return _spec(tr, te, ans, sample, md, truth)


# ───────────────────────── «грязные» task.md ─────────────────────────


@task
def t_noisy_metric_late(rng):
    """Английский шум; ранее упомянута accuracy прошлого конкурса, основная метрика (RMSE) — позже и без «слов-маркеров»."""
    n, nt = 3000, 1200
    x, y = _signal(rng, n + nt, 5)
    df = pd.DataFrame({f"f{i}": x[:, i] for i in range(5)})
    df.insert(0, "id", np.arange(n + nt))
    df["target"] = np.round(20 + 4 * y + rng.normal(0, 1, n + nt), 3)
    tr, te = _split(df, n)
    ans = te["target"].copy()
    te = te.drop(columns=["target"])
    sample = pd.DataFrame({"id": te["id"], "target": 0.0})
    md = (
        "# Quarterly forecast (v3, DRAFT — do not distribute)\n\nHistory: last year the sister team reached 0.83 accuracy "
        "on a classification variant of this problem; that variant is retired.\n\nData lives in `/data/input/train.csv` and "
        "`/data/input/test.csv` (the loader will copy them next to this file). Remember to be nice to the cluster.\n\n"
        "Predict the numeric column `target`. After the leaderboard freeze, entries are ranked by the main score: "
        "root mean squared error, smaller is better.\n"
    )
    truth = dict(metric="rmse", target="target", id="id", greater_is_better=False)
    return _spec(tr, te, ans, sample, md, truth)


@task
def t_negation_ru(rng):
    """Русское отрицание: «Не используйте accuracy; оценка — ROC AUC»."""
    n, nt = 3000, 1000
    x, z = _signal(rng, n + nt, 4)
    y = (rng.random(n + nt) < 1 / (1 + np.exp(-z))).astype(int)
    df = pd.DataFrame({f"f{i}": x[:, i] for i in range(4)})
    df.insert(0, "id", np.arange(n + nt))
    df["y"] = y
    tr, te = _split(df, n)
    ans = te["y"].copy()
    te = te.drop(columns=["y"])
    sample = pd.DataFrame({"id": te["id"], "y": 0.5})
    md = (
        "# Кредитный скоринг\n\nПредскажите вероятность события `y`.\n\nНе используйте accuracy; оценка — ROC AUC, "
        "больше — лучше.\n"
    )
    truth = dict(metric="roc_auc", target="y", id="id", greater_is_better=True)
    return _spec(tr, te, ans, sample, md, truth)


def write_task(name: str, spec: dict, out: Path) -> None:
    folder = out / name
    folder.mkdir(parents=True, exist_ok=True)
    kw = dict(index=False, sep=spec["sep"], decimal=spec["decimal"], encoding=spec["encoding"])
    for fname, frame in (
        ("train.csv", spec["train"]),
        ("test.csv", spec["test"]),
        ("sample_submission.csv", spec["sample"]),
    ):
        frame.to_csv(folder / fname, **kw)
        if spec.get("raw_header_dups") and fname != "sample_submission.csv":
            # настоящие дубли имён: pandas их не даст записать, поэтому правим строку заголовка вручную
            dup = ["value", "label", "value", "uid", "value", "value_dup"]
            dup = [c for c in dup if fname == "train.csv" or c != "label"]
            lines = (folder / fname).read_text(encoding="utf-8").split("\n")
            lines[0] = spec["sep"].join(dup)
            (folder / fname).write_text("\n".join(lines), encoding="utf-8")
    spec["truth"].update(sep=spec["sep"], decimal=spec["decimal"], encoding=spec["encoding"])
    (folder / "task.md").write_text(spec["md"], encoding=spec["task_md_encoding"])
    truth_dir = out / "_truth"
    truth_dir.mkdir(exist_ok=True)
    (truth_dir / f"{name}.json").write_text(json.dumps(spec["truth"], ensure_ascii=False, indent=1), encoding="utf-8")
    spec["answers"].rename(spec["answers_name"]).to_csv(truth_dir / f"{name}.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("/tmp/night_tasks"))
    parser.add_argument("--nick", default="thenlaooo")
    parser.add_argument("--only", default="")
    args = parser.parse_args()
    seed = sum(ord(c) for c in args.nick)
    only = [s for s in args.only.split(",") if s]
    for name, fn in TASKS.items():
        if only and name not in only:
            continue
        rng = np.random.default_rng(seed + sum(ord(c) for c in name))
        write_task(name, fn(rng), args.out)
        print("ok", name)
    print("seed", seed)


if __name__ == "__main__":
    main()
