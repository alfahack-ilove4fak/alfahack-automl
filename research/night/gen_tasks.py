# ruff: noqa: UP031
"""Генератор «чужих» синтетических задач для ночного аудита (фаза 3).

    uv run python research/night/gen_tasks.py runs/night/tasks

Каждая задача: <out>/<имя>/{task.md, train.csv, test.csv, sample_submission.csv, ...}; ответы вне папки задачи —
<out>/_answers/<имя>.csv, ожидания (метрика, направление) — <out>/_expected.json.
seed = сумма кодов букв ника «thenlaooo» = 969.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SEED = sum(map(ord, "thenlaooo"))


def rng(k: int) -> np.random.Generator:
    return np.random.default_rng(SEED + k)


def base_frame(n: int, g: np.random.Generator, k: int = 6) -> pd.DataFrame:
    x = {f"f{i}": g.normal(size=n) for i in range(k)}
    x["cat_a"] = g.choice(list("ABCDE"), n)
    x["cat_b"] = g.choice(["red", "green", "blue", "x", "y", "z", "q"], n)
    return pd.DataFrame(x)


def signal(df: pd.DataFrame, g: np.random.Generator) -> np.ndarray:
    return (
        2 * df["f0"]
        + np.sin(3 * df["f1"])
        + df["f2"] * (df["cat_a"] == "A")
        + 0.5 * df["cat_b"].map({"red": 1, "green": -1}).fillna(0)
        + 0.3 * g.normal(size=len(df))
    ).to_numpy()


def write(out: Path, name: str, md: str, train, test, sample, answer, expected, sep=",", decimal=".", encoding="utf-8"):
    d = out / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "task.md").write_text(md, encoding="utf-8")
    for fname, frame in (("train.csv", train), ("test.csv", test), ("sample_submission.csv", sample)):
        if frame is not None:
            frame.to_csv(d / fname, index=False, sep=sep, decimal=decimal, encoding=encoding)
    (out / "_answers").mkdir(exist_ok=True)
    answer.to_csv(out / "_answers" / f"{name}.csv", index=False)
    return expected


def main(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    exp: dict[str, dict] = {}

    # 1. ';' + запятая-десятичная + BOM + русские заголовки с пробелами, id не первая, target не последний. MAE
    g = rng(1)
    n = 6000
    df = base_frame(n, g)
    df["y"] = signal(df, g) * 10 + 100
    df = df.rename(columns={"f0": "Площадь дома", "f1": "Возраст здания", "f2": "Число этажей", "cat_a": "Тип района"})
    df.insert(2, "номер записи", np.arange(n))
    df = df[[c for c in df.columns if c != "y"][:4] + ["y"] + [c for c in df.columns if c != "y"][4:]]
    tr, te = df.iloc[:4500], df.iloc[4500:].drop(columns="y")
    ans = df.iloc[4500:][["номер записи", "y"]].rename(columns={"y": "Цена, тыс. руб"})
    tr = tr.rename(columns={"y": "Цена, тыс. руб"})
    sample = pd.DataFrame({"номер записи": te["номер записи"], "Цена, тыс. руб": 0.0})
    md = (
        "# Оценка стоимости\n\nПредскажите `Цена, тыс. руб`. Метрика — средняя абсолютная ошибка (MAE), "
        "чем меньше, тем лучше.\n\nФайлы: train.csv, test.csv, sample_submission.csv. Разделитель `;`, "
        "десятичный разделитель — запятая, кодировка UTF-8 с BOM.\n"
    )
    exp["t01_semicolon_bom_ru"] = write(
        out,
        "t01_semicolon_bom_ru",
        md,
        tr,
        te,
        sample,
        ans,
        {"metric": "mae", "greater": False},
        sep=";",
        decimal=",",
        encoding="utf-8-sig",
    )

    # 2. нет id: sample_submission из одной колонки, английский текст, ROC-AUC упомянут вскользь
    g = rng(2)
    n = 8000
    df = base_frame(n, g)
    p = 1 / (1 + np.exp(-signal(df, g)))
    df["churned"] = (g.random(n) < p).astype(int)
    tr, te = df.iloc[:6000], df.iloc[6000:].drop(columns="churned")
    ans = pd.DataFrame({"row": np.arange(len(te)), "churned": df.iloc[6000:]["churned"].to_numpy()})
    sample = pd.DataFrame({"churned": 0.5 * np.ones(len(te))})
    md = (
        "# Churn\n\nWe are a telecom. Lots of text about how great our company is, history since 1998, "
        "mission statement, CSR, etc.\n\nGiven customer features predict probability of `churned` for every test row "
        "(same order as test.csv). Submissions are ranked by ROC AUC. Thanks and good luck! The data is in "
        "train.csv / test.csv, there is no id column.\n"
    )
    exp["t02_no_id_auc"] = write(out, "t02_no_id_auc", md, tr, te, sample, ans, {"metric": "roc_auc", "greater": True})

    # 3. дубли имён колонок + кавычки/запятые в тексте. R²
    g = rng(3)
    n = 5000
    df = base_frame(n, g)
    df["note"] = g.choice(['said "hello", then left', "plain", 'quote " inside', "a,b,c"], n)
    df["y"] = signal(df, g)
    df["id"] = np.arange(n)
    tr, te = df.iloc[:3800].copy(), df.iloc[3800:].drop(columns="y").copy()
    ans = df.iloc[3800:][["id", "y"]]
    # дубль имени колонки f3 → в csv два столбца «f0»
    for d in (tr, te):
        d["f0_dup"] = d["f3"] + 1
    tr.columns = [("f0" if c == "f0_dup" else c) for c in tr.columns]
    te.columns = [("f0" if c == "f0_dup" else c) for c in te.columns]
    sample = pd.DataFrame({"id": te["id"], "y": 0.0})
    md = "# Dup columns\n\nPredict `y` for each `id` in test. Metric: R² (coefficient of determination), higher is better.\n"
    exp["t03_dup_cols_quotes_r2"] = write(
        out, "t03_dup_cols_quotes_r2", md, tr, te, sample, ans, {"metric": "r2", "greater": True}
    )

    # 4. 1.2 млн строк, MAPE «чем меньше, тем лучше» словами
    g = rng(4)
    n = 1_200_000
    df = pd.DataFrame({f"f{i}": g.normal(size=n).astype("float32") for i in range(10)})
    df["y"] = (np.exp(0.3 * df["f0"] + 0.2 * df["f1"]) * 50 + 5 + g.random(n)).astype("float32")
    df["id"] = np.arange(n)
    tr, te = df.iloc[:1_000_000], df.iloc[1_000_000:].drop(columns="y")
    ans = df.iloc[1_000_000:][["id", "y"]]
    sample = pd.DataFrame({"id": te["id"], "y": 0.0})
    md = (
        "# Big\n\nPredict `y`. Quality: mean absolute percentage error — the smaller the percentage, "
        "the better the model.\n"
    )
    exp["t04_big_1m_mape"] = write(
        out, "t04_big_1m_mape", md, tr, te, sample, ans, {"metric": "mape", "greater": False}
    )

    # 5. 1500 колонок, PR-AUC
    g = rng(5)
    n, w = 3000, 1500
    X = g.normal(size=(n, w)).astype("float32")
    p = 1 / (1 + np.exp(-(2 * X[:, 0] - 1.5 * X[:, 1] * X[:, 2] - 3)))
    y = (g.random(n) < p).astype(int)
    df = pd.DataFrame(X, columns=[f"v{i}" for i in range(w)])
    df["id"] = np.arange(n)
    df["target"] = y
    tr, te = df.iloc[:2200], df.iloc[2200:].drop(columns="target")
    ans = df.iloc[2200:][["id", "target"]]
    sample = pd.DataFrame({"id": te["id"], "target": 0.0})
    md = "# Wide\n\nBinary target `target`. Metric: PR-AUC (area under precision-recall curve), higher is better.\n"
    exp["t05_wide_1500_prauc"] = write(
        out, "t05_wide_1500_prauc", md, tr, te, sample, ans, {"metric": "pr_auc", "greater": True}
    )

    # 6. мусорные колонки: 100% NaN, константы, текст, огромные/отрицательные. SMAPE
    g = rng(6)
    n = 5000
    df = base_frame(n, g)
    df["all_nan"] = np.nan
    df["const_num"] = 7
    df["const_str"] = "same"
    df["text"] = [" ".join(g.choice(["good", "bad", "very", "slow", "fast", "ok"], 6)) for _ in range(n)]
    df["huge"] = g.normal(size=n) * 1e12
    df["neg"] = -np.abs(g.normal(size=n)) * 1e3
    df["y"] = np.abs(signal(df, g)) * 10 + 20
    df["id"] = np.arange(n)
    tr, te = df.iloc[:3800], df.iloc[3800:].drop(columns="y")
    ans = df.iloc[3800:][["id", "y"]]
    sample = pd.DataFrame({"id": te["id"], "y": 0.0})
    md = "# Junk\n\nPredict y. Metric: SMAPE (symmetric mean absolute percentage error), lower is better.\n"
    exp["t06_junk_cols_smape"] = write(
        out, "t06_junk_cols_smape", md, tr, te, sample, ans, {"metric": "smape", "greater": False}
    )

    # 7. смешанные форматы дат, binary logloss
    g = rng(7)
    n = 6000
    df = base_frame(n, g)
    dates = pd.Timestamp("2022-01-01") + pd.to_timedelta(np.arange(n) // 10, unit="D")
    fmts = ["%d.%m.%Y", "%Y-%m-%d", "%m/%d/%y", "%d %b %Y"]
    df["date"] = [d.strftime(fmts[i % 4]) for i, d in enumerate(dates)]
    p = 1 / (1 + np.exp(-signal(df, g) + 0.5))
    df["label"] = (g.random(n) < p).astype(int)
    df["id"] = np.arange(n)
    tr, te = df.iloc[:4500], df.iloc[4500:].drop(columns="label")
    ans = df.iloc[4500:][["id", "label"]]
    sample = pd.DataFrame({"id": te["id"], "label": 0.5})
    md = "# Dates\n\nПредскажите вероятность `label`. Метрика — logloss (логарифмическая потеря), меньше — лучше.\n"
    exp["t07_mixed_dates_logloss"] = write(
        out, "t07_mixed_dates_logloss", md, tr, te, sample, ans, {"metric": "logloss", "greater": False}
    )

    # 8. target с NaN в train, MAE
    g = rng(8)
    n = 5000
    df = base_frame(n, g)
    df["y"] = signal(df, g)
    df["id"] = np.arange(n)
    tr, te = df.iloc[:3800].copy(), df.iloc[3800:].drop(columns="y")
    tr.loc[tr.sample(frac=0.1, random_state=1).index, "y"] = np.nan
    ans = df.iloc[3800:][["id", "y"]]
    sample = pd.DataFrame({"id": te["id"], "y": 0.0})
    md = "# NaN target\n\nPredict y; MAE, lower is better. Some training targets are missing.\n"
    exp["t08_target_nan_mae"] = write(
        out, "t08_target_nan_mae", md, tr, te, sample, ans, {"metric": "mae", "greater": False}
    )

    # 9. 50 классов, колонка на класс, multi-class logloss
    g = rng(9)
    n, K = 12000, 50
    df = base_frame(n, g, k=8)
    logits = np.stack([df[f"f{j % 8}"] * ((j % 5) - 2) + g.normal(size=n) for j in range(K)], axis=1) * 1.5
    df["cls"] = ["class_%02d" % c for c in logits.argmax(1)]
    df["id"] = np.arange(n)
    tr, te = df.iloc[:9000], df.iloc[9000:].drop(columns="cls")
    ans = df.iloc[9000:][["id", "cls"]]
    sample = pd.DataFrame({"id": te["id"], **{"class_%02d" % c: 1 / K for c in range(K)}})
    md = (
        "# 50 classes\n\nPredict the class probabilities for each id: one column per class in sample_submission.csv. "
        "Metric: multi-class log loss, lower is better.\n"
    )
    exp["t09_50class_logloss"] = write(
        out, "t09_50class_logloss", md, tr, te, sample, ans, {"metric": "logloss", "greater": False}
    )

    # 10. 99.9% одного класса, F-beta (β=2)
    g = rng(10)
    n = 30000
    df = base_frame(n, g)
    p = 1 / (1 + np.exp(-(2.5 * df["f0"] + df["f1"] - 9.5)))
    df["is_bad"] = (g.random(n) < p).astype(int)
    df["id"] = np.arange(n)
    tr, te = df.iloc[:22000], df.iloc[22000:].drop(columns="is_bad")
    ans = df.iloc[22000:][["id", "is_bad"]]
    sample = pd.DataFrame({"id": te["id"], "is_bad": 0})
    md = (
        "# Rare event\n\nPredict the label `is_bad` (0/1). The metric is the F-beta score with beta = 2 "
        "(recall counts twice as much as precision); higher is better.\n"
    )
    exp["t10_rare_fbeta2"] = write(
        out, "t10_rare_fbeta2", md, tr, te, sample, ans, {"metric": "fbeta", "greater": True}
    )
    exp["t10_rare_fbeta2"]["pos_rate"] = float(df["is_bad"].mean())

    # 11. утечка: колонка = target в train и test (правдоподобная копия), accuracy
    g = rng(11)
    n = 5000
    df = base_frame(n, g)
    p = 1 / (1 + np.exp(-signal(df, g)))
    df["outcome"] = (g.random(n) < p).astype(int)
    df["outcome_copy"] = df["outcome"]
    df["id"] = np.arange(n)
    tr, te = df.iloc[:3800], df.iloc[3800:].drop(columns="outcome")
    ans = df.iloc[3800:][["id", "outcome"]]
    sample = pd.DataFrame({"id": te["id"], "outcome": 0})
    md = "# Leak\n\nPredict `outcome` (0/1). Metric: accuracy.\n"
    exp["t11_leak_accuracy"] = write(
        out, "t11_leak_accuracy", md, tr, te, sample, ans, {"metric": "accuracy", "greater": True}
    )

    # 12. порядковая классификация, QWK; нет sample_submission, лишний столбец только в train
    g = rng(12)
    n = 6000
    df = base_frame(n, g)
    s = signal(df, g)
    df["grade"] = pd.cut(s, [-np.inf, -2, -0.5, 0.5, 2, np.inf], labels=[1, 2, 3, 4, 5]).astype(int)
    df["id"] = np.arange(n)
    tr, te = df.iloc[:4500], df.iloc[4500:].drop(columns="grade")
    ans = df.iloc[4500:][["id", "grade"]]
    md = (
        "# Ordinal\n\nPredict `grade` from 1 to 5 (ordered). Metric: quadratic weighted kappa (Cohen), higher is better. "
        "Submission: `id,grade`.\n"
    )
    exp["t12_ordinal_qwk_nosample"] = write(
        out, "t12_ordinal_qwk_nosample", md, tr, te, None, ans, {"metric": "qwk", "greater": True}
    )

    # 13. мультитаргет, MAE
    g = rng(13)
    n = 5000
    df = base_frame(n, g)
    df["y1"] = signal(df, g)
    df["y2"] = 3 * df["f3"] - df["f4"] + g.normal(size=n) * 0.2
    df["id"] = np.arange(n)
    tr, te = df.iloc[:3800], df.iloc[3800:].drop(columns=["y1", "y2"])
    ans = df.iloc[3800:][["id", "y1", "y2"]]
    sample = pd.DataFrame({"id": te["id"], "y1": 0.0, "y2": 0.0})
    md = "# Multi\n\nPredict both `y1` and `y2`. Metric: mean absolute error averaged over both targets, lower is better.\n"
    exp["t13_multitarget_mae"] = write(
        out, "t13_multitarget_mae", md, tr, te, sample, ans, {"metric": "mae", "greater": False}
    )

    # 14. id из train в test, новые категории в test, balanced accuracy
    g = rng(14)
    n = 6000
    df = base_frame(n, g)
    df["id"] = np.arange(n) % 4500  # id повторяются между train и test
    p = 1 / (1 + np.exp(-signal(df, g) + 1.5))
    df["y"] = (g.random(n) < p).astype(int)
    tr, te = df.iloc[:4500].copy(), df.iloc[4500:].drop(columns="y").copy()
    te.loc[te.index[:300], "cat_b"] = "NEW_CATEGORY"
    te.loc[te.index[300:500], "cat_a"] = "Z_unseen"
    ans = pd.DataFrame({"id": te["id"], "y": df.iloc[4500:]["y"].to_numpy()})
    sample = pd.DataFrame({"id": te["id"], "y": 0})
    md = "# Reuse ids\n\nPredict y (0/1). Metric: balanced accuracy, higher is better.\n"
    exp["t14_ids_overlap_newcat_balacc"] = write(
        out, "t14_ids_overlap_newcat_balacc", md, tr, te, sample, ans, {"metric": "balanced_accuracy", "greater": True}
    )

    # 15. неверные пути в task.md, файлы в подпапке, квантильная потеря 0.9, шум в тексте
    g = rng(15)
    n = 6000
    df = base_frame(n, g)
    df["y"] = signal(df, g) + (1 + 0.5 * np.abs(df["f3"])) * g.normal(size=n)
    df["id"] = np.arange(n)
    tr, te = df.iloc[:4500], df.iloc[4500:].drop(columns="y")
    ans = df.iloc[4500:][["id", "y"]]
    sample = pd.DataFrame({"id": te["id"], "y": 0.0})
    md = (
        "# Demand\n\nSee https://example.com/info for details. Files are at `/data/input/train.csv`, "
        "`/data/input/test.csv` and `/data/input/sample_submission.csv` (they are really somewhere else).\n\n"
        "We need the 0.9 quantile of `y`: use the pinball loss (quantile loss) at alpha = 0.9 to compare models; "
        "lower is better. Previously we used RMSE but it is deprecated.\n"
    )
    exp["t15_wrong_paths_quantile"] = write(
        out, "t15_wrong_paths_quantile", md, tr, te, sample, ans, {"metric": "pinball@0.9", "greater": False}
    )
    d = out / "t15_wrong_paths_quantile"
    sub = d / "participants" / "v2"
    sub.mkdir(parents=True, exist_ok=True)
    for f in ("train.csv", "test.csv", "sample_submission.csv"):
        (d / f).rename(sub / f)

    # 16. MAP@k: ранжирование — для каждого пользователя топ-3 товара (классификация на 12 классов, MAP@3)
    g = rng(16)
    n, K = 8000, 12
    df = base_frame(n, g, k=6)
    logits = np.stack([df[f"f{j % 6}"] * ((j % 3) - 1) + g.normal(size=n) for j in range(K)], axis=1)
    df["item"] = ["i%02d" % c for c in logits.argmax(1)]
    df["id"] = np.arange(n)
    tr, te = df.iloc[:6000], df.iloc[6000:].drop(columns="item")
    ans = df.iloc[6000:][["id", "item"]]
    sample = pd.DataFrame({"id": te["id"], "item": "i00 i01 i02"})
    md = (
        "# Recs\n\nFor each id predict the three most likely items as a space-separated string, e.g. `i03 i07 i01` "
        "(`item` column). Metric: MAP@3, higher is better.\n"
    )
    exp["t16_map3_ranking"] = write(
        out, "t16_map3_ranking", md, tr, te, sample, ans, {"metric": "map@3", "greater": True}
    )

    (out / "_expected.json").write_text(json.dumps(exp, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Сгенерировано задач: {len(exp)}; seed={SEED}")


if __name__ == "__main__":
    main(Path(sys.argv[1] if len(sys.argv) > 1 else "runs/night/tasks"))
