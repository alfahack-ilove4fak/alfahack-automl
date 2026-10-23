"""Варианты формата одной и той же небольшой регрессии (фаза 3б ночного аудита): где ломается разбор файлов.

    python research/night/gen_variants.py runs/night/variants
Ответы — <out>/_answers/<имя>.csv, ожидания — <out>/_expected.json (метрика mae, меньше — лучше).
"""

from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SEED = sum(map(ord, "thenlaooo"))


def main(out: Path) -> None:
    g = np.random.default_rng(SEED + 100)
    n = 3000
    df = pd.DataFrame({f"x{i}": g.normal(size=n) for i in range(5)})
    df["cat"] = g.choice(list("ABCD"), n)
    df["target"] = 3 * df["x0"] + np.sin(df["x1"]) + (df["cat"] == "A") * 2 + 0.2 * g.normal(size=n)
    df["id"] = np.arange(n)
    tr, te = df.iloc[:2200], df.iloc[2200:].drop(columns="target")
    ans = df.iloc[2200:][["id", "target"]]
    sample = pd.DataFrame({"id": te["id"], "target": 0.0})
    md = "# Variant\n\nPredict `target` for each `id` in test.csv. Metric: MAE (mean absolute error), lower is better.\n"
    exp = {}
    (out / "_answers").mkdir(parents=True, exist_ok=True)

    def make(name, *, sep=",", decimal=".", encoding="utf-8", names=("train.csv", "test.csv", "sample_submission.csv"),
             text=md, quoting_num=False, gz=False, rename=None, md_encoding="utf-8", note=""):
        d = out / name
        d.mkdir(parents=True, exist_ok=True)
        frames = [tr, te, sample]
        if rename:
            frames = [f.rename(columns=rename) for f in frames]
        for fname, frame in zip(names, frames, strict=True):
            if quoting_num:
                import csv

                frame.to_csv(d / fname, index=False, sep=sep, decimal=decimal, encoding=encoding, quoting=csv.QUOTE_NONNUMERIC)
            elif gz:
                with gzip.open(d / (fname + ".gz"), "wt", encoding=encoding) as fh:
                    frame.to_csv(fh, index=False, sep=sep, decimal=decimal)
            else:
                frame.to_csv(d / fname, index=False, sep=sep, decimal=decimal, encoding=encoding)
        (d / "task.md").write_bytes(text.encode(md_encoding))
        a = ans.copy()
        if rename:
            a = a.rename(columns=rename)
        a.to_csv(out / "_answers" / f"{name}.csv", index=False)
        exp[name] = {"metric": "mae", "greater": False, "note": note, "target": (rename or {}).get("target", "target"),
                     "sep": sep, "decimal": decimal, "enc": encoding}

    make("v01_semicolon_dot", sep=";", note="; и точка")
    make("v02_decimal_comma_quoted", decimal=",", note="запятая-разделитель и запятая-десятичная: нужны кавычки")
    make("v03_tab", sep="\t", note="TSV в .csv")
    make("v04_cp1251_csv", sep=",", encoding="cp1251", rename={"cat": "категория"}, note="CSV в cp1251")
    make("v05_bom_comma", encoding="utf-8-sig", note="BOM + запятая")
    make("v06_names_data_train", names=("data_train.csv", "data_test.csv", "sample_submission.csv"), note="data_train/data_test")
    make("v07_names_other", names=("learn.csv", "holdout.csv", "submission_example.csv"), note="learn/holdout/submission_example")
    make("v08_uppercase", names=("TRAIN.CSV", "TEST.CSV", "Sample_Submission.CSV"), note="верхний регистр")
    make("v09_gz", gz=True, note="train.csv.gz")
    make("v10_quoted_all", quoting_num=True, note="все строки в кавычках, числа не в кавычках")
    make("v11_md_cp1251", text="# Задача\n\nПредскажите `target`. Метрика — MAE, меньше — лучше. Кириллица.\n",
         md_encoding="cp1251", note="task.md в cp1251")
    make("v12_md_bom", text="﻿" + md, note="task.md с BOM")
    make("v13_target_first_idlast", note="target не последний; id последний (уже так)")
    (out / "_expected.json").write_text(json.dumps(exp, ensure_ascii=False, indent=1), encoding="utf-8")
    print("вариантов:", len(exp))


if __name__ == "__main__":
    main(Path(sys.argv[1] if len(sys.argv) > 1 else "runs/night/variants"))
