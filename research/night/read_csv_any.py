"""Прототип чтения CSV «как в жизни» для предложения в agent/ и ml/ (ночной аудит). В агенте не используется.

    uv run python research/night/read_csv_any.py runs/night/alien/*/train.csv

Определяет по первым 64 КБ: кодировку (utf-8 с BOM и без, иначе cp1251), разделитель (самый частый из , ; TAB |
в строке заголовка) и десятичную запятую (разделитель «;» и в данных есть «цифра,цифра»).
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd

ENCODINGS = ("utf-8-sig", "cp1251")
SEPARATORS = (",", ";", "\t", "|")


def sniff(path: Path) -> dict:
    raw = Path(path).read_bytes()[:65536]
    for encoding in ENCODINGS:
        try:
            head = raw.decode(encoding)
            break
        except UnicodeDecodeError:
            continue
    else:
        encoding, head = "latin-1", raw.decode("latin-1")
    header = head.splitlines()[0] if head else ""
    sep = max(SEPARATORS, key=header.count)
    decimal = "," if sep == ";" and re.search(r"\d,\d", head) else "."
    return {"encoding": encoding, "sep": sep, "decimal": decimal}


def read_csv_any(path: Path, **kwargs) -> pd.DataFrame:
    return pd.read_csv(path, **sniff(path), **kwargs)


if __name__ == "__main__":
    for arg in sys.argv[1:]:
        opts = sniff(Path(arg))
        df = read_csv_any(Path(arg), nrows=5)
        print(
            f"{arg}: {opts} → {df.shape[1]} колонок: {list(df.columns)[:6]}, dtypes {list(df.dtypes.astype(str))[:6]}"
        )
