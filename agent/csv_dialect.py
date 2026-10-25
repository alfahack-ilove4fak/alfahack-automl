"""CSV «как в жизни» → обычный CSV (UTF-8, запятая). Владелец — LLM/agent-инженер.

Скрытые задачи — «любые табличные без гарантий формата» (вебинар). Ночной аудит (#230, #238): файл в cp1251 или
с разделителем «;» и десятичной запятой ронял разбор за секунду, и в ответ уходил шаблон с нулями. Поэтому файлы
задачи один раз переводим в обычный CSV во временной папке, а дальше все модули (разбор, профиль, ответ)
читают его как обычно — без правок в каждом месте, где вызывается pd.read_csv.

Значения не трогаем, кроме десятичной запятой в числовых колонках: читаем всё строками, поэтому id «007»
остаётся «007». Определение — по первым 64 КБ (идея и прототип — research/night/read_csv_any.py).

Сжатые файлы (.gz, .zip, .bz2, .xz) и таблицы с другим расширением (.tsv, .txt) тоже переводим в обычную копию .csv:
остальные модули читают пути из TaskSpec без догадок о формате (#276, приоритет 2). Десятичная запятая бывает и
при разделителе «,» — тогда число в кавычках: «1,"10,5"».
"""

from __future__ import annotations

import bz2
import gzip
import hashlib
import lzma
import re
import tempfile
import zipfile
from pathlib import Path

import pandas as pd

from runtime.log import get_logger

log = get_logger("csv")

ENCODINGS = ("utf-8-sig", "cp1251")
SEPARATORS = (",", ";", "\t", "|")  # при равенстве — запятая (первая)
SNIFF_BYTES = 65536
NUMBER_WITH_COMMA = re.compile(r"-?\d+(,\d+)?")
# Пропуски среди чисел: колонка «1,5 / NA / -2,25» всё равно числовая (ревью #246). Тысячные разделители
# («1 200,5») не обрабатываем: такая колонка останется строками, как без приведения
MISSING_TOKENS = {"na", "n/a", "nan", "null", "none", "-", "?"}
CACHE_DIR = Path(tempfile.gettempdir()) / "agent_csv"
COMPRESSED = {".gz": gzip.open, ".bz2": bz2.open, ".xz": lzma.open}  # .zip — отдельно: внутри архива первый файл
QUOTED_DECIMAL = re.compile(r'"-?\d+,\d+"')  # «"10,5"» при разделителе «,»


def head_bytes(path: Path, n: int = SNIFF_BYTES) -> bytes:
    """Первые n байт таблицы, в том числе из сжатого файла."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in COMPRESSED:
        with COMPRESSED[suffix](path, "rb") as f:
            return f.read(n)
    if suffix == ".zip":
        with zipfile.ZipFile(path) as z, z.open(_zip_member(z)) as f:
            return f.read(n)
    with open(path, "rb") as f:
        return f.read(n)


def _zip_member(z: zipfile.ZipFile) -> str:
    names = [i.filename for i in z.infolist() if not i.is_dir() and not i.filename.startswith("__MACOSX")]
    return names[0]


def _read_all(path: Path, dialect: dict) -> pd.DataFrame:
    """Вся таблица строками; .gz/.bz2/.xz pandas распаковывает сам, из .zip берём первый файл."""
    kw = {"sep": dialect["sep"], "encoding": dialect["encoding"], "dtype": str, "keep_default_na": False}
    if Path(path).suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as z, z.open(_zip_member(z)) as f:
            return pd.read_csv(f, **kw)
    return pd.read_csv(path, **kw)


def is_plain_csv(path: Path) -> bool:
    return Path(path).suffix.lower() == ".csv"


def sniff(path: Path) -> dict:
    """Кодировка, разделитель и десятичный знак файла по его началу."""
    raw = head_bytes(path)
    encoding = "latin-1"
    for candidate in ENCODINGS:
        try:
            head = raw.decode(candidate)
        except UnicodeDecodeError as exc:
            if candidate.startswith("utf-8") and exc.start >= len(raw) - 3 and len(raw) == SNIFF_BYTES:
                head = raw[: exc.start].decode(candidate)  # обрезали посреди многобайтового символа — это UTF-8
            else:
                continue
        encoding = candidate
        break
    else:
        head = raw.decode(encoding)
    lines = head.splitlines()
    header = re.sub(r'"[^"]*"', "", lines[0]) if lines else ""  # разделители внутри кавычек не считаем
    sep = max(SEPARATORS, key=header.count)
    body = "\n".join(lines[1:])
    if sep != ",":
        decimal = "," if re.search(r"\d,\d", body) else "."
    else:
        decimal = "," if QUOTED_DECIMAL.search(body) else "."
    return {"encoding": encoding, "sep": sep, "decimal": decimal}


def is_standard(dialect: dict) -> bool:
    return dialect["encoding"] == "utf-8-sig" and dialect["sep"] == "," and dialect["decimal"] == "."


def standard_csv(path: Path) -> Path:
    """Путь к файлу в обычном виде: сам файл, если он уже такой, иначе — копия в CACHE_DIR. Сбой — исходный путь."""
    try:
        dialect = sniff(path)
        if is_standard(dialect) and is_plain_csv(path):
            return path
        stat = path.stat()
        key = hashlib.sha1(f"{path.resolve()}|{stat.st_size}|{stat.st_mtime_ns}".encode()).hexdigest()[:12]
        out = CACHE_DIR / key / plain_name(path)
        if not out.exists():
            df = _read_all(path, dialect)
            if dialect["decimal"] == ",":
                for column in df.columns:
                    values = df[column][~df[column].str.strip().str.lower().isin(MISSING_TOKENS | {""})]
                    if len(values) and values.str.fullmatch(NUMBER_WITH_COMMA.pattern).all():
                        df[column] = df[column].str.replace(",", ".", regex=False)
            out.parent.mkdir(parents=True, exist_ok=True)
            tmp = out.with_suffix(".tmp")
            df.to_csv(tmp, index=False)
            tmp.replace(out)
        log.info(
            "%s: кодировка %s, разделитель %r, десятичный знак %r — читаем обычную копию",
            path.name, dialect["encoding"], dialect["sep"], dialect["decimal"],
        )  # fmt: skip
        return out
    except Exception as exc:  # noqa: BLE001 — не смогли перевести: пусть читают как есть, это не хуже, чем было
        log.warning("%s: не удалось привести CSV к обычному виду (%s) — читаем как есть", path.name, exc)
        return path


def plain_name(path: Path) -> str:
    """Имя обычной копии: train.csv.gz → train.csv, data.tsv → data.csv."""
    name = Path(path).name
    for suffix in (*COMPRESSED, ".zip"):
        if name.lower().endswith(suffix):
            name = name[: -len(suffix)]
            break
    stem = name[: -len(Path(name).suffix)] if Path(name).suffix.lower() in (".csv", ".tsv", ".txt") else name
    return f"{stem}.csv"


def read_text_any(path: Path) -> str:
    """Текст task.md в UTF-8, иначе cp1251; в крайнем случае — с заменой нечитаемых символов."""
    raw = Path(path).read_bytes()
    for encoding in ENCODINGS:
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")
