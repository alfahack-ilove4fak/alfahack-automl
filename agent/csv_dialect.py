"""CSV «как в жизни» → обычный CSV (UTF-8, запятая). Владелец — LLM/agent-инженер.

Скрытые задачи — «любые табличные без гарантий формата» (вебинар). Ночной аудит (#230, #238): файл в cp1251 или
с разделителем «;» и десятичной запятой ронял разбор за секунду, и в ответ уходил шаблон с нулями. Поэтому файлы
задачи один раз переводим в обычный CSV во временной папке, а дальше все модули (разбор, профиль, ответ)
читают его как обычно — без правок в каждом месте, где вызывается pd.read_csv.

Значения не трогаем, кроме десятичной запятой в числовых колонках: читаем всё строками, поэтому id «007»
остаётся «007». Определение — по первым 64 КБ (идея и прототип — research/night/read_csv_any.py).
"""

from __future__ import annotations

import hashlib
import re
import tempfile
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


def sniff(path: Path) -> dict:
    """Кодировка, разделитель и десятичный знак файла по его началу."""
    raw = Path(path).read_bytes()[:SNIFF_BYTES]
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
    decimal = "," if sep != "," and re.search(r"\d,\d", "\n".join(lines[1:])) else "."
    return {"encoding": encoding, "sep": sep, "decimal": decimal}


def is_standard(dialect: dict) -> bool:
    return dialect["encoding"] == "utf-8-sig" and dialect["sep"] == ","


def standard_csv(path: Path) -> Path:
    """Путь к файлу в обычном виде: сам файл, если он уже такой, иначе — копия в CACHE_DIR. Сбой — исходный путь."""
    try:
        dialect = sniff(path)
        if is_standard(dialect):
            return path
        stat = path.stat()
        key = hashlib.sha1(f"{path.resolve()}|{stat.st_size}|{stat.st_mtime_ns}".encode()).hexdigest()[:12]
        out = CACHE_DIR / key / path.name
        if not out.exists():
            df = pd.read_csv(path, sep=dialect["sep"], encoding=dialect["encoding"], dtype=str, keep_default_na=False)
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


def read_text_any(path: Path) -> str:
    """Текст task.md в UTF-8, иначе cp1251; в крайнем случае — с заменой нечитаемых символов."""
    raw = Path(path).read_bytes()
    for encoding in ENCODINGS:
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")
