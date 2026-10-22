"""Код признаков от LLM (A-L3-1): промпт, разбор ответа и проверка кода до песочницы. Владелец — agent.

Контракт кода (interfaces/schemas.py, Hypothesis.code; ADR 0008 капитана, D4): код обязан определять
`add_features(df) -> df`; df — признаки train и test вместе БЕЗ target, на выходе — те же строки
в том же порядке плюс новые колонки. Исполняется только в песочнице (runtime/sandbox.py):
apply_feature_code — обёртка над run_code, обмен — файлы во временной папке.

check_feature_code — дешёвый предфильтр по тексту (AST), а не защита: он отсеивает очевидно опасный
или сломанный код, чтобы не тратить на него запуск песочницы. Главный барьер — песочница.
"""

from __future__ import annotations

import ast
import json
import re
import tempfile
from pathlib import Path
from string import Template

import numpy as np
import pandas as pd
from pydantic import ValidationError

from agent.prompting import describe_profile, describe_task, load_prompt
from interfaces.schemas import DataProfile, ExperimentResult, Hypothesis, HypothesisKind, TaskSpec
from runtime.log import get_logger
from runtime.sandbox import run_code

log = get_logger("feature_code")

FUNCTION_NAME = "add_features"
MAX_CODE_CHARS = 5000
ALLOWED_IMPORTS = {"pandas", "numpy", "math", "re", "datetime", "itertools", "functools", "collections"}
FORBIDDEN_CALLS = {
    "open", "exec", "eval", "compile", "__import__", "globals", "locals", "vars", "getattr", "setattr",
    "delattr", "input", "breakpoint", "exit", "quit", "help", "memoryview",
}  # fmt: skip
# Чтение и запись файлов через pandas/numpy: df.to_csv, pd.read_parquet, np.save, np.load …
FILE_IO_ATTR = re.compile(
    r"^(read_\w+|to_(csv|parquet|pickle|excel|json|sql|feather|hdf|html|latex|stata|orc|clipboard|markdown|xml)|save\w*|load\w*|fromfile|tofile)$"
)  # noqa: E501
MAX_NEW_COLUMNS = 10  # промпт просит ≤ 5; запас на случай, когда LLM разворачивает идею шире
SANDBOX_TIMEOUT = 120.0  # сек: векторный код на 100–500 тыс. строк укладывается в секунды
SANDBOX_MEM_MB = 8192  # ≥ 2048 только на импорт pandas (runtime/sandbox.py); таблица ×2–3 на копии
LLM_FIELDS = ("id", "title", "rationale", "code")
MAX_ID_LEN = 40


def check_feature_code(code: str, target: str | None = None) -> str | None:
    """Причина отклонить код признаков или None, если по тексту он допустим."""
    if not code or not code.strip():
        return "код пустой"
    if len(code) > MAX_CODE_CHARS:
        return f"код длиннее {MAX_CODE_CHARS} символов"
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return f"синтаксическая ошибка: {exc.msg} (строка {exc.lineno})"
    funcs = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == FUNCTION_NAME]
    if not funcs:
        return f"нет функции {FUNCTION_NAME}(df) на верхнем уровне"
    if len(funcs[0].args.args) != 1:
        return f"{FUNCTION_NAME} должна принимать ровно один аргумент — таблицу признаков"
    for node in ast.walk(tree):
        if isinstance(node, ast.Import | ast.ImportFrom):
            names = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module or ""]
            bad = [n for n in names if n.split(".")[0] not in ALLOWED_IMPORTS]
            if bad:
                return f"запрещённый импорт: {', '.join(bad)} (можно: {', '.join(sorted(ALLOWED_IMPORTS))})"
        elif isinstance(node, ast.Name) and node.id in FORBIDDEN_CALLS:
            return f"запрещено: {node.id}"
        elif isinstance(node, ast.Attribute):
            if node.attr.startswith("__"):
                return f"запрещён доступ к служебному атрибуту {node.attr}"
            if FILE_IO_ATTR.match(node.attr):
                return f"запрещена работа с файлами: .{node.attr}"
        elif isinstance(node, ast.Constant) and target and node.value == target:
            return f"код обращается к target «{target}» — его в таблице нет, это признак утечки"
        elif isinstance(node, ast.Global | ast.Nonlocal):
            return "запрещены global/nonlocal"
    return None


def describe_columns(profile: DataProfile, limit: int = 60) -> str:
    """Колонки с типами — LLM должна знать точные имена, иначе код упадёт на KeyError."""
    cols = [f"{c} (число)" for c in profile.numeric_features] + [
        f"{c} (категория)" for c in profile.categorical_features
    ]
    shown = ", ".join(cols[:limit])
    return shown + (f" … и ещё {len(cols) - limit}" if len(cols) > limit else "")


def build_feature_messages(spec: TaskSpec, profile: DataProfile, tried: list[str]) -> list[dict] | None:
    """Запрос одной гипотезы-признаков. tried — названия уже проверенных идей признаков. None — промпт не собрался."""
    system, template = load_prompt("system"), load_prompt("features")
    if system is None or template is None:
        return None
    try:
        user = Template(template).substitute(
            task=describe_task(spec),
            profile=describe_profile(profile),
            columns=describe_columns(profile),
            tried="\n".join(f"- {t}" for t in tried) or "Пока ничего.",
            function=FUNCTION_NAME,
            imports=", ".join(sorted(ALLOWED_IMPORTS)),
        )
    except (KeyError, ValueError) as exc:
        log.warning("Промпт features.md не собран: %s", exc)
        return None
    return [{"role": "system", "content": system.strip()}, {"role": "user", "content": user.strip()}]


def parse_feature_hypothesis(
    raw, spec: TaskSpec, history: list[tuple[Hypothesis, ExperimentResult]]
) -> Hypothesis | None:
    """Ответ LLM (JSON) → Hypothesis вида features с проверенным кодом, или None."""
    if isinstance(raw, list) and raw:
        raw = raw[0]
    if not isinstance(raw, dict):
        return None
    item = {f: raw[f] for f in LLM_FIELDS if f in raw}
    reason = check_feature_code(str(item.get("code", "")), spec.target)
    if reason:
        log.info("Код признаков от LLM отброшен: %s", reason)
        return None
    seen = {h.id for h, _ in history}
    base_id = re.sub(r"\W+", "_", str(item.get("id", "")), flags=re.ASCII).strip("_")[:MAX_ID_LEN] or "llm_features"
    hyp_id, n = base_id, 2
    while hyp_id in seen:
        hyp_id, n = f"{base_id}_{n}", n + 1
    try:
        return Hypothesis.model_validate(
            item | {"id": hyp_id, "kind": HypothesisKind.FEATURES, "source": "llm", "params": {"model": "lightgbm"}}
        )
    except ValidationError as exc:
        log.info("Гипотеза признаков от LLM не по схеме: %s", exc.errors()[0]["msg"])
        return None


# Обвязка вокруг кода LLM: читает таблицу, зовёт add_features и сохраняет отчёт о форме ответа и новые
# числовые колонки. Назад в агента идут только JSON и числовая матрица .npy (без pickle): прочитать их
# безопасно, даже если код повёл себя неожиданно. Работа с файлами — здесь; коду LLM она запрещена.
_RUNNER = """

import json as _json

import numpy as _np
import pandas as _pd

_df = _pd.read_pickle("features_in.pkl")
_out = {function}(_df.copy())
if not isinstance(_out, _pd.DataFrame):
    raise TypeError(f"{function} вернула {{type(_out).__name__}}, а не DataFrame")
_new = [c for c in _out.columns if c not in _df.columns]
_num = [c for c in _new if _pd.api.types.is_numeric_dtype(_out[c])]
_report = {{
    "rows": len(_out),
    "same_index": bool(_out.index.equals(_df.index)),
    "missing": [str(c) for c in _df.columns if c not in _out.columns],
    "numeric": [str(c) for c in _num],
    "other": [str(c) for c in _new if c not in _num],
}}
if _num and _report["rows"] == len(_df):
    _np.save("features_out.npy", _out[_num].to_numpy(dtype="float64"), allow_pickle=False)
with open("features_report.json", "w", encoding="utf-8") as _f:
    _json.dump(_report, _f, ensure_ascii=False)
"""


def apply_feature_code(
    code: str,
    df: pd.DataFrame,
    target: str | None = None,
    workdir: Path | None = None,
    timeout: float = SANDBOX_TIMEOUT,
    mem_mb: int = SANDBOX_MEM_MB,
) -> tuple[pd.DataFrame | None, str]:
    """Выполнить код признаков в песочнице на df (train + test без target).

    Возвращает (исходный df + новые числовые колонки, примечание — обычно "") или (None, причина).
    Исходные колонки берутся из df, а не из ответа кода: код не может незаметно испортить или подменить
    имеющиеся признаки. Никогда не бросает исключений — любой сбой превращается в причину для истории.
    """
    reason = check_feature_code(code, target)
    if reason:
        return None, reason
    if target is not None and target in df.columns:
        return None, f"в таблицу попал target «{target}» — код признаков его видеть не должен"
    try:
        with tempfile.TemporaryDirectory(prefix="features-", dir=workdir) as tmp:
            folder = Path(tmp)
            df.to_pickle(folder / "features_in.pkl")  # pickle пишем мы, читает песочница — это безопасно
            res = run_code(code + _RUNNER.format(function=FUNCTION_NAME), folder, timeout=timeout, mem_mb=mem_mb)
            if res.timed_out:
                return None, f"код признаков не уложился в {timeout:.0f} с"
            if res.exit_code != 0:
                last = (res.stderr.strip().splitlines() or ["без сообщения"])[-1]
                return None, f"код признаков упал: {last[:200]}"
            report = json.loads((folder / "features_report.json").read_text(encoding="utf-8"))
            values = None
            if (folder / "features_out.npy").exists():
                values = np.load(folder / "features_out.npy", allow_pickle=False)
    except Exception as exc:  # noqa: BLE001 — сбой обмена файлами = гипотеза не удалась, агент работает дальше
        return None, f"обмен с песочницей не удался: {type(exc).__name__}: {exc}"[:300]
    return _attach(df, report, values)


def _attach(df: pd.DataFrame, report: dict, values: np.ndarray | None) -> tuple[pd.DataFrame | None, str]:
    """Проверить отчёт кода и приклеить к df только новые числовые колонки."""
    if report.get("rows") != len(df) or not report.get("same_index"):
        return None, f"код изменил строки: было {len(df)}, стало {report.get('rows')} (или другой порядок)"
    if report.get("missing"):
        return None, f"код удалил исходные колонки: {', '.join(report['missing'][:5])}"
    names = list(report.get("numeric") or [])
    if not names or values is None:
        return None, "код не добавил ни одной числовой колонки"
    if len(names) > MAX_NEW_COLUMNS:
        return None, f"код добавил {len(names)} колонок — больше {MAX_NEW_COLUMNS}"
    if values.shape != (len(df), len(names)) or len(set(names)) != len(names) or set(names) & set(map(str, df.columns)):
        return None, "ответ кода признаков не сходится с таблицей (размер или имена колонок)"
    added = pd.DataFrame(values, index=df.index, columns=names).replace([np.inf, -np.inf], np.nan)
    constant = [c for c in names if added[c].nunique(dropna=True) <= 1]
    added = added.drop(columns=constant)
    if added.empty:
        return None, "все новые колонки константные — пользы для модели нет"
    skipped = list(report.get("other") or []) + constant
    note = f"пропущены нечисловые/константные: {', '.join(skipped[:5])}" if skipped else ""
    return pd.concat([df, added], axis=1), note
