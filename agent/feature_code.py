"""Код признаков от LLM (A-L3-1): промпт, разбор ответа и проверка кода до песочницы. Владелец — agent.

Контракт кода (interfaces/schemas.py, Hypothesis.code; ADR 0008 капитана, D4): код обязан определять
`add_features(df) -> df`; df — признаки train и test вместе БЕЗ target, на выходе — те же строки
в том же порядке плюс новые колонки. Исполняется только в песочнице (runtime/sandbox.py).

check_feature_code — дешёвый предфильтр по тексту (AST), а не защита: он отсеивает очевидно опасный
или сломанный код, чтобы не тратить на него запуск песочницы. Главный барьер — песочница.
"""

from __future__ import annotations

import ast
import re
from string import Template

from pydantic import ValidationError

from agent.prompting import describe_profile, describe_task, load_prompt
from interfaces.schemas import DataProfile, ExperimentResult, Hypothesis, HypothesisKind, TaskSpec
from runtime.log import get_logger

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
