"""Сборка промптов для LLM из файлов agent/prompts/*.md. Владелец — LLM/agent-инженер.

Промпты — шаблоны string.Template: плейсхолдеры вида $task. Фигурные скобки JSON в тексте
промпта поэтому не мешают (с str.format пришлось бы экранировать каждую).
В промпт идут только сжатые описания TaskSpec, DataProfile и истории — никаких сырых данных.
Любая проблема (нет файла, лишний плейсхолдер) → None: тогда гипотезы берутся из playbook.
"""

from __future__ import annotations

import json
import re
import textwrap
from pathlib import Path
from string import Template

import ml.experiment
from agent.hypotheses import _applies, load_playbook
from agent.memory import ResearchMemory
from interfaces.schemas import DataProfile, ExperimentResult, ExperimentStatus, Hypothesis, TaskSpec
from runtime.log import get_logger

log = get_logger("prompting")

PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"
MAX_DESCRIPTION_CHARS = 1500  # task.md целиком может быть длинным — LLM хватает начала
MAX_FEATURES_SHOWN = 30
MAX_CLASSES_SHOWN = 20
MAX_MISSING_SHOWN = 10
HISTORY_LAST_N = 8
MAX_NOTES_SHOWN = 6
HISTORY_BEST_K = 3


def load_prompt(name: str) -> str | None:
    path = PROMPTS_DIR / f"{name}.md"
    try:
        return path.read_text(encoding="utf-8")
    except OSError as exc:
        log.warning("Промпт %s не прочитан: %s", path.name, exc)
        return None


def _num(value) -> str:
    return f"{value:.4g}" if isinstance(value, float) else str(value)


def _short_list(items: list[str], limit: int) -> str:
    if not items:
        return "нет"
    shown = ", ".join(items[:limit])
    return shown + (f" … и ещё {len(items) - limit}" if len(items) > limit else "")


def describe_task(spec: TaskSpec) -> str:
    direction = "больше — лучше" if spec.metric.greater_is_better else "меньше — лучше"
    lines = [
        f"- Тип: {spec.task_type.value}",
        f"- Target: {spec.target}",
        f"- Метрика: {spec.metric.name} ({direction})",
    ]
    if spec.class_labels:
        lines.append(f"- Классов: {len(spec.class_labels)}")
    for note in spec.notes[:MAX_NOTES_SHOWN]:  # что заметили при разборе: формат ответа, время, условия из текста
        lines.append(f"- Заметка: {note}")
    description = spec.description.strip()
    if description:
        if len(description) > MAX_DESCRIPTION_CHARS:
            description = description[:MAX_DESCRIPTION_CHARS] + " …"
        lines.append(f"- Описание задачи:\n{description}")
    return "\n".join(lines)


def describe_profile(profile: DataProfile) -> str:
    lines = [
        f"- Строк: train {profile.n_train}, test {profile.n_test}",
        f"- Числовые признаки ({len(profile.numeric_features)}): "
        + _short_list(profile.numeric_features, MAX_FEATURES_SHOWN),
        f"- Категориальные признаки ({len(profile.categorical_features)}): "
        + _short_list(profile.categorical_features, MAX_FEATURES_SHOWN),
    ]
    if profile.missing_share:
        top = sorted(profile.missing_share.items(), key=lambda kv: -kv[1])[:MAX_MISSING_SHOWN]
        lines.append("- Пропуски (доля): " + ", ".join(f"{k} {v:.2f}" for k, v in top))
    summary = dict(profile.target_summary)
    counts = summary.pop("class_counts", None)
    if summary:
        lines.append("- Target: " + ", ".join(f"{k}={_num(v)}" for k, v in summary.items()))
    if counts:
        top = sorted(counts.items(), key=lambda kv: -kv[1])[:MAX_CLASSES_SHOWN]
        lines.append("- Размеры классов: " + ", ".join(f"{k}: {v}" for k, v in top))
    lines += [f"- Предупреждение: {w}" for w in profile.warnings]  # без эмодзи: модель копирует стиль промпта
    return "\n".join(lines)


def _outcome(res: ExperimentResult) -> str:
    if res.status == ExperimentStatus.OK and res.mean_score is not None:
        return f"{res.mean_score:.5f} ± {res.std_score or 0:.5f}"
    return f"{res.status.value}: {(res.error or '')[:80]}"


def describe_history(
    history: list[tuple[Hypothesis, ExperimentResult]],
    greater_is_better: bool,
    memory: ResearchMemory | None = None,
    last_n: int = HISTORY_LAST_N,
    best_k: int = HISTORY_BEST_K,
) -> str:
    """Сжатая история: лучшие best_k по метрике + последние last_n, в порядке проверки, с решением по каждой."""
    if not history:
        return "Пока не было ни одного эксперимента."
    scored = [i for i, (_, r) in enumerate(history) if r.status == ExperimentStatus.OK and r.mean_score is not None]
    scored.sort(key=lambda i: history[i][1].mean_score, reverse=greater_is_better)
    # «лучший» — тот, кого принял цикл (memory.best_id), а не максимум среднего: после правила принятия
    # шумовая гипотеза может иметь лучшее среднее и быть отклонённой (ревью #95)
    best_id = memory.best_id if memory and memory.best_id else (history[scored[0]][0].id if scored else None)
    best_rows = {i for i, (h, _) in enumerate(history) if h.id == best_id}
    keep = sorted(set(scored[:best_k]) | best_rows | set(range(max(0, len(history) - last_n), len(history))))
    verdicts = memory.verdicts if memory else {}
    rows = ["| id | kind | params | результат | решение |", "|---|---|---|---|---|"]
    for i in keep:
        hyp, res = history[i]
        mark = " (лучший)" if i in best_rows else ""
        params = json.dumps(hyp.params, ensure_ascii=False)
        verdict = verdicts.get(hyp.id, "—")
        rows.append(f"| {hyp.id} | {hyp.kind.value} | {params} | {_outcome(res)}{mark} | {verdict} |")
    hidden = len(history) - len(keep)
    if hidden > 0:
        rows.insert(0, f"(всего экспериментов {len(history)}: показаны лучшие {best_k} и последние {last_n})")
    return "\n".join(rows)


def describe_lessons(memory: ResearchMemory | None) -> str:
    if not memory or not memory.lessons:
        return "Пока нет."
    return "\n".join(f"- {lesson}" for lesson in memory.lessons)


def params_doc() -> str:
    """Список ключей params берём из docstring ml/experiment.py — так он не разойдётся с исполнителем."""
    doc = ml.experiment.__doc__ or ""
    marker = "Поддерживаемые params"
    if marker in doc:
        doc = doc.split(marker, 1)[1].split("\n", 1)[-1]
    return textwrap.dedent(doc).strip()


def allowed_param_keys() -> set[str]:
    """Ключи params, которые понимает исполнитель: первые слова строк вида «ключ — описание»."""
    return set(re.findall(r"^(\w+)\s+—", params_doc(), re.MULTILINE))


def playbook_examples(spec: TaskSpec, profile: DataProfile, n: int = 2) -> str:
    """Few-shot пример ответа: подходящие задаче гипотезы playbook, по одной на строку (короче для LLM)."""
    fields = ("id", "kind", "title", "rationale", "params")
    items = [{f: item[f] for f in fields if f in item} for item in load_playbook() if _applies(item, spec, profile)]
    return "[\n" + ",\n".join("  " + json.dumps(item, ensure_ascii=False) for item in items[:n]) + "\n]"


def _render(name: str, **values) -> str | None:
    template = load_prompt(name)
    if template is None:
        return None
    try:
        return Template(template).substitute(**values).strip()
    except (KeyError, ValueError) as exc:
        log.warning("Промпт %s.md не собран: %s", name, exc)
        return None


def build_hypothesis_messages(
    spec: TaskSpec,
    profile: DataProfile,
    history: list[tuple[Hypothesis, ExperimentResult]],
    k: int = 3,
    memory: ResearchMemory | None = None,
) -> list[dict] | None:
    """Сообщения для LLMClient.chat_json: system + запрос k гипотез. None — промпт собрать не удалось."""
    system = load_prompt("system")
    user = _render(
        "hypotheses",
        task=describe_task(spec),
        profile=describe_profile(profile),
        history=describe_history(history, spec.metric.greater_is_better, memory),
        lessons=describe_lessons(memory),
        params_doc=params_doc(),
        examples=playbook_examples(spec, profile),
        k=k,
    )
    if system is None or user is None:
        return None
    return [{"role": "system", "content": system.strip()}, {"role": "user", "content": user}]


def build_reflection_messages(
    spec: TaskSpec,
    best: tuple[Hypothesis, ExperimentResult],
    batch: list[tuple[Hypothesis, ExperimentResult]],
) -> list[dict] | None:
    """Запрос короткого вывода «почему пачка гипотез не сработала». None — промпт собрать не удалось."""
    system = load_prompt("system")
    user = _render(
        "reflection",
        task=describe_task(spec),
        best=describe_history([best], spec.metric.greater_is_better),
        batch=describe_history(batch, spec.metric.greater_is_better),
    )
    if system is None or user is None:
        return None
    return [{"role": "system", "content": system.strip()}, {"role": "user", "content": user}]


def build_critic_messages(
    spec: TaskSpec,
    profile: DataProfile,
    history: list[tuple[Hypothesis, ExperimentResult]],
    candidates: str,
    memory: ResearchMemory | None = None,
) -> list[dict] | None:
    """Запрос к LLM-критику: проверить пачку гипотез по чек-листу. candidates — гипотезы в JSON. None — не собрался."""
    system = load_prompt("system")
    user = _render(
        "critic",
        task=describe_task(spec),
        profile=describe_profile(profile),
        history=describe_history(history, spec.metric.greater_is_better, memory),
        candidates=candidates,
    )
    if system is None or user is None:
        return None
    return [{"role": "system", "content": system.strip()}, {"role": "user", "content": user}]
