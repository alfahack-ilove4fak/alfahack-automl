"""Гипотезы от LLM: промпт → JSON → проверенные Hypothesis. Владелец — LLM/agent-инженер.

Ответу модели не доверяем: каждая гипотеза проходит pydantic-валидацию и фильтры
(допустимый kind, только известные ключи params, не повтор истории). Всё, что не прошло, — выбрасываем.
Любой сбой → пустой список, и propose_hypotheses добирает гипотезы из playbook.
"""

from __future__ import annotations

import re

from pydantic import ValidationError

from agent.hypotheses import params_key, run_scheme
from agent.llm_client import LLMClient
from agent.memory import ResearchMemory
from agent.prompting import allowed_param_keys, build_hypothesis_messages, build_reflection_messages
from interfaces.schemas import DataProfile, ExperimentResult, Hypothesis, HypothesisKind, RunBudget, TaskSpec
from runtime.log import get_logger

log = get_logger("llm_hypotheses")

# features/ensemble требуют кода и песочницы (A-L3-1) — пока LLM их не предлагает
ALLOWED_KINDS = {HypothesisKind.HYPERPARAMS, HypothesisKind.PREPROCESSING, HypothesisKind.MODEL}
LLM_FIELDS = ("id", "kind", "title", "rationale", "params")
MAX_ID_LEN = 40
TEMPERATURE = 0.4  # чуть выше дефолта: нужны разные идеи, а не одна и та же
MAX_TOKENS = 800  # 3 гипотезы с короткими rationale укладываются с запасом


def _items(raw) -> list:
    if isinstance(raw, dict):
        raw = raw.get("hypotheses", [raw])  # модель иногда оборачивает список в объект
    return raw if isinstance(raw, list) else []


def parse_hypotheses(raw, history: list[tuple[Hypothesis, ExperimentResult]], k: int) -> list[Hypothesis]:
    """Ответ LLM (уже JSON) → не более k новых гипотез. Мусорные элементы пропускаются по одному."""
    allowed_params = allowed_param_keys()
    seen_ids = {h.id for h, _ in history}
    scheme = run_scheme(history)
    seen_params = {params_key(h.params, scheme) for h, _ in history}
    out: list[Hypothesis] = []
    for item in _items(raw):
        if not isinstance(item, dict):
            continue
        try:
            hyp = Hypothesis.model_validate({f: item[f] for f in LLM_FIELDS if f in item} | {"source": "llm"})
        except ValidationError as exc:
            log.info("Гипотеза LLM отброшена — не по схеме: %s", exc.errors()[0]["msg"])
            continue
        unknown = set(hyp.params) - allowed_params
        if hyp.kind not in ALLOWED_KINDS or unknown:
            log.info("Гипотеза LLM %s отброшена: kind=%s, неизвестные params=%s", hyp.id, hyp.kind.value, unknown)
            continue
        if not isinstance(hyp.params.get("model_params", {}), dict):
            continue
        key = params_key(hyp.params, scheme)
        if key in seen_params:
            log.info("Гипотеза LLM %s отброшена — повтор уже проверенных params", hyp.id)
            continue
        hyp_id = re.sub(r"\W+", "_", hyp.id, flags=re.ASCII).strip("_")[:MAX_ID_LEN] or "llm"
        base_id, n = hyp_id, 2
        while hyp_id in seen_ids:  # тот же id, но новые params — переименовываем, а не теряем идею
            hyp_id, n = f"{base_id}_{n}", n + 1
        hyp = hyp.model_copy(update={"id": hyp_id})
        seen_ids.add(hyp_id)
        seen_params.add(key)
        out.append(hyp)
        if len(out) == k:
            break
    return out


def llm_hypotheses(
    spec: TaskSpec,
    profile: DataProfile,
    history: list[tuple[Hypothesis, ExperimentResult]],
    k: int,
    llm: LLMClient,
    memory: ResearchMemory | None = None,
    budget: RunBudget | None = None,
) -> list[Hypothesis]:
    if not llm.available():
        return []
    messages = build_hypothesis_messages(spec, profile, history, k=k, memory=memory)
    if messages is None:
        return []
    try:
        raw = llm.chat_json(
            messages, temperature=TEMPERATURE, max_tokens=MAX_TOKENS, seconds_left=_left(budget), purpose="hypotheses"
        )
        out = parse_hypotheses(raw, history, k)
    except Exception:  # noqa: BLE001 — сбой LLM-части не должен останавливать цикл
        log.exception("Гипотезы от LLM не получены")
        return []
    log.info("LLM предложила %d гипотез (потрачено %.2f ₽ за %d запросов)", len(out), llm.rub_spent, llm.calls)
    return out


REFLECTION_MAX_TOKENS = 150


def _left(budget: RunBudget | None) -> float | None:
    return None if budget is None else budget.seconds_left()


def reflect(
    spec: TaskSpec,
    best: tuple[Hypothesis, ExperimentResult],
    batch: list[tuple[Hypothesis, ExperimentResult]],
    llm: LLMClient,
    memory: ResearchMemory,
    budget: RunBudget | None = None,
) -> None:
    """Короткий вывод LLM «почему пачка не сработала» → memory.lessons. Без LLM — ничего не делаем."""
    if not batch or not llm.available():
        return
    messages = build_reflection_messages(spec, best, batch)
    if messages is None:
        return
    try:
        memory.add_lesson(
            llm.chat(
                messages,
                temperature=0.2,
                max_tokens=REFLECTION_MAX_TOKENS,
                seconds_left=_left(budget),
                purpose="reflect",
            )
        )
    except Exception:  # noqa: BLE001 — рефлексия необязательна
        log.exception("Рефлексия LLM не получена")
