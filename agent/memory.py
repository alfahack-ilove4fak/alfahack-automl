"""Память цикла AutoResearch: решения по гипотезам и выводы LLM после неудач. Владелец — agent.

История [(Hypothesis, ExperimentResult)] — контракт и не хранит, приняли ли гипотезу и почему.
Это знание живёт здесь, рядом с циклом, и попадает в промпт в сжатом виде.
"""

from __future__ import annotations

from dataclasses import dataclass, field

MAX_LESSONS = 3  # в промпт идут только последние выводы — старые уже учтены в новых гипотезах
MAX_LESSON_CHARS = 300


@dataclass
class ResearchMemory:
    verdicts: dict[str, str] = field(default_factory=dict)  # id гипотезы → «принята» / «отклонена: почему»
    lessons: list[str] = field(default_factory=list)  # короткие выводы LLM после неудачных пачек
    best_id: str | None = None  # текущий лучший — то, что принял цикл и что лежит в submission.csv
    llm_rejected: list[str] = field(default_factory=list)  # «id: причина» гипотез LLM, отброшенных при разборе

    def record(self, hypothesis_id: str, accepted: bool, reason: str = "") -> None:
        if accepted:
            self.best_id = hypothesis_id
        verdict = "принята" if accepted else "отклонена"
        self.verdicts[hypothesis_id] = f"{verdict}: {reason}" if reason else verdict

    def add_lesson(self, text: str | None) -> None:
        text = " ".join((text or "").split())[:MAX_LESSON_CHARS]
        if text:
            self.lessons = (self.lessons + [text])[-MAX_LESSONS:]
