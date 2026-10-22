"""Бюджет запуска: время по этапам, память, рубли на LLM. Владелец — капитан.

Время делится на этапы долями от общего бюджета (константы ниже, на запуске переопределяются окружением):

    |-- разбор + бейзлайн --|------ AutoResearch ------|-- финал --|-- запас --|
           до 20%                   остаток (65%)         ≥ 10%       ≥ 5%

- «разбор + бейзлайн» — мягкий потолок: бейзлайн нельзя обрывать, иначе останется тривиальный ответ;
  при превышении агент пишет предупреждение, а AutoResearch просто получает меньше времени;
- AutoResearch заканчивается за `safety_seconds` до конца бюджета — это резерв «финал + запас»;
  за ним следит `RunBudget.has_time()` в цикле и в фолдах CV;
- финал — обучение на всём train и запись ответа после последней принятой гипотезы;
- запас — время между жёсткой отсечкой сторожа (`runtime/watchdog.py`) и лимитом платформы.

Переменные окружения (для локальных прогонов; на платформе не задаются — берутся значения по умолчанию):
  AGENT_TIME_BUDGET_SEC   — сколько секунд даём агенту (по умолчанию 3 часа, как на платформе)
  AGENT_SAFETY_SEC        — резерв в конце (финал + запас) в секундах; перекрывает доли ниже
  AGENT_BASELINE_SHARE    — доля на разбор + бейзлайн (по умолчанию 0.20)
  AGENT_FINAL_SHARE       — доля на финальное обучение (по умолчанию 0.10)
  AGENT_SAFETY_SHARE      — доля запаса до лимита платформы (по умолчанию 0.05)
  AGENT_RAM_LIMIT_MB      — потолок памяти; по умолчанию 90% от доступной на машине
  LLM_RUB_LIMIT           — сколько рублей агент может потратить на LLM за запуск
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass

import psutil

from interfaces.schemas import RunBudget

PLATFORM_SECONDS = 3 * 3600

BASELINE_SHARE = 0.20  # разбор + бейзлайн: не больше
FINAL_SHARE = 0.10  # резерв на финальное обучение и запись ответа: не меньше
SAFETY_SHARE = 0.05  # запас до лимита платформы: не меньше
# AutoResearch получает остаток: 1 - 0.20 - 0.10 - 0.05 = 65%

FINAL_FIT_FACTOR = 3.0  # во сколько раз финал последней гипотезы может быть дольше замеренного финала бейзлайна
MAX_RESERVE_SHARE = 0.5  # резерв в конце никогда не съедает больше половины бюджета


@dataclass(frozen=True)
class Shares:
    baseline: float = BASELINE_SHARE
    final: float = FINAL_SHARE
    safety: float = SAFETY_SHARE

    @property
    def research(self) -> float:
        return 1.0 - self.baseline - self.final - self.safety

    @property
    def reserve(self) -> float:
        """Резерв в конце бюджета: финал + запас."""
        return self.final + self.safety


@dataclass(frozen=True)
class StagePlan:
    """Моменты (unix-время), к которым этапы должны закончиться. Считается один раз на старте."""

    started_at: float
    baseline_end: float  # мягкий потолок разбора + бейзлайна
    research_end: float  # конец AutoResearch на старте (потом сдвигается, если финал оказался долгим)
    hard_deadline: float  # жёсткая отсечка сторожа: к этому моменту процесс обязан завершиться
    platform_end: float  # лимит, после которого платформа остановит процесс принудительно

    def describe(self) -> str:
        def at(ts: float) -> str:
            return f"+{ts - self.started_at:.0f} с"

        return (
            f"бейзлайн до {at(self.baseline_end)}, AutoResearch до {at(self.research_end)}, "
            f"сторож {at(self.hard_deadline)}, лимит платформы {at(self.platform_end)}"
        )


def _env_share(name: str, default: float) -> float:
    try:
        value = float(os.getenv(name, default))
    except ValueError:
        return default
    return value if 0.0 <= value < 1.0 else default


def get_shares() -> Shares:
    """Доли из окружения; если они не оставляют AutoResearch хотя бы 10% — возвращаем значения по умолчанию."""
    shares = Shares(
        baseline=_env_share("AGENT_BASELINE_SHARE", BASELINE_SHARE),
        final=_env_share("AGENT_FINAL_SHARE", FINAL_SHARE),
        safety=_env_share("AGENT_SAFETY_SHARE", SAFETY_SHARE),
    )
    return shares if shares.research >= 0.1 else Shares()


def make_budget(started_at: float | None = None) -> RunBudget:
    total = float(os.getenv("AGENT_TIME_BUDGET_SEC", PLATFORM_SECONDS))
    default_safety = get_shares().reserve * total
    return RunBudget(
        started_at=time.time() if started_at is None else started_at,
        total_seconds=total,
        safety_seconds=float(os.getenv("AGENT_SAFETY_SEC", default_safety)),
        ram_limit_mb=float(os.getenv("AGENT_RAM_LIMIT_MB", available_ram_mb() * 0.9)),
        llm_rub_limit=float(os.getenv("LLM_RUB_LIMIT", 0)),
    )


def hard_margin_seconds(budget: RunBudget) -> float:
    """За сколько секунд до лимита платформы сработает сторож.

    По умолчанию — доля SAFETY_SHARE. Если резерв задан вручную (AGENT_SAFETY_SEC) и он мал, запас
    уменьшается пропорционально, чтобы жёсткая отсечка оставалась позже конца AutoResearch.
    """
    shares = get_shares()
    by_share = shares.safety * budget.total_seconds
    by_reserve = budget.safety_seconds * shares.safety / max(shares.reserve, 1e-9)
    return min(by_share, by_reserve)


def plan_stages(budget: RunBudget) -> StagePlan:
    shares = get_shares()
    platform_end = budget.started_at + budget.total_seconds
    return StagePlan(
        started_at=budget.started_at,
        baseline_end=budget.started_at + shares.baseline * budget.total_seconds,
        research_end=platform_end - budget.safety_seconds,
        hard_deadline=platform_end - hard_margin_seconds(budget),
        platform_end=platform_end,
    )


def ensure_final_reserve(budget: RunBudget, measured_final_seconds: float) -> float:
    """Увеличить резерв в конце, если замеренное финальное обучение не помещается в него с запасом.

    Вызывается один раз после первого сохранения ответа (бейзлайн): это единственный честный замер
    финала на полных данных. Резерв только растёт и не больше MAX_RESERVE_SHARE бюджета.
    Возвращает новое значение `safety_seconds`.
    """
    needed = hard_margin_seconds(budget) + FINAL_FIT_FACTOR * measured_final_seconds
    cap = MAX_RESERVE_SHARE * budget.total_seconds
    budget.safety_seconds = min(max(budget.safety_seconds, needed), cap)
    return budget.safety_seconds


def available_ram_mb() -> float:
    """Сколько памяти реально доступно сейчас (учитывает и лимит Docker через cgroup)."""
    avail = psutil.virtual_memory().available / 2**20
    for cgroup_file in ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"):
        try:
            raw = open(cgroup_file).read().strip()
            if raw.isdigit():
                avail = min(avail, int(raw) / 2**20)
        except OSError:
            pass
    return avail


def used_ram_mb() -> float:
    return psutil.Process().memory_info().rss / 2**20
