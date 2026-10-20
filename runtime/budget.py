"""Бюджет запуска: время, память, рубли на LLM. Владелец — капитан.

Переменные окружения (для локальных прогонов; на платформе не задаются — берутся значения по умолчанию):
  AGENT_TIME_BUDGET_SEC   — сколько секунд даём агенту (по умолчанию 3 часа, как на платформе)
  AGENT_SAFETY_SEC        — запас в конце на финальное обучение и запись ответа
  AGENT_RAM_LIMIT_MB      — потолок памяти; по умолчанию 90% от доступной на машине
  LLM_RUB_LIMIT           — сколько рублей агент может потратить на LLM за запуск
"""

from __future__ import annotations

import os
import time

import psutil

from interfaces.schemas import RunBudget

PLATFORM_SECONDS = 3 * 3600


def make_budget(started_at: float | None = None) -> RunBudget:
    total = float(os.getenv("AGENT_TIME_BUDGET_SEC", PLATFORM_SECONDS))
    # Запас по умолчанию: 15 мин на платформе, пропорционально меньше на коротких локальных прогонах
    default_safety = min(15 * 60, total * 0.15)
    return RunBudget(
        started_at=started_at or time.time(),
        total_seconds=total,
        safety_seconds=float(os.getenv("AGENT_SAFETY_SEC", default_safety)),
        ram_limit_mb=float(os.getenv("AGENT_RAM_LIMIT_MB", available_ram_mb() * 0.9)),
        llm_rub_limit=float(os.getenv("LLM_RUB_LIMIT", 0)),
    )


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
