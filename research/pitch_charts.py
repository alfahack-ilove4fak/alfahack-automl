"""Графики для питча (docs/pitch/). Данные — docs/pitch/data/*.csv (выгрузка стенда research/bench_playbook.py).

    uv run python research/pitch_charts.py      # → docs/pitch/img/*.png

График 1 «CV против честной проверки»: каждая точка — гипотеза на задаче. По X — выигрыш над бейзлайном
по CV агента, по Y — на честной отложенной части. Оранжевым — случаи, где решение агента по CV неверное:
принял, а честно хуже бейзлайна, или отверг, а честно лучше больше чем на 1%.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
PITCH = ROOT / "docs" / "pitch"

# Палитра — категориальные слоты 1–2 (проверены на дальтонизм, all-pairs), текст — нейтральные чернила
SURFACE, INK, INK_2, INK_MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1"
OK_COLOR, WRONG_COLOR = "#2a78d6", "#eb6834"
MISSED_GAIN_PCT = 1.0  # «отверг, а честно лучше» считаем ошибкой, только если выигрыш заметный


def cv_vs_honest(data: Path, out: Path) -> dict:
    d = pd.read_csv(data)
    wrong = (d.agent_accepts & (d.honest_gain_pct < 0)) | (~d.agent_accepts & (d.honest_gain_pct > MISSED_GAIN_PCT))
    stats = {
        "pairs": len(d),
        "wrong": int(wrong.sum()),
        "accepted_worse": int((d.agent_accepts & (d.honest_gain_pct < 0)).sum()),
        "rejected_better": int((~d.agent_accepts & (d.honest_gain_pct > MISSED_GAIN_PCT)).sum()),
        "tasks": d.task.nunique(),
    }

    fig, ax = plt.subplots(figsize=(12, 6.75), dpi=150, facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    ax.axhline(0, color=INK_MUTED, lw=1, zorder=1)
    ax.axvline(0, color=INK_MUTED, lw=1, zorder=1)
    ax.grid(color=GRID, lw=0.8, zorder=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)

    # Форма — второе кодирование: цвет никогда не единственный носитель смысла
    ax.scatter(
        d.cv_gain_pct[~wrong],
        d.honest_gain_pct[~wrong],
        s=70,
        marker="o",
        color=OK_COLOR,
        edgecolor=SURFACE,
        linewidth=2,
        zorder=3,
        label="CV подсказала верное решение",
    )
    ax.scatter(
        d.cv_gain_pct[wrong],
        d.honest_gain_pct[wrong],
        s=110,
        marker="D",
        color=WRONG_COLOR,
        edgecolor=SURFACE,
        linewidth=2,
        zorder=4,
        label="CV подсказала неверное решение",
    )

    # Подписи — только у самых показательных точек
    top = d[wrong].reindex(d[wrong].honest_gain_pct.abs().sort_values(ascending=False).index).head(2)
    for _, r in top.iterrows():
        ax.annotate(
            f"{r.hypothesis}\n{r.task}: CV {r.cv_gain_pct:+.1f}%, честно {r.honest_gain_pct:+.1f}%",
            (r.cv_gain_pct, r.honest_gain_pct),
            xytext=(14, -4),
            textcoords="offset points",
            fontsize=10,
            color=INK_2,
            va="center",
        )

    lim_x = max(abs(d.cv_gain_pct).max(), 1) * 1.15
    ax.set_xlim(-lim_x, lim_x)
    ax.text(0.99, 0.02, "агент примет →", transform=ax.transAxes, ha="right", fontsize=10, color=INK_MUTED)
    ax.text(0.01, 0.02, "← агент отвергнет", transform=ax.transAxes, ha="left", fontsize=10, color=INK_MUTED)
    ax.set_xlabel("Выигрыш над бейзлайном по кросс-валидации агента, %", color=INK_2, fontsize=11)
    ax.set_ylabel("Выигрыш на честной проверке, %", color=INK_2, fontsize=11)
    ax.tick_params(colors=INK_2, labelsize=10)
    fig.suptitle(
        f"Обычная кросс-валидация ведёт агента к неверному решению в {stats['wrong']} случаях из " f"{stats['pairs']}",
        x=0.06,
        y=0.97,
        ha="left",
        fontsize=16,
        color=INK,
        weight="bold",
    )
    ax.set_title(
        f"{stats['tasks']} задач × гипотезы playbook. Неверно: принял, а честно хуже бейзлайна "
        f"({stats['accepted_worse']}), или отверг, а честно лучше > {MISSED_GAIN_PCT:.0f}% "
        f"({stats['rejected_better']})",
        loc="left",
        fontsize=10.5,
        color=INK_2,
        pad=10,
    )
    ax.legend(loc="lower right", bbox_to_anchor=(1, 0.07), frameon=False, fontsize=10, labelcolor=INK_2)
    ax.text(
        0.99,
        0.2,
        "порог «принять» — чуть правее нуля:\nзапас на шум 0.25·разброс по фолдам",
        transform=ax.transAxes,
        ha="right",
        fontsize=9,
        color=INK_MUTED,
    )
    fig.tight_layout(rect=(0.02, 0, 1, 0.955))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, facecolor=SURFACE)
    plt.close(fig)
    return stats


if __name__ == "__main__":
    print(cv_vs_honest(PITCH / "data" / "cv_vs_honest.csv", PITCH / "img" / "cv_vs_honest.png"))
