"""Графики для питча (docs/pitch/). Данные — docs/pitch/data/*.csv (выгрузка стенда research/bench_playbook.py).

    uv run python research/pitch_charts.py      # → docs/pitch/img/*.png

График 1 «CV против честной проверки»: каждая точка — гипотеза на задаче. По X — выигрыш над бейзлайном
по CV агента (в его схеме валидации), по Y — на честной отложенной части. Цвет — решение агента по его правилу
(agent.acceptance.accept): принял и честно лучше, принял и честно хуже, отклонил. Данные — стенд v2 и #237.
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
MISSED_GAIN_PCT = 1.0  # «отверг, а честно лучше» считаем упущенным, только если выигрыш заметный
LIM = 40.0  # оси, %: дальше — выбросы целей «не под метрику» (−10⁶ %)


def cv_vs_honest(data: Path, out: Path) -> dict:
    d = pd.read_csv(data)
    acc = d.agent_accepts.astype(bool)
    acc_ok, acc_bad = acc & (d.honest_gain_pct >= 0), acc & (d.honest_gain_pct < 0)
    stats = {
        "pairs": len(d),
        "accepted": int(acc.sum()),
        "accepted_worse": int(acc_bad.sum()),
        "rejected_better": int((~acc & (d.honest_gain_pct > MISSED_GAIN_PCT)).sum()),
        "tasks": d.task.nunique(),
    }
    # Цели «не под метрику» дают −10⁶ %: прижимаем к краю, иначе остальные точки сливаются в одну
    x, y = d.cv_gain_pct.clip(-LIM, LIM), d.honest_gain_pct.clip(-LIM, LIM)

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
        x[~acc],
        y[~acc],
        s=40,
        marker="o",
        color=INK_MUTED,
        alpha=0.45,
        edgecolor="none",
        zorder=2,
        label="агент отклонил",
    )
    ax.scatter(
        x[acc_ok],
        y[acc_ok],
        s=80,
        marker="o",
        color=OK_COLOR,
        edgecolor=SURFACE,
        linewidth=2,
        zorder=3,
        label="агент принял — честно лучше",
    )
    ax.scatter(
        x[acc_bad],
        y[acc_bad],
        s=120,
        marker="D",
        color=WRONG_COLOR,
        edgecolor=SURFACE,
        linewidth=2,
        zorder=4,
        label="агент принял — честно хуже",
    )

    for _, r in d[acc_bad].iterrows():
        ax.annotate(
            f"{r.hypothesis}\n{r.task}: CV {r.cv_gain_pct:+.1f}%, честно {r.honest_gain_pct:+.1f}%",
            (min(max(r.cv_gain_pct, -LIM), LIM), min(max(r.honest_gain_pct, -LIM), LIM)),
            xytext=(-14, -18),
            textcoords="offset points",
            fontsize=10,
            color=INK_2,
            ha="right",
        )

    ax.set_xlim(-LIM * 1.05, LIM * 1.05)
    ax.set_ylim(-LIM * 1.05, LIM * 1.05)
    ax.set_xlabel("Выигрыш над бейзлайном по кросс-валидации агента, %", color=INK_2, fontsize=11)
    ax.set_ylabel("Выигрыш на честной отложенной части, %", color=INK_2, fontsize=11)
    ax.tick_params(colors=INK_2, labelsize=10)
    fig.suptitle(
        f"Агент ошибся с принятием гипотезы {stats['accepted_worse']} раз из {stats['accepted']}",
        x=0.06,
        y=0.97,
        ha="left",
        fontsize=16,
        color=INK,
        weight="bold",
    )
    ax.set_title(
        f"{stats['tasks']} задач × гипотезы playbook, {stats['pairs']} проверок. Принимает только то, что лучше "
        f"на тех же фолдах с запасом на шум (t ≥ 2); платит за это осторожностью — отклонено {stats['rejected_better']} "
        f"честно лучших > {MISSED_GAIN_PCT:.0f}%",
        loc="left",
        fontsize=10.5,
        color=INK_2,
        pad=10,
        wrap=True,
    )
    ax.legend(loc="upper left", frameon=False, fontsize=10, labelcolor=INK_2)
    ax.text(
        0.99,
        0.02,
        f"значения за ±{LIM:.0f}% показаны на краю",
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
