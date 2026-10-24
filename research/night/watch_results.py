"""Монитор результатов исследования и система контроля: живые прогоны, «до/после», гейт, ловушки — с вердиктами.

    uv run python research/night/watch_results.py --plan runs/oh/plan.json              # обновление каждые 15 с
    uv run python research/night/watch_results.py --plan runs/oh/plan.json --once       # один снимок
    uv run python research/night/watch_results.py --plan runs/oh/plan.json --once --markdown > report.md
    uv run python research/night/watch_results.py --plan runs/oh/plan.json --serve      # http://127.0.0.1:8765
    uv run python research/night/watch_results.py --plan runs/oh/plan.json --export snapshot.json  # снимок в файл
    uv run python research/night/watch_results.py --snapshot snapshot.json --serve     # показать его на другой машине

Скрипт только читает файлы (agent.log, CSV харнесса, jsonl run_alien, gate.md) — прогонам не мешает.
--serve — страница в браузере: на каждое обновление (раз в --interval секунд) файлы читаются заново;
/report.md — тот же снимок в markdown. По умолчанию слушает только 127.0.0.1. Наружу (--host 0.0.0.0 и т. п.)
не открывать: авторизации нет, сводку прогонов увидит любой в сети.
Код выхода --once: 0 — всё проверено и нет ❌; 1 — есть ❌; 2 — нет ❌, но что-то ещё идёт (⏳).

План — JSON (пример: research/night/control_plan.example.json). Пути — от текущей папки или абсолютные:
  title       — заголовок отчёта
  hypothesis  — id гипотезы playbook, решения по которой ищем в agent.log (например object_history)
  runs        — папки, где рекурсивно ищем agent.log (живые прогоны и решения по гипотезе)
  pairs       — «до/после»: [{name, before, after, criterion}] — CSV evaluation/harness.py или jsonl run_alien;
                criterion: false — пара не входит в К4 (например, перепроверка на 5 минутах)
  gate        — {report: gate.md, results: [harness.csv и alien_results.jsonl этого прогона гейта]}
  rechecks    — перепроверка провалов гейта: [{name, before: результат dev, after: результат ветки}]
  leak        — ловушки утечек: {before: jsonl dev, after: jsonl ветки} (run_alien.py --set leak)
  criteria    — пороги, см. DEFAULTS

Проверки:
  К1 Прогоны завершены   — нет живых agent.log (менялся недавно и нет «Финал»); брошенный прогон — ⚠️
  К2 Без падений         — у всех строк статус OK, у run_alien — код выхода 0 и нет problems
  К3 Без потерь          — ни одна задача «до/после» не потеряла больше noise баллов к сильному бейзлайну
  К4 Честный выигрыш     — метрика лучше на ≥ min_gain_pct % на ≥ min_tasks задачах (пары с criterion)
  К5 Принятие гипотезы   — где гипотеза принята, t ≥ t_min; ниже — ⚠️: принята по правилу последнего периода
  К6 Гейт                — gate.md проходит; провал, который повторился на dev (rechecks), — ⚠️ «не от ветки»
  К7 Ловушки утечек      — на каждой ловушке ветка не хуже dev больше noise и нет новых problems
  К8 Время               — каждый прогон уложился в бюджет + grace_sec
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from html import escape
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

OK, FAIL, WARN, WAIT, NA = "✅", "❌", "⚠️", "⏳", "—"
DEFAULTS = {"min_gain_pct": 2.0, "min_tasks": 2, "noise": 2.0, "t_min": 2.0, "grace_sec": 90, "stale_sec": 180}

STAMP = re.compile(r"^\[\s*([\d.]+)s\]")
LIMIT = re.compile(r"лимит платформы \+(\d+) с")
ITER = re.compile(r"Итерация (\d+): (\S+)")
BETTER = re.compile(r"✓ лучше: (\S+) → (\S+)")
REJECTED = re.compile(r"· не лучше: (\S+) \(лучший (\S+)\)")
BASELINE = re.compile(r"Шаг 1: бейзлайн \S+?=([-\d.eE+]+)")
T_STAT = re.compile(r"t=([-\d.]+)")
FOLDS = re.compile(r"лучше на (\d+/\d+) фолдах")


@dataclass
class Run:
    """Один agent.log: где агент сейчас и что он решил по гипотезе."""

    name: str
    elapsed: float = 0.0
    budget: int | None = None
    finished: bool = False
    age: float = 0.0
    iteration: str = ""
    best: str = ""
    decision: str = ""
    detail: str = ""
    t: float | None = None

    def state(self, stale_sec: float, grace_sec: float = 0) -> str:
        """Живой, пока лог менялся недавно или ещё не вышел бюджет: длинный шаг CV пишет в лог редко."""
        if self.finished:
            return "готов"
        quiet = max(stale_sec, (self.budget or 0) - self.elapsed + grace_sec)
        return "идёт" if self.age < quiet else "брошен?"


@dataclass
class Rec:
    """Строка результата задачи — из CSV харнесса или jsonl run_alien."""

    task: str
    metric: str
    value: float | None
    score: float  # балл к сильному бейзлайну — как в гейте
    ok: bool
    seconds: float | None
    minutes: float | None


def short(task: str) -> str:
    return task.rstrip("/").split("/")[-1]


def _float(text: str) -> float | None:
    try:
        return float(text)
    except (TypeError, ValueError):
        return None


def same_as_logged(text: str, value: float | None) -> bool:
    """Совпадает ли число из лога (округлённое при выводе) с точным значением — с точностью записи в логе."""
    logged = _float(text)
    if logged is None or value is None:
        return False
    if "e" in text.lower() or "." not in text:
        return abs(logged - value) <= 1e-4 * max(1.0, abs(value))
    return abs(logged - value) <= 0.5 * 10 ** -len(text.split(".")[1]) + 1e-12


def parse_log(path: Path, hypothesis: str, name: str, now: float | None = None) -> Run:
    """Разобрать agent.log: время, итерация, лучший CV и решение по гипотезе (с соседних строк лога)."""
    run = Run(name=name, age=(now or time.time()) - path.stat().st_mtime)
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    baseline = None
    hyp_at = None
    for i, line in enumerate(lines):
        if m := STAMP.match(line):
            run.elapsed = float(m.group(1))
        if m := LIMIT.search(line):
            run.budget = int(m.group(1))
        if m := BASELINE.search(line):
            baseline = _float(m.group(1))
        if m := ITER.search(line):
            run.iteration = f"{m.group(1)}: {m.group(2)}"
            if m.group(2) == hypothesis and hyp_at is None:
                hyp_at = i
        if m := BETTER.search(line):
            run.best = m.group(2)
        if "Финал" in line:
            run.finished = True
    if hyp_at is None:
        run.decision = "не дошла" if run.finished else "ещё не дошла"
        return run
    run.decision = "проверяется"
    for line in lines[hyp_at + 1 :]:
        if ITER.search(line):
            break
        if "✓ лучше" in line:
            m = BETTER.search(line)
            run.t = _float(T_STAT.search(line).group(1)) if T_STAT.search(line) else None
            folds = FOLDS.search(line)
            run.decision = "принята"
            run.detail = f"CV {m.group(1)} → {m.group(2)}" if m else ""
            run.detail += f", t={run.t}" if run.t is not None else ""
            run.detail += f", лучше на {folds.group(1)} фолдах" if folds else ""
            break
        if "· не лучше" in line:
            run.decision = "отклонена"
            m = REJECTED.search(line)
            if m:
                run.detail = f"CV {m.group(1)} против лучшего {m.group(2)}"
                if same_as_logged(m.group(1), baseline):
                    run.detail += " — CV = бейзлайну: операция ничего не добавила"
            break
        if "⊘" in line:
            run.decision = "пропущена"
            run.detail = line.split("⊘", 1)[1].strip()
            break
        if "✗" in line:
            run.decision = "упала"
            run.detail = line.split("✗", 1)[1].strip()[:120]
            break
    return run


def find_runs(dirs: list[str], hypothesis: str) -> list[Run]:
    runs, now = [], time.time()
    for d in dirs:
        base = Path(d)
        for log in sorted(base.rglob("agent.log")) if base.exists() else []:
            parts = [p for p in log.relative_to(base).parts[:-1] if p != "run"]
            runs.append(parse_log(log, hypothesis, "/".join([base.name, *parts]), now))
    return runs


def load_records(path: str | None) -> dict[str, Rec] | None:
    """Последняя строка по задаче; файла нет — None (прогон ещё не дописал результат)."""
    if not path or not Path(path).exists():
        return None
    out: dict[str, Rec] = {}
    if path.endswith(".jsonl"):
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            out[row["task"]] = Rec(
                task=row["task"],
                metric=str(row.get("metric", "")),
                value=_float(row.get("value")),
                score=float(row.get("score_vs_strong") or 0.0),
                ok=row.get("exit_code") == 0 and not row.get("problems"),
                seconds=_float(row.get("seconds")),
                minutes=_float(row.get("minutes")),
            )
        return out
    df = pd.read_csv(path)
    score_col = "model_score" if "model_score" in df.columns else "score"
    for _, row in df.groupby("task", sort=True).tail(1).iterrows():
        out[row["task"]] = Rec(
            task=row["task"],
            metric=str(row.get("metric", "")),
            value=None if pd.isna(row.get("value")) else float(row["value"]),
            score=float(row[score_col]) if not pd.isna(row[score_col]) else 0.0,
            ok=row.get("status") == "OK",
            seconds=_float(row.get("seconds")),
            minutes=_float(row.get("minutes")),
        )
    return out


def greater_is_better(before: Rec, after: Rec) -> bool | None:
    """Направление метрики: из реестра evaluation/metrics.py, иначе — по тому, куда сдвинулся балл."""
    name = before.metric.split("@")[0].lower()
    try:
        from evaluation.metrics import METRICS

        for metric in METRICS.values():
            if name == metric.name or name in metric.aliases:
                return metric.greater_is_better
    except Exception:  # noqa: BLE001 — монитор работает и без реестра
        pass
    if before.value is not None and after.value is not None and before.value != after.value:
        if before.score != after.score:
            return (after.value > before.value) == (after.score > before.score)
    return None


def gain_pct(before: Rec, after: Rec) -> float | None:
    """На сколько % метрика стала лучше (минус — хуже)."""
    if before.value is None or after.value is None or before.value == 0:
        return None
    if before.value == after.value:
        return 0.0
    higher = greater_is_better(before, after)
    if higher is None:
        return None
    diff = after.value - before.value if higher else before.value - after.value
    return 100 * diff / abs(before.value)


def find_rec(records: dict[str, Rec] | None, name: str) -> Rec | None:
    if not records:
        return None
    for task, rec in records.items():
        if task == name or short(task) == name:
            return rec
    return None


def load_gate(path: str | None) -> dict | None:
    if not path or not Path(path).exists():
        return None
    text = Path(path).read_text(encoding="utf-8")
    failures = [line[2:].strip() for line in text.splitlines() if line.startswith("- ❌")]
    names = [re.search(r"«([^»]+)»", f).group(1) for f in failures if re.search(r"«([^»]+)»", f)]
    return {"passed": "НЕ ПРОХОДИТ" not in text and "ПРОХОДИТ" in text, "failures": failures, "names": names}


def check(code: str, title: str, status: str, detail: str) -> dict:
    return {"code": code, "title": title, "status": status, "detail": detail}


def evaluate(plan: dict) -> dict:
    """Собрать снимок: прогоны, сравнения и проверки К1–К8."""
    crit = {**DEFAULTS, **plan.get("criteria", {})}
    noise, hyp = crit["noise"], plan.get("hypothesis", "")
    runs = find_runs(plan.get("runs", []), hyp)
    all_recs: list[tuple[str, Rec]] = []

    pairs = []
    for pair in plan.get("pairs", []):
        before, after = load_records(pair.get("before")), load_records(pair.get("after"))
        rows = []
        for task in sorted(set(before or {}) | set(after or {})):
            b, a = (before or {}).get(task), (after or {}).get(task)
            if b is None or a is None:
                rows.append({"task": task, "before": b, "after": a, "gain": None, "delta": None, "verdict": WAIT})
                continue
            gain, delta = gain_pct(b, a), a.score - b.score
            if delta < -noise:
                verdict = "🔴 хуже"
            elif gain is not None and gain >= crit["min_gain_pct"]:
                verdict = "🟢 лучше"
            else:
                verdict = "⚪ в пределах шума"
            rows.append({"task": task, "before": b, "after": a, "gain": gain, "delta": delta, "verdict": verdict})
        all_recs += [(pair["name"], r) for side in (before, after) for r in (side or {}).values()]
        pairs.append({**pair, "rows": rows, "ready": before is not None and after is not None})

    gate = load_gate((plan.get("gate") or {}).get("report"))
    gate_results: dict[str, Rec] = {}
    for path in (plan.get("gate") or {}).get("results", []):
        gate_results.update(load_records(path) or {})
    all_recs += [("гейт", r) for r in gate_results.values()]
    attribution = []
    if gate and not gate["passed"]:
        for name in gate["names"]:
            dev = branch = None
            for recheck in plan.get("rechecks", []):
                dev = dev or find_rec(load_records(recheck.get("before")), name)
                branch = branch or find_rec(load_records(recheck.get("after")), name)
            branch = branch or find_rec(gate_results, name)
            if dev is None or branch is None:
                attribution.append({"name": name, "status": WAIT, "text": "нет перепроверки на dev"})
            elif dev.score <= branch.score + noise:
                text = f"на dev {dev.score:.1f}, на ветке {branch.score:.1f} — провал повторяется на dev"
                attribution.append({"name": name, "status": WARN, "text": text})
            else:
                text = f"на dev {dev.score:.1f}, на ветке {branch.score:.1f} — на dev лучше: провал от ветки"
                attribution.append({"name": name, "status": FAIL, "text": text})

    leak_cfg = plan.get("leak") or {}
    leak_before, leak_after = load_records(leak_cfg.get("before")), load_records(leak_cfg.get("after"))
    leak_rows = []
    if leak_cfg:
        for task in sorted(set(leak_before or {}) | set(leak_after or {})):
            b, a = (leak_before or {}).get(task), (leak_after or {}).get(task)
            leak_rows.append({"task": task, "before": b, "after": a})
        all_recs += [("ловушки", r) for r in (leak_after or {}).values()]

    checks = []
    active = [r for r in runs if r.state(crit["stale_sec"], crit["grace_sec"]) == "идёт"]
    stale = [r for r in runs if r.state(crit["stale_sec"], crit["grace_sec"]) == "брошен?"]
    if active:
        checks.append(check("К1", "Прогоны завершены", WAIT, "идут: " + ", ".join(r.name for r in active)))
    elif stale:
        checks.append(check("К1", "Прогоны завершены", WARN, "без «Финал»: " + ", ".join(r.name for r in stale)))
    else:
        checks.append(check("К1", "Прогоны завершены", OK if runs else NA, f"{len(runs)} agent.log"))

    failed = [f"{where}: {short(r.task)}" for where, r in all_recs if not r.ok]
    checks.append(
        check("К2", "Без падений", FAIL if failed else (OK if all_recs else NA), ", ".join(failed) or "всё OK")
    )

    ready_pairs = [p for p in pairs if p["ready"]]
    losses = [
        f"{p['name']}: {short(r['task'])} {r['delta']:+.1f}"
        for p in ready_pairs
        for r in p["rows"]
        if r["delta"] is not None and r["delta"] < -noise
    ]
    status = FAIL if losses else (WAIT if len(ready_pairs) < len(pairs) else (OK if pairs else NA))
    checks.append(check("К3", "Без потерь", status, ", ".join(losses) or f"шум ±{noise} балла"))

    crit_pairs = [p for p in pairs if p.get("criterion", True)]
    gains = {short(r["task"]): r["gain"] for p in crit_pairs for r in p["rows"] if r["gain"] is not None}
    won = sorted(t for t, g in gains.items() if g >= crit["min_gain_pct"])
    need = int(crit["min_tasks"])
    if not crit_pairs:
        status = NA
    elif len(won) >= need:
        status = OK
    elif any(not p["ready"] for p in crit_pairs) or any(r["gain"] is None for p in crit_pairs for r in p["rows"]):
        status = WAIT
    else:
        status = FAIL
    detail = f"≥ {crit['min_gain_pct']}% на {len(won)} из {len(gains)} (нужно {need})"
    detail += ": " + ", ".join(f"{t} {gains[t]:+.1f}%" for t in won) if won else ""
    checks.append(check("К4", "Честный выигрыш", status, detail))

    accepted = [r for r in runs if r.decision == "принята"]
    weak = [r for r in accepted if r.t is not None and r.t < crit["t_min"]]
    if not accepted:
        status, detail = NA, f"«{hyp}» нигде не принята"
    elif weak:
        status = WARN
        detail = "t ниже порога: " + ", ".join(f"{r.name} (t={r.t})" for r in weak)
    else:
        status, detail = OK, f"принята в {len(accepted)}, везде t ≥ {crit['t_min']}"
    checks.append(check("К5", "Принятие гипотезы", status, detail))

    if not plan.get("gate"):
        checks.append(check("К6", "Гейт", NA, "в плане нет гейта"))
    elif gate is None:
        checks.append(check("К6", "Гейт", WAIT, "gate.md ещё нет"))
    elif gate["passed"]:
        checks.append(check("К6", "Гейт", OK, "проходит"))
    else:
        statuses = {a["status"] for a in attribution} or {FAIL}
        status = FAIL if FAIL in statuses else (WAIT if WAIT in statuses else WARN)
        detail = "; ".join(f"{a['name']}: {a['text']}" for a in attribution) or "; ".join(gate["failures"])
        checks.append(check("К6", "Гейт", status, detail))

    if not leak_cfg:
        checks.append(check("К7", "Ловушки утечек", NA, "в плане нет ловушек"))
    elif not leak_rows or any(r["before"] is None or r["after"] is None for r in leak_rows):
        checks.append(check("К7", "Ловушки утечек", WAIT, "нет результата ветки или dev"))
    else:
        worse = [
            short(r["task"])
            for r in leak_rows
            if r["after"].score < r["before"].score - noise or (r["before"].ok and not r["after"].ok)
        ]
        same = sum(r["after"].value == r["before"].value for r in leak_rows)
        detail = f"совпало с dev {same} из {len(leak_rows)}" + (f"; хуже dev: {', '.join(worse)}" if worse else "")
        checks.append(check("К7", "Ловушки утечек", FAIL if worse else OK, detail))

    late = [
        f"{where}: {short(r.task)} {r.seconds:.0f} с"
        for where, r in all_recs
        if r.seconds is not None and r.minutes and r.seconds > r.minutes * 60 + crit["grace_sec"]
    ]
    status = FAIL if late else (OK if all_recs else NA)
    checks.append(check("К8", "Время", status, ", ".join(late) or f"всё ≤ бюджет + {crit['grace_sec']} с"))

    missing = [p for p in plan_paths(plan) if not Path(p).exists()]
    return {
        "missing": missing,
        "plan": plan,
        "crit": crit,
        "runs": runs,
        "pairs": pairs,
        "gate": gate,
        "attribution": attribution,
        "leak": leak_rows,
        "checks": checks,
    }


def exit_code(snapshot: dict) -> int:
    statuses = {c["status"] for c in snapshot["checks"]}
    return 1 if FAIL in statuses else (2 if WAIT in statuses else 0)


FORMATS = ("text", "markdown", "html")
CELL_CLASS = {FAIL: "bad", "🔴": "bad", WARN: "warn", WAIT: "wait", OK: "good", "🟢": "good"}
LEGEND = (
    f"{OK} пройдено · {WARN} с оговоркой · {FAIL} не пройдено · {WAIT} ещё идёт или нет данных · {NA} нечего проверять"
)
# Что проверяет каждая проверка — для страницы; {…} подставляются из criteria плана
CHECK_HELP = {
    "К1": "Все ли прогоны агента закончились. Пока что-то идёт, остальные итоги предварительные.",
    "К2": "Ни один прогон не упал и не сдал невалидный ответ: на платформе это ноль.",
    "К3": "Ни одна задача не стала хуже больше чем на шум (±{noise} балла к сильному бейзлайну): гипотеза не вредит.",
    "К4": "Метрика лучше хотя бы на {min_gain_pct}% минимум на {min_tasks} задачах — критерий синка. "
    "Выигрыш на одной задаче может оказаться подгонкой.",
    "К5": "Где гипотезу приняли, статистика убедительна: t ≥ {t_min}. Ниже — её приняли по правилу «лучше на "
    "последнем периоде», и выигрыш может не перенестись на платформу.",
    "К6": "Гейт перед сабмитом не стал хуже. Провал, который повторяется и на dev, — не от ветки.",
    "К7": "На задачах со спрятанной утечкой ветка не хуже dev: новая операция не подхватывает утечку.",
    "К8": "Каждый прогон уложился в бюджет времени: на платформе за перерасход обрывают без ответа.",
}
SECTION_HELP = {
    "runs": "Прогоны агента, которые ещё пишут agent.log: сколько прошло из бюджета, какая гипотеза сейчас "
    "проверяется и лучший CV на этот момент.",
    "pairs": "Те же задачи и то же разбиение: агент без гипотезы («до», dev) и с ней («после», ветка). "
    "«Метрика лучше на» — изменение в процентах с учётом направления метрики; «балл к сильному» — как в гейте, "
    "разница меньше ±{noise} — шум.",
    "decisions": "Что агент решил по гипотезе в каждом прогоне. «CV = бейзлайну» — операция не нашла, что "
    "добавить, и ничего не изменила; «не дошла» — до неё не хватило времени.",
    "gate": "Провалы гейта и перепроверка тех же задач на dev. Повторилось на dev — дело не в ветке, а в машине "
    "или в самом dev.",
    "leak": "Синтетические задачи со спрятанной утечкой. Результат как у dev — гипотеза утечку не добавила.",
}
PAGE = """<!doctype html>
<html lang="ru"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<meta http-equiv="refresh" content="{refresh}"><title>Контроль результата</title>
<style>
:root {{ --bg: #fff; --fg: #1d1d1f; --line: #d9d9de; --head: #f3f3f6; --good: #e3f6e8; --bad: #fde4e4;
  --warn: #fff4d6; --wait: #e8eefc; }}
@media (prefers-color-scheme: dark) {{ :root {{ --bg: #17181c; --fg: #e8e8ea; --line: #34363d; --head: #22242a;
  --good: #1f3a28; --bad: #4a2224; --warn: #4a3d1a; --wait: #23304d; }} }}
body {{ background: var(--bg); color: var(--fg); font: 14px/1.45 system-ui, sans-serif; margin: 0 16px 32px; }}
table {{ border-collapse: collapse; margin: 6px 0 4px; max-width: 100%; display: block; overflow-x: auto; }}
th, td {{ border: 1px solid var(--line); padding: 4px 8px; text-align: left; vertical-align: top; }}
th {{ background: var(--head); }} td.good {{ background: var(--good); }} td.bad {{ background: var(--bad); }}
td.warn {{ background: var(--warn); }} td.wait {{ background: var(--wait); }}
h1 {{ font-size: 20px; }} h2 {{ font-size: 16px; margin-top: 22px; }} .muted {{ opacity: .7; }}
.help {{ opacity: .75; font-size: 13px; margin: 2px 0 6px; max-width: 75ch; }} td .help {{ margin: 2px 0 0; }}
.summary {{ border: 1px solid var(--line); background: var(--head); padding: 4px 14px 8px; margin: 8px 0;
  max-width: 1100px; }} .summary h2 {{ margin-top: 8px; }} .verdict {{ font-weight: 600; }}
.missing {{ margin: 8px 0; }} .missing summary {{ cursor: pointer; }} code {{ font-size: 12px; word-break: break-all; }}
</style></head><body>
{body}
<p class="muted">Страница обновляется каждые {refresh} с · данные — файлы прогонов на этой машине</p>
</body></html>
"""


class Raw(str):
    """Готовый HTML для ячейки таблицы — не экранируется."""


def _cell_class(text: str) -> str:
    for mark, css in CELL_CLASS.items():
        if text.startswith(mark):
            return f' class="{css}"'
    return ""


def _table(headers: list[str], rows: list[list], fmt: str) -> str:
    cells = [[c if isinstance(c, Raw) else str(c) for c in row] for row in rows]
    if fmt == "markdown":
        lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
        return "\n".join(lines + ["| " + " | ".join(r) + " |" for r in cells])
    if fmt == "html":
        head = "".join(f"<th>{escape(h)}</th>" for h in headers)
        body = "".join(
            "<tr>"
            + "".join(f"<td{_cell_class(c)}>{c if isinstance(c, Raw) else escape(c)}</td>" for c in row)
            + "</tr>"
            for row in cells
        )
        return f"<table><tr>{head}</tr>{body}</table>"
    widths = [max(len(h), *(len(r[i]) for r in cells)) if cells else len(h) for i, h in enumerate(headers)]
    line = "  ".join(f"{{:<{w}}}" for w in widths)
    return "\n".join(
        [line.format(*headers), line.format(*("-" * w for w in widths))] + [line.format(*r) for r in cells]
    )


def _num(value: float | None) -> str:
    return NA if value is None else f"{value:.5g}"


def plan_paths(plan: dict) -> list[str]:
    """Все файлы и папки, которые читает план."""
    paths = list(plan.get("runs", []))
    for item in [*plan.get("pairs", []), *plan.get("rechecks", []), plan.get("leak") or {}]:
        paths += [item[k] for k in ("before", "after") if item.get(k)]
    gate = plan.get("gate") or {}
    paths += [gate["report"]] if gate.get("report") else []
    return paths + list(gate.get("results", []))


def summary(snapshot: dict) -> tuple[str, list[str]]:
    """Итог простыми словами и список того, на что смотреть."""
    checks = snapshot["checks"]
    statuses = {c["status"] for c in checks}
    paths = plan_paths(snapshot["plan"])
    if paths and len(snapshot.get("missing", [])) == len(paths):
        verdict = (
            "На этой машине нет ни одного файла из плана — показывать нечего. Замеры, которые идут на другой "
            "машине, отсюда не видны: смотри их там или укажи в плане свои прогоны."
        )
    elif FAIL in statuses:
        verdict = "Есть проблемы (❌): в таком виде гипотезу не берём, сначала разбираемся с ними."
    elif WAIT in statuses:
        verdict = "Замер ещё идёт (⏳): итог предварительный."
    elif WARN in statuses:
        verdict = "Проблем нет, но есть оговорки (⚠️): решение — с их учётом."
    elif statuses <= {NA}:
        verdict = "Проверять пока нечего."
    else:
        verdict = "Все проверки пройдены: гипотезу можно брать."
    items = [f"{c['code']} {c['title']}: {c['detail']}" for c in checks if c["status"] in (FAIL, WARN, WAIT)]
    return verdict, items


def render(snapshot: dict, fmt: str = "text") -> str:
    """Снимок как текст для терминала, markdown для PR/issue или HTML-фрагмент страницы (с объяснениями)."""
    plan, crit = snapshot["plan"], snapshot["crit"]
    html = fmt == "html"

    def head(text: str, level: int = 2) -> str:
        if html:
            return f"<h{level}>{escape(text)}</h{level}>"
        return ("#" * level + " " if fmt == "markdown" else ("" if level == 1 else "== ")) + text

    def para(text: str, css: str = "") -> str:
        if not html:
            return text
        return f'<p class="{css}">{escape(text)}</p>' if css else f"<p>{escape(text)}</p>"

    def explain(key: str) -> list[str]:
        return [para(SECTION_HELP[key].format(**crit), "help")] if html else []

    counts = {s: sum(c["status"] == s for c in snapshot["checks"]) for s in (OK, WARN, FAIL, WAIT)}
    now = datetime.now().strftime("%H:%M:%S")
    verdict, items = summary(snapshot)
    out = [
        head(f"{plan.get('title', 'Контроль результата')} · гипотеза «{plan.get('hypothesis', '')}»", 1),
        para(f"{now} · " + " · ".join(f"{s} {n}" for s, n in counts.items()), "stamp"),
    ]
    if html:
        lis = "".join(f"<li>{escape(i)}</li>" for i in items)
        out.append(
            f'<section class="summary"><h2>Итог сейчас</h2><p class="verdict">{escape(verdict)}</p>'
            + (f"<p>На что смотреть:</p><ul>{lis}</ul>" if lis else "")
            + "</section>"
        )
        if snapshot.get("missing"):
            lis = "".join(f"<li><code>{escape(m)}</code></li>" for m in snapshot["missing"])
            out.append(
                f'<details class="missing"><summary>Нет на этой машине: {len(snapshot["missing"])} из '
                f"{len(plan_paths(plan))} файлов плана</summary><p>Пути берутся от папки, из которой запущен "
                "скрипт. Укажи в плане свои прогоны (пример — research/night/control_plan.example.json).</p>"
                f"<ul>{lis}</ul></details>"
            )
    else:
        out += ["", f"Итог: {verdict}", *[f"  · {i}" for i in items]]
    out += ["", head("Контроль")]
    if html:
        out.append(para(LEGEND, "help"))
        rows = [
            [
                c["code"],
                Raw(
                    f'<b>{escape(c["title"])}</b><div class="help">{escape(CHECK_HELP[c["code"]].format(**crit))}</div>'
                ),
                c["status"],
                c["detail"],
            ]
            for c in snapshot["checks"]
        ]
    else:
        rows = [[c["code"], c["title"], c["status"], c["detail"]] for c in snapshot["checks"]]
    out.append(_table(["", "Проверка", "Итог", "Подробности"], rows, fmt))
    active = [r for r in snapshot["runs"] if r.state(crit["stale_sec"], crit["grace_sec"]) != "готов"]
    out += ["", head("Идут сейчас"), *explain("runs")]
    if active:
        rows = [
            [
                r.name,
                r.state(crit["stale_sec"], crit["grace_sec"]),
                f"{r.elapsed:.0f} / {r.budget or '?'} с",
                r.iteration,
                r.best or NA,
            ]
            for r in active
        ]
        out.append(_table(["Прогон", "Состояние", "Прошло", "Итерация", "Лучший CV"], rows, fmt))
    else:
        out.append(para("нет живых прогонов"))
    for i, pair in enumerate(snapshot["pairs"]):
        out += ["", head(f"До/после: {pair['name']}"), *(explain("pairs") if i == 0 else [])]
        rows = [
            [
                short(r["task"]),
                (r["before"] or r["after"]).metric,
                _num(r["before"] and r["before"].value),
                _num(r["after"] and r["after"].value),
                NA if r["gain"] is None else f"{r['gain']:+.1f}%",
                NA if r["delta"] is None else f"{r['before'].score:.1f} → {r['after'].score:.1f}",
                r["verdict"],
            ]
            for r in pair["rows"]
        ]
        headers = ["Задача", "Метрика", "До", "После", "Метрика лучше на", "Балл к сильному", "Итог"]
        out.append(_table(headers, rows, fmt) if rows else para(f"{WAIT} результатов ещё нет"))
    decided = [r for r in snapshot["runs"] if r.decision]
    if decided and plan.get("hypothesis"):
        out += ["", head(f"Решения по «{plan['hypothesis']}»"), *explain("decisions")]
        out.append(_table(["Прогон", "Решение", "Подробности"], [[r.name, r.decision, r.detail] for r in decided], fmt))
    if snapshot["gate"] and not snapshot["gate"]["passed"]:
        out += ["", head("Провалы гейта"), *explain("gate")]
        rows = [[a["name"], a["status"], a["text"]] for a in snapshot["attribution"]]
        failures = [[f] for f in snapshot["gate"]["failures"]]
        out.append(_table(["Задача", "Итог", "Перепроверка"], rows, fmt) if rows else _table(["Провал"], failures, fmt))
    if snapshot["leak"]:
        out += ["", head("Ловушки утечек"), *explain("leak")]
        rows = [
            [
                short(r["task"]),
                _num(r["before"] and r["before"].value),
                _num(r["after"] and r["after"].value),
                NA if not (r["before"] and r["after"]) else f"{r['before'].score:.1f} → {r['after'].score:.1f}",
            ]
            for r in snapshot["leak"]
        ]
        out.append(_table(["Ловушка", "dev", "Ветка", "Балл к сильному"], rows, fmt))
    return "\n".join(out) + "\n"


def export_snapshot(snapshot: dict) -> dict:
    """Готовый снимок во всех форматах — чтобы показать его на другой машине (--export → --snapshot)."""
    return {
        "version": 1,
        "generated": datetime.now().strftime("%d.%m %H:%M:%S"),
        "exit_code": exit_code(snapshot),
        **{fmt: render(snapshot, fmt) for fmt in FORMATS},
    }


def load_view(plan: dict | None, snapshot_file: Path | None, fmt: str) -> tuple[str, int]:
    """Снимок в нужном формате и код выхода: свежий по плану или готовый из файла (перечитывается каждый раз)."""
    if snapshot_file is None:
        snapshot = evaluate(plan or {})
        return render(snapshot, fmt), exit_code(snapshot)
    if not snapshot_file.exists():
        note = (
            f"Файла снимка нет: {snapshot_file.resolve()}. Путь считается от папки, из которой запущен скрипт "
            f"({Path.cwd()}). Положи файл туда или укажи в --snapshot полный путь."
        )
        return (f'<p class="verdict">{escape(note)}</p>' if fmt == "html" else note + "\n"), 2
    data = json.loads(snapshot_file.read_text(encoding="utf-8"))
    note = f"Снимок из файла {snapshot_file.name}, собран {data['generated']} на машине, где идут прогоны."
    note += " Положи новую версию файла — страница покажет её при следующем обновлении."
    text = data[fmt]
    text = f'<p class="help">{escape(note)}</p>\n{text}' if fmt == "html" else f"{note}\n\n{text}"
    return text, int(data.get("exit_code", 2))


def make_handler(plan: dict | None, refresh: float, snapshot_file: Path | None = None) -> type[BaseHTTPRequestHandler]:
    """Обработчик для --serve: на каждый запрос заново читает файлы и отдаёт страницу (/) или markdown (/report.md)."""

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802 — имя задаёт http.server
            path = self.path.split("?", 1)[0]
            if path not in ("/", "/report.md"):
                self.send_error(404)
                return
            try:
                if path == "/":
                    body = PAGE.format(refresh=max(1, int(refresh)), body=load_view(plan, snapshot_file, "html")[0])
                    ctype = "text/html; charset=utf-8"
                else:
                    body, ctype = load_view(plan, snapshot_file, "markdown")[0], "text/markdown; charset=utf-8"
            except Exception as exc:  # noqa: BLE001 — страница не падает от недописанного файла
                body = f"Снимок не собрался: {type(exc).__name__}: {exc}"
                ctype = "text/plain; charset=utf-8"
            data = body.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *args):  # не засоряем консоль строкой на каждый запрос
            pass

    return Handler


LOCAL_HOSTS = {"127.0.0.1", "localhost", "::1"}


def exposure_warning(host: str) -> str | None:
    """Предупреждение, если --serve слушает не только этот компьютер: авторизации у страницы нет."""
    if host in LOCAL_HOSTS:
        return None
    return f"⚠️ --host {host}: страница без авторизации, сводку прогонов увидит любой, кто достучится до порта."


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--plan", type=Path, help="JSON-план: что и с какими порогами проверяем")
    parser.add_argument("--snapshot", type=Path, help="показать готовый снимок (файл от --export) вместо плана")
    parser.add_argument("--export", type=Path, help="сохранить снимок по плану в файл для другой машины и выйти")
    parser.add_argument("--once", action="store_true", help="один снимок и выход (код: 0 / 1 — есть ❌ / 2 — идёт)")
    parser.add_argument("--markdown", action="store_true", help="markdown — для комментария в PR или issue")
    parser.add_argument("--interval", type=float, default=15, help="период обновления, секунд (по умолчанию 15)")
    parser.add_argument("--serve", action="store_true", help="страница в браузере: http://HOST:PORT, сама обновляется")
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="адрес для --serve: по умолчанию только этот компьютер; наружу не открывать — авторизации нет",
    )
    parser.add_argument("--port", type=int, default=8765, help="порт для --serve (по умолчанию 8765)")
    args = parser.parse_args(argv)
    if (args.plan is None) == (args.snapshot is None):
        parser.error("укажи --plan (прогоны на этой машине) или --snapshot (готовый снимок), но не оба")
    if args.export and args.plan is None:
        parser.error("--export собирает снимок по --plan")
    plan = json.loads(args.plan.read_text(encoding="utf-8")) if args.plan else None
    fmt = "markdown" if args.markdown else "text"
    if args.export:
        snapshot = evaluate(plan)
        args.export.write_text(json.dumps(export_snapshot(snapshot), ensure_ascii=False), encoding="utf-8")
        print(f"Снимок сохранён: {args.export}")
        return exit_code(snapshot)
    if args.once:
        text, code = load_view(plan, args.snapshot, fmt)
        print(text, end="")
        return code
    if args.serve:
        server = ThreadingHTTPServer((args.host, args.port), make_handler(plan, args.interval, args.snapshot))
        if warning := exposure_warning(args.host):
            print(warning, flush=True)
        print(f"Открой http://{args.host}:{server.server_port} — markdown: /report.md · Ctrl+C — выход", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            server.server_close()
        return 0
    try:
        while True:
            sys.stdout.write("\033[2J\033[H" + load_view(plan, args.snapshot, fmt)[0])
            sys.stdout.write(f"\nобновление каждые {args.interval:g} с · Ctrl+C — выход\n")
            sys.stdout.flush()
            time.sleep(args.interval)
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    for _stream in (sys.stdout, sys.stderr):  # Windows/Git Bash: не UTF-8 → кириллица роняет вывод
        _stream.reconfigure(encoding="utf-8", errors="replace")
    sys.exit(main())
