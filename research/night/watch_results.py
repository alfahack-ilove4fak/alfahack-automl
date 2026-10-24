"""Монитор результатов исследования и система контроля: живые прогоны, «до/после», гейт, ловушки — с вердиктами.

    uv run python research/night/watch_results.py --plan runs/oh/plan.json              # обновление каждые 15 с
    uv run python research/night/watch_results.py --plan runs/oh/plan.json --once       # один снимок
    uv run python research/night/watch_results.py --plan runs/oh/plan.json --once --markdown > report.md

Скрипт только читает файлы (agent.log, CSV харнесса, jsonl run_alien, gate.md) — прогонам не мешает.
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

    return {
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


def _table(headers: list[str], rows: list[list], markdown: bool) -> str:
    cells = [[str(c) for c in row] for row in rows]
    if markdown:
        lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
        return "\n".join(lines + ["| " + " | ".join(r) + " |" for r in cells])
    widths = [max(len(h), *(len(r[i]) for r in cells)) if cells else len(h) for i, h in enumerate(headers)]
    fmt = "  ".join(f"{{:<{w}}}" for w in widths)
    return "\n".join([fmt.format(*headers), fmt.format(*("-" * w for w in widths))] + [fmt.format(*r) for r in cells])


def _num(value: float | None) -> str:
    return NA if value is None else f"{value:.5g}"


def render(snapshot: dict, markdown: bool = False) -> str:
    plan, crit = snapshot["plan"], snapshot["crit"]
    h = "## " if markdown else "== "
    counts = {s: sum(c["status"] == s for c in snapshot["checks"]) for s in (OK, WARN, FAIL, WAIT)}
    now = datetime.now().strftime("%H:%M:%S")
    out = [
        f"{'# ' if markdown else ''}{plan.get('title', 'Контроль результата')} · гипотеза «{plan.get('hypothesis', '')}»",
        f"{now} · " + " · ".join(f"{s} {n}" for s, n in counts.items()),
        "",
        h + "Контроль",
        _table(
            ["", "Проверка", "Итог", "Подробности"],
            [[c["code"], c["title"], c["status"], c["detail"]] for c in snapshot["checks"]],
            markdown,
        ),
    ]
    active = [r for r in snapshot["runs"] if r.state(crit["stale_sec"], crit["grace_sec"]) != "готов"]
    out += ["", h + "Идут сейчас"]
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
        out.append(_table(["Прогон", "Состояние", "Прошло", "Итерация", "Лучший CV"], rows, markdown))
    else:
        out.append("нет живых прогонов")
    for pair in snapshot["pairs"]:
        out += ["", h + f"До/после: {pair['name']}"]
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
        out.append(_table(headers, rows, markdown) if rows else f"{WAIT} результатов ещё нет")
    decided = [r for r in snapshot["runs"] if r.decision]
    if decided and plan.get("hypothesis"):
        out += ["", h + f"Решения по «{plan['hypothesis']}»"]
        out.append(
            _table(["Прогон", "Решение", "Подробности"], [[r.name, r.decision, r.detail] for r in decided], markdown)
        )
    if snapshot["gate"] and not snapshot["gate"]["passed"]:
        out += ["", h + "Провалы гейта"]
        rows = [[a["name"], a["status"], a["text"]] for a in snapshot["attribution"]]
        out.append(
            _table(["Задача", "Итог", "Перепроверка"], rows, markdown)
            if rows
            else "\n".join(snapshot["gate"]["failures"])
        )
    if snapshot["leak"]:
        out += ["", h + "Ловушки утечек"]
        rows = [
            [
                short(r["task"]),
                _num(r["before"] and r["before"].value),
                _num(r["after"] and r["after"].value),
                NA if not (r["before"] and r["after"]) else f"{r['before'].score:.1f} → {r['after'].score:.1f}",
            ]
            for r in snapshot["leak"]
        ]
        out.append(_table(["Ловушка", "dev", "Ветка", "Балл к сильному"], rows, markdown))
    return "\n".join(out) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--plan", type=Path, required=True, help="JSON-план: что и с какими порогами проверяем")
    parser.add_argument("--once", action="store_true", help="один снимок и выход (код: 0 / 1 — есть ❌ / 2 — идёт)")
    parser.add_argument("--markdown", action="store_true", help="markdown — для комментария в PR или issue")
    parser.add_argument("--interval", type=float, default=15, help="период обновления, секунд (по умолчанию 15)")
    args = parser.parse_args(argv)
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    if args.once:
        snapshot = evaluate(plan)
        print(render(snapshot, args.markdown), end="")
        return exit_code(snapshot)
    try:
        while True:
            snapshot = evaluate(plan)
            sys.stdout.write("\033[2J\033[H" + render(snapshot, args.markdown))
            sys.stdout.write(f"\nобновление каждые {args.interval:g} с · Ctrl+C — выход\n")
            sys.stdout.flush()
            time.sleep(args.interval)
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    for _stream in (sys.stdout, sys.stderr):  # Windows/Git Bash: не UTF-8 → кириллица роняет вывод
        _stream.reconfigure(encoding="utf-8", errors="replace")
    sys.exit(main())
