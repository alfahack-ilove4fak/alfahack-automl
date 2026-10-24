"""Монитор прогона внешних задач: веб-страница с живым обновлением или цветная таблица в терминале.

    uv run python research/external/watch.py            # страница http://localhost:8765 (обновляется сама)
    uv run python research/external/watch.py --text     # то же в терминале, цветная таблица
    uv run python research/external/watch.py --once     # один снимок текстом и выход
    uv run python research/external/watch.py --root ../wt-external --port 9000

Только читает файлы прогона (`run_alien.py --set external`) и ничего не запускает — можно держать открытым рядом.
Как читать «выигрыш»: на сколько баллов агент лучше простой модели (LightGBM по умолчанию) по формуле организаторов.
0 — не лучше простой модели, и на платформе такая задача дала бы 0. Реальные таблицы дают 0–10, нули — главный сигнал.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

BUDGET_MIN = 5.0  # минут на задачу в прогоне (run_alien.py --minutes); для полосы «сколько прошло»
EXTRA_SEC = 40  # на ориентиры LightGBM после агента — для оценки «осталось»


def is_zero(row: dict) -> bool:
    if row.get("exit_code") not in (0, None) or row.get("problems") or row.get("value") is None:
        return True
    score = row.get("score_vs_strong")
    return score is None or score <= 0


def why_zero(row: dict) -> str:
    if row.get("exit_code") not in (0, None):
        return f"агент упал (код {row['exit_code']})"
    if row.get("problems"):
        return "ответ не принят: " + "; ".join(row["problems"])[:120]
    if row.get("value") is None:
        return "метрика не посчиталась"
    return "не лучше LightGBM по умолчанию"


def read_rows(path: Path) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                rows[row["task"]] = row
    return rows


def log_progress(log: Path) -> dict:
    if not log.exists():
        return {"iterations": 0, "accepted": 0, "last": "запускается…"}
    text = log.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    last = re.sub(r"^\[\s*[\d.]+s\]\s*\w+\s+[\w.]+:\s*", "", lines[-1])[:120] if lines else ""
    return {
        "iterations": len(re.findall(r"Итерация \d+:", text)),
        "accepted": len(re.findall(r"✓ лучше", text)),
        "last": last,
    }


def state(root: Path) -> dict:
    base = root / "runs" / "night"
    answers = base / "external" / "_answers"
    names = sorted(p.stem for p in answers.glob("*.json"))
    rows = read_rows(base / "external_results.jsonl")
    runs = base / "external_runs"
    pending = [n for n in names if n not in rows and (runs / n).exists()]
    current = max(pending, key=lambda n: (runs / n).stat().st_mtime) if pending else None
    tasks = []
    for name in names:
        meta = json.loads((answers / f"{name}.json").read_text(encoding="utf-8"))
        row = rows.get(name)
        item = {"name": name, "type": meta["task_type"], "metric": meta["metric"], "source": meta.get("source", "")}
        if row is not None:
            ref = row.get("ref", {})
            item |= {
                "status": "zero" if is_zero(row) else "ok",
                "score": row.get("score_vs_strong"),
                "seconds": row.get("seconds"),
                "agent": row.get("value"),
                "lgbm": ref.get("lgbm_default"),
                "trivial": ref.get("trivial"),
                "parsed": (row.get("spec") or {}).get("metric") == meta["metric"],
                "parsed_as": (row.get("spec") or {}).get("metric"),
                "note": why_zero(row) if is_zero(row) else "",
            }
        elif name == current:
            elapsed = time.time() - (runs / name).stat().st_ctime
            item |= {"status": "running", "elapsed": round(elapsed), **log_progress(runs / name / "agent.log")}
        else:
            item["status"] = "waiting"
        tasks.append(item)
    done = [t for t in tasks if t["status"] in ("ok", "zero")]
    scores = [t["score"] for t in done if t["score"] is not None]
    seconds = [t["seconds"] for t in done if t["seconds"]]
    left = len(tasks) - len(done)
    eta = round(left * (sum(seconds) / len(seconds) + EXTRA_SEC)) if seconds and left else None
    return {
        "updated": time.strftime("%H:%M:%S"),
        "total": len(tasks),
        "done": len(done),
        "zeros": sum(t["status"] == "zero" for t in done),
        "mean": round(sum(scores) / len(scores), 2) if scores else None,
        "unparsed": sum(not t["parsed"] for t in done),
        "eta_seconds": eta,
        "budget_seconds": round(BUDGET_MIN * 60),
        "finished": bool(tasks) and len(done) == len(tasks),
        "tasks": tasks,
    }


# ------------------------------------------------------------------ терминал

COLORS = {"ok": "\033[32m", "zero": "\033[31m", "running": "\033[33m", "waiting": "\033[90m"}
MARK = {"ok": "✅", "zero": "❌", "running": "⏳", "waiting": "· "}


def render_text(s: dict, color: bool = True) -> str:
    def paint(status: str, text: str) -> str:
        return f"{COLORS[status]}{text}\033[0m" if color else text

    bar_len = 30
    filled = round(bar_len * s["done"] / s["total"]) if s["total"] else 0
    eta = f", осталось ≈ {s['eta_seconds'] / 60:.0f} мин" if s["eta_seconds"] else ""
    out = [
        f"Прогон внешних задач: [{'█' * filled}{'░' * (bar_len - filled)}] {s['done']}/{s['total']}{eta}",
        f"Нулей: {s['zeros']} (задач, где агент не лучше простой модели) · средний выигрыш: "
        f"{s['mean'] if s['mean'] is not None else '—'} · метрику поняла неверно: {s['unparsed']}",
        "",
        f"{'':3}{'задача':<30}{'тип':<11}{'метрика':<18}{'выигрыш':>8}  комментарий",
    ]
    for t in s["tasks"]:
        score = f"{t['score']:.2f}" if t.get("score") is not None else ""
        if t["status"] == "running":
            note = f"идёт {t['elapsed'] // 60}:{t['elapsed'] % 60:02d} из {BUDGET_MIN:.0f}:00 · гипотез {t['iterations']}, принято {t['accepted']} · {t['last']}"
        elif t["status"] in ("ok", "zero"):
            note = (
                t["note"] or f"агент {t['agent']:.4g} · LightGBM {t['lgbm']:.4g}"
                if t.get("agent") is not None
                else t["note"]
            )
            if not t["parsed"]:
                note += f" · ⚠️ метрика понята как {t['parsed_as']}"
        else:
            note = ""
        out.append(
            paint(t["status"], f"{MARK[t['status']]} {t['name']:<30}{t['type']:<11}{t['metric']:<18}{score:>8}  {note}")
        )
    if s["finished"]:
        out += ["", "Прогон закончен. Итоговая таблица и сравнение с базой: uv run python research/external/report.py"]
    return "\n".join(out)


# ------------------------------------------------------------------ веб

PAGE = """<!doctype html><html lang="ru"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Внешние задачи — прогон</title><style>
:root{--bg:#f6f7f9;--card:#fff;--text:#1d2330;--mute:#6b7385;--line:#e3e6ec;--ok:#1a8f4c;--zero:#c93a3a;--run:#c98a00;--wait:#9aa1b2;--bar:#3b6fe0}
@media (prefers-color-scheme:dark){:root{--bg:#14171d;--card:#1d222b;--text:#e8ebf2;--mute:#9099ad;--line:#2c3340;--ok:#4cc78a;--zero:#ff7b7b;--run:#f0b840;--wait:#6b7385;--bar:#6e97ff}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:15px/1.45 system-ui,-apple-system,Segoe UI,sans-serif}
main{max-width:1100px;margin:0 auto;padding:20px 16px 40px}h1{font-size:20px;margin:0 0 4px}.sub{color:var(--mute);margin:0 0 16px}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px;margin-bottom:16px}
.tile{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px 14px}.tile b{display:block;font-size:26px}.tile span{color:var(--mute);font-size:13px}
.bar{height:10px;background:var(--line);border-radius:6px;overflow:hidden;margin:6px 0 16px}.bar i{display:block;height:100%;background:var(--bar);transition:width .6s}
table{width:100%;border-collapse:collapse;background:var(--card);border:1px solid var(--line);border-radius:10px;overflow:hidden}
th,td{padding:9px 10px;text-align:left;border-bottom:1px solid var(--line);vertical-align:top}th{font-size:12px;color:var(--mute);font-weight:600}
tr:last-child td{border-bottom:0}.num{text-align:right;font-variant-numeric:tabular-nums}
.pill{display:inline-block;padding:2px 9px;border-radius:99px;font-size:12px;font-weight:600;color:#fff}
.ok{background:var(--ok)}.zero{background:var(--zero)}.running{background:var(--run)}.waiting{background:var(--wait)}
.mini{height:6px;background:var(--line);border-radius:4px;margin-top:5px;overflow:hidden}.mini i{display:block;height:100%;background:var(--run)}
.note{color:var(--mute);font-size:13px}.warn{color:var(--zero)}details{margin-top:18px;color:var(--mute);font-size:13px}
@media (max-width:640px){.hide{display:none}}
</style></head><body><main>
<h1>Внешние задачи: прогон агента</h1><p class="sub" id="sub">загрузка…</p>
<div class="tiles" id="tiles"></div><div class="bar"><i id="bar" style="width:0"></i></div>
<table><thead><tr><th>Статус</th><th>Задача</th><th class="hide">Тип · метрика</th><th class="num">Выигрыш</th><th class="hide num">Агент</th><th class="hide num">LightGBM</th><th>Комментарий</th></tr></thead><tbody id="rows"></tbody></table>
<details><summary>Как читать</summary><p><b>Выигрыш</b> — на сколько баллов агент лучше простой модели (LightGBM по умолчанию) по формуле организаторов.
<b>0</b> — не лучше, на платформе такая задача дала бы 0 баллов: это главный сигнал. На реальных таблицах выигрыш обычно 0–10.
«Агент» и «LightGBM» — значения метрики задачи (направление зависит от метрики).</p></details>
</main><script>
const L={ok:'лучше',zero:'ноль',running:'идёт',waiting:'ждёт'};
const f=v=>v==null?'':(Math.abs(v)>=100?v.toFixed(1):Number(v).toPrecision(4));
const mm=s=>Math.floor(s/60)+':'+String(s%60).padStart(2,'0');
async function tick(){try{const s=await (await fetch('/api/state')).json();
document.getElementById('sub').textContent='обновлено '+s.updated+(s.finished?' · прогон закончен':'');
const eta=s.eta_seconds?Math.round(s.eta_seconds/60)+' мин':(s.finished?'—':'считаю…');
const tile=(b,t,c)=>`<div class="tile"><b ${c?'style="color:var(--'+c+')"':''}>${b}</b><span>${t}</span></div>`;
document.getElementById('tiles').innerHTML=tile(s.done+' / '+s.total,'задач готово')+tile(s.zeros,'нулей (не лучше простой модели)',s.zeros?'zero':'ok')+tile(s.mean??'—','средний выигрыш, баллов')+tile(s.unparsed,'метрика понята неверно',s.unparsed?'zero':'ok')+tile(eta,'осталось ≈');
document.getElementById('bar').style.width=(s.total?100*s.done/s.total:0)+'%';
document.getElementById('rows').innerHTML=s.tasks.map(t=>{let note='';
if(t.status=='running'){note=`гипотез ${t.iterations}, принято ${t.accepted} · <span class="note">${t.last}</span><div class="mini"><i style="width:${Math.min(100,100*t.elapsed/s.budget_seconds)}%"></i></div>`}
else if(t.status=='zero'){note=`<span class="warn">${t.note}</span>`}
if((t.status=='ok'||t.status=='zero')&&!t.parsed)note+=` <span class="warn">⚠️ метрика понята как ${t.parsed_as}</span>`;
const lab=t.status=='running'?L.running+' '+mm(t.elapsed):L[t.status];
return `<tr><td><span class="pill ${t.status}">${lab}</span></td><td><a href="${t.source}" target="_blank" rel="noopener" style="color:inherit">${t.name}</a></td><td class="hide">${t.type} · ${t.metric}</td><td class="num">${t.score==null?'':t.score.toFixed(2)}</td><td class="hide num">${f(t.agent)}</td><td class="hide num">${f(t.lgbm)}</td><td>${note}</td></tr>`}).join('');
}catch(e){document.getElementById('sub').textContent='нет связи с монитором…'}}
tick();setInterval(tick,3000);
</script></body></html>"""


def serve(root: Path, port: int) -> None:
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            if self.path.startswith("/api/state"):
                body, kind = json.dumps(state(root), ensure_ascii=False).encode(), "application/json"
            else:
                body, kind = PAGE.encode(), "text/html"
            self.send_response(200)
            self.send_header("Content-Type", f"{kind}; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args) -> None:  # тишина в терминале
            pass

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"Монитор: http://localhost:{port}  (Ctrl+C — выход)", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--text", action="store_true", help="цветная таблица в терминале вместо веб-страницы")
    parser.add_argument("--once", action="store_true", help="один снимок текстом и выход")
    parser.add_argument("--interval", type=float, default=5, help="период обновления в режиме --text")
    args = parser.parse_args()
    root = args.root.resolve()
    if args.once:
        print(render_text(state(root), color=False))
        return 0
    if args.text:
        try:
            while True:
                print("\033[2J\033[H" + time.strftime("%H:%M:%S") + "  (Ctrl+C — выход)\n\n" + render_text(state(root)))
                time.sleep(args.interval)
        except KeyboardInterrupt:
            return 0
    serve(root, args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
