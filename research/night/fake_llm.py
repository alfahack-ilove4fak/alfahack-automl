"""Ночной аудит: поддельный OpenAI-совместимый сервер для проверки устойчивости агента к плохой LLM.

    uv run python research/night/fake_llm.py --mode garbage --port 8765

Агент: OPENAI_BASE_URL=http://127.0.0.1:8765/v1 OPENAI_API_KEY=fake OPENAI_MODEL=fake NO_PROXY=127.0.0.1
Режимы (--mode):
  garbage      — ответ не JSON («Конечно! Вот мои идеи…»)
  badjson      — оборванный JSON
  empty        — пустой ответ
  error500     — HTTP 500 на каждый запрос
  slow         — валидный ответ, но через --delay секунд
  hang         — соединение открыто, ответа нет (до --delay секунд)
  valid        — валидные гипотезы (LightGBM с другими params); код признаков — безобидный
  feat_loop    — гипотезы валидные, код признаков с бесконечным циклом
  feat_mem     — код признаков, выделяющий десятки ГБ
  feat_raise   — код признаков, который падает
  feat_shape   — код признаков, который теряет строки
  feat_nan     — код признаков из одних NaN и inf
  feat_cheat   — код признаков, пытающийся читать файл через pandas.core.common.builtins
  hyp_evil     — гипотезы с неизвестными ключами params, огромными числами, отрицательными n_estimators
Каждый запрос пишется в stdout одной строкой: время, назначение (по тексту промпта), режим.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

FEATURE_CODES = {
    "valid": "import numpy as np\ndef add_features(df):\n    out = df.copy()\n    num = out.select_dtypes('number')\n    out['night_sum'] = num.sum(axis=1)\n    return out\n",
    "feat_loop": "def add_features(df):\n    while True:\n        pass\n    return df\n",
    "feat_mem": "import numpy as np\ndef add_features(df):\n    big = np.ones((60000, 60000))\n    df = df.copy()\n    df['x'] = big.sum()\n    return df\n",
    "feat_raise": "def add_features(df):\n    return df['нет_такой_колонки'] + 1\n",
    "feat_shape": "def add_features(df):\n    return df.iloc[: len(df) // 2].copy()\n",
    "feat_nan": "import numpy as np\ndef add_features(df):\n    out = df.copy()\n    out['n1'] = np.nan\n    out['n2'] = np.inf\n    out['n3'] = -np.inf\n    return out\n",
    "feat_cheat": "import pandas as pd\ndef add_features(df):\n    f = pd.core.common.builtins.open('/etc/hostname')\n    out = df.copy()\n    out['h'] = len(f.read())\n    return out\n",
}
GOOD_HYPS = [
    {
        "id": "night_leaves_63",
        "kind": "hyperparams",
        "title": "Больше листьев",
        "rationale": "Больше ёмкость модели.",
        "params": {"model": "lightgbm", "model_params": {"num_leaves": 63, "learning_rate": 0.05}},
    },
    {
        "id": "night_min_child_40",
        "kind": "hyperparams",
        "title": "Сильнее регуляризация",
        "rationale": "Против переобучения.",
        "params": {"model": "lightgbm", "model_params": {"min_child_samples": 40, "reg_lambda": 5}},
    },
    {
        "id": "night_dart_like",
        "kind": "hyperparams",
        "title": "Малый шаг",
        "rationale": "Плавнее обучение.",
        "params": {"model": "lightgbm", "model_params": {"learning_rate": 0.02, "n_estimators": 900}},
    },
]
EVIL_HYPS = [
    {
        "id": "../../etc/passwd",
        "kind": "hyperparams",
        "title": "x" * 500,
        "rationale": "y",
        "params": {"model": "lightgbm", "model_params": {"n_estimators": -5}},
    },
    {
        "id": "huge",
        "kind": "hyperparams",
        "title": "огромные числа",
        "rationale": "z",
        "params": {"model": "lightgbm", "model_params": {"n_estimators": 10**9, "num_leaves": 10**7}},
    },
    {
        "id": "rm_rf",
        "kind": "model",
        "title": "неизвестный ключ",
        "rationale": "z",
        "params": {"model": "lightgbm", "run_shell": "rm -rf /", "model_params": {"max_depth": "десять"}},
    },
]


def purpose_of(prompt: str) -> str:
    low = prompt.lower()
    if "add_features" in prompt:
        return "features"
    if "критик" in low and "гипотез" in low and "отклон" in low:
        return "critic"
    if "target" in low and "метрик" in low and "id_column" in low:
        return "task"
    if "почему" in low and "пачк" in low:
        return "reflect"
    return "hypotheses"


def answer(mode: str, purpose: str) -> str | None:
    if mode == "garbage":
        return "Конечно! Вот мои идеи: попробуйте больше деревьев и нормализацию. Удачи!"
    if mode == "badjson":
        return '[{"id": "a", "kind": "hyperparams", "title": "обрыв", "params": {"model": "lightg'
    if mode == "empty":
        return ""
    if purpose == "task":
        return json.dumps({"target": None, "id_column": None, "metric": "totally_unknown_metric"})
    if purpose == "reflect":
        return "Пачка не сработала: слишком мало данных."
    if purpose == "critic":
        return json.dumps({"rejected": []})
    if purpose == "features":
        return json.dumps(
            {
                "id": f"night_{mode}",
                "title": "ночной признак",
                "rationale": "проверка",
                "code": FEATURE_CODES.get(mode, FEATURE_CODES["valid"]),
            }
        )
    if mode == "hyp_evil":
        return json.dumps(EVIL_HYPS)
    return json.dumps(GOOD_HYPS)


class Handler(BaseHTTPRequestHandler):
    mode = "valid"
    delay = 0.0

    def log_message(self, *a):  # тихо
        pass

    def do_GET(self):  # /v1/models
        self._send(200, {"object": "list", "data": [{"id": "fake", "object": "model"}]})

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
        prompt = "\n".join(str(m.get("content", "")) for m in body.get("messages", []))
        purpose = purpose_of(prompt)
        print(
            f"{time.strftime('%H:%M:%S')} {purpose:10} mode={self.mode} stream={bool(body.get('stream'))}", flush=True
        )
        if self.mode == "error500":
            return self._send(500, {"error": {"message": "internal error", "type": "server_error"}})
        if self.mode in ("slow", "hang"):
            time.sleep(self.delay)
        text = answer(self.mode, purpose)
        usage = {"prompt_tokens": 1000, "completion_tokens": 200, "total_tokens": 1200, "cost": 0.05}
        if body.get("stream"):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            chunk = {
                "id": "c",
                "object": "chat.completion.chunk",
                "model": "fake",
                "choices": [{"index": 0, "delta": {"content": text}, "finish_reason": "stop"}],
            }
            self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
            last = {"id": "c", "object": "chat.completion.chunk", "model": "fake", "choices": [], "usage": usage}
            self.wfile.write(f"data: {json.dumps(last)}\n\ndata: [DONE]\n\n".encode())
            return
        self._send(
            200,
            {
                "id": "c",
                "object": "chat.completion",
                "model": "fake",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
                "usage": usage,
            },
        )

    def _send(self, code, obj):
        data = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="valid")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--delay", type=float, default=0.0)
    args = ap.parse_args()
    Handler.mode, Handler.delay = args.mode, args.delay
    print(f"fake LLM mode={args.mode} delay={args.delay} port={args.port}", file=sys.stderr, flush=True)
    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
