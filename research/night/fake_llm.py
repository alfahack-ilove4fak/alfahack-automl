"""Ночной аудит: поддельный OpenAI-совместимый сервер для проверки устойчивости агента к плохой LLM.

    uv run python research/night/fake_llm.py --port 8765 --mode garbage
    OPENAI_BASE_URL=http://127.0.0.1:8765/v1 OPENAI_API_KEY=x OPENAI_MODEL=fake NO_PROXY=127.0.0.1 ...

Режимы: valid | garbage (не JSON) | slow (ждёт --delay с до ответа) | error500 | loopcode (код признаков —
бесконечный цикл) | memhog (код признаков съедает память) | badparams (гипотезы с кривыми params).
Каждый запрос пишется в stdout одной строкой (что спросили, что ответили).
"""

from __future__ import annotations

import argparse
import json
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

VALID_HYPS = [
    {
        "id": "fake_lr",
        "kind": "hyperparams",
        "title": "Меньше шаг",
        "rationale": "проверка",
        "params": {"model": "lightgbm", "model_params": {"learning_rate": 0.03}},
    },
    {
        "id": "fake_leaves",
        "kind": "hyperparams",
        "title": "Меньше листьев",
        "rationale": "проверка",
        "params": {"model": "lightgbm", "model_params": {"num_leaves": 15}},
    },
    {"id": "fake_linear", "kind": "model", "title": "Линейная", "rationale": "проверка", "params": {"model": "linear"}},
]
BAD_HYPS = [
    {"id": "bad_folds", "kind": "hyperparams", "title": "x", "rationale": "x", "params": {"n_folds": "много"}},
    {"id": "bad_model", "kind": "model", "title": "x", "rationale": "x", "params": {"model": "superboost"}},
    {
        "id": "bad_lr",
        "kind": "hyperparams",
        "title": "x",
        "rationale": "x",
        "params": {"model": "lightgbm", "model_params": {"learning_rate": -1, "num_leaves": 0}},
    },
    {"id": "bad_ops", "kind": "preprocessing", "title": "x", "rationale": "x", "params": {"feature_ops": ["rm_rf"]}},
    {
        "id": "bad_valid",
        "kind": "hyperparams",
        "title": "x",
        "rationale": "x",
        "params": {"validation": "time:нет_такой"},
    },
]
LOOP_CODE = (
    "import math\ndef add_features(df):\n    x = 0\n    while True:\n        x = math.sqrt(x + 1)\n    return df"
)
MEM_CODE = "import numpy as np\ndef add_features(df):\n    big = [np.ones(10**8) for _ in range(1000)]\n    return df"


class Handler(BaseHTTPRequestHandler):
    mode = "valid"
    delay = 0.0

    def log_message(self, *args):  # тихо
        pass

    def do_POST(self):  # noqa: N802
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
        text = " ".join(str(m.get("content", "")) for m in body.get("messages", []))
        purpose = "features" if "add_features" in text else "hypotheses" if "JSON-список" in text else "other"
        mode = self.mode
        if mode == "error500":
            self.send_response(500)
            self.end_headers()
            self.wfile.write(b'{"error": "boom"}')
            print(json.dumps({"purpose": purpose, "answer": "500"}), flush=True)
            return
        if mode == "slow":
            time.sleep(self.delay)
        if mode == "garbage":
            answer = "Извините, я не могу помочь с этим. ```python\nprint('hi')\n``` {broken json"
        elif purpose == "features":
            code = (
                LOOP_CODE
                if mode == "loopcode"
                else MEM_CODE
                if mode == "memhog"
                else (
                    "import numpy as np\ndef add_features(df):\n    num = df.select_dtypes('number')\n"
                    "    df['row_mean_fake'] = num.mean(axis=1)\n    return df"
                )
            )
            answer = json.dumps({"id": f"feat_{mode}", "title": "тест", "rationale": "тест", "code": code})
        elif purpose == "hypotheses":
            answer = json.dumps(BAD_HYPS if mode == "badparams" else VALID_HYPS, ensure_ascii=False)
        else:
            answer = "{}"
        print(json.dumps({"purpose": purpose, "mode": mode, "answer": answer[:80]}, ensure_ascii=False), flush=True)
        if body.get("stream"):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            for i in range(0, len(answer), 50):
                chunk = {
                    "id": "x",
                    "object": "chat.completion.chunk",
                    "created": 0,
                    "model": "fake",
                    "choices": [{"index": 0, "delta": {"content": answer[i : i + 50]}, "finish_reason": None}],
                }
                self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
            end = {
                "id": "x",
                "object": "chat.completion.chunk",
                "created": 0,
                "model": "fake",
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                "usage": {
                    "prompt_tokens": len(text) // 4,
                    "completion_tokens": len(answer) // 4,
                    "total_tokens": (len(text) + len(answer)) // 4,
                },
            }
            self.wfile.write(f"data: {json.dumps(end)}\n\ndata: [DONE]\n\n".encode())
            return
        resp = {
            "id": "x",
            "object": "chat.completion",
            "created": 0,
            "model": "fake",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": answer}, "finish_reason": "stop"}],
            "usage": {
                "prompt_tokens": len(text) // 4,
                "completion_tokens": len(answer) // 4,
                "total_tokens": (len(text) + len(answer)) // 4,
            },
        }
        data = json.dumps(resp).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--mode", default="valid")
    ap.add_argument("--delay", type=float, default=300)
    args = ap.parse_args()
    Handler.mode, Handler.delay = args.mode, args.delay
    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
