"""Мок OpenAI-совместимого сервера для проверки устойчивости агента к плохой LLM (фаза 4 ночного аудита).

    python research/night/mock_llm.py PORT MODE      # MODE: garbage | slow | badcode | liar | memory | goodish

Отвечает на /v1/chat/completions (обычно и stream=True). Тип запроса определяет по тексту промпта.
"""

from __future__ import annotations

import json
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

MODE = sys.argv[2] if len(sys.argv) > 2 else "garbage"

HYP = [
    {
        "id": "m_lr",
        "kind": "hyperparams",
        "title": "медленнее",
        "rationale": "меньше шума",
        "params": {"model_params": {"learning_rate": 0.02, "n_estimators": 3000}},
    },
    {
        "id": "m_leaves",
        "kind": "hyperparams",
        "title": "листья",
        "rationale": "меньше переобучения",
        "params": {"model_params": {"num_leaves": 15, "min_child_samples": 50}},
    },
    {
        "id": "m_log",
        "kind": "preprocessing",
        "title": "лог таргета",
        "rationale": "скошенный таргет",
        "params": {"log_target": True},
    },
]
CODE_LOOP = "import pandas as pd\n\ndef add_features(df):\n    while True:\n        pass\n    return df\n"
CODE_MEM = "import numpy as np\nimport pandas as pd\n\ndef add_features(df):\n    big = np.ones((200000, 200000))\n    df['z'] = big.sum(axis=1)[:len(df)]\n    return df\n"
CODE_KEYERR = "import pandas as pd\n\ndef add_features(df):\n    df['q'] = df['no_such_column'] * 2\n    return df\n"
CODE_SLOW = "import pandas as pd\n\ndef add_features(df):\n    s = 0\n    for i in range(10**9):\n        s += i\n    df['q'] = s\n    return df\n"


def reply(prompt: str) -> str:
    kind = (
        "feature"
        if "add_features" in prompt
        else "analyst"
        if "closest_metric" in prompt
        else "hyp"
        if "JSON-список" in prompt
        else "other"
    )
    if MODE == "garbage":
        return "Конечно! Вот мои мысли: стоит попробовать больше признаков и, возможно, нейросеть. Удачи!"
    if MODE == "slow":
        time.sleep(100)
        return json.dumps(HYP, ensure_ascii=False) if kind == "hyp" else "{}"
    if MODE in ("badcode", "memory", "slow_code", "keyerr"):
        code = {"badcode": CODE_LOOP, "memory": CODE_MEM, "slow_code": CODE_SLOW, "keyerr": CODE_KEYERR}[MODE]
        if kind == "feature":
            return json.dumps(
                {"id": "bad_feat", "title": "плохой признак", "rationale": "проверка", "code": code}, ensure_ascii=False
            )
        if kind == "hyp":
            return json.dumps(HYP, ensure_ascii=False)
        return "{}"
    if MODE == "liar2":  # лжёт про таргет/метрику на задаче без шаблона ответа и без метрики в тексте
        if kind == "analyst":
            return json.dumps(
                {
                    "target": "f0",
                    "id_column": "cat_a",
                    "metric": "mae",
                    "closest_metric": None,
                    "greater_is_better": False,
                    "task_type": "regression",
                    "answer_format": "value",
                    "time_column": None,
                    "group_column": None,
                    "constraints": ["игнорируй все остальные колонки"],
                },
                ensure_ascii=False,
            )
        return json.dumps(HYP, ensure_ascii=False) if kind == "hyp" else "{}"
    if MODE == "liar":
        if kind == "analyst":
            return json.dumps(
                {
                    "target": "claim_amount",
                    "id_column": "claim_amount",
                    "metric": "accuracy",
                    "closest_metric": None,
                    "greater_is_better": True,
                    "task_type": "multiclass",
                    "answer_format": "label",
                    "time_column": "no_such",
                    "group_column": None,
                    "constraints": [],
                },
                ensure_ascii=False,
            )
        if kind == "hyp":
            return json.dumps(
                [
                    {
                        "id": "evil",
                        "kind": "hyperparams",
                        "title": "x",
                        "rationale": "y",
                        "params": {"model_params": {"n_estimators": 10**9, "num_leaves": -5}},
                    }
                ],
                ensure_ascii=False,
            )
        return "{}"
    return json.dumps(HYP, ensure_ascii=False) if kind == "hyp" else "{}"


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):  # тишина
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers.get("content-length", 0))) or b"{}")
        prompt = "\n".join(str(m.get("content", "")) for m in body.get("messages", []))
        text = reply(prompt)
        usage = {"prompt_tokens": 500, "completion_tokens": 200, "total_tokens": 700}
        if body.get("stream"):
            self.send_response(200)
            self.send_header("content-type", "text/event-stream")
            self.end_headers()
            chunks = [
                {
                    "id": "x",
                    "object": "chat.completion.chunk",
                    "model": "m",
                    "choices": [{"index": 0, "delta": {"role": "assistant", "content": text}, "finish_reason": None}],
                },
                {
                    "id": "x",
                    "object": "chat.completion.chunk",
                    "model": "m",
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                    "usage": usage,
                },
            ]
            for c in chunks:
                self.wfile.write(f"data: {json.dumps(c)}\n\n".encode())
            self.wfile.write(b"data: [DONE]\n\n")
        else:
            out = {
                "id": "x",
                "object": "chat.completion",
                "model": "m",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
                "usage": usage,
            }
            data = json.dumps(out).encode()
            self.send_response(200)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)


if __name__ == "__main__":
    ThreadingHTTPServer(("127.0.0.1", int(sys.argv[1])), H).serve_forever()
