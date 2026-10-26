"""Поддельный OpenAI-совместимый сервер для проверки устойчивости агента к плохой LLM (ночной аудит, фаза 4).

    uv run python research/night/fake_llm.py <режим> <порт>
    OPENAI_BASE_URL=http://127.0.0.1:<порт>/v1 OPENAI_API_KEY=fake OPENAI_MODEL=fake python solution.py task.md

Режимы:
  valid      — осмысленные ответы (аналитик, 3 гипотезы, код признаков)
  garbage    — текст без JSON
  badcode    — JSON по схеме, но внутри мусор: несуществующая модель, кривые params, код признаков с вечным циклом,
               с чтением файлов, с огромной памятью, с синтаксической ошибкой
  trickle    — потоковый ответ по 1 символу раз в 3 с (бесконечно долгий ответ «медленной» LLM)
  hang       — соединение принято, ответа нет 15 минут
  e500       — HTTP 500 на каждый запрос
  disconnect — соединение закрывается без ответа
Каждый запрос пишется в stderr: время, режим, о чём спросили.
"""

from __future__ import annotations

import json
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from itertools import count

MODE = sys.argv[1] if len(sys.argv) > 1 else "valid"
PORT = int(sys.argv[2]) if len(sys.argv) > 2 else 8765
_counter = count(1)

ANALYST_OK = {"target": None, "id_column": None, "metric": None, "greater_is_better": None, "constraints": []}
HYPS_OK = [
    {
        "id": "llm_l1",
        "kind": "hyperparams",
        "title": "меньше листьев",
        "rationale": "шум",
        "params": {"model": "lightgbm", "model_params": {"num_leaves": 15, "min_child_samples": 40}},
    },
    {
        "id": "llm_xgb",
        "kind": "model",
        "title": "xgboost",
        "rationale": "другое семейство",
        "params": {"model": "xgboost"},
    },
    {
        "id": "llm_lr",
        "kind": "hyperparams",
        "title": "медленный шаг",
        "rationale": "точнее",
        "params": {"model": "lightgbm", "model_params": {"learning_rate": 0.02}},
    },
]
HYPS_BAD = [
    {"id": "bad_model", "kind": "model", "title": "суперсеть", "rationale": "x", "params": {"model": "superboost9000"}},
    {"id": "bad_folds", "kind": "hyperparams", "title": "фолды", "rationale": "x", "params": {"n_folds": "five"}},
    {
        "id": "bad_leaves",
        "kind": "hyperparams",
        "title": "огромные деревья",
        "rationale": "x",
        "params": {
            "model_params": {
                "num_leaves": 1000000,
                "min_child_samples": 1,
                "n_estimators": 100000,
                "learning_rate": 0.0001,
            }
        },
    },
    {
        "id": "bad_valid",
        "kind": "hyperparams",
        "title": "схема",
        "rationale": "x",
        "params": {"validation": "time:no_such_column"},
    },
    {
        "id": "bad_optuna",
        "kind": "hyperparams",
        "title": "optuna навсегда",
        "rationale": "x",
        "params": {"optuna": {"trials": 100000, "timeout": 100000}},
    },
    {
        "id": "bad_param_type",
        "kind": "hyperparams",
        "title": "строка вместо словаря",
        "rationale": "x",
        "params": {"model_params": "много деревьев"},
    },
    {"kind": "hyperparams", "title": "без id"},
    "просто строка",
]
CODES_BAD = [
    "def add_features(df):\n    while True:\n        pass\n    return df",
    "def add_features(df):\n    x = open('/etc/hostname').read()\n    df['h'] = len(x)\n    return df",
    "def add_features(df):\n    import numpy as np\n    big = np.ones((200000, 200000))\n    df['b'] = big[0, 0]\n    return df",
    "def add_features(df):\n    return df[",
    "def add_features(df):\n    return df.iloc[:10]",
]
CODE_OK = (
    "def add_features(df):\n    import numpy as np\n    num = df.select_dtypes('number')\n"
    "    df['row_mean'] = num.mean(axis=1)\n    df['row_std'] = num.std(axis=1)\n    return df"
)


# Инженер данных (agent/data_engineer.py): первый план — с ошибкой (проверка обратной связи), дальше — рабочий
DE_CODE_BROKEN = "def prepare(df):\n    return df['нет_такой_колонки']"
DE_CODE_OK = (
    "def prepare(df):\n    import numpy as np\n    out = df.copy()\n"
    "    for c in list(out.columns):\n"
    "        if out[c].dtype == object and out[c].nunique() > 0.9 * len(out):\n"
    "            out = out.drop(columns=c)\n"
    "    num = out.select_dtypes('number')\n"
    "    if num.shape[1] > 1:\n"
    "        out['row_mean'] = num.mean(axis=1)\n"
    "    return out"
)
_de_rounds = count(1)


def kind_of(messages: list[dict]) -> str:
    text = " ".join(str(m.get("content", "")) for m in messages)
    if "Выбери вариант для ответа" in text:
        return "verdict"
    if "инженер данных" in text:
        return "data_engineer"
    if "add_features" in text:
        return "features"
    if "что именно надо предсказать" in text or "answer_format" in text:
        return "analyst"
    if "критик" in text.lower() or "critic" in text.lower():
        return "critic"
    if "Верни JSON-список" in text:
        return "hypotheses"
    return "other"


def answer(kind: str, n: int) -> str:
    if MODE == "garbage":
        return "Думаю, стоит попробовать бустинг!!! {не json: [1, 2,"
    if MODE == "badcode":
        if kind == "features":
            code = CODES_BAD[n % len(CODES_BAD)]
            return json.dumps({"id": f"bad_code_{n}", "title": "плохой код", "rationale": "x", "code": code})
        if kind == "hypotheses":
            return "```json\n" + json.dumps(HYPS_BAD, ensure_ascii=False) + "\n```"
        if kind == "analyst":
            return json.dumps(
                {
                    "target": "нет_такой_колонки",
                    "id_column": 42,
                    "metric": "super_metric",
                    "greater_is_better": "да",
                    "task_type": "regression",
                    "constraints": "x",
                }
            )
        return "ok"
    if kind == "data_engineer":
        k = next(_de_rounds)
        code = DE_CODE_BROKEN if k == 1 else DE_CODE_OK
        model = "lightgbm" if k <= 2 else "xgboost"
        plan = "убрать идентификаторы-строки, добавить среднее числовых признаков строки"
        return json.dumps({"plan": plan, "code": code, "model": model, "model_params": {}}, ensure_ascii=False)
    if kind == "verdict":
        return json.dumps({"choice": "qwen_data_2", "conclusion": "Подготовка данных дала устойчивый выигрыш."})
    if kind == "features":
        return json.dumps({"id": f"row_stats_{n}", "title": "статистики строки", "rationale": "x", "code": CODE_OK})
    if kind == "hypotheses":
        return json.dumps(HYPS_OK, ensure_ascii=False)
    if kind == "analyst":
        return json.dumps(ANALYST_OK)
    return "Пачка не сработала: гипотезы слишком похожи на бейзлайн."


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):  # тише стандартного лога
        pass

    def do_POST(self):  # noqa: N802
        n = next(_counter)
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length) or b"{}")
        kind = kind_of(body.get("messages", []))
        stream = bool(body.get("stream"))
        print(f"{time.strftime('%H:%M:%S')} #{n} {MODE} {kind} stream={stream}", file=sys.stderr, flush=True)
        if MODE == "disconnect":
            self.close_connection = True
            self.connection.close()
            return
        if MODE == "hang":
            time.sleep(900)
            return
        if MODE == "e500":
            data = b'{"error": {"message": "internal error"}}'
            self.send_response(500)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return
        text = answer(kind, n)
        usage = {"prompt_tokens": 1000, "completion_tokens": 500, "total_tokens": 1500}
        if not stream:
            data = json.dumps(
                {
                    "id": f"c{n}",
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": "fake",
                    "choices": [
                        {"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}
                    ],
                    "usage": usage,
                }
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()

        def send(obj) -> None:
            payload = ("data: " + (obj if isinstance(obj, str) else json.dumps(obj)) + "\n\n").encode()
            self.wfile.write(f"{len(payload):X}\r\n".encode() + payload + b"\r\n")
            self.wfile.flush()

        def chunk(content: str | None, finish: str | None = None) -> dict:
            delta = {"content": content} if content is not None else {}
            return {
                "id": f"c{n}",
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": "fake",
                "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
            }

        try:
            if MODE == "trickle":
                for ch in "x" * 100000:  # бесконечно долгий ответ: символ раз в 3 с
                    send(chunk(ch))
                    time.sleep(3)
            for i in range(0, len(text), 40):
                send(chunk(text[i : i + 40]))
            send(chunk(None, "stop"))
            send(
                {
                    "id": f"c{n}",
                    "object": "chat.completion.chunk",
                    "created": 0,
                    "model": "fake",
                    "choices": [],
                    "usage": usage,
                }
            )
            send("[DONE]")
            self.wfile.write(b"0\r\n\r\n")
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            print(f"{time.strftime('%H:%M:%S')} #{n} клиент оборвал поток", file=sys.stderr, flush=True)


if __name__ == "__main__":
    print(f"fake LLM: режим {MODE}, порт {PORT}", file=sys.stderr, flush=True)
    ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
