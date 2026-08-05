"""Run the item set against a local Ollama model, once with thinking on and once off.

Everything the analysis needs is written to data/raw/<model>__<condition>.jsonl,
so the analysis, the tests and the independent checker all run on cached bytes and
need no GPU.

Two things this guards against:

1. A silently ignored `think` flag. Ollama accepts unknown fields without
   complaining, and the OpenAI compatible endpoint drops `think` entirely. If the
   flag did nothing, both conditions would be the same run and the experiment
   would produce a beautiful null result that means nothing. So every response
   records the raw `thinking` payload, and `analyze` refuses to report a model
   whose two conditions do not differ in whether that payload is present.

2. Silent truncation. `done_reason` is recorded per response, and a response that
   hit the token cap without emitting an answer is counted as unusable rather
   than as wrong.
"""

from __future__ import annotations

import argparse
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path

from .items import build_items

OLLAMA = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
NUM_PREDICT = 4096
TEMPERATURE = 0.0
ROOT = Path(__file__).resolve().parents[2]


def raw_path(model: str, think: bool) -> Path:
    safe = model.replace(":", "-").replace("/", "-")
    cond = "think-on" if think else "think-off"
    return ROOT / "data" / "raw" / f"{safe}__{cond}.jsonl"


def chat(model: str, prompt: str, think: bool, timeout: int = 600) -> dict:
    payload = {
        "model": model,
        "think": think,
        "stream": False,
        "options": {
            "temperature": TEMPERATURE,
            "top_k": 1,
            "seed": 7,
            "num_predict": NUM_PREDICT,
        },
        "messages": [{"role": "user", "content": prompt}],
    }
    req = urllib.request.Request(
        f"{OLLAMA}/api/chat",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    started = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    body["_wall_s"] = time.time() - started
    return body


def run(model: str, think: bool, limit: int | None = None, resume: bool = True) -> Path:
    items = build_items()
    if limit:
        items = items[:limit]
    out = raw_path(model, think)
    out.parent.mkdir(parents=True, exist_ok=True)
    done: set[str] = set()
    if resume and out.exists():
        for line in out.read_text(encoding="utf-8").splitlines():
            if line.strip():
                done.add(json.loads(line)["item_id"])
    todo = [it for it in items if it["id"] not in done]
    print(f"{model} think={think}: {len(todo)} to run, {len(done)} cached", flush=True)
    with out.open("a", encoding="utf-8") as fh:
        for n, item in enumerate(todo, 1):
            body = chat(model, item["prompt"], think)
            msg = body.get("message", {}) or {}
            rec = {
                "item_id": item["id"],
                "type": item["type"],
                "model": model,
                "think_requested": think,
                "content": msg.get("content", "") or "",
                "thinking": msg.get("thinking", "") or "",
                "done_reason": body.get("done_reason", ""),
                "eval_count": body.get("eval_count", 0) or 0,
                "prompt_eval_count": body.get("prompt_eval_count", 0) or 0,
                "total_duration_ns": body.get("total_duration", 0) or 0,
                "eval_duration_ns": body.get("eval_duration", 0) or 0,
                "load_duration_ns": body.get("load_duration", 0) or 0,
                "wall_s": body["_wall_s"],
                "num_predict": NUM_PREDICT,
                "temperature": TEMPERATURE,
            }
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fh.flush()
            if n % 10 == 0 or n == len(todo):
                print(f"  {n}/{len(todo)} {item['id']} {rec['eval_count']}tok "
                      f"{rec['wall_s']:.1f}s", flush=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--think", choices=["on", "off", "both"], default="both")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--no-resume", action="store_true")
    args = ap.parse_args()
    conds = {"on": [True], "off": [False], "both": [False, True]}[args.think]
    for think in conds:
        path = run(args.model, think, args.limit, resume=not args.no_resume)
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
