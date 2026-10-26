#!/usr/bin/env python3
"""Concurrent streaming benchmark.

Each request counts `usage.completion_tokens`; aggregate throughput is total
completion tokens divided by wall time.  Outputs JSON per concurrency level.
"""

from __future__ import annotations

import argparse
import json
import statistics
import threading
import time
import urllib.request


def one_stream(args, results: list, idx: int) -> None:
    payload = {
        "model": args.model,
        "messages": [{"role": "user", "content": args.prompt}],
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if args.ignore_eos:
        payload["ignore_eos"] = True
    headers = {"Content-Type": "application/json"}
    if args.api_key:
        headers["Authorization"] = f"Bearer {args.api_key}"
    req = urllib.request.Request(
        args.base_url.rstrip("/") + "/chat/completions",
        data=json.dumps(payload).encode(),
        headers=headers,
    )
    t0 = time.time()
    first = None
    last = t0
    chunks = 0
    usage_tokens = None
    try:
        with urllib.request.urlopen(req, timeout=1800) as resp:
            for raw in resp:
                line = raw.strip()
                if not line.startswith(b"data:"):
                    continue
                data = line[5:].strip()
                if data == b"[DONE]":
                    break
                try:
                    obj = json.loads(data)
                except json.JSONDecodeError:
                    continue
                if obj.get("usage"):
                    usage_tokens = obj["usage"].get("completion_tokens") or usage_tokens
                choices = obj.get("choices") or []
                if not choices:
                    continue
                delta = choices[0].get("delta") or {}
                piece = delta.get("content") or delta.get("reasoning_content") or delta.get("reasoning")
                if piece:
                    now = time.time()
                    if first is None:
                        first = now
                    chunks += 1
                    last = now
    except Exception as exc:  # noqa: BLE001
        results[idx] = {"error": str(exc)}
        return
    if first is None:
        results[idx] = {"error": "no tokens"}
        return
    tokens = usage_tokens or chunks
    results[idx] = {
        "ttft_s": round(first - t0, 3),
        "completion_tokens": tokens,
        "decode_tok_s": round(tokens / (last - first), 1),
    }


def bench(args, level: int) -> dict:
    results: list = [None] * level
    threads = [threading.Thread(target=one_stream, args=(args, results, i)) for i in range(level)]
    wall0 = time.time()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall = time.time() - wall0
    ok = [r for r in results if r and "error" not in r]
    errors = [r for r in results if not r or "error" in r]
    total_tokens = sum(r["completion_tokens"] for r in ok)
    return {
        "concurrency": level,
        "ok": len(ok),
        "errors": len(errors),
        "wall_s": round(wall, 3),
        "aggregate_tok_s": round(total_tokens / wall, 1) if ok else 0,
        "ttft_s": [min(r["ttft_s"] for r in ok), max(r["ttft_s"] for r in ok)] if ok else [],
        "per_stream_tok_s": [min(r["decode_tok_s"] for r in ok), max(r["decode_tok_s"] for r in ok)] if ok else [],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:12345/v1")
    parser.add_argument("--model", default="qwen38-flash-next-nvfp4")
    parser.add_argument("--api-key", default="")
    parser.add_argument("--levels", default="1,4,8,16")
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--prompt", default="Write a short technical note about local LLM inference.")
    parser.add_argument("--ignore-eos", action="store_true")
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    levels = [int(x) for x in args.levels.split(",")]
    results = [bench(args, n) for n in levels]
    summary = {
        "base_url": args.base_url,
        "model": args.model,
        "max_tokens": args.max_tokens,
        "levels": results,
        "median_aggregate_tok_s": round(statistics.median([r["aggregate_tok_s"] for r in results]), 1),
    }
    text = json.dumps(summary, ensure_ascii=False, indent=2)
    print(text)
    if args.output:
        with open(args.output, "w") as f:
            f.write(text + "\n")


if __name__ == "__main__":
    main()