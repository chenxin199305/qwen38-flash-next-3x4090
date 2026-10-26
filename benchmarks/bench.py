#!/usr/bin/env python3
"""Single-stream streaming benchmark.

Counts `usage.completion_tokens` (not SSE chunks) so speculative decoding is
measured correctly.  Outputs JSON with TTFT and decode tok/s.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
import urllib.request


def run_once(args) -> dict:
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
    if first is None:
        raise RuntimeError("no tokens received")
    tokens = usage_tokens or chunks
    return {
        "ttft_s": round(first - t0, 3),
        "completion_tokens": tokens,
        "decode_s": round(last - first, 3),
        "decode_tok_s": round(tokens / (last - first), 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:12345/v1")
    parser.add_argument("--model", default="qwen38-flash-next-nvfp4")
    parser.add_argument("--api-key", default="")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--prompt", default="Write a short technical note about local LLM inference.")
    parser.add_argument("--ignore-eos", action="store_true")
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    for _ in range(args.warmup):
        run_once(args)
    runs = [run_once(args) for _ in range(args.runs)]
    rates = [r["decode_tok_s"] for r in runs]
    summary = {
        "base_url": args.base_url,
        "model": args.model,
        "max_tokens": args.max_tokens,
        "runs": runs,
        "median_decode_tok_s": round(statistics.median(rates), 1),
        "min_decode_tok_s": min(rates),
        "max_decode_tok_s": max(rates),
    }
    text = json.dumps(summary, ensure_ascii=False, indent=2)
    print(text)
    if args.output:
        with open(args.output, "w") as f:
            f.write(text + "\n")


if __name__ == "__main__":
    main()