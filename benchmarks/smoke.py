#!/usr/bin/env python3
"""Functional smoke test for the TP3+EP3 deployment.

Checks:
  * /health and /v1/models
  * 3 arithmetic prompts (catches the MARLIN EP top-k bug)
  * one tool call
  * one streaming request
Optional:
  * repeat-prefix TTFT (prefix cache)
  * long-context needle (slow)

Exit code 0 only if every non-optional check passes.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request


def request(args, path: str, payload: dict | None = None, stream: bool = False):
    url = args.base_url.rstrip("/") + path
    headers = {"Content-Type": "application/json"}
    if args.api_key:
        headers["Authorization"] = f"Bearer {args.api_key}"
    data = json.dumps(payload).encode() if payload is not None else None
    return urllib.request.urlopen(urllib.request.Request(url, data=data, headers=headers), timeout=1800)


def chat(args, messages, **kwargs) -> dict:
    payload = {"model": args.model, "messages": messages, "temperature": 0, **kwargs}
    with request(args, "/chat/completions", payload) as resp:
        return json.loads(resp.read())


def text_of(obj: dict) -> str:
    msg = obj["choices"][0]["message"]
    return (msg.get("content") or "") + (msg.get("reasoning_content") or "")


def check_arithmetic(args) -> bool:
    cases = [(17, 23, 391), (48, 37, 1776), (123, 456, 56088)]
    ok = True
    for a, b, expected in cases:
        obj = chat(args, [{"role": "user", "content": f"{a} * {b} = ? Answer with the number only."}],
                   max_tokens=32)
        got = text_of(obj).strip()
        passed = str(expected) in got
        print(f"[{'PASS' if passed else 'FAIL'}] {a}*{b}: {got[:80]!r}")
        ok &= passed
    return ok


def check_tool_call(args) -> bool:
    tools = [{
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the weather for a city",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    }]
    obj = chat(
        args,
        [{"role": "user", "content": "What is the weather in Shanghai? Use the get_weather tool."}],
        max_tokens=128,
        tools=tools,
        tool_choice="auto",
    )
    calls = obj["choices"][0]["message"].get("tool_calls") or []
    passed = bool(calls) and calls[0].get("function", {}).get("name") == "get_weather"
    print(f"[{'PASS' if passed else 'FAIL'}] tool_call: {calls[:1]}")
    return passed


def check_streaming(args) -> bool:
    payload = {
        "model": args.model,
        "messages": [{"role": "user", "content": "Count from 1 to 5."}],
        "max_tokens": 32,
        "temperature": 0,
        "stream": True,
    }
    chunks = 0
    with request(args, "/chat/completions", payload, stream=True) as resp:
        for raw in resp:
            if raw.strip().startswith(b"data:"):
                chunks += 1
    passed = chunks > 2
    print(f"[{'PASS' if passed else 'FAIL'}] streaming: {chunks} chunks")
    return passed


def check_prefix_cache(args) -> bool:
    prompt = "The following is a repeated system prefix. " * 200 + "\nReply with OK."
    messages = [{"role": "user", "content": prompt}]
    t0 = time.time()
    chat(args, messages, max_tokens=8)
    first = time.time() - t0
    t0 = time.time()
    chat(args, messages, max_tokens=8)
    second = time.time() - t0
    passed = second <= first * 1.05
    print(f"[{'PASS' if passed else 'WARN'}] prefix repeat: first {first:.2f}s, second {second:.2f}s")
    return True  # informational; small prefixes may not hit


def check_long_context(args) -> bool:
    needle = "QWEN38-NEEDLE-7F3A"
    filler = "The quick brown fox jumps over the lazy dog. " * 24000
    prompt = f"Read the text and reply with the exact token.\n{filler}\nToken: {needle}\nQuestion: what is the token?"
    t0 = time.time()
    obj = chat(args, [{"role": "user", "content": prompt}], max_tokens=32)
    got = text_of(obj)
    passed = needle in got
    print(f"[{'PASS' if passed else 'FAIL'}] long context ~288K: {time.time() - t0:.1f}s, {got[:80]!r}")
    return passed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:12345/v1")
    parser.add_argument("--model", default="qwen38-flash-next-nvfp4")
    parser.add_argument("--api-key", default="")
    parser.add_argument("--long-context", action="store_true")
    parser.add_argument("--prefix-cache", action="store_true")
    args = parser.parse_args()

    with request(args, "/health") as resp:
        print(f"[PASS] health: {resp.status}")
    with request(args, "/models") as resp:
        models = json.loads(resp.read())
        print(f"[PASS] models: {[m['id'] for m in models.get('data', [])]}")

    ok = True
    ok &= check_arithmetic(args)
    ok &= check_tool_call(args)
    ok &= check_streaming(args)
    if args.prefix_cache:
        check_prefix_cache(args)
    if args.long_context:
        ok &= check_long_context(args)

    print("SMOKE " + ("PASS" if ok else "FAIL"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()