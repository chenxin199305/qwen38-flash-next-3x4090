#!/usr/bin/env python3
"""OpenAI Python client example. pip install openai"""

import os

from openai import OpenAI

client = OpenAI(
    base_url=os.environ.get("VLLM_BASE_URL", "http://127.0.0.1:12345/v1"),
    api_key=os.environ.get("VLLM_API_KEY", "local-vllm"),
)

response = client.chat.completions.create(
    model=os.environ.get("MODEL_ID", "qwen38-flash-next-nvfp4"),
    messages=[
        {"role": "system", "content": "You are a concise deployment assistant."},
        {"role": "user", "content": "Why does EP3 avoid the MoE 640 to 768 padding?"},
    ],
    max_tokens=256,
    temperature=0,
)
print(response.choices[0].message.content)